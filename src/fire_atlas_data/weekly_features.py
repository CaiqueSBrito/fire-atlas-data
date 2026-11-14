"""Causal detection-history features; no outcome columns participate in computation."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .normalization import sha256

CELL_WINDOWS = (3, 7, 14, 30, 60, 90)
RING_WINDOWS = (3, 7, 30)
ORIGINAL_FEATURE_NAMES = (
    'detection_rows_on_issue', 'cell_detection_rows_7d', 'regional_detection_rows_7d',
)
FEATURE_NAMES = ORIGINAL_FEATURE_NAMES + tuple(
    name for window in CELL_WINDOWS
    for name in (f'cell_trailing_detection_rows_{window}d', f'cell_active_days_{window}d')
) + (
    'days_since_last_detection', 'history_days', 'history_incomplete_90d',
    'latitude', 'longitude', 'doy_sin', 'doy_cos',
) + tuple(
    name for window in RING_WINDOWS
    for name in (f'ring1_detection_rows_{window}d', f'ring1_active_cell_days_{window}d')
) + (
    'ring1_neighbor_count', 'regional_detection_rows_30d',
    'historical_cell_month_active_fraction', 'historical_cell_month_support_days',
)
FEATURE_UNITS = {
    name: ('detection rows' if 'detection_rows' in name else
           'active cell-days' if 'active_cell_days' in name else
           'days' if ('days' in name or 'active_days' in name) else
           'degrees north' if name == 'latitude' else
           'degrees east' if name == 'longitude' else
           'cells' if name == 'ring1_neighbor_count' else 'dimensionless')
    for name in FEATURE_NAMES
}


def _panel(con, grid):
    """Validate the calendar without inspecting labels or observation status."""
    con.read_parquet(str(grid)).create_view('grid')
    schema = {row[0]: row[1] for row in con.execute('DESCRIBE grid').fetchall()}
    required = set(ORIGINAL_FEATURE_NAMES) | {
        'cell_id', 'issue_date', 'target_start_date', 'target_end_date', 'horizon_days',
    }
    if not required <= schema.keys():
        raise ValueError(f'Missing grid columns: {sorted(required - schema.keys())}')
    if any(name in schema for name in FEATURE_NAMES[len(ORIGINAL_FEATURE_NAMES):]):
        raise ValueError('Grid already contains expanded feature columns')
    if any(schema[name] != 'DATE' for name in
           ('issue_date', 'target_start_date', 'target_end_date')):
        raise ValueError('Grid issue and target boundaries must be DATE columns')
    invalid = con.execute('''SELECT count(*) FROM grid WHERE
        cell_id IS NULL OR issue_date IS NULL OR horizon_days IS NULL OR horizon_days != 7
        OR target_start_date IS NULL OR target_end_date IS NULL
        OR target_start_date != issue_date + 1 OR target_end_date != issue_date + 7
        OR detection_rows_on_issue IS NULL OR NOT isfinite(detection_rows_on_issue)
        OR detection_rows_on_issue < 0 OR detection_rows_on_issue != trunc(detection_rows_on_issue)
        ''').fetchone()[0]
    if invalid:
        raise ValueError('Invalid grid keys, horizon7 target bounds, or detection counts')
    rows, unique_rows, start, end = con.execute('''SELECT count(*),
        count(DISTINCT (issue_date,cell_id)),min(issue_date),max(issue_date) FROM grid''').fetchone()
    cells = [row[0] for row in con.execute('SELECT DISTINCT cell_id FROM grid ORDER BY cell_id').fetchall()]
    if not rows or any(not h3.is_valid_cell(cell) or h3.get_resolution(cell) != 5 for cell in cells):
        raise ValueError('Require nonempty H3 resolution5 panel')
    days = (end - start).days + 1
    if rows != unique_rows or rows != days * len(cells):
        raise ValueError('Require unique complete daily fixed-cell panel')
    # Carry outcomes unchanged, but they are never selected for a feature calculation.
    table = pq.read_table(grid).sort_by([('issue_date', 'ascending'), ('cell_id', 'ascending')])
    return table, cells, start, end


def _rolling(values, window):
    """Inclusive calendar rolling sum; the caller masks unsupported prefixes."""
    cumulative = np.empty((len(values) + 1, values.shape[1]), dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(values, axis=0, dtype=np.int64, out=cumulative[1:])
    result = cumulative[1:].copy()
    result[window:] -= cumulative[1:-window]
    return result


def _expand(table, cells, start, end):
    days, ncell = (end - start).days + 1, len(cells)
    counts = table['detection_rows_on_issue'].to_numpy().astype(np.int64, copy=False).reshape(days, ncell)
    active = counts > 0
    day_index = np.arange(days, dtype=np.int64)

    def append(name, values, mask=None):
        nonlocal table
        values = np.broadcast_to(values, (days, ncell)).reshape(-1)
        if mask is not None:
            mask = np.broadcast_to(mask, (days, ncell)).reshape(-1)
        table = table.append_column(name, pa.array(values, mask=mask))

    for window in CELL_WINDOWS:
        unsupported = (day_index < window - 1)[:, None]
        append(f'cell_trailing_detection_rows_{window}d', _rolling(counts, window), unsupported)
        append(f'cell_active_days_{window}d', _rolling(active, window), unsupported)
    last = np.maximum.accumulate(np.where(active, day_index[:, None], -1), axis=0)
    append('days_since_last_detection', day_index[:, None] - last, last < 0)
    append('history_days', (day_index + 1)[:, None])
    append('history_incomplete_90d', (day_index < 89).astype(np.int8)[:, None])
    coordinates = np.array([h3.cell_to_latlng(cell) for cell in cells])
    append('latitude', coordinates[:, 0])
    append('longitude', coordinates[:, 1])
    dates = [start + timedelta(days=int(day)) for day in day_index]
    phase = 2 * np.pi * np.array([day.timetuple().tm_yday - 1 for day in dates]) / 365.25
    append('doy_sin', np.sin(phase)[:, None])
    append('doy_cos', np.cos(phase)[:, None])
    positions = {cell: index for index, cell in enumerate(cells)}
    neighbors = [sorted(positions[other] for other in h3.grid_disk(cell, 1)
                        if other != cell and other in positions) for cell in cells]
    ring_counts = np.zeros_like(counts)
    ring_active = np.zeros_like(counts)
    for index, adjacent in enumerate(neighbors):
        ring_counts[:, index] = counts[:, adjacent].sum(axis=1)
        ring_active[:, index] = active[:, adjacent].sum(axis=1)
    for window in RING_WINDOWS:
        unsupported = (day_index < window - 1)[:, None]
        append(f'ring1_detection_rows_{window}d', _rolling(ring_counts, window), unsupported)
        append(f'ring1_active_cell_days_{window}d', _rolling(ring_active, window), unsupported)
    append('ring1_neighbor_count', np.array([len(adjacent) for adjacent in neighbors], dtype=np.int64))
    regional = _rolling(counts.sum(axis=1, keepdims=True), 30)
    append('regional_detection_rows_30d', regional, (day_index < 29)[:, None])
    # Update history only after computing every month in a complete issue year.
    # Partial source years contribute only their actually represented calendar days.
    support = np.zeros(12, dtype=np.int64)
    prior_active = np.zeros((12, ncell), dtype=np.int64)
    fractions = np.zeros((days, ncell), dtype=np.float64)
    supports = np.zeros((days, 1), dtype=np.int64)
    years = np.array([day.year for day in dates])
    months = np.array([day.month - 1 for day in dates])
    for year in np.unique(years):
        for month in range(12):
            selected = np.flatnonzero((years == year) & (months == month))
            supports[selected, 0] = support[month]
            if support[month]:
                fractions[selected] = prior_active[month] / support[month]
        for month in range(12):
            selected = (years == year) & (months == month)
            support[month] += selected.sum()
            prior_active[month] += active[selected].sum(axis=0)
    append('historical_cell_month_active_fraction', fractions, supports == 0)
    append('historical_cell_month_support_days', supports)
    return table


def build(grid, output_dir):
    """Publish features.parquet and manifest.json atomically, refusing overwrite.

    History starts at the first grid issue date, not an external source date.
    All added windows include D. Recency is zero on an active issue day and
    null until the first known detection. Original columns are passed through.
    """
    grid = Path(grid).resolve(strict=True)
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    grid_hash = sha256(grid)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}-', dir=target.parent))
    try:
        with duckdb.connect(str(stage / 'work.duckdb')) as con:
            con.execute("SET memory_limit='512MB'")
            con.execute('SET threads=2')
            table, cells, start, end = _panel(con, grid)
        (stage / 'work.duckdb').unlink()
        table = _expand(table, cells, start, end)
        pq.write_table(table, stage / 'features.parquet', compression='zstd', row_group_size=65536)
        if sha256(grid) != grid_hash:
            raise ValueError('Input grid changed during feature construction')
        manifest = dict(
            schema='weekly-features/1.0.0', feature_names=list(FEATURE_NAMES),
            feature_types={name: str(table.schema.field(name).type) for name in FEATURE_NAMES},
            feature_units=FEATURE_UNITS, grid_sha256=grid_hash, code_sha256=sha256(Path(__file__)),
            outputs=[dict(path='features.parquet', sha256=sha256(stage / 'features.parquet'))],
            rows=table.num_rows, cells=len(cells), cell_ids=cells, source_start=str(start),
            issue_dates=[str(start), str(end)], horizon_days=7,
            history_limits=dict(rolling_max_days=90, available_days=(end - start).days + 1,
                                climatology='only issue years strictly before current issue year'),
            feature_sources=['detection_rows_on_issue', 'issue_date', 'cell_id'],
            seasonality_period_days=365.25,
            limitations=[
                'History begins at earliest grid issue; pre-grid detections are unavailable.',
                'New rolling counts are null until their full calendar window is represented.',
                'Original three predictors are unchanged, including any initial left censoring.',
                'Ring1 excludes the focal cell and cells outside the fixed input grid.',
                'Calendar support does not establish satellite observability; zeros are not verified absence.',
                'Retrospective SP and unknown available_at do not establish operational availability.',
            ],
        )
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        stage.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', required=True)
    parser.add_argument('--output-dir', required=True)
    print(json.dumps(build(**vars(parser.parse_args(argv))), indent=2))


if __name__ == '__main__':
    main()
