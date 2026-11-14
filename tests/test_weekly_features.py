"""Small artificial panels test mechanics only, not scientific performance."""
from datetime import date, timedelta
import json

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data.normalization import sha256
from fire_atlas_data.weekly_features import build, FEATURE_NAMES, ORIGINAL_FEATURE_NAMES


def _cells():
    focal = h3.latlng_to_cell(-16, -47, 5)
    adjacent = sorted(set(h3.grid_disk(focal, 1)) - {focal})[0]
    isolated = h3.latlng_to_cell(10, 10, 5)
    return focal, adjacent, isolated


def _fixture(path, start=date(2020, 1, 1), days=95, events=None):
    events = events or {}
    rows = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        for cell in sorted(_cells()):
            rows.append(dict(
                cell_id=cell, issue_date=day, target_start_date=day + timedelta(days=1),
                target_end_date=day + timedelta(days=7), horizon_days=7,
                detection_rows_on_issue=events.get((cell, day), 0),
                # Deliberately unrelated constants prove these are carried through,
                # never recomputed, nor used to generate expanded history features.
                cell_detection_rows_7d=123, regional_detection_rows_7d=456,
                label=0, target_detection_rows=0, target_active_days=None,
                fire_observation_status='unknown', extra_original_column='preserved',
            ))
    pq.write_table(pa.Table.from_pylist(rows), path)
    return rows


def _build(path, output):
    manifest = build(path, output)
    table = pq.read_table(output / 'features.parquet')
    indexed = {(row['cell_id'], row['issue_date']): row for row in table.to_pylist()}
    return manifest, table, indexed


def test_calendar_boundaries_support_and_original_population(tmp_path):
    focal, _, isolated = _cells()
    start = date(2020, 1, 1)
    grid = tmp_path / 'grid.parquet'
    original = _fixture(grid, events={(focal, start): 2, (focal, start + timedelta(days=2)): 3})
    output = tmp_path / 'expanded'
    manifest, table, rows = _build(grid, output)
    assert table.num_rows == len(original)
    assert table.select(list(original[0])).equals(pq.read_table(grid))
    assert list(FEATURE_NAMES[:3]) == list(ORIGINAL_FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert manifest['feature_names'] == list(FEATURE_NAMES)
    assert manifest['feature_types'] == {name: str(table.schema.field(name).type) for name in FEATURE_NAMES}
    assert set(manifest['feature_units']) == set(FEATURE_NAMES)
    assert manifest['grid_sha256'] == sha256(grid)
    assert manifest['outputs'] == [{'path': 'features.parquet', 'sha256': sha256(output / 'features.parquet')}]
    assert json.loads((output / 'manifest.json').read_text()) == manifest
    first = rows[focal, start]
    assert first['cell_trailing_detection_rows_3d'] is None
    assert first['cell_detection_rows_7d'] == 123
    assert first['history_days'] == 1 and first['history_incomplete_90d'] == 1
    assert first['days_since_last_detection'] == 0
    third = rows[focal, start + timedelta(days=2)]
    assert third['cell_trailing_detection_rows_3d'] == 5
    assert third['cell_active_days_3d'] == 2
    fourth = rows[focal, start + timedelta(days=3)]
    assert fourth['cell_trailing_detection_rows_3d'] == 3
    assert fourth['cell_active_days_3d'] == 1
    assert fourth['days_since_last_detection'] == 1
    assert rows[isolated, start + timedelta(days=94)]['days_since_last_detection'] is None
    for window in (3, 7, 14, 30, 60, 90):
        name = f'cell_trailing_detection_rows_{window}d'
        assert rows[focal, start + timedelta(days=window - 2)][name] is None
        assert rows[focal, start + timedelta(days=window - 1)][name] == 5
    assert rows[focal, start + timedelta(days=89)]['history_incomplete_90d'] == 0
    with pytest.raises(FileExistsError):
        build(grid, output)


def test_future_and_outcome_mutations_do_not_change_past_features(tmp_path):
    focal, _, _ = _cells()
    grid = tmp_path / 'grid.parquet'
    original = _fixture(grid)
    _, before, _ = _build(grid, tmp_path / 'before')
    cutoff = date(2020, 2, 1)
    for row in original:
        if row['issue_date'] > cutoff and row['cell_id'] == focal:
            row['detection_rows_on_issue'] = 17
    pq.write_table(pa.Table.from_pylist(original), grid)
    _, after, _ = _build(grid, tmp_path / 'after')
    n = sum(row['issue_date'] <= cutoff for row in original)
    assert before.select(FEATURE_NAMES).slice(0, n).equals(after.select(FEATURE_NAMES).slice(0, n))
    for row in original:
        row['label'] = 1
        row['target_detection_rows'] = 999
        row['target_active_days'] = 7
        row['fire_observation_status'] = 'arbitrary technical mutation'
    pq.write_table(pa.Table.from_pylist(original), grid)
    _, mutated, _ = _build(grid, tmp_path / 'outcomes')
    assert after.select(FEATURE_NAMES).equals(mutated.select(FEATURE_NAMES))


def test_ring_excludes_focal_and_outside_grid_and_masks_short_history(tmp_path):
    focal, adjacent, isolated = _cells()
    start = date(2020, 1, 1)
    grid = tmp_path / 'grid.parquet'
    _fixture(grid, days=31, events={
        (focal, start): 2, (adjacent, start): 5,
        (adjacent, start + timedelta(days=1)): 7, (isolated, start): 11,
    })
    _, _, rows = _build(grid, tmp_path / 'expanded')
    assert rows[focal, start]['ring1_neighbor_count'] == 1
    assert rows[isolated, start]['ring1_neighbor_count'] == 0
    assert rows[focal, start]['ring1_detection_rows_3d'] is None
    third = rows[focal, start + timedelta(days=2)]
    assert third['ring1_detection_rows_3d'] == 12
    assert third['ring1_active_cell_days_3d'] == 2
    assert rows[adjacent, start + timedelta(days=2)]['ring1_detection_rows_3d'] == 2
    assert rows[isolated, start + timedelta(days=2)]['ring1_detection_rows_3d'] == 0
    assert rows[focal, start + timedelta(days=28)]['regional_detection_rows_30d'] is None
    assert rows[focal, start + timedelta(days=29)]['regional_detection_rows_30d'] == 25
    assert rows[focal, start + timedelta(days=30)]['regional_detection_rows_30d'] == 7
    assert third['latitude'] == h3.cell_to_latlng(focal)[0]
    assert third['longitude'] == h3.cell_to_latlng(focal)[1]


def test_month_history_uses_only_strictly_previous_years(tmp_path):
    focal, _, isolated = _cells()
    start, end = date(2019, 12, 30), date(2021, 1, 3)
    grid = tmp_path / 'grid.parquet'
    _fixture(grid, start, (end - start).days + 1, events={
        (focal, date(2019, 12, 30)): 3,
        (focal, date(2020, 1, 1)): 4,
        (focal, date(2020, 12, 1)): 8,
        (focal, date(2021, 1, 1)): 999,
    })
    _, _, rows = _build(grid, tmp_path / 'expanded')
    for day in (date(2020, 1, 1), date(2020, 1, 31)):
        assert rows[focal, day]['historical_cell_month_active_fraction'] is None
        assert rows[focal, day]['historical_cell_month_support_days'] == 0
    december = rows[focal, date(2020, 12, 31)]
    assert december['historical_cell_month_active_fraction'] == 0.5
    assert december['historical_cell_month_support_days'] == 2
    for day in (date(2021, 1, 1), date(2021, 1, 3)):
        assert rows[focal, day]['historical_cell_month_active_fraction'] == pytest.approx(1 / 31)
        assert rows[focal, day]['historical_cell_month_support_days'] == 31
        assert rows[isolated, day]['historical_cell_month_active_fraction'] == 0


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'gap', 'horizon', 'bounds', 'negative'])
def test_rejects_invalid_panel_without_publishing(tmp_path, mutation):
    grid = tmp_path / 'grid.parquet'
    rows = _fixture(grid, days=4)
    if mutation == 'duplicate':
        rows.append(rows[0].copy())
    elif mutation == 'missing':
        rows.pop(0)
    elif mutation == 'gap':
        rows = [row for row in rows if row['issue_date'] != date(2020, 1, 2)]
    elif mutation == 'horizon':
        rows[0]['horizon_days'] = 1
    elif mutation == 'bounds':
        rows[0]['target_end_date'] += timedelta(days=1)
    else:
        rows[0]['detection_rows_on_issue'] = -1
    pq.write_table(pa.Table.from_pylist(rows), grid)
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        build(grid, output)
    assert not output.exists()
