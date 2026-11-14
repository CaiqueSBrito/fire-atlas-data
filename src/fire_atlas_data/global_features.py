"""Out-of-core, causal multisensor features for the global seven-day target."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import duckdb

from .normalization import identifier, literal, sha256
from .ranking import _json

METHOD_VERSION = '1.0.0'
HORIZON_DAYS = 7
WINDOWS = (1, 7, 30, 90, 365)
FEATURE_NAMES = tuple(
    f'{family}_{window}d'
    for family in ('cell_active_days', 'cell_viirs_detections', 'cell_modis_detections')
    for window in WINDOWS
) + tuple(
    f'cell_{sensor}_frp_mean_{window}d_mw'
    for sensor in ('viirs', 'modis') for window in (7, 30)
) + ('days_since_last_detection', 'days_since_first_detection') + tuple(
    f'{region}_active_cell_days_{window}d'
    for region in ('ring1', 'parent4', 'parent3') for window in (7, 30)
) + (
    'modis_prior_month_active_fraction', 'modis_prior_month_supported_days',
    'viirs_prior_month_active_fraction', 'viirs_prior_month_supported_days',
    'modis_prior_active_years', 'modis_prior_supported_years',
    'doy_sin', 'doy_cos', 'latitude', 'longitude',
    'viirs_platforms_7d', 'modis_platforms_7d',
)
LIMITATIONS = [
    'Support means global reporting days, not local clear-sky satellite coverage; zero counts are not verified absence of fire.',
    'Features use UTC acquisition dates through D, not operational availability timestamps.',
    'Labels are any non-static detection during D+1 through D+7, not ignition or burned area.',
    'Unknown FIRMS type outside the static mask can include unrecognized static sources.',
    'Sensor counts are not harmonized; overlapping platforms can detect the same fire.',
    'The frozen static mask must have reference_end on or before every issue date.',
]


def _stage(output_dir):
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, Path(tempfile.mkdtemp(prefix=f'.{target.name}-', dir=target.parent))


def _configure(con, stage, memory_limit, threads):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('threads must be a positive integer')
    con.execute(f'SET memory_limit={literal(memory_limit)}')
    con.execute(f'SET threads={threads}')
    con.execute(f"SET temp_directory={literal(str(stage / 'spill'))}")
    con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')


def _activity_outputs(root, manifest):
    outputs = manifest.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        raise ValueError('Activity manifest must declare hashed outputs')
    declared = set()
    for output in outputs:
        relative = Path(output['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in declared:
            raise ValueError('Activity output paths must be unique and relative')
        path = root / relative
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Activity output resolves outside its directory')
        if path.stat().st_size != output.get('size_bytes') or sha256(path) != output.get('sha256'):
            raise ValueError(f'Activity output integrity mismatch: {relative}')
        declared.add(relative.as_posix())
    parquet = {name for name in declared if name.endswith('.parquet')}
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*.parquet')}
    if parquet != actual:
        raise ValueError('Activity Parquet files must match declared outputs exactly')
    if not {'cells.parquet', 'neighbors.parquet', 'platform_days.parquet'} <= parquet:
        raise ValueError('Activity manifest is missing required output tables')
    daily = sorted(name for name in parquet if name.startswith('daily/'))
    if not daily:
        raise ValueError('Activity manifest must declare daily Parquet output')
    return '[' + ','.join(literal(str(root / name)) for name in daily) + ']'


def _activity(con, activity_dir):
    root = Path(activity_dir).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != 'global-cell-activity/1.0.0' or manifest.get('h3_resolution') != 5:
        raise ValueError('Require global-cell-activity/1.0.0 at H3 resolution 5')
    daily_paths = _activity_outputs(root, manifest)
    for name in ('cells', 'neighbors', 'platform_days'):
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet('
                    f'{literal(str(root / (name + ".parquet")))})')
    con.execute(f'CREATE VIEW daily AS SELECT * FROM read_parquet('
                f'{daily_paths}, hive_partitioning=true)')
    con.execute('''CREATE TABLE support AS SELECT acquisition_date,
        bool_or(sensor='VIIRS') AS viirs, bool_or(sensor='MODIS') AS modis
        FROM platform_days WHERE detections > 0 GROUP BY acquisition_date''')
    return manifest, {
        'manifest_sha256': sha256(root / 'manifest.json'),
        'mask_manifest_sha256': manifest['mask']['manifest_sha256'],
        'acquisition_range': manifest['acquisition_range'],
    }


def _asof(manifest, start, end):
    acquisition = manifest['acquisition_range']
    if start < date.fromisoformat(acquisition['min']) or end > date.fromisoformat(acquisition['max']):
        raise ValueError('Issue dates must be within the activity acquisition range')
    cutoff = date.fromisoformat(manifest['mask']['parameters']['reference_end'])
    if cutoff > start:
        raise ValueError('Static mask reference_end must be on or before the earliest issue date')


def _publish(stage, target, manifest, filename):
    (stage / 'work.duckdb').unlink()
    shutil.rmtree(stage / 'spill', ignore_errors=True)
    output = stage / filename
    manifest['outputs'] = [{'path': filename, 'size_bytes': output.stat().st_size,
                            'sha256': sha256(output)}]
    manifest['limitations'] = LIMITATIONS
    _json(stage / 'manifest.json', manifest)
    if target.exists():
        raise FileExistsError(target)
    os.rename(stage, target)
    return manifest


def _keys(con, keys, manifest, labels):
    con.execute(f'CREATE TABLE keys AS SELECT * FROM read_parquet({literal(str(keys))})')
    schema = dict(row[:2] for row in con.execute('DESCRIBE keys').fetchall())
    if schema.get('cell_id') != 'VARCHAR' or schema.get('issue_date') != 'DATE':
        raise ValueError('Keys require cell_id VARCHAR and issue_date DATE')
    rows, unique, start, end = con.execute('''SELECT count(*),
        count(DISTINCT (cell_id, issue_date)), min(issue_date), max(issue_date) FROM keys''').fetchone()
    invalid = con.execute('''SELECT count(*) FROM keys k LEFT JOIN cells c USING(cell_id)
        WHERE k.cell_id IS NULL OR k.issue_date IS NULL OR c.cell_id IS NULL
        OR c.first_detection_date IS NULL OR c.first_detection_date > k.issue_date''').fetchone()[0]
    if not rows or rows != unique or invalid:
        raise ValueError('Require nonempty unique, nonnull keys in the causally eligible cell domain')
    _asof(manifest, start, end)
    if labels and end + timedelta(days=HORIZON_DAYS) > date.fromisoformat(manifest['acquisition_range']['max']):
        raise ValueError('Labels require a complete seven-day future horizon')
    con.execute('CREATE TABLE issues AS SELECT DISTINCT issue_date FROM keys')
    if labels and con.execute('''SELECT count(*) FROM (
        SELECT i.issue_date FROM issues i LEFT JOIN support s
        ON s.acquisition_date BETWEEN i.issue_date+1 AND i.issue_date+7
        GROUP BY i.issue_date HAVING count(s.acquisition_date) != 7)''').fetchone()[0]:
        raise ValueError('Labels require reporting support on every future horizon day')
    return rows, start, end


def _support_features(con):
    fields = []
    for window in WINDOWS:
        predicate = f's.acquisition_date >= i.issue_date-{window - 1}'
        fields.append(f'count(*) FILTER (WHERE {predicate}) AS all_{window}')
        for sensor in ('viirs', 'modis'):
            fields.append(f'count(*) FILTER (WHERE {predicate} AND s.{sensor}) AS {sensor}_{window}')
    for sensor in ('modis', 'viirs'):
        fields.append(f'''count(*) FILTER (WHERE s.{sensor}
            AND year(s.acquisition_date)<year(i.issue_date)
            AND month(s.acquisition_date)=month(i.issue_date)) AS {sensor}_month_days''')
    fields.append('''count(DISTINCT year(s.acquisition_date)) FILTER (WHERE s.modis
        AND year(s.acquisition_date)<year(i.issue_date)) AS modis_years''')
    con.execute(f'''CREATE TABLE supported AS SELECT i.issue_date, {', '.join(fields)}
        FROM issues i LEFT JOIN support s ON s.acquisition_date<=i.issue_date GROUP BY i.issue_date''')
    con.execute('''CREATE TABLE platforms AS SELECT i.issue_date,
        count(DISTINCT p.platform) FILTER (WHERE p.sensor='VIIRS') AS viirs,
        count(DISTINCT p.platform) FILTER (WHERE p.sensor='MODIS') AS modis
        FROM issues i LEFT JOIN platform_days p
        ON p.acquisition_date BETWEEN i.issue_date-6 AND i.issue_date AND p.detections>0
        GROUP BY i.issue_date''')


def _cell_features(con):
    fields = []
    for window in WINDOWS:
        predicate = f'd.acquisition_date >= k.issue_date-{window - 1}'
        fields.append(f'count(d.acquisition_date) FILTER (WHERE {predicate}) AS active_{window}')
        for sensor in ('viirs', 'modis'):
            fields.append(f'coalesce(sum(d.{sensor}_detections) FILTER (WHERE {predicate}),0) AS {sensor}_{window}')
    for sensor in ('viirs', 'modis'):
        for window in (7, 30):
            predicate = f'd.acquisition_date >= k.issue_date-{window - 1}'
            fields.append(f'''sum(d.{sensor}_frp_sum_mw) FILTER (WHERE {predicate}) /
                nullif(sum(d.{sensor}_frp_rows) FILTER (WHERE {predicate}),0) AS {sensor}_frp_{window}''')
        fields.append(f'''count(*) FILTER (WHERE d.{sensor}_detections>0
            AND year(d.acquisition_date)<year(k.issue_date)
            AND month(d.acquisition_date)=month(k.issue_date)) AS {sensor}_month_active''')
    fields += [
        'date_diff(\'day\', max(d.acquisition_date), k.issue_date) AS recency',
        '''count(DISTINCT year(d.acquisition_date)) FILTER (WHERE d.modis_detections>0
            AND year(d.acquisition_date)<year(k.issue_date)) AS modis_active_years''',
    ]
    con.execute(f'''CREATE TABLE cell_features AS SELECT k.cell_id, k.issue_date, {', '.join(fields)}
        FROM keys k LEFT JOIN daily d ON d.cell_id=k.cell_id AND d.acquisition_date<=k.issue_date
        GROUP BY k.cell_id, k.issue_date''')


def _context_features(con):
    # Aggregate from the full activity domain, never the requested key subset.
    for parent in ('parent4', 'parent3'):
        con.execute(f'''CREATE TABLE {parent}_daily AS
            SELECT c.{parent}_id AS region_id, d.acquisition_date, count(*) AS active
            FROM daily d JOIN cells c USING(cell_id) GROUP BY ALL''')
        con.execute(f'''CREATE TABLE {parent}_features AS
            SELECT k.cell_id, k.issue_date,
            coalesce(sum(d.active) FILTER (WHERE d.acquisition_date>=k.issue_date-6),0) AS active_7,
            coalesce(sum(d.active),0) AS active_30
            FROM keys k JOIN cells c USING(cell_id) LEFT JOIN {parent}_daily d
            ON d.region_id=c.{parent}_id AND d.acquisition_date BETWEEN k.issue_date-29 AND k.issue_date
            GROUP BY k.cell_id, k.issue_date''')
    con.execute('''CREATE TABLE ring1_features AS SELECT k.cell_id, k.issue_date,
        count(d.acquisition_date) FILTER (WHERE d.acquisition_date>=k.issue_date-6) AS active_7,
        count(d.acquisition_date) AS active_30
        FROM keys k LEFT JOIN neighbors n ON n.cell_id=k.cell_id LEFT JOIN daily d
        ON d.cell_id=n.neighbor_id AND d.acquisition_date BETWEEN k.issue_date-29 AND k.issue_date
        GROUP BY k.cell_id, k.issue_date''')


def _feature_sql(labels):
    expressions = {}
    for window in WINDOWS:
        expressions[f'cell_active_days_{window}d'] = f'CASE WHEN s.all_{window}={window} THEN f.active_{window} END'
        for sensor in ('viirs', 'modis'):
            expressions[f'cell_{sensor}_detections_{window}d'] = (
                f'CASE WHEN s.{sensor}_{window}={window} THEN f.{sensor}_{window} END')
    for sensor in ('viirs', 'modis'):
        for window in (7, 30):
            expressions[f'cell_{sensor}_frp_mean_{window}d_mw'] = (
                f'CASE WHEN s.{sensor}_{window}={window} THEN f.{sensor}_frp_{window} END')
        expressions[f'{sensor}_prior_month_active_fraction'] = (
            f'f.{sensor}_month_active / nullif(s.{sensor}_month_days,0)')
        expressions[f'{sensor}_prior_month_supported_days'] = f's.{sensor}_month_days'
        expressions[f'{sensor}_platforms_7d'] = f'CASE WHEN s.all_7=7 THEN p.{sensor} END'
    for region in ('ring1', 'parent4', 'parent3'):
        for window in (7, 30):
            expressions[f'{region}_active_cell_days_{window}d'] = (
                f'CASE WHEN s.all_{window}={window} THEN {region}.active_{window} END')
    expressions.update({
        'days_since_last_detection': 'f.recency',
        'days_since_first_detection': "date_diff('day', c.first_detection_date, k.issue_date)",
        'modis_prior_active_years': 'CASE WHEN s.modis_years>0 THEN f.modis_active_years END',
        'modis_prior_supported_years': 's.modis_years',
        'doy_sin': 'sin(2*pi()*(dayofyear(k.issue_date)-1)/365.25)',
        'doy_cos': 'cos(2*pi()*(dayofyear(k.issue_date)-1)/365.25)',
        'latitude': 'c.latitude', 'longitude': 'c.longitude',
    })
    columns = ', '.join(f'({expressions[name]})::DOUBLE AS {identifier(name)}' for name in FEATURE_NAMES)
    outcomes = ', (l.target_active_days>0)::INTEGER AS label, l.target_active_days' if labels else ''
    label_join = 'JOIN labels l USING(cell_id, issue_date)' if labels else ''
    return f'''SELECT k.cell_id, k.issue_date, k.issue_date+1 AS target_start_date,
        k.issue_date+7 AS target_end_date, 7::INTEGER AS horizon_days, {columns}{outcomes}
        FROM keys k JOIN cells c USING(cell_id) JOIN cell_features f USING(cell_id,issue_date)
        JOIN supported s USING(issue_date) JOIN platforms p USING(issue_date)
        JOIN ring1_features ring1 USING(cell_id,issue_date)
        JOIN parent4_features parent4 USING(cell_id,issue_date)
        JOIN parent3_features parent3 USING(cell_id,issue_date) {label_join}
        ORDER BY k.issue_date, k.cell_id'''


def sample(activity_dir, output_dir, *, start: date, end: date, rate: float, seed: str = '42',
           memory_limit='2GB', threads=2) -> dict:
    """Uniform deterministic Bernoulli sampling of causally eligible cell-day keys."""
    if start > end or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError('Require start<=end and finite rate in (0,1]')
    target, stage = _stage(output_dir)
    try:
        with duckdb.connect(str(stage / 'work.duckdb')) as con:
            _configure(con, stage, memory_limit, threads)
            manifest, provenance = _activity(con, activity_dir)
            _asof(manifest, start, end)
            # Fixed-width date and H3 components make the seed-delimited key unambiguous.
            con.execute(f'''CREATE TABLE keys AS SELECT c.cell_id, d.issue_date::DATE AS issue_date
                FROM cells c CROSS JOIN generate_series(DATE {literal(start.isoformat())},
                    DATE {literal(end.isoformat())}, INTERVAL 1 DAY) d(issue_date)
                WHERE c.first_detection_date<=d.issue_date AND
                md5_number_lower(c.cell_id || '|' || d.issue_date::DATE::VARCHAR || '|' || {literal(str(seed))})
                    < {int(rate * (1 << 64))}''')
            rows, first, last = con.execute('SELECT count(*),min(issue_date),max(issue_date) FROM keys').fetchone()
            con.execute(f'''COPY (SELECT * FROM keys ORDER BY issue_date,cell_id)
                TO {literal(str(stage / 'keys.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)''')
        return _publish(stage, target, {
            'schema': 'global-weekly-keys/1.0.0', 'rows': rows,
            'issue_range': {'min': str(first) if first else None, 'max': str(last) if last else None},
            'parameters': {'start': start.isoformat(), 'end': end.isoformat(), 'rate': rate, 'seed': str(seed)},
            'activity': provenance,
        }, 'keys.parquet')
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build(activity_dir, keys, output_dir, *, labels: bool, memory_limit='2GB', threads=2) -> dict:
    """Build fixed-context features; read future activity only for separate optional labels."""
    if not isinstance(labels, bool):
        raise ValueError('labels must be a boolean')
    keys = Path(keys).resolve(strict=True)
    keys_hash = sha256(keys)
    target, stage = _stage(output_dir)
    try:
        with duckdb.connect(str(stage / 'work.duckdb')) as con:
            _configure(con, stage, memory_limit, threads)
            manifest, provenance = _activity(con, activity_dir)
            rows, start, end = _keys(con, keys, manifest, labels)
            _support_features(con)
            _cell_features(con)
            _context_features(con)
            positives = None
            if labels:
                con.execute('''CREATE TABLE labels AS SELECT k.cell_id, k.issue_date,
                    count(d.acquisition_date)::INTEGER AS target_active_days
                    FROM keys k LEFT JOIN daily d ON d.cell_id=k.cell_id
                    AND d.acquisition_date BETWEEN k.issue_date+1 AND k.issue_date+7
                    GROUP BY k.cell_id,k.issue_date''')
                positives = con.execute('SELECT count(*) FROM labels WHERE target_active_days>0').fetchone()[0]
            con.execute(f'''COPY ({_feature_sql(labels)}) TO {literal(str(stage / 'features.parquet'))}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)''')
        if sha256(keys) != keys_hash:
            raise ValueError('Keys changed during feature construction')
        return _publish(stage, target, {
            'schema': 'global-weekly-features/1.0.0', 'feature_names': list(FEATURE_NAMES),
            'feature_method_version': METHOD_VERSION,
            'labels': labels, 'rows': rows, 'positives': positives,
            'issue_range': {'min': start.isoformat(), 'max': end.isoformat()},
            'activity': provenance, 'keys_sha256': keys_hash,
        }, 'features.parquet')
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('sample', 'build'):
        sub = commands.add_parser(command)
        sub.add_argument('--activity-dir', required=True)
        sub.add_argument('--output-dir', required=True)
        sub.add_argument('--memory-limit', default='2GB')
        sub.add_argument('--threads', type=int, default=2)
        if command == 'sample':
            sub.add_argument('--start', type=date.fromisoformat, required=True)
            sub.add_argument('--end', type=date.fromisoformat, required=True)
            sub.add_argument('--rate', type=float, required=True)
            sub.add_argument('--seed', default='42')
        else:
            sub.add_argument('--keys', required=True)
            sub.add_argument('--labels', action=argparse.BooleanOptionalAction, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    print(json.dumps((sample if command == 'sample' else build)(**args), indent=2))


if __name__ == '__main__':
    main()
