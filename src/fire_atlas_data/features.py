"""Fixed-geography, retrospective cell-day feature construction."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import glob
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3

from .normalization import literal, sha256


# Keep the original model order; enhanced consumers opt in explicitly.
BASE_FEATURES = [
    'seasonal_modis', 'seasonal_viirs', 'doy_sin', 'doy_cos',
    'neighbors_3d', 'neighbors_7d', 'days_since_last', 'regional_7d', 'regional_30d',
]
ENHANCED_FEATURES = BASE_FEATURES + [
    'cell_active_days_7d', 'cell_active_days_14d', 'cell_active_days_30d',
    'cell_detections_7d', 'cell_detections_30d', 'cell_prior_novel_events',
    'ring1_active_cells_1d', 'ring1_active_cells_3d', 'ring1_active_cells_7d',
    'ring1_active_cells_14d', 'ring1_active_cells_30d',
    'ring2_active_cells_3d', 'ring2_active_cells_7d',
    'ring2_active_cells_14d', 'ring2_active_cells_30d', 'days_since_neighbor',
    'nearby_frp_max_3d', 'nearby_frp_mean_3d', 'nearby_frp_max_7d',
    'nearby_frp_mean_7d', 'latitude', 'longitude', 'history_days',
]
HISTORY_FEATURES = ENHANCED_FEATURES + [
    'snpp_cell_active_days', 'snpp_cell_new_days', 'snpp_cell_detections',
    'snpp_days_since_last', 'snpp_ring1_active_cell_days',
    'snpp_ring2_active_cell_days', 'snpp_cell_frp_mean', 'snpp_cell_frp_max',
]


def _snpp_history(connection) -> None:
    """All supplied Suomi-NPP thermal anomalies through issue D, never targets."""
    connection.execute("""CREATE VIEW snpp_detections AS SELECT * FROM detections
        WHERE sensor='VIIRS' AND platform='Suomi-NPP' AND source_product='VIIRS_SNPP_SP'""")
    connection.execute("""CREATE TABLE snpp_daily AS
        SELECT cell_id,acquisition_date,count(*) detections
        FROM snpp_detections GROUP BY ALL""")
    connection.execute("""CREATE VIEW snpp_novel AS SELECT o.* FROM snpp_daily o
        WHERE NOT EXISTS (SELECT 1 FROM snpp_daily p WHERE p.cell_id=o.cell_id
        AND p.acquisition_date BETWEEN o.acquisition_date-3 AND o.acquisition_date-1)""")
    connection.execute("""CREATE TABLE with_history AS SELECT f.*,
        cell.active::DOUBLE snpp_cell_active_days,novel.n::DOUBLE snpp_cell_new_days,
        coalesce(cell.detections,0)::DOUBLE snpp_cell_detections,
        date_diff('day',cell.last_date,f.issue_date)::DOUBLE snpp_days_since_last,
        nearby.r1::DOUBLE snpp_ring1_active_cell_days,
        nearby.r2::DOUBLE snpp_ring2_active_cell_days,
        frp.mean::DOUBLE snpp_cell_frp_mean,frp.max::DOUBLE snpp_cell_frp_max
        FROM features f
        LEFT JOIN LATERAL (SELECT count(*) active,sum(o.detections) detections,
            max(o.acquisition_date) last_date FROM snpp_daily o
            WHERE o.cell_id=f.cell_id AND o.acquisition_date<=f.issue_date) cell ON true
        LEFT JOIN LATERAL (SELECT count(*) n FROM snpp_novel o
            WHERE o.cell_id=f.cell_id AND o.acquisition_date<=f.issue_date) novel ON true
        LEFT JOIN LATERAL (SELECT count(*) FILTER(WHERE n.is_ring1) r1,count(*) r2
            FROM snpp_daily o JOIN neighbors2 n ON n.neighbor_id=o.cell_id AND n.cell_id=f.cell_id
            WHERE o.acquisition_date<=f.issue_date) nearby ON true
        LEFT JOIN LATERAL (SELECT avg(o.frp_mw) mean,max(o.frp_mw) max
            FROM snpp_detections o WHERE o.cell_id=f.cell_id
            AND o.acquisition_date<=f.issue_date) frp ON true""")
    connection.execute('DROP TABLE features')
    connection.execute('ALTER TABLE with_history RENAME TO features')


def _dates(start: str, end: str):
    current, stop = date.fromisoformat(start), date.fromisoformat(end)
    while current <= stop:
        yield current
        current += timedelta(days=1)


def _grid(config: dict) -> list[str]:
    west, south, east, north = config['bbox']
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('bbox must be non-antimeridian west,south,east,north')
    polygon = {'type': 'Polygon', 'coordinates': [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}
    cells = sorted(h3.geo_to_cells(polygon, config['h3_resolution']))
    if not cells:
        raise ValueError('Declared bounding box contains no H3 cell centers')
    return cells


def _calendar(config: dict) -> list[tuple]:
    rows = []
    seen = set()
    splits = dict(config['splits'])
    if config.get('publish_date'):
        splits['publish'] = [config['publish_date'], config['publish_date']]
    for split, (start, end) in splits.items():
        if start > end:
            raise ValueError('Split start must precede end')
        for day in _dates(start, end):
            if day in seen:
                raise ValueError('Splits must not overlap')
            seen.add(day)
            target = day + timedelta(days=1)
            # Leap-year reference makes the circular seasonal calendar stable.
            seasonal_day = date(2000, target.month, target.day).timetuple().tm_yday
            angle = 2 * math.pi * seasonal_day / 366
            rows.append((day, target, split, math.sin(angle), math.cos(angle), seasonal_day))
    if not rows or not (config['splits']['train'][1] < config['splits']['validation'][0] <= config['splits']['validation'][1] < config['splits']['test'][0]):
        raise ValueError('Require chronologically ordered train, validation, test splits')
    return rows


def _enhance(connection, warmup: date) -> int:
    """Add NOAA-20 recurrence without altering baseline columns or eligibility."""
    connection.execute("""CREATE VIEW target_detections AS SELECT * FROM detections
        WHERE sensor='VIIRS' AND platform='NOAA-20' AND source_product='VIIRS_NOAA20_NRT'""")
    connection.execute("""CREATE TABLE target_daily AS
        SELECT cell_id,acquisition_date,count(*)::BIGINT detections
        FROM target_detections GROUP BY ALL""")
    connection.execute("""CREATE VIEW target_novel AS SELECT o.cell_id,o.acquisition_date
        FROM target_daily o WHERE NOT EXISTS (SELECT 1 FROM target_daily p
            WHERE p.cell_id=o.cell_id
            AND p.acquisition_date BETWEEN o.acquisition_date-3 AND o.acquisition_date-1)""")
    connection.execute(f"""CREATE TABLE enhanced AS SELECT f.*,
        cell.n7::DOUBLE cell_active_days_7d,cell.n14::DOUBLE cell_active_days_14d,
        cell.n30::DOUBLE cell_active_days_30d,
        coalesce(cell.d7,0)::DOUBLE cell_detections_7d,coalesce(cell.d30,0)::DOUBLE cell_detections_30d,
        novel.n::DOUBLE cell_prior_novel_events,
        nearby.r1_1::DOUBLE ring1_active_cells_1d,nearby.r1_3::DOUBLE ring1_active_cells_3d,
        nearby.r1_7::DOUBLE ring1_active_cells_7d,nearby.r1_14::DOUBLE ring1_active_cells_14d,
        nearby.r1_30::DOUBLE ring1_active_cells_30d,
        nearby.r2_3::DOUBLE ring2_active_cells_3d,nearby.r2_7::DOUBLE ring2_active_cells_7d,
        nearby.r2_14::DOUBLE ring2_active_cells_14d,nearby.r2_30::DOUBLE ring2_active_cells_30d,
        date_diff('day',last_neighbor.last_date,f.issue_date)::DOUBLE days_since_neighbor,
        frp.max3::DOUBLE nearby_frp_max_3d,frp.mean3::DOUBLE nearby_frp_mean_3d,
        frp.max7::DOUBLE nearby_frp_max_7d,frp.mean7::DOUBLE nearby_frp_mean_7d,
        c.latitude,c.longitude,
        least(30,date_diff('day',DATE {literal(str(warmup))},f.issue_date)+1)::INTEGER history_days
        FROM features f JOIN candidates c USING(cell_id)
        LEFT JOIN LATERAL (
            SELECT count(*) FILTER(WHERE o.acquisition_date>=f.issue_date-6) n7,
                count(*) FILTER(WHERE o.acquisition_date>=f.issue_date-13) n14,count(*) n30,
                sum(o.detections) FILTER(WHERE o.acquisition_date>=f.issue_date-6) d7,sum(o.detections) d30
            FROM target_daily o WHERE o.cell_id=f.cell_id
            AND o.acquisition_date BETWEEN f.issue_date-29 AND f.issue_date) cell ON true
        LEFT JOIN LATERAL (SELECT count(*) n FROM target_novel o
            WHERE o.cell_id=f.cell_id AND o.acquisition_date<=f.issue_date) novel ON true
        LEFT JOIN LATERAL (
            SELECT count(DISTINCT o.cell_id) FILTER(WHERE n.is_ring1 AND o.acquisition_date=f.issue_date) r1_1,
                count(DISTINCT o.cell_id) FILTER(WHERE n.is_ring1 AND o.acquisition_date>=f.issue_date-2) r1_3,
                count(DISTINCT o.cell_id) FILTER(WHERE n.is_ring1 AND o.acquisition_date>=f.issue_date-6) r1_7,
                count(DISTINCT o.cell_id) FILTER(WHERE n.is_ring1 AND o.acquisition_date>=f.issue_date-13) r1_14,
                count(DISTINCT o.cell_id) FILTER(WHERE n.is_ring1) r1_30,
                count(DISTINCT o.cell_id) FILTER(WHERE o.acquisition_date>=f.issue_date-2) r2_3,
                count(DISTINCT o.cell_id) FILTER(WHERE o.acquisition_date>=f.issue_date-6) r2_7,
                count(DISTINCT o.cell_id) FILTER(WHERE o.acquisition_date>=f.issue_date-13) r2_14,
                count(DISTINCT o.cell_id) r2_30
            FROM target_daily o JOIN neighbors2 n ON n.neighbor_id=o.cell_id AND n.cell_id=f.cell_id
            WHERE o.acquisition_date BETWEEN f.issue_date-29 AND f.issue_date) nearby ON true
        LEFT JOIN LATERAL (SELECT max(o.acquisition_date) last_date FROM target_daily o
            JOIN neighbors2 n ON n.neighbor_id=o.cell_id AND n.cell_id=f.cell_id
            WHERE o.acquisition_date<=f.issue_date) last_neighbor ON true
        LEFT JOIN LATERAL (
            SELECT max(o.frp_mw) FILTER(WHERE o.acquisition_date>=f.issue_date-2) max3,
                avg(o.frp_mw) FILTER(WHERE o.acquisition_date>=f.issue_date-2) mean3,
                max(o.frp_mw) max7,avg(o.frp_mw) mean7
            FROM target_detections o JOIN neighbors2 n ON n.neighbor_id=o.cell_id AND n.cell_id=f.cell_id
            WHERE o.acquisition_date BETWEEN f.issue_date-6 AND f.issue_date) frp ON true""")
    connection.execute('DROP TABLE features')
    connection.execute('ALTER TABLE enhanced RENAME TO features')
    return connection.execute('SELECT count(*) FROM features WHERE history_days<30').fetchone()[0]


def build(config_path: str | Path, output_dir: str | Path) -> dict:
    """Build all eligible candidates, never sampling controls or using future geometry."""
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    target_mode = config.get('target_mode', 'novel_activity')
    if target_mode not in ('novel_activity', 'any_activity'):
        raise ValueError('target_mode must be novel_activity or any_activity')
    sources = sorted({Path(p).resolve() for pattern in config['sources'] for p in glob.glob(pattern, recursive=True)})
    if not sources:
        raise ValueError('No normalized source Parquet files matched')
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    cells, calendar = _grid(config), _calendar(config)
    warmup = date.fromisoformat(config['warmup_start'])
    snpp_history = config.get('snpp_history', False)
    if not isinstance(snpp_history, bool):
        raise ValueError('snpp_history must be a boolean')
    enhanced = config.get('enhanced_features', False)
    if not isinstance(enhanced, bool):
        raise ValueError('enhanced_features must be a boolean')
    enhanced = enhanced or snpp_history
    minimum_history = config.get('minimum_history_days', 30)
    if isinstance(minimum_history, bool) or not isinstance(minimum_history, int) or minimum_history < 7:
        raise ValueError('minimum_history_days must be an integer >=7')
    if (min(r[0] for r in calendar) - warmup).days < minimum_history:
        raise ValueError(f'At least {minimum_history} days of acquisition warmup are required')
    source_metadata = [{'path': str(p), 'size_bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha256(p)} for p in sources]
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}-', dir=target.parent))
    connection = None
    try:
        connection = duckdb.connect(str(stage / 'work.duckdb'))
        connection.execute(f"SET memory_limit={literal(config.get('memory_limit', '512MB'))}")
        connection.execute(f"SET threads={int(config.get('threads', 2))}")
        connection.execute(f"SET temp_directory={literal(str(stage / 'spill'))}")
        connection.execute("SET TimeZone='UTC'")
        connection.execute('SET preserve_insertion_order=false')
        connection.execute('CREATE TABLE candidates(cell_id VARCHAR PRIMARY KEY, latitude DOUBLE, longitude DOUBLE, region VARCHAR)')
        connection.executemany('INSERT INTO candidates VALUES (?,?,?,?)', [(c, *h3.cell_to_latlng(c), config['region']) for c in cells])
        connection.execute('CREATE TABLE neighbors(cell_id VARCHAR, neighbor_id VARCHAR)')
        connection.executemany('INSERT INTO neighbors VALUES (?,?)', [(c, n) for c in cells for n in sorted(h3.grid_disk(c, 1)) if n != c])
        if enhanced:
            connection.execute('CREATE TABLE neighbors2(cell_id VARCHAR, neighbor_id VARCHAR, is_ring1 BOOLEAN)')
            edges = []
            for cell in cells:
                ring1 = set(h3.grid_disk(cell, 1))
                edges.extend((cell, neighbor, neighbor in ring1)
                             for neighbor in sorted(h3.grid_disk(cell, 2)) if neighbor != cell)
            connection.executemany('INSERT INTO neighbors2 VALUES (?,?,?)', edges)
        connection.execute('CREATE TABLE calendar(issue_date DATE, target_date DATE, split VARCHAR, doy_sin DOUBLE, doy_cos DOUBLE, seasonal_day INTEGER)')
        connection.executemany('INSERT INTO calendar VALUES (?,?,?,?,?,?)', calendar)
        paths = '[' + ','.join(literal(str(p)) for p in sources) + ']'
        west, south, east, north = config['bbox']
        west, south, east, north = west - 2, south - 2, east + 2, north + 2
        # Baseline halo is unchanged; enhanced reads include every radius-two cell.
        if config['h3_resolution'] < 4:
            raise ValueError('Prototype requires H3 resolution >=4 for its two-degree neighbor halo')
        if enhanced:
            boundary = [point for cell in {n for c in cells for n in h3.grid_disk(c, 2)}
                        for point in h3.cell_to_boundary(cell)]
            west, east = min(west, min(lon for lat, lon in boundary)), max(east, max(lon for lat, lon in boundary))
            south, north = min(south, min(lat for lat, lon in boundary)), max(north, max(lat for lat, lon in boundary))
        connection.execute(f"""CREATE VIEW source AS SELECT * FROM read_parquet({paths}, union_by_name=true)
            WHERE longitude BETWEEN {west} AND {east} AND latitude BETWEEN {south} AND {north}
            AND acquisition_date <= DATE {literal(str(max(r[1] for r in calendar)))}""")
        connection.create_function('coordinate_cell', lambda lat, lon: h3.latlng_to_cell(lat, lon, config['h3_resolution']), ['DOUBLE', 'DOUBLE'], 'VARCHAR')
        # H3 is called once per distinct coordinate, not once per observation.
        connection.execute('CREATE TABLE coordinates AS SELECT latitude,longitude,coordinate_cell(latitude,longitude) cell_id FROM (SELECT DISTINCT latitude,longitude FROM source)')
        # Exact semantic duplicates ignore file provenance and source row number.
        halo = 'SELECT neighbor_id FROM neighbors2' if enhanced else 'SELECT neighbor_id FROM neighbors'
        source_columns = {row[0] for row in connection.execute('DESCRIBE source').fetchall()}
        # Raw anomaly type is not filtered and participates in semantic identity.
        anomaly_type = (",json_extract_string(to_json(s.source_fields),'$.type') AS source_anomaly_type"
                        if snpp_history and 'source_fields' in source_columns else '')
        connection.execute(f'''CREATE TABLE detections AS SELECT DISTINCT c.cell_id,s.latitude,s.longitude,s.acquired_at,s.acquisition_date,s.sensor,s.platform,
            s.source_product,s.source_version,s.confidence_category,s.confidence_percent,s.brightness_i4_k,s.brightness_i5_k,
            s.brightness_k,s.bright_t31_k,s.frp_mw,s.scan_km,s.track_km,s.daynight{anomaly_type}
            FROM source s JOIN coordinates c USING(latitude,longitude)
            WHERE c.cell_id IN (SELECT cell_id FROM candidates UNION {halo})''')
        # Archive covariates must not change any preexisting logical features.
        history_exclusion = "WHERE sensor='VIIRS' AND platform='NOAA-20' AND source_product='VIIRS_NOAA20_NRT'" if snpp_history or target_mode == 'any_activity' else ''
        connection.execute(f'''CREATE TABLE daily AS SELECT cell_id,acquisition_date,sensor,platform,source_product,count(*)::BIGINT detections
            FROM detections {history_exclusion} GROUP BY ALL''')
        # Novel cell-days require three preceding inactive days on the same
        # sensor/platform, independent of product repackaging and pixel counts.
        connection.execute('''CREATE TABLE new_cell_days AS
            SELECT DISTINCT o.cell_id,o.acquisition_date,o.sensor,o.platform
            FROM daily o WHERE NOT EXISTS (
                SELECT 1 FROM daily p WHERE p.cell_id=o.cell_id AND p.sensor=o.sensor
                AND p.platform=o.platform AND p.acquisition_date BETWEEN o.acquisition_date-3 AND o.acquisition_date-1)''')
        connection.execute('''CREATE VIEW seasonal_daily AS SELECT DISTINCT cell_id,acquisition_date,sensor
            FROM new_cell_days''')
        minimum, maximum, unknown = connection.execute('SELECT min(acquisition_date),max(acquisition_date),count(*) FILTER(WHERE available_at IS NULL) FROM source').fetchone()
        required_end = max(r[0] if r[2] == 'publish' else r[1] for r in calendar)
        if minimum is None or minimum > warmup or maximum < required_end:
            raise ValueError('Source acquisition range does not cover warmup and all next-day labels')
        connection.execute("CREATE VIEW label_daily AS SELECT cell_id,acquisition_date,sum(detections) detections FROM daily WHERE sensor='VIIRS' AND platform='NOAA-20' AND source_product='VIIRS_NOAA20_NRT' GROUP BY ALL")
        eligibility = '' if target_mode == 'any_activity' else '''WHERE NOT EXISTS (
            SELECT 1 FROM label_daily o WHERE o.cell_id=c.cell_id
            AND o.acquisition_date BETWEEN d.issue_date-2 AND d.issue_date)'''
        connection.execute(f'''CREATE TABLE eligible AS
            SELECT c.*,d.* FROM candidates c CROSS JOIN calendar d {eligibility}''')
        connection.execute('''CREATE TABLE features AS SELECT e.cell_id,e.issue_date,e.target_date,e.region,e.split,e.doy_sin,e.doy_cos,
            coalesce(season.modis,0)::DOUBLE seasonal_modis,coalesce(season.viirs,0)::DOUBLE seasonal_viirs,
            coalesce(season.support,0)::INTEGER seasonal_support_years,
            coalesce(nearby.n3,0)::DOUBLE neighbors_3d,coalesce(nearby.n7,0)::DOUBLE neighbors_7d,
            date_diff('day',last_seen.last_date,e.issue_date)::DOUBLE days_since_last,
            coalesce(regional.r7,0)::DOUBLE regional_7d,coalesce(regional.r30,0)::DOUBLE regional_30d,
            CASE WHEN e.split='publish' THEN NULL WHEN EXISTS(SELECT 1 FROM label_daily y WHERE y.cell_id=e.cell_id AND y.acquisition_date=e.target_date) THEN 1 ELSE 0 END::INTEGER label
            FROM eligible e
            LEFT JOIN LATERAL (SELECT count(DISTINCT o.cell_id) FILTER(WHERE o.acquisition_date>=e.issue_date-2) n3,count(DISTINCT o.cell_id) n7
                FROM daily o JOIN neighbors n ON n.neighbor_id=o.cell_id AND n.cell_id=e.cell_id
                WHERE o.acquisition_date BETWEEN e.issue_date-6 AND e.issue_date) nearby ON true
            LEFT JOIN LATERAL (SELECT max(o.acquisition_date) last_date FROM daily o WHERE o.cell_id=e.cell_id AND o.acquisition_date<=e.issue_date) last_seen ON true
            LEFT JOIN LATERAL (SELECT count(DISTINCT o.cell_id) FILTER(WHERE o.acquisition_date>=e.issue_date-6) r7,count(DISTINCT o.cell_id) r30
                FROM daily o JOIN candidates c ON c.cell_id=o.cell_id WHERE o.acquisition_date BETWEEN e.issue_date-29 AND e.issue_date) regional ON true
            LEFT JOIN LATERAL (SELECT count(*) FILTER(WHERE o.sensor='MODIS')::DOUBLE modis,
                count(*) FILTER(WHERE o.sensor='VIIRS')::DOUBLE viirs,count(DISTINCT year(o.acquisition_date)) support
                FROM seasonal_daily o WHERE o.cell_id=e.cell_id AND year(o.acquisition_date)<year(e.issue_date)
                AND ((dayofyear(make_date(2000,month(o.acquisition_date),day(o.acquisition_date)))-e.seasonal_day+366+15)%366)<30) season ON true''')
        left_censored = _enhance(connection, warmup) if enhanced else 0
        if snpp_history:
            _snpp_history(connection)
        counts = connection.execute('SELECT split,count(*),sum(label),count(*) FILTER(WHERE seasonal_support_years>0) FROM features GROUP BY split ORDER BY split').fetchall()
        platforms = connection.execute('SELECT sensor,platform,source_product,count(*) FROM detections GROUP BY ALL ORDER BY 1,2,3').fetchall()
        for table, name in [('daily', 'daily.parquet'), ('candidates', 'candidates.parquet'), ('features', 'features.parquet')]:
            connection.execute(f"COPY {table} TO {literal(str(stage / name))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)")
        connection.close()
        connection = None
        (stage / 'work.duckdb').unlink()
        shutil.rmtree(stage / 'spill', ignore_errors=True)
        for item in source_metadata:
            current = Path(item['path']).stat()
            if (current.st_size, current.st_mtime_ns) != (item['size_bytes'], item['mtime_ns']):
                raise RuntimeError('Source changed while building features')
        manifest = {
            'method_version': '1.3.0' if target_mode == 'any_activity' else ('1.2.0' if snpp_history else ('1.1.0' if enhanced else '1.0.0')), 'config': config, 'config_sha256': sha256(config_path), 'sources': source_metadata,
            'engine': {'duckdb': duckdb.__version__, 'h3': h3.__version__}, 'candidate_count': len(cells),
            'target_product': 'VIIRS_NOAA20_NRT', 'target_platform': 'NOAA-20', 'target_sensor': 'VIIRS',
            'target_mode': target_mode,
            'target_definition': 'Any supplied NOAA-20 next-day detection, including continuing activity' if target_mode == 'any_activity' else 'Supplied NOAA-20 next-day detection with no detection on issue D, D-1 or D-2',
            'candidate_eligibility': 'All fixed-geography cells' if target_mode == 'any_activity' else 'Exclude cells active on issue D, D-1 or D-2',
            'feature_units': {'neighbors_3d': 'distinct active neighboring cells', 'neighbors_7d': 'distinct active neighboring cells',
                'regional_7d': 'distinct active candidate cells', 'regional_30d': 'distinct active candidate cells',
                'seasonal_modis': 'prior-year new MODIS cell-days', 'seasonal_viirs': 'prior-year new VIIRS cell-days'},
            'source_acquisition_range': [str(minimum), str(maximum)], 'unknown_available_at_rows_in_halo': unknown,
            'split_counts': {s: {'eligible': n, 'observed_next_day': p, 'historically_supported_rows': h} for s,n,p,h in counts},
            'platform_counts': [{'sensor': s, 'platform': p, 'source_product': product, 'detections': n} for s,p,product,n in platforms],
            'seasonal_baseline_supported': any(r[3] > 0 for r in counts),
            'outputs': [{'path': p.name, 'sha256': sha256(p), 'size_bytes': p.stat().st_size} for p in sorted(stage.glob('*.parquet'))],
            'limitations': [
                'Retrospective acquisition-time experiment only: available_at is unknown; no operational validity claim.',
                'Fixed bbox center-contained H3 candidates, not land-certified; no land mask is available.',
                'Zeros are unlabeled controls, not verified absence of fire. Labels are observed NOAA-20 next-day detections.',
                'No historical baseline claim without prior-year support. Seasonal zeros with support=0 mean unavailable history, not estimated risk.',
                'Seasonal values count prior-year new cell-days in a circular 30-day window [-15,+14] around target day on a 366-day calendar. Novelty requires three preceding inactive days per sensor/platform; family counts collapse same-day platform overlap.',
                'Neighbor and regional features count distinct active cells, not pixels or detection totals; neighbors exclude the focal cell.',
                'Acquisition range coverage does not establish complete daily satellite coverage. Missing detections can reflect observation gaps.',
                'days_since_last is null when no earlier supplied detection exists; history is left-censored.',
                'DuckDB spills to disk; H3 geometry and distinct-coordinate Python scalar UDF are bounded to declared geography; source and output hashing add streaming I/O.'
            ]
        }
        if enhanced:
            manifest['enhanced_features'] = True
            manifest['feature_columns'] = ENHANCED_FEATURES
            manifest['history'] = {
                'warmup_start': str(warmup), 'minimum_history_days': minimum_history,
                'maximum_window_days': 30, 'left_censored_30d_rows': left_censored,
                'definition': 'history_days=min(30,issue_date-warmup_start+1); calendar exposure, not certified coverage',
            }
            manifest['feature_units'].update({
                **{name: 'distinct supplied NOAA-20 active cell-days' for name in ENHANCED_FEATURES if name.startswith('cell_active_days_')},
                **{name: 'supplied NOAA-20 detections after semantic deduplication' for name in ENHANCED_FEATURES if name.startswith('cell_detections_')},
                **{name: 'distinct supplied NOAA-20 neighboring active cells' for name in ENHANCED_FEATURES if name.startswith(('ring1_', 'ring2_'))},
                **{name: 'MW; null when no non-null supplied FRP exists' for name in ENHANCED_FEATURES if name.startswith('nearby_frp_')},
                'cell_prior_novel_events': 'supplied NOAA-20 novel cell-days through issue date',
                'days_since_neighbor': 'days since last supplied radius-two NOAA-20 detection; nullable',
                'latitude': 'H3 center latitude degrees', 'longitude': 'H3 center longitude degrees',
                'history_days': 'calendar days since configured warmup, inclusive, capped at 30',
            })
            manifest['limitations'].extend([
                'Enhanced columns use only VIIRS NOAA-20 VIIRS_NOAA20_NRT. Radius two includes radius one and excludes the focal cell.',
                'Enhanced 30-day windows are left-censored when history_days<30; history_days does not certify complete satellite observation.',
                'Prior novel-event totals and elapsed times use all supplied history and are left-censored at its beginning; first observed events may have unknown antecedents.',
                'Nearby FRP summaries use supplied non-null detections within radius two, not cell-weighted averages; null is not replaced by zero.',
                'Cell-center coordinates describe this fixed known region; no geographic transfer claim is supported.',
            ])
        if snpp_history:
            manifest['snpp_history'] = True
            manifest['feature_columns'] = HISTORY_FEATURES
            manifest['feature_units'].update({
                'snpp_cell_active_days': 'supplied Suomi-NPP active cell-days through issue D',
                'snpp_cell_new_days': 'supplied Suomi-NPP cell-days with three preceding inactive days',
                'snpp_cell_detections': 'supplied Suomi-NPP detections after semantic deduplication',
                'snpp_days_since_last': 'days since supplied Suomi-NPP detection; nullable',
                'snpp_ring1_active_cell_days': 'neighbor cell-day pairs in disk one excluding center',
                'snpp_ring2_active_cell_days': 'neighbor cell-day pairs in disk two including disk one excluding center',
                'snpp_cell_frp_mean': 'MW; detection-weighted, nullable',
                'snpp_cell_frp_max': 'MW; nullable',
            })
            manifest['limitations'].append(
                'Suomi-NPP VIIRS_SNPP_SP scientific archive covariates use all supplied thermal-anomaly types '
                'without filtering, acquisition_date<=issue_date, and all supplied history (not warmup-truncated). '
                'They do not change NOAA-20 labels, eligibility, or original logical features. Novelty and elapsed '
                'times are left-censored; unknown available_at means retrospective use only.')
            manifest['limitations'].append(
                'In snpp_history mode existing daily/base features use only VIIRS NOAA-20 '
                'VIIRS_NOAA20_NRT, including last_seen; archive additions cannot fill missing '
                'baseline elapsed times or contaminate the frozen original nine-feature comparison.')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        os.rename(stage, target)
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
