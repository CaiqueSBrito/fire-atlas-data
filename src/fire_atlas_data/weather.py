"""Exact-cycle GFS D+1 covariates, with immutable raw Open-Meteo receipts."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import h3

from .normalization import literal, sha256

ENDPOINT = 'https://single-runs-api.open-meteo.com/v1/forecast'
MODEL = 'gfs_global'
ERA5_ENDPOINT = 'https://archive-api.open-meteo.com/v1/archive'
VARIABLES = ('temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m')
UNITS = ('°C', '%', 'mm', 'm/s')
WEATHER_FEATURES = ['wx_tmax_c', 'wx_rhmin_pct', 'wx_precip_mm', 'wx_windmax_ms', 'wx_valid_fraction']
ERA5_FEATURES = ['era5_tmax_c', 'era5_rhmin_pct', 'era5_precip_mm', 'era5_windmax_ms',
                 'era5_precip_7d_mm', 'era5_valid_fraction']
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_URL_BYTES = 6000
QUOTA_WINDOWS = ((60, 480), (3600, 4800), (86400, 9500))


def _json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _target_hours(issue: date) -> list[str]:
    target = issue + timedelta(days=1)
    return [f'{target.isoformat()}T{hour:02d}:00' for hour in range(24)]


def _sampling_node(latitude: float, longitude: float) -> tuple[str, float, float]:
    lat, lon = round(latitude * 4) / 4, round(longitude * 4) / 4
    return f'{lat:.2f},{lon:.2f}', lat, lon


def _request_url(issue: date, points: list[tuple]) -> str:
    if not points or len(points) > 64:
        raise ValueError('Each request requires 1..64 locations')
    for _, lat, lon in points:
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Invalid WGS84 cell center')
    params = {
        'latitude': ','.join(format(p[1], '.8f') for p in points),
        'longitude': ','.join(format(p[2], '.8f') for p in points),
        'run': f'{issue.isoformat()}T00:00', 'models': MODEL,
        'timezone': 'UTC', 'forecast_days': 2, 'hourly': ','.join(VARIABLES),
        'temperature_unit': 'celsius', 'precipitation_unit': 'mm', 'wind_speed_unit': 'ms',
        'timeformat': 'iso8601',
    }
    url = ENDPOINT + '?' + urlencode(params)
    if len(url.encode()) > MAX_URL_BYTES:
        raise ValueError('Request URL exceeds the bounded URL size')
    return url


def _aggregate(payload: dict, issue: date) -> tuple:
    """All 24 named target timestamps are required independently for each field."""
    if not isinstance(payload, dict) or payload.get('error'):
        raise ValueError('Weather response is not a successful location object')
    if payload.get('utc_offset_seconds') != 0:
        raise ValueError('Weather response must use UTC')
    hourly, units = payload.get('hourly'), payload.get('hourly_units')
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError('Missing hourly data or units')
    timestamps = hourly.get('time')
    if not isinstance(timestamps, list) or not timestamps:
        raise ValueError('Missing hourly timestamps')
    try:
        parsed = [datetime.strptime(t, '%Y-%m-%dT%H:%M') for t in timestamps]
    except (ValueError, TypeError):
        raise ValueError('Malformed hourly timestamp') from None
    if (any(t.minute != 0 for t in parsed) or len(set(timestamps)) != len(timestamps)
            or parsed != sorted(parsed)):
        raise ValueError('Hourly timestamps must be unique, increasing, whole hours')
    positions = {value: index for index, value in enumerate(timestamps)}
    indices = [positions.get(hour) for hour in _target_hours(issue)]
    aggregates = []
    valid = 0
    for variable, unit, reducer in zip(VARIABLES, UNITS, (max, min, sum, max)):
        values = hourly.get(variable)
        if units.get(variable) != unit:
            raise ValueError(f'Unexpected unit for {variable}')
        if not isinstance(values, list) or len(values) != len(timestamps):
            raise ValueError(f'Malformed hourly array for {variable}')
        for value in values:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f'Non-numeric or non-finite {variable}')
            if value is not None and ((variable == 'relative_humidity_2m' and not 0 <= value <= 100)
                                      or (variable in ('precipitation', 'wind_speed_10m') and value < 0)):
                raise ValueError(f'Out-of-domain {variable}')
        target_values = [values[index] if index is not None else None for index in indices]
        count = sum(value is not None for value in target_values)
        valid += count
        aggregates.append(float(reducer(target_values)) if count == 24 else None)
    return (*aggregates, valid / 96.0)


def _location_objects(payload, count: int) -> list[dict]:
    objects = payload if isinstance(payload, list) else [payload]
    if len(objects) != count or any(not isinstance(value, dict) for value in objects):
        raise ValueError('Response location count does not match request')
    # Open-Meteo omits the zero-valued location_id on the first object.
    if count > 1 and [value.get('location_id', 0 if index == 0 else None)
                      for index, value in enumerate(objects)] != list(range(count)):
        raise ValueError('Response location_id order does not match request')
    return objects


def _input_keys(features: Path, *, today: date | None = None,
                minimum: date = date(2026, 4, 2)) -> list[tuple]:
    # Project only request keys. Labels, split and all other data are never read.
    with duckdb.connect() as connection:
        keys = connection.execute(f'''SELECT DISTINCT cell_id, issue_date
            FROM read_parquet({literal(str(features))}) ORDER BY issue_date, cell_id''').fetchall()
    if not keys:
        raise ValueError('Feature input has no keys')
    today = today or datetime.now(timezone.utc).date()
    rows = []
    centers = {}
    for cell, issue in keys:
        if not isinstance(cell, str) or not h3.is_valid_cell(cell):
            raise ValueError('Feature input requires valid H3 cell IDs')
        if type(issue) is not date or not minimum <= issue < today:
            raise ValueError(f'Issue dates must be archived past DATE values, from {minimum}')
        if cell not in centers:
            centers[cell] = h3.cell_to_latlng(cell)
        rows.append((cell, issue, *centers[cell]))
    return rows


class _Quota:
    """Conservatively count each location as one call, including failed requests."""
    def __init__(self):
        self.events = deque()
        self.lock = threading.Lock()

    def reserve(self, locations: int) -> None:
        with self.lock:
            while True:
                now = time.monotonic()
                while self.events and self.events[0][0] <= now - 86400:
                    self.events.popleft()
                waits = []
                for seconds, limit in QUOTA_WINDOWS:
                    active = [(stamp, size) for stamp, size in self.events if stamp > now - seconds]
                    total = sum(size for _, size in active)
                    for stamp, size in active:
                        if total + locations <= limit:
                            break
                        waits.append(max(0.01, stamp + seconds - now + 0.1))
                        total -= size
                if not waits:
                    self.events.append((now, locations))
                    return
                time.sleep(max(waits))


def _fetch(job: tuple, stage: Path, quota: _Quota) -> tuple[list[tuple], dict]:
    issue, batch, points = job
    url = _request_url(issue, points)
    prefix = f'{issue.isoformat()}-{batch:03d}'
    quota.reserve(len(points))
    started = datetime.now(timezone.utc).isoformat()
    request = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'fire-atlas-data/0.1'})
    response = None
    status, headers, body, failure = None, {}, b'', None
    try:
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            response = error
        with response:
            status = response.status
            headers = dict(response.headers.items())
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            failure = 'Response exceeded bounded size; raw file is the received prefix'
        elif status != 200:
            failure = f'Weather source returned HTTP {status}'
    except Exception as error:
        failure = f'HTTPS acquisition failed: {type(error).__name__}: {error}'
    raw = stage / 'raw' / (prefix + '.json')
    with raw.open('xb') as stream:
        stream.write(body)
    receipt = {
        'url': url, 'http_status': status, 'response_headers': headers,
        'request_started_at_utc': started, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': MODEL, 'model_version': None, 'init_time_utc': f'{issue}T00:00:00+00:00',
        'run_semantics': 'UTC initialization, not publication', 'available_at': None,
        'target_valid_time_start_utc': _target_hours(issue)[0] + ':00+00:00',
        'target_valid_time_end_utc': _target_hours(issue)[-1] + ':00+00:00',
        'target_lead_hours': [24, 47], 'precipitation_interval': 'preceding hour at each target timestamp',
        'locations': [{'sampling_node': c, 'latitude': lat, 'longitude': lon} for c, lat, lon in points],
        'raw': {'path': str(raw.relative_to(stage)), 'size_bytes': len(body),
                'sha256': hashlib.sha256(body).hexdigest()},
        'failure': failure,
    }
    rows = []
    if failure is None:
        try:
            payload = json.loads(body)
            objects = _location_objects(payload, len(points))
            for point, obj in zip(points, objects):
                rows.append((point[0], issue, *_aggregate(obj, issue)))
            receipt['response_grid_centers'] = [
                {'latitude': obj.get('latitude'), 'longitude': obj.get('longitude'),
                 'elevation': obj.get('elevation')} for obj in objects]
        except (ValueError, TypeError, KeyError) as error:
            failure = f'Malformed weather data: {error}'
            receipt['failure'] = failure
    _json(stage / 'raw' / (prefix + '.receipt.json'), receipt)
    if failure is not None:
        raise ValueError(f'{failure}; raw receipt: {raw.parent / (prefix + ".receipt.json")}')
    return rows, receipt


def _reuse_index(directories: list[Path]) -> dict:
    index = {}
    for directory in directories:
        directory = directory.resolve(strict=True)
        for path in sorted((directory / 'raw').glob('*.receipt.json')):
            receipt = json.loads(path.read_text())
            if receipt.get('http_status') == 200 and receipt.get('failure') is None:
                raw = directory / receipt['raw']['path']
                if (raw.stat().st_size != receipt['raw']['size_bytes']
                        or sha256(raw) != receipt['raw']['sha256']):
                    raise ValueError(f'Reuse raw hash/size mismatch: {raw}')
                index[receipt['url']] = (path, raw, receipt)
    return index


def _fetch_or_reuse(job: tuple, stage: Path, quota: _Quota, reuse: dict) -> tuple[list[tuple], dict]:
    issue, batch, points = job
    cached = reuse.get(_request_url(issue, points))
    if cached is None:
        return _fetch(job, stage, quota)
    path, raw, original = cached
    objects = _location_objects(json.loads(raw.read_bytes()), len(points))
    rows = [(point[0], issue, *_aggregate(obj, issue)) for point, obj in zip(points, objects)]
    prefix = f'{issue.isoformat()}-{batch:03d}'
    destination = stage / 'raw' / (prefix + '.json')
    shutil.copyfile(raw, destination)
    receipt = dict(original)
    receipt['raw'] = {**original['raw'], 'path': str(destination.relative_to(stage))}
    receipt['reused_from'] = {'receipt_path': str(path), 'receipt_sha256': sha256(path),
                            'revalidated_at_utc': datetime.now(timezone.utc).isoformat()}
    _json(stage / 'raw' / (prefix + '.receipt.json'), receipt)
    return rows, receipt


def _publish(stage: Path, target: Path) -> None:
    # Reserve without replacement; atomically link each complete file, manifest last.
    target.mkdir()
    (target / 'raw').mkdir()
    for source in sorted((stage / 'raw').iterdir()):
        os.link(source, target / 'raw' / source.name)
    os.link(stage / 'covariates.parquet', target / 'covariates.parquet')
    os.link(stage / 'manifest.json', target / 'manifest.json')
    shutil.rmtree(stage)


def _historical_url(start: date, end: date, points: list[tuple]) -> str:
    if not points or len(points) > 64 or not date(1940, 1, 1) <= start <= end:
        raise ValueError('ERA5 requires 1..64 locations and an ordered archive interval from 1940')
    params = {'latitude': ','.join(format(p[1], '.8f') for p in points),
              'longitude': ','.join(format(p[2], '.8f') for p in points),
              'start_date': str(start), 'end_date': str(end), 'models': 'era5',
              'hourly': ','.join(VARIABLES), 'timezone': 'UTC',
              'temperature_unit': 'celsius', 'precipitation_unit': 'mm', 'wind_speed_unit': 'ms'}
    url = ERA5_ENDPOINT + '?' + urlencode(params)
    if len(url.encode()) > MAX_URL_BYTES:
        raise ValueError('ERA5 request URL exceeds bounded size')
    return url


def _era5_aggregate(payload: dict, issue: date) -> tuple:
    # Existing daily reducer selects day argument+1: pass D-1 to select D, never D+1.
    daily = _aggregate(payload, issue - timedelta(days=1))
    hourly = payload['hourly']
    precipitation = dict(zip(hourly['time'], hourly['precipitation']))
    window = [precipitation.get(hour)
              for offset in range(6, -1, -1)
              for hour in _target_hours(issue - timedelta(days=offset + 1))]
    seven_day = float(sum(window)) if all(value is not None for value in window) else None
    return (*daily[:4], seven_day, daily[-1])


def _fetch_era5(job: tuple, stage: Path, start: date, end: date, quota: _Quota,
                reuse: dict) -> tuple[dict, dict]:
    batch, points = job
    url = _historical_url(start, end, points)
    prefix = f'{start}-{end}-{batch:03d}'
    raw = stage / 'raw' / (prefix + '.json')
    cached = reuse.get(url)
    if cached is not None:
        original_path, original_raw, original = cached
        body = original_raw.read_bytes()
        receipt = dict(original)
        receipt['reused_from'] = {'receipt_path': str(original_path),
                                'receipt_sha256': sha256(original_path),
                                'revalidated_at_utc': datetime.now(timezone.utc).isoformat()}
    else:
        # A 43-day response costs up to four 14-day location-calls per node.
        quota.reserve(len(points) * math.ceil(((end - start).days + 1) / 14))
        started = datetime.now(timezone.utc).isoformat()
        status, headers, body, failure = None, {}, b'', None
        try:
            try:
                response = urlopen(Request(url, headers={'Accept': 'application/json',
                                   'User-Agent': 'fire-atlas-data/0.1'}), timeout=120)
            except HTTPError as error:
                response = error
            with response:
                status, headers = response.status, dict(response.headers.items())
                body = response.read(MAX_RESPONSE_BYTES + 1)
            if status != 200:
                failure = f'ERA5 source returned HTTP {status}'
            elif len(body) > MAX_RESPONSE_BYTES:
                failure = 'ERA5 response exceeded bounded size; raw is received prefix'
        except Exception as error:
            failure = f'ERA5 HTTPS acquisition failed: {type(error).__name__}: {error}'
        receipt = {'url': url, 'http_status': status, 'response_headers': headers,
                   'request_started_at_utc': started,
                   'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                   'model': 'era5', 'model_version': None, 'product_type': 'reanalysis',
                   'analysis_date_start': str(start), 'analysis_date_end': str(end),
                   'produced_at': None, 'available_at': None, 'failure': failure,
                   'locations': [{'sampling_node': node, 'latitude': lat, 'longitude': lon}
                                 for node, lat, lon in points]}
    with raw.open('xb') as stream:
        stream.write(body)
    receipt['raw'] = {'path': str(raw.relative_to(stage)), 'size_bytes': len(body),
                      'sha256': hashlib.sha256(body).hexdigest()}
    objects = []
    if receipt.get('failure') is None:
        try:
            objects = _location_objects(json.loads(body), len(points))
            # Validate all raw fields, but do not require complete values to manufacture data.
            for obj in objects:
                _aggregate(obj, end - timedelta(days=1))
            receipt['response_grid_centers'] = [
                {'latitude': obj.get('latitude'), 'longitude': obj.get('longitude'),
                 'elevation': obj.get('elevation')} for obj in objects]
        except (ValueError, TypeError, KeyError) as error:
            receipt['failure'] = f'Malformed ERA5 data: {error}'
    _json(stage / 'raw' / (prefix + '.receipt.json'), receipt)
    if receipt.get('failure') is not None:
        raise ValueError(receipt['failure'])
    return {point[0]: obj for point, obj in zip(points, objects)}, receipt


def acquire(features: str | Path, output_dir: str | Path, *, batch_size: int = 32,
            concurrency: int = 3, reuse_raw_dirs: list[str | Path] | None = None) -> dict:
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('batch_size must be 1..64')
    if type(concurrency) is not int or not 1 <= concurrency <= 3:
        raise ValueError('concurrency must be 1..3')
    features, target = Path(features).resolve(strict=True), Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('Weather output must not already exist')
    input_hash = sha256(features)
    code_hash = sha256(Path(__file__))
    keys = _input_keys(features)
    by_date, mapping = {}, {}
    for cell, issue, lat, lon in keys:
        node, node_lat, node_lon = _sampling_node(lat, lon)
        by_date.setdefault(issue, {})[node] = (node, node_lat, node_lon)
        mapping[cell] = {'sampling_node': node, 'cell_latitude': lat, 'cell_longitude': lon,
                         'request_latitude': node_lat, 'request_longitude': node_lon}
    by_date = {issue: sorted(points.values()) for issue, points in by_date.items()}
    jobs = [(issue, offset // batch_size, points[offset:offset + batch_size])
            for issue, points in sorted(by_date.items()) for offset in range(0, len(points), batch_size)]
    location_calls = sum(len(points) for points in by_date.values())
    if location_calls > 4800:
        raise ValueError('Acquisition exceeds conservative 4800/hour location quota; use a smaller dated input')
    reuse = _reuse_index([Path(path) for path in (reuse_raw_dirs or [])])
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.weather-acquisition-', dir=target.parent))
    (stage / 'raw').mkdir()
    started = datetime.now(timezone.utc).isoformat()
    try:
        quota = _Quota()
        rows, receipts = [], []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # map preserves request order while bounded workers acquire in parallel.
            for result, receipt in executor.map(lambda job: _fetch_or_reuse(job, stage, quota, reuse), jobs):
                rows.extend(result)
                receipts.append(receipt)
        values = {(row[0], row[1]): row[2:] for row in rows}
        if len(values) != location_calls or len(rows) != location_calls:
            raise ValueError('Response sampling-node keys are duplicated or incomplete')
        rows = [(cell, issue, *values[(mapping[cell]['sampling_node'], issue)])
                for cell, issue, _, _ in keys]
        if sha256(features) != input_hash:
            raise ValueError('Feature input changed during acquisition')
        if {(row[0], row[1]) for row in rows} != {(row[0], row[1]) for row in keys} or len(rows) != len(keys):
            raise ValueError('Output keys do not match distinct input keys')
        with duckdb.connect() as connection:
            connection.execute('''CREATE TABLE covariates(cell_id VARCHAR, issue_date DATE,
                wx_tmax_c DOUBLE, wx_rhmin_pct DOUBLE, wx_precip_mm DOUBLE,
                wx_windmax_ms DOUBLE, wx_valid_fraction DOUBLE)''')
            connection.executemany('INSERT INTO covariates VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
            connection.execute(f'''COPY (SELECT * FROM covariates ORDER BY issue_date,cell_id)
                TO {literal(str(stage / 'covariates.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)''')
        manifest = {
            'schema_version': '1.0.0', 'method': 'gfs_single_run_d00_target_d_plus_1_v1',
            'started_at_utc': started, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'implementation': {'path': 'src/fire_atlas_data/weather.py', 'sha256_at_start': code_hash,
                               'python': sys.version, 'duckdb': duckdb.__version__, 'h3': h3.__version__},
            'config': {'endpoint': ENDPOINT, 'model': MODEL, 'cycle_utc': '00:00',
                       'forecast_days': 2, 'timezone': 'UTC', 'hourly': list(VARIABLES),
                       'batch_size': batch_size, 'concurrency': concurrency,
                       'quota_location_limits': {'minute': 480, 'hour': 4800, 'day': 9500},
                       'sampling_lattice_degrees': 0.25, 'sampling_rounding': 'round(coordinate*4)/4, Python nearest ties-to-even',
                       'max_response_bytes': MAX_RESPONSE_BYTES, 'max_url_bytes': MAX_URL_BYTES},
            'columns': WEATHER_FEATURES, 'column_units': dict(zip(WEATHER_FEATURES, (*UNITS, 'fraction'))),
            'key_types': {'cell_id': 'VARCHAR', 'issue_date': 'DATE'}, 'feature_type': 'DOUBLE',
            'inputs_hashes': [{'path': str(features), 'sha256': input_hash}],
            'outputs': {'covariates.parquet': {'sha256': sha256(stage / 'covariates.parquet'),
                        'size_bytes': (stage / 'covariates.parquet').stat().st_size, 'rows': len(rows)}},
            'source': {'provider': 'NOAA GFS via Open-Meteo', 'product': MODEL,
                       'model_version': None, 'model_selection': 'fixed gfs_global; never best_match or seamless',
                       'available_at': None, 'init_time_is_publication_time': False,
                       'units': dict(zip(VARIABLES, UNITS)), 'access': 'public noncommercial, no API key',
                       'attribution': 'Open-Meteo and NOAA GFS; weather data CC BY 4.0',
                       'documentation_consulted_on': '2026-11-12',
                       'documentation': ['https://open-meteo.com/en/docs/single-runs-api',
                                         'https://open-meteo.com/en/docs/gfs-api',
                                         'https://open-meteo.com/en/pricing']},
            'semantics': {'valid_times': 'D+1 UTC 00:00..23:00 inclusive; forecast from D00 UTC',
                          'lead_hours': [24, 47], 'complete_field_hours_required': 24,
                          'precipitation': 'Sum preceding-hour amounts timestamped D+1 00..23; includes D23..D+1 00 interval, excludes D+1 23..D+2 00',
                          'missing': 'Incomplete field aggregate is null; valid fraction is non-null values /96',
                          'spatial': 'Output sampled on predeclared 0.25-degree lattice; nearest node propagated to H3; API default land selection and elevation downscaling; response grid centers preserved'},
            'cell_sampling_map': mapping,
            'response_grid_centers': [{'request_url': r['url'], 'locations': r['locations'],
                                       'returned_coordinates': r['response_grid_centers']} for r in receipts],
            'counts': {'cells': len({row[0] for row in keys}), 'issue_dates': len(by_date),
                       'rows': len(rows), 'http_requests': sum('reused_from' not in r for r in receipts),
                       'response_batches': len(receipts), 'reused_batches': sum('reused_from' in r for r in receipts),
                       'location_calls': location_calls,
                       'sampling_nodes': len({value['sampling_node'] for value in mapping.values()}),
                       'raw_response_bytes': sum(r['raw']['size_bytes'] for r in receipts),
                       'complete_rows': sum(row[-1] == 1.0 for row in rows),
                       'nulls': {column: sum(row[index + 2] is None for row in rows)
                                 for index, column in enumerate(WEATHER_FEATURES)}},
            'receipts': [{'path': 'raw/' + Path(r['raw']['path']).stem + '.receipt.json',
                          'sha256': sha256(stage / 'raw' / (Path(r['raw']['path']).stem + '.receipt.json')),
                          'raw': r['raw']} for r in receipts],
            'limitations': [
                'Retrospective acquisition; historical public availability before issue cutoff is unverified.',
                'Initialization is not publication; available_at remains unknown, not backdated from download time.',
                'Fixed archive model identifier and run do not pin an upstream executable version; API does not expose one.',
                'The 0.25-degree lattice is output sampling, not a claim of native/effective GFS resolution; nearest-node propagation limits representativeness and creates shared values, not independent H3 observations.',
                'No operational readiness claim; no additional spatial interpolation is performed.',
                'Free public API has no SLA and is for noncommercial use; quotas assume no concurrent external usage.',
            ],
        }
        _json(stage / 'manifest.json', manifest)
        _publish(stage, target)
        return manifest
    except Exception as error:
        _json(stage / 'failure.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'error': str(error), 'inputs_hashes': [{'path': str(features), 'sha256': input_hash}]})
        raise RuntimeError(f'Weather acquisition failed; preserved evidence at {stage}: {error}') from error


def acquire_historical(features: str | Path, output_dir: str | Path, *, batch_size: int = 32,
                       concurrency: int = 3, reuse_raw_dirs: list[str | Path] | None = None) -> dict:
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('batch_size must be 1..64')
    if type(concurrency) is not int or not 1 <= concurrency <= 3:
        raise ValueError('concurrency must be 1..3')
    features, target = Path(features).resolve(strict=True), Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('ERA5 output must not already exist')
    input_hash, code_hash = sha256(features), sha256(Path(__file__))
    keys = _input_keys(features, minimum=date(1940, 1, 7))
    start, end = min(row[1] for row in keys) - timedelta(days=6), max(row[1] for row in keys)
    nodes, mapping = {}, {}
    for cell, _, lat, lon in keys:
        node, node_lat, node_lon = _sampling_node(lat, lon)
        nodes[node] = (node, node_lat, node_lon)
        mapping[cell] = {'sampling_node': node, 'cell_latitude': lat, 'cell_longitude': lon,
                         'request_latitude': node_lat, 'request_longitude': node_lon}
    points = sorted(nodes.values())
    weighted_calls = len(points) * math.ceil(((end - start).days + 1) / 14)
    if weighted_calls > 4800:
        raise ValueError('ERA5 interval/location budget exceeds conservative 4800 hourly calls')
    reuse = _reuse_index([Path(path) for path in (reuse_raw_dirs or [])])
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.era5-acquisition-', dir=target.parent))
    (stage / 'raw').mkdir()
    started = datetime.now(timezone.utc).isoformat()
    try:
        jobs = [(offset // batch_size, points[offset:offset + batch_size])
                for offset in range(0, len(points), batch_size)]
        payloads, receipts = {}, []
        quota = _Quota()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for result, receipt in executor.map(
                    lambda job: _fetch_era5(job, stage, start, end, quota, reuse), jobs):
                payloads.update(result)
                receipts.append(receipt)
        derived = {(node, issue): _era5_aggregate(payloads[node], issue)
                   for node, issue in {(mapping[cell]['sampling_node'], issue) for cell, issue, _, _ in keys}}
        rows = [(cell, issue, *derived[(mapping[cell]['sampling_node'], issue)])
                for cell, issue, _, _ in keys]
        if len({row[:2] for row in rows}) != len(keys) or sha256(features) != input_hash:
            raise ValueError('ERA5 key invariant failed or feature input changed')
        with duckdb.connect() as connection:
            columns = ','.join(column + ' DOUBLE' for column in ERA5_FEATURES)
            connection.execute(f'CREATE TABLE covariates(cell_id VARCHAR, issue_date DATE,{columns})')
            connection.executemany('INSERT INTO covariates VALUES (?,?,?,?,?,?,?,?)', rows)
            connection.execute(f'''COPY (SELECT * FROM covariates ORDER BY issue_date,cell_id)
                TO {literal(str(stage / 'covariates.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)''')
        manifest = {
            'schema_version': '1.0.0', 'method': 'era5_retrospective_through_issue_day_v1',
            'started_at_utc': started, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'implementation': {'path': 'src/fire_atlas_data/weather.py', 'sha256_at_start': code_hash,
                               'python': sys.version, 'duckdb': duckdb.__version__, 'h3': h3.__version__},
            'config': {'endpoint': ERA5_ENDPOINT, 'model': 'era5', 'timezone': 'UTC',
                       'start_date': str(start), 'end_date': str(end), 'hourly': list(VARIABLES),
                       'batch_size': batch_size, 'concurrency': concurrency,
                       'sampling_lattice_degrees': 0.25,
                       'sampling_rounding': 'round(coordinate*4)/4, Python nearest ties-to-even',
                       'quota_location_limits': {'minute': 480, 'hour': 4800, 'day': 9500},
                       'max_response_bytes': MAX_RESPONSE_BYTES, 'max_url_bytes': MAX_URL_BYTES},
            'columns': ERA5_FEATURES,
            'column_units': dict(zip(ERA5_FEATURES, (*UNITS, 'mm', 'fraction'))),
            'key_types': {'cell_id': 'VARCHAR', 'issue_date': 'DATE'}, 'feature_type': 'DOUBLE',
            'inputs_hashes': [{'path': str(features), 'sha256': input_hash}],
            'outputs': {'covariates.parquet': {'sha256': sha256(stage / 'covariates.parquet'),
                        'size_bytes': (stage / 'covariates.parquet').stat().st_size, 'rows': len(rows)}},
            'source': {'provider': 'Copernicus/ECMWF ERA5 via Open-Meteo', 'product': 'era5',
                       'product_type': 'reanalysis', 'model_version': None, 'produced_at': None,
                       'available_at': None, 'historically_available_at_issue': 'unverified; retrospective only',
                       'model_selection': 'fixed era5; not auto, era5_land, era5_seamless or IFS',
                       'units': dict(zip(VARIABLES, UNITS)), 'access': 'public noncommercial, no API key',
                       'attribution': 'Open-Meteo and Copernicus/ECMWF ERA5; weather data CC BY 4.0',
                       'documentation_consulted_on': '2026-11-12',
                       'documentation': ['https://open-meteo.com/en/docs/historical-weather-api',
                                         'https://open-meteo.com/en/pricing']},
            'semantics': {'daily_fields': 'D UTC timestamps 00..23; no D+1 observed weather',
                          'precipitation_7d': '168 preceding-hour amounts timestamped D-6 00..D23; requires all 168',
                          'missing': 'Incomplete daily field or 168-hour precipitation stays null; no partial sums',
                          'valid_fraction': 'Non-null values across four D daily fields /96; seven-day completeness separate',
                          'precipitation_boundary': 'API amounts refer to preceding hour; calendar-boundary caveat applies'},
            'cell_sampling_map': mapping,
            'response_grid_centers': [{'request_url': r['url'], 'locations': r['locations'],
                                       'returned_coordinates': r['response_grid_centers']} for r in receipts],
            'counts': {'cells': len(mapping), 'issue_dates': len({row[1] for row in keys}),
                       'rows': len(rows), 'sampling_nodes': len(points),
                       'http_requests': sum('reused_from' not in r for r in receipts),
                       'response_batches': len(receipts), 'reused_batches': sum('reused_from' in r for r in receipts),
                       'conservative_weighted_location_calls': weighted_calls,
                       'raw_response_bytes': sum(r['raw']['size_bytes'] for r in receipts),
                       'complete_rows': sum(row[-1] == 1.0 and row[-2] is not None for row in rows),
                       'nulls': {column: sum(row[index + 2] is None for row in rows)
                                 for index, column in enumerate(ERA5_FEATURES)}},
            'receipts': [{'path': 'raw/' + Path(r['raw']['path']).stem + '.receipt.json',
                          'sha256': sha256(stage / 'raw' / (Path(r['raw']['path']).stem + '.receipt.json')),
                          'raw': r['raw']} for r in receipts],
            'limitations': [
                'Retrospective reanalysis acquired now; analysis dates do not establish availability on issue D.',
                'ERA5 latency/revisions prevent claiming operational availability; available_at remains unknown.',
                'No observations after D are selected; later production/revisions still remain retrospective information.',
                'Shared nearest-node samples are not independent H3 observations; no additional spatial interpolation.',
                'Free public API is noncommercial with no SLA; quota accounting excludes other users of the same IP.',
            ],
        }
        _json(stage / 'manifest.json', manifest)
        _publish(stage, target)
        return manifest
    except Exception as error:
        _json(stage / 'failure.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'error': str(error), 'inputs_hashes': [{'path': str(features), 'sha256': input_hash}]})
        raise RuntimeError(f'ERA5 acquisition failed; preserved evidence at {stage}: {error}') from error


def main() -> int:
    parser = argparse.ArgumentParser(description='Acquire fixed GFS D00 single-run D+1 weather covariates')
    parser.add_argument('--features', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--historical-only', action='store_true',
                        help='Acquire retrospective ERA5 through issue D, instead of GFS D+1 forecasts')
    parser.add_argument('--reuse-raw-dir', action='append', default=[],
                        help='Reuse exact URL/hash-matched successful raw receipts from a prior attempt')
    args = parser.parse_args()
    try:
        acquisition = acquire_historical if args.historical_only else acquire
        manifest = acquisition(args.features, args.output_dir, batch_size=args.batch_size,
                               concurrency=args.concurrency, reuse_raw_dirs=args.reuse_raw_dir)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({'counts': manifest['counts'], 'outputs': manifest['outputs']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
