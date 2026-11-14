"""Synthetic engineering fixtures only; these do not verify upstream GFS availability."""
from datetime import date, datetime, timedelta, timezone
from email.message import Message
import hashlib
import io
import json
from urllib.parse import parse_qs, urlsplit

import duckdb
import h3
import pytest

from fire_atlas_data import weather

ISSUE = date(2026, 10, 15)


def forecast(issue=ISSUE):
    start = datetime.combine(issue, datetime.min.time())
    times = [(start + timedelta(hours=hour)).strftime('%Y-%m-%dT%H:%M') for hour in range(48)]
    return {
        'latitude': -16.0, 'longitude': -47.0, 'elevation': 1000,
        'utc_offset_seconds': 0,
        'hourly_units': dict(zip(weather.VARIABLES, weather.UNITS)),
        'hourly': {'time': times, 'temperature_2m': list(range(48)),
                   'relative_humidity_2m': [80.0] * 24 + list(range(24, 48)),
                   'precipitation': [None] + [1.0] * 47, 'wind_speed_10m': [2.5] * 48},
    }


def base_parquet(path, keys):
    with duckdb.connect() as connection:
        connection.execute('CREATE TABLE features(cell_id VARCHAR, issue_date DATE, label INTEGER)')
        connection.executemany('INSERT INTO features VALUES (?, ?, ?)', [(c, d, 999) for c, d in keys])
        connection.execute(f"COPY features TO '{path}' (FORMAT PARQUET)")


def test_exact_target_hours_and_first_null_precipitation():
    payload = forecast()
    assert weather._aggregate(payload, ISSUE) == (47.0, 24.0, 24.0, 2.5, 1.0)
    # D23 is outside selection, including a large amount must not alter D+1.
    payload['hourly']['precipitation'][23] = 1000.0
    assert weather._aggregate(payload, ISSUE)[2] == 24.0


@pytest.mark.parametrize('index,variable', list(enumerate(weather.VARIABLES)))
def test_field_null_means_null_aggregate_not_partial_reducer(index, variable):
    payload = forecast()
    payload['hourly'][variable][24] = None
    result = weather._aggregate(payload, ISSUE)
    assert result[index] is None
    assert result[-1] == 95 / 96
    assert all(value is not None for i, value in enumerate(result[:-1]) if i != index)


def test_absent_target_hour_nulls_every_field_and_counts_fraction():
    payload = forecast()
    for values in payload['hourly'].values():
        del values[30]
    assert weather._aggregate(payload, ISSUE) == (None, None, None, None, 92 / 96)


def test_all_missing_stays_missing_not_zero():
    payload = forecast()
    for variable in weather.VARIABLES:
        payload['hourly'][variable] = [None] * 48
    assert weather._aggregate(payload, ISSUE) == (None, None, None, None, 0.0)


@pytest.mark.parametrize('issue,first,last', [
    (date(2026, 12, 31), '2027-01-01T00:00', '2027-01-01T23:00'),
    (date(2028, 2, 28), '2028-02-29T00:00', '2028-02-29T23:00'),
    (date(2028, 2, 29), '2028-03-01T00:00', '2028-03-01T23:00'),
])
def test_calendar_target_boundaries(issue, first, last):
    hours = weather._target_hours(issue)
    assert len(hours) == len(set(hours)) == 24
    assert (hours[0], hours[-1]) == (first, last)
    assert weather._aggregate(forecast(issue), issue)[-1] == 1.0


@pytest.mark.parametrize('change', ['duplicate', 'unordered', 'minute', 'units', 'array', 'nan', 'bool', 'negative_rain', 'rh', 'offset'])
def test_malformed_responses_fail_visibly(change):
    payload = forecast()
    if change == 'duplicate':
        payload['hourly']['time'][25] = payload['hourly']['time'][24]
    elif change == 'unordered':
        payload['hourly']['time'].reverse()
    elif change == 'minute':
        payload['hourly']['time'][24] = '2026-10-16T00:01'
    elif change == 'units':
        payload['hourly_units']['wind_speed_10m'] = 'km/h'
    elif change == 'array':
        payload['hourly']['precipitation'].pop()
    elif change == 'nan':
        payload['hourly']['temperature_2m'][24] = float('nan')
    elif change == 'bool':
        payload['hourly']['wind_speed_10m'][24] = True
    elif change == 'negative_rain':
        payload['hourly']['precipitation'][24] = -0.1
    elif change == 'rh':
        payload['hourly']['relative_humidity_2m'][24] = 101
    else:
        payload['utc_offset_seconds'] = -10800
    with pytest.raises(ValueError):
        weather._aggregate(payload, ISSUE)


def test_request_pins_cycle_model_units_and_batch():
    params = parse_qs(urlsplit(weather._request_url(ISSUE, [('a', -16.0, -47.0), ('b', -15.75, -47.25)])).query)
    assert params['models'] == ['gfs_global']
    assert params['run'] == ['2026-10-15T00:00']
    assert params['wind_speed_unit'] == ['ms']
    assert params['timezone'] == ['UTC']
    assert params['forecast_days'] == ['2']
    assert params['latitude'] == ['-16.00000000,-15.75000000']
    assert params['hourly'] == [','.join(weather.VARIABLES)]
    assert 'best_match' not in json.dumps(params)


def test_lattice_sampling_nearest_node_and_shared_values():
    assert weather._sampling_node(-16.01, -47.04) == ('-16.00,-47.00', -16.0, -47.0)
    assert weather._sampling_node(-16.10, -47.11) == weather._sampling_node(-16.01, -47.04)
    assert weather._sampling_node(-16.14, -47.14) == ('-16.25,-47.25', -16.25, -47.25)


def test_distinct_input_keys_only_and_actual_h3_centers(tmp_path):
    cell = h3.latlng_to_cell(-16, -47, 5)
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(cell, ISSUE), (cell, ISSUE)])
    assert weather._input_keys(path, today=date(2026, 11, 12)) == [(cell, ISSUE, *h3.cell_to_latlng(cell))]


@pytest.mark.parametrize('issue', [date(2026, 4, 1), date(2026, 11, 12), date(2026, 11, 13), date(2026, 12, 12)])
def test_archive_and_reserved_future_dates_are_refused(tmp_path, issue):
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(h3.latlng_to_cell(-16, -47, 5), issue)])
    with pytest.raises(ValueError, match='archived past DATE'):
        weather._input_keys(path, today=date(2026, 11, 12))


def test_batch_location_identity_is_not_silently_reordered():
    assert weather._location_objects([{}, {'location_id': 1}], 2) == [{}, {'location_id': 1}]
    with pytest.raises(ValueError, match='location_id'):
        weather._location_objects([{'location_id': 1}, {'location_id': 0}], 2)
    with pytest.raises(ValueError, match='location count'):
        weather._location_objects(forecast(), 2)


class Response(io.BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.headers = Message()
        self.headers['Content-Type'] = 'application/json'


def test_offline_acquisition_receipts_key_invariants_and_no_overwrite(tmp_path, monkeypatch):
    cell = h3.latlng_to_cell(-16, -47, 5)
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(cell, ISSUE), (cell, ISSUE)])
    raw = json.dumps(forecast()).encode()
    monkeypatch.setattr(weather, 'urlopen', lambda request, timeout: Response(raw))
    target = tmp_path / 'weather'
    manifest = weather.acquire(path, target, concurrency=1)
    assert manifest['counts']['rows'] == manifest['counts']['location_calls'] == 1
    assert manifest['counts']['complete_rows'] == 1
    assert manifest['columns'] == weather.WEATHER_FEATURES
    assert manifest['source']['available_at'] is None
    assert manifest['source']['init_time_is_publication_time'] is False
    receipt = json.loads(next((target / 'raw').glob('*.receipt.json')).read_text())
    assert receipt['raw']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['target_lead_hours'] == [24, 47]
    assert receipt['available_at'] is None
    assert (target / receipt['raw']['path']).read_bytes() == raw
    with duckdb.connect() as connection:
        rows = connection.execute(f"SELECT * FROM read_parquet('{target / 'covariates.parquet'}')").fetchall()
    assert rows == [(cell, ISSUE, 47.0, 24.0, 24.0, 2.5, 1.0)]
    with pytest.raises(FileExistsError):
        weather.acquire(path, target)


def test_two_h3_cells_share_one_sampled_request_node(tmp_path, monkeypatch):
    cells = [h3.latlng_to_cell(-16.01, -47.01, 7), h3.latlng_to_cell(-16.03, -47.01, 7)]
    assert cells[0] != cells[1]
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(cell, ISSUE) for cell in cells])
    requests = []

    def open_url(request, timeout):
        requests.append(parse_qs(urlsplit(request.full_url).query))
        return Response(json.dumps(forecast()).encode())

    monkeypatch.setattr(weather, 'urlopen', open_url)
    target = tmp_path / 'weather'
    manifest = weather.acquire(path, target, concurrency=1)
    assert manifest['counts']['cells'] == manifest['counts']['rows'] == 2
    assert manifest['counts']['sampling_nodes'] == manifest['counts']['location_calls'] == 1
    assert len(requests) == 1
    assert requests[0]['latitude'] == ['-16.00000000']
    assert requests[0]['longitude'] == ['-47.00000000']
    with duckdb.connect() as connection:
        values = connection.execute(f"SELECT DISTINCT wx_tmax_c,wx_precip_mm FROM read_parquet('{target / 'covariates.parquet'}')").fetchall()
    assert values == [(47.0, 24.0)]


def test_failed_payload_retains_raw_without_publishing_covariates(tmp_path, monkeypatch):
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(h3.latlng_to_cell(-16, -47, 5), ISSUE)])
    monkeypatch.setattr(weather, 'urlopen', lambda request, timeout: Response(b'{"error":true}'))
    target = tmp_path / 'weather'
    with pytest.raises(RuntimeError, match='preserved evidence'):
        weather.acquire(path, target, concurrency=1)
    assert not target.exists()
    stage = next(tmp_path.glob('.weather-acquisition-*'))
    assert (stage / 'failure.json').exists()
    assert list((stage / 'raw').glob('*.receipt.json'))
    assert not (stage / 'covariates.parquet').exists()


def reanalysis(issue=ISSUE):
    payload = forecast(issue)
    start = datetime.combine(issue - timedelta(days=6), datetime.min.time())
    payload['hourly'] = {
        'time': [(start + timedelta(hours=hour)).strftime('%Y-%m-%dT%H:%M') for hour in range(168)],
        'temperature_2m': [30.0] * 168, 'relative_humidity_2m': [60.0] * 168,
        'precipitation': [1.0] * 168, 'wind_speed_10m': [4.0] * 168,
    }
    return payload


@pytest.mark.parametrize('issue', [ISSUE, date(2028, 3, 1), date(2027, 1, 3)])
def test_era5_daily_and_seven_day_window_boundaries(issue):
    assert weather._era5_aggregate(reanalysis(issue), issue) == (30.0, 60.0, 24.0, 4.0, 168.0, 1.0)


def test_era5_historical_missing_hour_does_not_reduce_sum_to_partial():
    payload = reanalysis()
    payload['hourly']['precipitation'][0] = None
    assert weather._era5_aggregate(payload, ISSUE) == (30.0, 60.0, 24.0, 4.0, None, 1.0)
    for values in payload['hourly'].values():
        del values[0]
    assert weather._era5_aggregate(payload, ISSUE)[4] is None


def test_era5_null_on_issue_day_invalidates_daily_and_window():
    payload = reanalysis()
    payload['hourly']['precipitation'][144] = None
    assert weather._era5_aggregate(payload, ISSUE) == (30.0, 60.0, None, 4.0, None, 95 / 96)


def test_era5_never_selects_observed_target_day_values():
    payload = reanalysis()
    payload['hourly']['time'].extend(weather._target_hours(ISSUE))
    for variable, value in zip(weather.VARIABLES, (999.0, 0.0, 999.0, 999.0)):
        payload['hourly'][variable].extend([value] * 24)
    assert weather._era5_aggregate(payload, ISSUE) == (30.0, 60.0, 24.0, 4.0, 168.0, 1.0)


def test_era5_request_pins_model_not_seamless_and_end_is_issue_day():
    params = parse_qs(urlsplit(weather._historical_url(ISSUE - timedelta(days=6), ISSUE,
                                                     [('node', -16.0, -47.0)])).query)
    assert params['models'] == ['era5']
    assert params['start_date'] == ['2026-10-09']
    assert params['end_date'] == ['2026-10-15']
    assert params['wind_speed_unit'] == ['ms']
    assert params['timezone'] == ['UTC']
    assert 'run' not in params


def test_offline_era5_receipt_rows_nulls_and_retrospective_limitation(tmp_path, monkeypatch):
    cell = h3.latlng_to_cell(-16, -47, 5)
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(cell, ISSUE)])
    raw = json.dumps(reanalysis()).encode()
    monkeypatch.setattr(weather, 'urlopen', lambda request, timeout: Response(raw))
    target = tmp_path / 'era5'
    manifest = weather.acquire_historical(path, target, concurrency=1)
    assert manifest['columns'] == weather.ERA5_FEATURES
    assert manifest['counts']['rows'] == manifest['counts']['complete_rows'] == 1
    assert manifest['source']['product'] == 'era5'
    assert manifest['source']['product_type'] == 'reanalysis'
    assert manifest['source']['available_at'] is None
    assert manifest['source']['historically_available_at_issue'] == 'unverified; retrospective only'
    with duckdb.connect() as connection:
        result = connection.execute(f"SELECT * FROM read_parquet('{target / 'covariates.parquet'}')").fetchall()
    assert result == [(cell, ISSUE, 30.0, 60.0, 24.0, 4.0, 168.0, 1.0)]


def test_explicit_raw_reuse_preserves_download_time_and_avoids_http(tmp_path, monkeypatch):
    cell = h3.latlng_to_cell(-16, -47, 5)
    path = tmp_path / 'base.parquet'
    base_parquet(path, [(cell, ISSUE)])
    monkeypatch.setattr(weather, 'urlopen', lambda request, timeout: Response(json.dumps(forecast()).encode()))
    first = tmp_path / 'first'
    weather.acquire(path, first, concurrency=1)
    original = json.loads(next((first / 'raw').glob('*.receipt.json')).read_text())

    def no_http(*args, **kwargs):
        raise AssertionError('Exact raw reuse must not request the network')

    monkeypatch.setattr(weather, 'urlopen', no_http)
    second = tmp_path / 'second'
    manifest = weather.acquire(path, second, concurrency=1, reuse_raw_dirs=[first])
    receipt = json.loads(next((second / 'raw').glob('*.receipt.json')).read_text())
    assert manifest['counts']['http_requests'] == 0
    assert manifest['counts']['reused_batches'] == 1
    assert receipt['downloaded_at_utc'] == original['downloaded_at_utc']
    assert receipt['raw']['sha256'] == original['raw']['sha256']
    assert receipt['reused_from']['receipt_sha256'] == hashlib.sha256(
        next((first / 'raw').glob('*.receipt.json')).read_bytes()).hexdigest()
