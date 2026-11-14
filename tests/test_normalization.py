"""Consumer-visible observation normalization, not scientific/model validation."""
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pytest
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
HEADER = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'frp', 'daynight']
BASE = ['-15.12', '-47.75', '339.13', '0.5', '0.49', '2026-09-01', '0007', 'N20', 'VIIRS', 'n', '2.0NRT', '276.93', '5.39', 'N']


def write_csv(path, rows, header=HEADER):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def execute(source, output, product='VIIRS_NOAA20_NRT', check=True):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run([sys.executable, '-m', 'fire_atlas_data', 'normalize', str(source),
        '--output-dir', str(output), '--source-product', product,
        '--source-url', 'https://firms.modaps.eosdis.nasa.gov/', '--memory-limit', '256MB', '--threads', '1'],
        cwd=ROOT, env=env, text=True, capture_output=True)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def rows_at(output, kind):
    with duckdb.connect() as connection:
        cursor = connection.execute('SELECT * FROM read_parquet(?, hive_partitioning=true, union_by_name=true) ORDER BY source_record_number', [str(output / kind / '**/*.parquet')])
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def test_typed_observations_and_missingness(tmp_path):
    source, output = tmp_path / 'input.csv', tmp_path / 'out'
    missing = BASE.copy()
    missing[6], missing[12] = '7', ''
    zero = BASE.copy()
    zero[12] = '0'
    write_csv(source, [BASE, missing, zero])
    execute(source, output)
    rows = rows_at(output, 'accepted')
    assert [r['frp_mw'] for r in rows] == [5.39, None, 0.0]
    assert rows[0]['acquired_at'] == datetime(2026, 9, 1, 0, 7, tzinfo=timezone.utc)
    assert rows[1]['acquired_at'] == rows[0]['acquired_at']
    assert rows[0]['latitude'] == -15.12 and rows[0]['longitude'] == -47.75
    assert rows[0]['confidence_category'] == 'n' and rows[0]['confidence_percent'] is None
    assert all(r['available_at'] is None for r in rows)
    assert rows[0]['source_fields']['acq_time'] == '0007'
    assert rows[0]['source_fields']['bright_t31'] == '276.93'
    assert rows[0]['observation_type'] == 'active_fire_detection'
    assert [r['source_record_number'] for r in rows] == [1, 2, 3]
    assert rows[0]['record_id'] != rows[1]['record_id']  # Duplicated observations retained, not silently deduplicated.
    logical = json.loads(json.dumps(rows[0], default=lambda value: value.isoformat()))
    schema = json.loads((ROOT / 'contracts/normalized-observation.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(logical)


@pytest.mark.parametrize('column,value', [('latitude','91'), ('longitude','NaN'), ('acq_date','2026-02-30'), ('acq_time','2400'), ('confidence','95'), ('frp','abc'), ('scan','0'), ('satellite','NPP')])
def test_invalid_observation_is_accounted_not_silently_dropped(tmp_path, column, value):
    source, output = tmp_path / 'input.csv', tmp_path / 'out'
    invalid = BASE.copy()
    invalid[HEADER.index(column)] = value
    write_csv(source, [BASE, invalid])
    execute(source, output)
    accepted, rejected = rows_at(output, 'accepted'), rows_at(output, 'rejected')
    assert [r['source_record_number'] for r in accepted] == [1]
    assert [r['source_record_number'] for r in rejected] == [2]
    assert rejected[0]['reason']
    assert rejected[0]['source_fields'][column] == value
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['counts'] == {'total': 2, 'accepted': 1, 'rejected': 1}
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert accepted[0]['source_sha256'] == rejected[0]['source_sha256'] == manifest['source']['sha256'] == expected
    for artifact in manifest['outputs']:
        assert hashlib.sha256((output / artifact['path']).read_bytes()).hexdigest() == artifact['sha256']


def test_output_refuses_overwrite_and_preserves_raw(tmp_path):
    source, output = tmp_path / 'input.csv', tmp_path / 'out'
    write_csv(source, [BASE])
    original = source.read_bytes()
    execute(source, output)
    manifest = (output / 'manifest.json').read_bytes()
    assert execute(source, output, check=False).returncode != 0
    assert source.read_bytes() == original
    assert (output / 'manifest.json').read_bytes() == manifest


def test_malformed_csv_does_not_publish_completed_dataset(tmp_path):
    source, output = tmp_path / 'input.csv', tmp_path / 'out'
    source.write_text(','.join(HEADER) + '\n"unterminated,-47.75\n')
    assert execute(source, output, check=False).returncode != 0
    assert not output.exists()


def test_modis_confidence_is_not_viirs_probability(tmp_path):
    source, output = tmp_path / 'modis.csv', tmp_path / 'out'
    row = BASE.copy()
    row[7:12] = ['Terra', 'MODIS', '80', '6.1NRT', '280']
    write_csv(source, [row])
    execute(source, output, product='MODIS_NRT')
    result = rows_at(output, 'accepted')[0]
    assert result['sensor'] == 'MODIS'
    assert result['confidence_percent'] == 80
    assert result['confidence_category'] is None
    assert result['brightness_i4_k'] is None
    assert result['brightness_k'] == 339.13


def test_logical_lineage_is_deterministic_and_unknown_columns_survive(tmp_path):
    source = tmp_path / 'input.csv'
    write_csv(source, [BASE + ['opaque source value'], BASE + ['same detection']], HEADER + ['vendor_note'])
    first, second = tmp_path / 'first', tmp_path / 'second'
    execute(source, first)
    execute(source, second)
    rows = rows_at(first, 'accepted')
    assert rows == rows_at(second, 'accepted')
    assert [row['source_fields']['vendor_note'] for row in rows] == ['opaque source value', 'same detection']


@pytest.mark.parametrize('all_invalid', [False, True])
def test_empty_output_sides_remain_readable(tmp_path, all_invalid):
    source, output = tmp_path / 'input.csv', tmp_path / 'out'
    invalid = BASE.copy()
    invalid[0] = 'invalid'
    write_csv(source, [invalid] if all_invalid else [])
    execute(source, output)
    assert rows_at(output, 'accepted') == []
    rejected = rows_at(output, 'rejected')
    assert [row['source_record_number'] for row in rejected] == ([1] if all_invalid else [])


def test_public_viirs_schema_preserves_full_confidence_and_missing_instrument(tmp_path):
    header = [name for name in HEADER if name != 'instrument']
    header[header.index('brightness')] = 'bright_ti4'
    header[header.index('bright_t31')] = 'bright_ti5'
    rows = []
    for confidence in ('low', 'nominal', 'high', 'unknown'):
        row = dict(zip(HEADER, BASE))
        row['confidence'] = confidence
        rows.append([row['brightness' if name == 'bright_ti4' else
                         'bright_t31' if name == 'bright_ti5' else name] for name in header])
    wrong_platform = rows[1].copy()
    wrong_platform[header.index('satellite')] = 'N21'
    rows.append(wrong_platform)
    source, output = tmp_path / 'public.csv', tmp_path / 'out'
    write_csv(source, rows, header)
    execute(source, output)
    accepted = rows_at(output, 'accepted')
    assert [row['confidence_category'] for row in accepted] == ['l', 'n', 'h']
    assert [row['source_fields']['confidence'] for row in accepted] == ['low', 'nominal', 'high']
    assert all(row['sensor'] == 'VIIRS' and 'instrument' not in row['source_fields'] for row in accepted)
    assert [(row['brightness_i4_k'], row['brightness_i5_k']) for row in accepted] == [(339.13, 276.93)] * 3
    assert [row['reason'] for row in rows_at(output, 'rejected')] == ['invalid_confidence', 'platform_product_mismatch']


def test_present_instrument_still_must_match_declared_product(tmp_path):
    row = BASE.copy()
    row[HEADER.index('instrument')] = 'MODIS'
    source, output = tmp_path / 'mismatch.csv', tmp_path / 'out'
    write_csv(source, [row])
    execute(source, output)
    assert rows_at(output, 'accepted') == []
    assert rows_at(output, 'rejected')[0]['reason'] == 'sensor_product_mismatch'


def test_scientific_snpp_archive_preserves_identity_and_raw_labels(tmp_path):
    source, output = tmp_path / 'archive.csv', tmp_path / 'out'
    rows = []
    for label in ('0', '1', '2', '3'):
        row = BASE.copy()
        row[5], row[7], row[8], row[10] = '2026-01-01', 'SNPP', 'SNPP', 2
        rows.append(row + [label])
    write_csv(source, rows, HEADER + ['type'])
    execute(source, output, product='VIIRS_SNPP_SP')
    accepted = rows_at(output, 'accepted')
    assert len(accepted) == 4 and rows_at(output, 'rejected') == []
    for row, original in zip(accepted, rows):
        assert row['source_product'] == 'VIIRS_SNPP_SP'
        assert row['platform'] == 'Suomi-NPP' and row['sensor'] == 'VIIRS'
        assert row['schema_version'] == '1.3.0' and row['source_version'] == '2'
        assert row['confidence_category'] == 'n' and row['confidence_percent'] is None
        assert row['source_fields'] == dict(zip(HEADER + ['type'], map(str, original)))
        assert row['brightness_i5_k'] is None and row['bright_t31_k'] == 276.93
        assert row['available_at'] is None
        logical = json.loads(json.dumps(row, default=lambda value: value.isoformat()))
        schema = json.loads((ROOT / 'contracts/normalized-observation.schema.json').read_text())
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(logical)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['method_version'] == '1.6.0'
    assert manifest['sensor_origin'] == 'source_instrument_or_snpp_archive_alias_checked_against_product_and_platform'


@pytest.mark.parametrize('product,sensor,platform,satellite,instrument', [
    ('VIIRS_NOAA20_SP', 'VIIRS', 'NOAA-20', 'N20', 'VIIRS'),
    ('VIIRS_NOAA21_SP', 'VIIRS', 'NOAA-21', 'N21', 'VIIRS'),
    ('MODIS_SP', 'MODIS', 'Terra', 'Terra', 'MODIS'),
    ('MODIS_SP', 'MODIS', 'Aqua', 'Aqua', 'MODIS'),
])
def test_science_processing_products_keep_label_and_platform(tmp_path, product, sensor, platform, satellite, instrument):
    row = BASE.copy()
    row[7], row[8], row[9], row[10] = satellite, instrument, ('n' if sensor == 'VIIRS' else '82'), ('2' if sensor == 'VIIRS' else '61.03')
    source, output = tmp_path / 'sp.csv', tmp_path / 'out'
    write_csv(source, [row + ['0']], HEADER + ['type'])
    execute(source, output, product=product)
    accepted = rows_at(output, 'accepted')
    assert len(accepted) == 1 and rows_at(output, 'rejected') == []
    result = accepted[0]
    assert result['source_product'] == product
    assert result['sensor'] == sensor and result['platform'] == platform
    assert result['source_version'] == row[10]
    logical = json.loads(json.dumps(result, default=lambda value: value.isoformat()))
    schema = json.loads((ROOT / 'contracts/normalized-observation.schema.json').read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(logical)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['sensor_origin'] == 'source_instrument_checked_against_product'


@pytest.mark.parametrize('product,satellite,instrument', [
    ('VIIRS_NOAA20_SP', 'N21', 'VIIRS'),
    ('VIIRS_NOAA21_SP', 'N20', 'VIIRS'),
    ('VIIRS_NOAA20_SP', 'N20', 'SNPP'),
    ('MODIS_SP', 'N20', 'VIIRS'),
])
def test_science_processing_products_keep_platform_and_sensor_checks(tmp_path, product, satellite, instrument):
    source, output = tmp_path / 'mismatch.csv', tmp_path / 'out'
    row = BASE.copy()
    row[7], row[8], row[10] = satellite, instrument, '2'
    write_csv(source, [row])
    execute(source, output, product=product)
    assert rows_at(output, 'accepted') == []
    rejected = rows_at(output, 'rejected')
    assert len(rejected) == 1
    assert rejected[0]['reason'] in {
        'sensor_product_mismatch', 'invalid_platform', 'platform_product_mismatch'}


@pytest.mark.parametrize('product,satellite,instrument', [
    ('VIIRS_NOAA20_NRT', 'SNPP', 'SNPP'),
    ('VIIRS_NOAA20_NRT', 'SNPP', 'VIIRS'),
    ('VIIRS_NOAA20_NRT', 'N20', 'SNPP'),
    ('VIIRS_SNPP_NRT', 'N', 'SNPP'),
    ('VIIRS_SNPP_SP', 'SNPP', 'MODIS'),
    ('VIIRS_SNPP_SP', 'SNPP', 'unexpected'),
    ('VIIRS_SNPP_SP', 'N20', 'SNPP'),
    ('VIIRS_SNPP_SP', 'N21', 'VIIRS'),
])
def test_archive_alias_never_weakens_product_sensor_platform_checks(tmp_path, product, satellite, instrument):
    source, output = tmp_path / 'mismatch.csv', tmp_path / 'out'
    row = BASE.copy()
    row[7], row[8], row[10] = satellite, instrument, '2'
    write_csv(source, [row])
    execute(source, output, product=product)
    assert rows_at(output, 'accepted') == []
    rejected = rows_at(output, 'rejected')
    assert len(rejected) == 1
    assert rejected[0]['reason'] in {
        'sensor_product_mismatch', 'invalid_platform', 'platform_product_mismatch'}
    assert rejected[0]['source_fields'] == dict(zip(HEADER, row))


def sp_row(hotspot_type, **overrides):
    row = dict(zip(HEADER, BASE), version='2', **overrides)
    return [row[name] for name in HEADER] + [hotspot_type]


def test_archive_type_becomes_typed_hotspot_class_and_empty_is_unknown(tmp_path):
    source, output = tmp_path / 'archive.csv', tmp_path / 'out'
    labels = ['0', '1', '2', '3', '', ' 2 ']
    write_csv(source, [sp_row(label) for label in labels], HEADER + ['type'])
    execute(source, output, product='VIIRS_NOAA20_SP')
    accepted = rows_at(output, 'accepted')
    assert rows_at(output, 'rejected') == []
    assert [row['hotspot_type'] for row in accepted] == [0, 1, 2, 3, None, 2]
    assert [row['source_fields']['type'] for row in accepted] == labels
    schema = json.loads((ROOT / 'contracts/normalized-observation.schema.json').read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for row in accepted:
        validator.validate(json.loads(json.dumps(row, default=lambda value: value.isoformat())))
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['schema_version'] == '1.3.0'
    assert manifest['hotspot_type_counts'] == [
        {'hotspot_type': 0, 'count': 1}, {'hotspot_type': 1, 'count': 1}, {'hotspot_type': 2, 'count': 2},
        {'hotspot_type': 3, 'count': 1}, {'hotspot_type': None, 'count': 1}]


def test_stream_without_type_column_has_unknown_hotspot_class(tmp_path):
    source, output = tmp_path / 'nrt.csv', tmp_path / 'out'
    write_csv(source, [BASE, BASE])
    execute(source, output)
    accepted = rows_at(output, 'accepted')
    assert [row['hotspot_type'] for row in accepted] == [None, None]
    assert all('type' not in row['source_fields'] for row in accepted)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['hotspot_type_counts'] == [{'hotspot_type': None, 'count': 2}]


def test_invalid_type_is_rejected_without_displacing_earlier_reasons(tmp_path):
    source, output = tmp_path / 'archive.csv', tmp_path / 'out'
    invalid = ['4', '-1', 'x', '0.0', '1.5', '01']
    rows = [sp_row(label) for label in invalid]
    rows += [sp_row('9', latitude='91'), sp_row('9', frp='nan'), sp_row('9', satellite='N21'), sp_row('1')]
    write_csv(source, rows, HEADER + ['type'])
    execute(source, output, product='VIIRS_NOAA20_SP')
    accepted, rejected = rows_at(output, 'accepted'), rows_at(output, 'rejected')
    assert [row['reason'] for row in rejected] == ['invalid_type'] * len(invalid) + [
        'invalid_latitude', 'invalid_frp', 'platform_product_mismatch']
    assert [row['source_fields']['type'] for row in rejected] == invalid + ['9', '9', '9']
    assert [(row['source_record_number'], row['hotspot_type']) for row in accepted] == [(len(rows), 1)]
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['rejection_counts'] == {
        'invalid_type': len(invalid), 'invalid_latitude': 1, 'invalid_frp': 1, 'platform_product_mismatch': 1}


def test_impossible_frp_and_brightness_drop_only_the_value(tmp_path):
    source, output = tmp_path / 'archive.csv', tmp_path / 'out'
    rows = [sp_row('0', frp='-0.5'), sp_row('0', frp='0'), sp_row('0', bright_t31='0'),
            sp_row('0', brightness='-3'), sp_row('0', frp='abc')]
    write_csv(source, rows, HEADER + ['type'])
    execute(source, output, product='VIIRS_NOAA20_SP')
    accepted, rejected = rows_at(output, 'accepted'), rows_at(output, 'rejected')
    assert [row['frp_mw'] for row in accepted] == [None, 0.0, 5.39, 5.39]
    assert [row['bright_t31_k'] for row in accepted][2] is None
    assert accepted[3]['brightness_k'] is None and accepted[3]['brightness_i4_k'] is None
    assert accepted[0]['source_fields']['frp'] == '-0.5'
    assert [row['reason'] for row in rejected] == ['invalid_frp']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['nullified_value_counts']['frp_negative'] == 1
    assert manifest['nullified_value_counts']['bright_t31_non_positive'] == 1
    assert manifest['nullified_value_counts']['brightness_non_positive'] == 1


def test_snpp_nrt_platform_alias_is_accepted_only_for_snpp(tmp_path):
    snpp = dict(zip(HEADER, BASE), satellite='SNPP', instrument='SNPP')
    row = [snpp[name] for name in HEADER]
    source = tmp_path / 'snpp.csv'
    write_csv(source, [row])
    execute(source, tmp_path / 'snpp', product='VIIRS_SNPP_NRT')
    accepted = rows_at(tmp_path / 'snpp', 'accepted')
    assert [(r['platform'], r['sensor']) for r in accepted] == [('Suomi-NPP', 'VIIRS')]
    execute(source, tmp_path / 'n20', product='VIIRS_NOAA20_NRT')
    assert rows_at(tmp_path / 'n20', 'accepted') == []
    assert len(rows_at(tmp_path / 'n20', 'rejected')) == 1


def test_hotspot_type_counts_cover_exactly_the_accepted_rows(tmp_path):
    source, output = tmp_path / 'archive.csv', tmp_path / 'out'
    labels = ['3', '0', '', '0', '7', '2', '', '0', 'volcano']
    write_csv(source, [sp_row(label) for label in labels], HEADER + ['type'])
    execute(source, output, product='VIIRS_NOAA20_SP')
    accepted = rows_at(output, 'accepted')
    manifest = json.loads((output / 'manifest.json').read_text())
    counts = manifest['counts']
    assert counts == {'total': 9, 'accepted': 7, 'rejected': 2}
    assert counts['total'] == counts['accepted'] + counts['rejected']
    assert len(rows_at(output, 'rejected')) == counts['rejected']
    observed = [row['hotspot_type'] for row in accepted]
    assert manifest['hotspot_type_counts'] == [
        {'hotspot_type': value, 'count': observed.count(value)} for value in (0, 2, 3, None)]
    assert sum(entry['count'] for entry in manifest['hotspot_type_counts']) == counts['accepted'] == len(accepted)
