"""H3 daily map pyramid over normalized Parquet: hierarchy, sensor separation, filters."""
import csv
from datetime import date
import json

import duckdb
import pytest

from fire_atlas_data.map_grid import aggregate, load_h3
from fire_atlas_data.normalization import normalize

HEADER = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'frp', 'daynight']


def viirs(lat, lon, day='2024-01-01', frp='5.0', satellite='N20'):
    return [lat, lon, '330', '0.5', '0.5', day, '0100', satellite, 'VIIRS', 'n', '2', '280', frp, 'D']


def modis(lat, lon, day='2024-01-01', frp='7.0'):
    return [lat, lon, '330', '1.0', '1.0', day, '0100', 'Aqua', 'MODIS', '80', '61.03', '280', frp, 'D']


def dataset(tmp_path, name, product, rows):
    source = tmp_path / f'{name}.csv'
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    normalize(source, tmp_path / name, source_product=product, memory_limit='256MB', threads=1)
    return tmp_path / name


def cells(output, resolution):
    with duckdb.connect() as connection:
        connection.execute("SET TimeZone='UTC'")
        cursor = connection.execute('SELECT * FROM read_parquet(?, hive_partitioning=true, union_by_name=true) '
                                    f'WHERE resolution = {resolution} ORDER BY period_start, sensor, cell',
                                    [str(output / 'cells' / '**' / '*.parquet')])
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def test_coarse_levels_are_exact_parent_rollups_and_sensors_never_merge(tmp_path):
    v = dataset(tmp_path, 'v', 'VIIRS_NOAA20_SP', [
        viirs('90', '180'), viirs('-90', '-180'), viirs('10.2', '20.7', frp=''),
        viirs('10.21', '20.71', frp='4.0'), viirs('10.2', '20.7', day='2025-06-01')])
    m = dataset(tmp_path, 'm', 'MODIS_SP', [modis('10.2', '20.7')])
    manifest = aggregate([v, m], tmp_path / 'grid', resolutions=(8, 2, 5))
    assert manifest['counts']['detections'] == 6
    assert manifest['parameters']['resolutions'] == [2, 5, 8]
    with duckdb.connect() as connection:
        load_h3(connection)
        fine, coarse = cells(tmp_path / 'grid', 8), cells(tmp_path / 'grid', 2)
        rollup = {}
        for row in fine:
            parent = connection.execute('SELECT h3_cell_to_parent(?::UBIGINT, 2)', [row['cell']]).fetchone()[0]
            key = (parent, row['period_start'], row['sensor'])
            rollup[key] = rollup.get(key, 0) + row['detections']
        assert rollup == {(r['cell'], r['period_start'], r['sensor']): r['detections'] for r in coarse}
        assert all(connection.execute('SELECT h3_is_valid_cell(?::UBIGINT)', [r['cell']]).fetchone()[0] for r in fine)
        target = connection.execute('SELECT h3_latlng_to_cell(10.2, 20.7, 2)').fetchone()[0]
    shared = [r for r in coarse if r['cell'] == target and r['period_start'] == date(2024, 1, 1)]
    assert sorted(r['sensor'] for r in shared) == ['MODIS', 'VIIRS']
    viirs_cell = next(r for r in shared if r['sensor'] == 'VIIRS')
    assert viirs_cell['detections'] == 2 and viirs_cell['detections_with_frp'] == 1
    assert viirs_cell['frp_sum_mw'] == 4.0 and viirs_cell['frp_max_mw'] == 4.0
    assert {r['period_year'] for r in coarse} == {2024, 2025}


def test_inclusive_dates_bbox_daily_rows_and_platform_grouping(tmp_path):
    v = dataset(tmp_path, 'v', 'VIIRS_NOAA20_NRT', [
        viirs('-15.1', '-47.9', '2024-01-31'), viirs('-15.2', '-47.8', '2024-02-01'),
        viirs('-15.3', '-47.7', '2024-02-29', frp='3.0'), viirs('-15.4', '-47.6', '2024-03-01'),
        viirs('40.0', '10.0', '2024-02-10')])
    s = dataset(tmp_path, 's', 'VIIRS_SNPP_NRT', [viirs('-15.5', '-47.5', '2024-02-15', satellite='N')])
    aggregate([v, s], tmp_path / 'grid', resolutions=(3,), group_by='platform',
              start=date(2024, 2, 1), end=date(2024, 2, 29), bbox=(-48, -16, -47, -15))
    rows = cells(tmp_path / 'grid', 3)
    assert [(r['period_start'], r['platform']) for r in rows] == [
        (date(2024, 2, 1), 'NOAA-20'), (date(2024, 2, 15), 'Suomi-NPP'), (date(2024, 2, 29), 'NOAA-20')]
    assert rows[2]['frp_sum_mw'] == 3.0


def test_empty_selection_publishes_readable_grid(tmp_path):
    v = dataset(tmp_path, 'v', 'VIIRS_NOAA20_SP', [viirs('1', '1')])
    manifest = aggregate([v], tmp_path / 'grid', resolutions=(4,), start=date(2025, 1, 1))
    assert manifest['counts'] == {'detections': 0, 'cells_by_resolution': {'4': 0}}
    assert duckdb.sql(f"SELECT count(*) FROM '{tmp_path / 'grid' / 'cells' / 'empty.parquet'}'").fetchall() == [(0,)]
    assert json.loads((tmp_path / 'grid' / 'manifest.json').read_text())['sources'][0]['accepted'] == 1


@pytest.mark.parametrize('kwargs', [{'resolutions': (16,)}, {'resolutions': ()}, {'resolutions': (5, 5)},
                                    {'bbox': (10, 0, -10, 5)},
                                    {'start': date(2024, 2, 1), 'end': date(2024, 1, 1)}])
def test_invalid_parameters_publish_nothing(tmp_path, kwargs):
    v = dataset(tmp_path, 'v', 'VIIRS_NOAA20_SP', [viirs('1', '1')])
    with pytest.raises(ValueError):
        aggregate([v], tmp_path / 'grid', **kwargs)
    assert not (tmp_path / 'grid').exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.grid')] == []


def test_existing_output_is_refused(tmp_path):
    v = dataset(tmp_path, 'v', 'VIIRS_NOAA20_SP', [viirs('1', '1')])
    aggregate([v], tmp_path / 'grid', resolutions=(2,))
    before = (tmp_path / 'grid' / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        aggregate([v], tmp_path / 'grid', resolutions=(3,))
    assert (tmp_path / 'grid' / 'manifest.json').read_bytes() == before
