"""Permanent synthetic regression checks; these are not scientific validation."""
from datetime import date
import json

import duckdb
import h3
import pytest

from fire_atlas_data.features import BASE_FEATURES, ENHANCED_FEATURES, _grid, build
from fire_atlas_data.normalization import literal


def _config(tmp_path, *, winter=False):
    year, month = (2027, 1) if winter else (2026, 10)
    start = f'{year}-{month:02d}'
    return {
        'sources': [str(tmp_path / 'source.parquet')], 'bbox': [-48, -17, -46, -15],
        'h3_resolution': 5, 'region': 'fixture', 'warmup_start': f'{year-1}-12-01' if winter else '2026-09-01',
        'splits': {'train': [start+'-01', start+'-01'], 'validation': [start+'-02', start+'-02'], 'test': [start+'-03', start+'-03']},
        'publish_date': start+'-05', 'memory_limit': '256MB', 'threads': 1,
    }


def _build(tmp_path, config, rows, name='result'):
    connection = duckdb.connect()
    connection.execute('CREATE TABLE input(latitude DOUBLE,longitude DOUBLE, acquisition_date DATE,frp_mw DOUBLE,platform VARCHAR)')
    connection.executemany('INSERT INTO input VALUES (?,?,?,?,?)', [
        (*h3.cell_to_latlng(row[0]), row[1], row[2] if len(row)>2 else 1,
         row[3] if len(row)>3 else 'NOAA-20') for row in rows
    ])
    connection.execute(f'''COPY (SELECT *, acquisition_date::TIMESTAMP AT TIME ZONE 'UTC' acquired_at,
        NULL::TIMESTAMPTZ available_at,'VIIRS' sensor,
        CASE WHEN platform='NOAA-20' THEN 'VIIRS_NOAA20_NRT' WHEN platform='Suomi-NPP' THEN 'VIIRS_SNPP_SP' ELSE 'VIIRS_NOAA21_NRT' END source_product,
        'fixture' source_version,'n' confidence_category,NULL::SMALLINT confidence_percent,
        300::DOUBLE brightness_i4_k,290::DOUBLE brightness_i5_k,NULL::DOUBLE brightness_k,NULL::DOUBLE bright_t31_k,
        1::DOUBLE scan_km,1::DOUBLE track_km,'D' daynight FROM input)
        TO {literal(config['sources'][0])} (FORMAT PARQUET)''')
    connection.close()
    config_file = tmp_path / (name+'.json')
    config_file.write_text(json.dumps(config))
    build(config_file, tmp_path / name)
    return duckdb.connect(), tmp_path / name


def test_eligibility_next_day_isolation_and_fixed_geometry(tmp_path):
    config = _config(tmp_path)
    cells = _grid(config)
    a,b,c,d,e = cells[:5]
    rows = [(a,'2026-10-01'),(a,'2026-09-30'),(b,'2026-09-30'),(c,'2026-09-29'),(d,'2026-09-28'),
            (d,'2026-09-27'),(d,'2026-10-02'),(e,'2026-09-01'),(e,'2026-10-05')]
    connection, output = _build(tmp_path, config, rows)
    table = f"read_parquet({literal(str(output / 'features.parquet'))})"
    eligible = dict(connection.execute(f"SELECT cell_id,label FROM {table} WHERE issue_date=DATE '2026-10-01'").fetchall())
    assert a not in eligible and b not in eligible and c not in eligible
    assert eligible[d] == 1
    assert connection.execute(f"SELECT days_since_last FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-10-01'", [d]).fetchone()[0] == 3
    assert connection.execute(f"SELECT regional_7d FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-10-01'", [d]).fetchone()[0] == 4
    neighbor_ids = set(h3.grid_disk(d, 1)) - {d}
    assert connection.execute(f"SELECT neighbors_3d,neighbors_7d FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-10-01'", [d]).fetchone() == (len(neighbor_ids & {a,b,c}), len(neighbor_ids & {a,b,c}))
    assert connection.execute(f"SELECT count(*) FROM {table} WHERE split='publish' AND label IS NOT NULL").fetchone()[0] == 0
    candidates = connection.execute(f"SELECT cell_id FROM read_parquet({literal(str(output / 'candidates.parquet'))}) ORDER BY cell_id").fetchall()
    assert [r[0] for r in candidates] == cells
    # Adding a future detection changes a label, never the candidate geometry or D features.
    connection.close()
    (tmp_path / 'source.parquet').unlink()
    connection, other = _build(tmp_path, config, rows+[(d,'2026-10-03')], 'other')
    assert connection.execute(f"SELECT * EXCLUDE(label) FROM read_parquet({literal(str(output/'features.parquet'))}) WHERE issue_date=DATE '2026-10-01' EXCEPT SELECT * EXCLUDE(label) FROM read_parquet({literal(str(other/'features.parquet'))}) WHERE issue_date=DATE '2026-10-01'").fetchall() == []
    connection.close()


def test_any_activity_includes_persistence_without_future_features(tmp_path):
    config = _config(tmp_path)
    config['target_mode'] = 'any_activity'
    a, b, c = _grid(config)[:3]
    rows = [(a, '2026-09-01'), (a, '2026-10-01'), (a, '2026-10-02'),
            (b, '2026-10-02'), (c, '2026-10-02', 1, 'NOAA-21'),
            (a, '2026-10-05')]
    connection, output = _build(tmp_path, config, rows)
    table = f"read_parquet({literal(str(output/'features.parquet'))})"
    assert connection.execute(f"SELECT count(*) FROM {table} WHERE issue_date=DATE '2026-10-01'").fetchone()[0] == len(_grid(config))
    labels = dict(connection.execute(f"SELECT cell_id,label FROM {table} WHERE issue_date=DATE '2026-10-01'").fetchall())
    assert (labels[a], labels[b], labels[c]) == (1, 1, 0)
    assert connection.execute(f"SELECT days_since_last FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-10-01'", [a]).fetchone()[0] == 0
    assert connection.execute(f"SELECT count(*) FROM {table} WHERE split='publish' AND label IS NOT NULL").fetchone()[0] == 0
    connection.close()
    (tmp_path / 'source.parquet').unlink()
    connection, other = _build(tmp_path, config, rows + [(c, '2026-10-02')], 'future')
    assert connection.execute(f"SELECT label FROM read_parquet({literal(str(other/'features.parquet'))}) WHERE cell_id=? AND issue_date=DATE '2026-10-01'", [c]).fetchone()[0] == 1
    assert connection.execute(f"SELECT * EXCLUDE(label) FROM {table} WHERE issue_date=DATE '2026-10-01' EXCEPT SELECT * EXCLUDE(label) FROM read_parquet({literal(str(other/'features.parquet'))}) WHERE issue_date=DATE '2026-10-01'").fetchall() == []
    connection.close()


def test_unknown_target_fails_before_output(tmp_path):
    config = _config(tmp_path)
    config['target_mode'] = 'typo'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='target_mode'):
        build(path, tmp_path / 'result')
    assert not (tmp_path / 'result').exists()


def test_seasonal_circular_window_and_prior_year_only(tmp_path):
    config = _config(tmp_path, winter=True)
    a,b = _grid(config)[:2]
    rows = [(b,'2026-12-01'),(b,'2027-01-05'),
            (a,'2025-12-16'),(a,'2025-12-20'), (a,'2026-01-15'), (a,'2026-01-16'),
            (a,'2026-01-20'),(a,'2027-01-02')]
    connection, output = _build(tmp_path, config, rows)
    row = connection.execute(f"SELECT seasonal_viirs,seasonal_support_years FROM read_parquet({literal(str(output/'features.parquet'))}) WHERE cell_id=? AND issue_date=DATE '2027-01-01'", [a]).fetchone()
    # Jan2 target: Dec20 wraps into window; Dec16/Jan20 fall outside.
    # Jan16 follows Jan15, so is not a new cell-day; same-year target is excluded.
    assert row == (2.0, 2)
    connection.close()


def _enhanced_config(tmp_path):
    config = _config(tmp_path)
    config.update(enhanced_features=True, minimum_history_days=14,
                  splits={'train': ['2026-09-15', '2026-09-15'],
                          'validation': ['2026-09-16', '2026-09-16'],
                          'test': ['2026-09-17', '2026-09-17']},
                  publish_date='2026-09-19')
    return config


def test_enhanced_recurrence_rings_frp_and_censored_history(tmp_path):
    config = _enhanced_config(tmp_path)
    cells = _grid(config)
    # Explicitly put an active distance-two neighbor outside the candidate grid.
    a = next(cell for cell in cells if set(h3.grid_disk(cell, 2)) - set(cells) - set(h3.grid_disk(cell, 1)))
    b = sorted(set(h3.grid_disk(a, 1)) - {a})[0]
    c = sorted(set(h3.grid_disk(a, 2)) - set(cells) - set(h3.grid_disk(a, 1)))[0]
    d = next(cell for cell in cells if cell not in h3.grid_disk(a, 3))
    rows = [(d, '2026-09-01'), (d, '2026-09-19'),
            (a, '2026-09-03'), (a, '2026-09-04'), (a, '2026-09-10'),
            (a, '2026-09-11', 2), (a, '2026-09-11', 3), (a, '2026-09-16'),
            (b, '2026-09-10', 0), (b, '2026-09-12', 8),
            (b, '2026-09-14', None), (b, '2026-09-15', None),
            (b, '2026-09-15', 5000, 'NOAA-21'),
            (c, '2026-09-13', 0), (c, '2026-09-14', 0)]
    connection, output = _build(tmp_path, config, rows)
    table = f"read_parquet({literal(str(output/'features.parquet'))})"
    row = connection.execute(f"""SELECT cell_active_days_7d,cell_active_days_14d,cell_active_days_30d,
        cell_detections_7d,cell_detections_30d,cell_prior_novel_events,
        ring1_active_cells_1d,ring1_active_cells_3d,ring1_active_cells_7d,
        ring1_active_cells_14d,ring1_active_cells_30d,
        ring2_active_cells_3d,ring2_active_cells_7d,ring2_active_cells_14d,ring2_active_cells_30d,
        days_since_neighbor,nearby_frp_max_3d,nearby_frp_mean_3d,nearby_frp_max_7d,nearby_frp_mean_7d,
        latitude,longitude,history_days,label,seasonal_modis,seasonal_viirs,seasonal_support_years
        FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-09-15'""", [a]).fetchone()
    assert row[:6] == (2, 4, 4, 3, 5, 2)
    assert row[6:15] == (1, 1, 1, 1, 1, 2, 2, 2, 2)
    assert row[15:20] == (0, 0, 0, 8, 2)
    assert row[20:22] == h3.cell_to_latlng(a)
    assert row[22:] == (15, 1, 0, 0, 0)
    assert ENHANCED_FEATURES[:len(BASE_FEATURES)] == BASE_FEATURES
    assert set(ENHANCED_FEATURES) <= {column[0] for column in connection.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}
    assert connection.execute(f"SELECT count(*) FROM {table} WHERE split='train' AND issue_date<DATE '2026-10-01'").fetchone()[0] > 0
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['history']['minimum_history_days'] == 14
    assert manifest['history']['left_censored_30d_rows'] == sum(s['eligible'] for s in manifest['split_counts'].values())
    candidates = connection.execute(f"SELECT cell_id FROM read_parquet({literal(str(output/'candidates.parquet'))}) ORDER BY cell_id").fetchall()
    assert [r[0] for r in candidates] == cells
    connection.close()
    # Future FRP (including an extreme value) can never change any feature at D.
    (tmp_path/'source.parquet').unlink()
    connection, future = _build(tmp_path, config, rows+[(c, '2026-09-16', 99999)], 'future')
    assert connection.execute(f"""SELECT * EXCLUDE(label) FROM {table} WHERE issue_date=DATE '2026-09-15'
        EXCEPT SELECT * EXCLUDE(label) FROM read_parquet({literal(str(future/'features.parquet'))})
        WHERE issue_date=DATE '2026-09-15'""").fetchall() == []
    connection.close()


def test_enhanced_null_frp_is_not_zero_and_unknown_elapsed_is_null(tmp_path):
    config = _enhanced_config(tmp_path)
    cells = _grid(config)
    a = cells[len(cells)//2]
    b = sorted(set(h3.grid_disk(a, 1)) - {a})[0]
    d = next(cell for cell in cells if cell not in h3.grid_disk(a, 5))
    rows = [(d, '2026-09-01'), (d, '2026-09-19'), (b, '2026-09-15', None)]
    connection, output = _build(tmp_path, config, rows)
    table = f"read_parquet({literal(str(output/'features.parquet'))})"
    assert connection.execute(f"""SELECT ring2_active_cells_3d,nearby_frp_max_3d,nearby_frp_mean_3d,
        nearby_frp_max_7d,nearby_frp_mean_7d,days_since_neighbor FROM {table}
        WHERE cell_id=? AND issue_date=DATE '2026-09-15'""", [a]).fetchone() == (1, None, None, None, None, 0)
    empty = next(cell for cell in cells if cell not in set(h3.grid_disk(b, 2)) | set(h3.grid_disk(d, 2)))
    assert connection.execute(f"""SELECT ring2_active_cells_3d,days_since_neighbor,nearby_frp_mean_3d
        FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-09-15'""", [empty]).fetchone() == (0, None, None)
    connection.close()


def test_enhanced_history_cap_and_base_columns_unchanged(tmp_path):
    config = _config(tmp_path)
    config['enhanced_features'] = True
    a,b = _grid(config)[:2]
    rows = [(a, '2026-09-01'), (b, '2026-10-05')]
    connection, enhanced = _build(tmp_path, config, rows, 'enhanced')
    assert connection.execute(f"SELECT min(history_days),max(history_days) FROM read_parquet({literal(str(enhanced/'features.parquet'))})").fetchone() == (30, 30)
    assert json.loads((enhanced/'manifest.json').read_text())['history']['left_censored_30d_rows'] == 0
    connection.close()
    (tmp_path/'source.parquet').unlink()
    config.pop('enhanced_features')
    connection, baseline = _build(tmp_path, config, rows, 'baseline')
    columns = ','.join(['cell_id','issue_date','target_date','region','split','label', *BASE_FEATURES, 'seasonal_support_years'])
    for left,right in [(enhanced,baseline), (baseline,enhanced)]:
        assert connection.execute(f"""SELECT {columns} FROM read_parquet({literal(str(left/'features.parquet'))})
            EXCEPT SELECT {columns} FROM read_parquet({literal(str(right/'features.parquet'))})""").fetchall() == []
    connection.close()


@pytest.mark.parametrize('minimum', [6, 0, -1, 7.5, True])
def test_minimum_history_rejects_invalid_values(tmp_path, minimum):
    source = tmp_path/'source.parquet'
    source.touch()  # Validation must reject the config before opening the input.
    config = _enhanced_config(tmp_path)
    config['minimum_history_days'] = minimum
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='integer >=7'):
        build(path, tmp_path/'result')


def test_default_history_requirement_remains_thirty_days(tmp_path):
    (tmp_path/'source.parquet').touch()
    config = _enhanced_config(tmp_path)
    config.pop('minimum_history_days')
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='30 days'):
        build(path, tmp_path/'result')


def test_snpp_separation_future_isolation_and_null_frp(tmp_path):
    config = _enhanced_config(tmp_path)
    a = _grid(config)[0]
    b = sorted(set(h3.grid_disk(a, 1)) - {a})[0]
    c = sorted(set(h3.grid_disk(a, 2)) - set(h3.grid_disk(a, 1)))[0]
    d = next(cell for cell in _grid(config) if cell not in h3.grid_disk(a, 4))
    noaa = [(d, '2026-09-01'), (d, '2026-09-19'), (a, '2026-09-16')]
    connection, baseline = _build(tmp_path, config, noaa, 'baseline')
    connection.close()
    config['snpp_history'] = True
    snpp = [(a, '2026-01-01', None, 'Suomi-NPP'),
            (a, '2026-01-02', None, 'Suomi-NPP'),
            (a, '2026-01-06', None, 'Suomi-NPP'),
            (b, '2026-01-01', 4, 'Suomi-NPP'),
            (b, '2026-01-02', 5, 'Suomi-NPP'),
            (c, '2026-01-01', 6, 'Suomi-NPP')]
    connection, history = _build(tmp_path, config, noaa + snpp, 'history')
    table = f"read_parquet({literal(str(history/'features.parquet'))})"
    assert connection.execute(f"""SELECT snpp_cell_active_days,snpp_cell_new_days,
        snpp_cell_detections,snpp_days_since_last,snpp_ring1_active_cell_days,
        snpp_ring2_active_cell_days,snpp_cell_frp_mean,snpp_cell_frp_max
        FROM {table} WHERE cell_id=? AND issue_date=DATE '2026-09-15'""", [a]).fetchone() == (
            3, 2, 3, (date(2026,9,15)-date(2026,1,6)).days, 2, 3, None, None)
    columns = ','.join(['cell_id', 'issue_date', 'label', 'split', *ENHANCED_FEATURES])
    for left, right in [(baseline, history), (history, baseline)]:
        assert connection.execute(f"""SELECT {columns} FROM read_parquet({literal(str(left/'features.parquet'))})
            EXCEPT SELECT {columns} FROM read_parquet({literal(str(right/'features.parquet'))})""").fetchall() == []
    connection.close()
    connection, future = _build(tmp_path, config, noaa + snpp + [
        (a, '2026-09-16', 99999, 'Suomi-NPP')], 'future')
    assert connection.execute(f"""SELECT * FROM {table} WHERE issue_date=DATE '2026-09-15'
        EXCEPT SELECT * FROM read_parquet({literal(str(future/'features.parquet'))})
        WHERE issue_date=DATE '2026-09-15'""").fetchall() == []
    connection.close()
