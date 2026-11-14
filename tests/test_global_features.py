from datetime import date, timedelta
import json

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data.global_features import FEATURE_NAMES, build, sample
from fire_atlas_data.normalization import sha256

ISSUE = date(2014, 1, 10)
CELL = h3.latlng_to_cell(-12, -53, 5)
NEIGHBOR = next(cell for cell in sorted(h3.grid_ring(CELL, 1))
                if h3.cell_to_parent(cell, 4) == h3.cell_to_parent(CELL, 4))
FAR = h3.latlng_to_cell(40, 100, 5)
START = date(2012, 1, 1)
END = ISSUE + timedelta(days=9)
DAILY_SCHEMA = pa.schema([
    ('cell_id', pa.string()), ('acquisition_date', pa.date32()),
    ('viirs_detections', pa.int64()), ('modis_detections', pa.int64()),
    ('viirs_frp_sum_mw', pa.float64()), ('viirs_frp_rows', pa.int64()),
    ('modis_frp_sum_mw', pa.float64()), ('modis_frp_rows', pa.int64()),
    ('year', pa.int32()),
])


def _event(cell, day, viirs=1, modis=0, frp=None):
    return dict(cell_id=cell, acquisition_date=day, viirs_detections=viirs,
                modis_detections=modis, viirs_frp_sum_mw=frp or 0.,
                viirs_frp_rows=1 if frp is not None else 0,
                modis_frp_sum_mw=0., modis_frp_rows=0, year=day.year)


def _outputs(root):
    return [dict(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size,
                 sha256=sha256(path)) for path in sorted(root.rglob('*.parquet'))]


def _activity(root, events, *, missing=(), sensors=('VIIRS', 'MODIS'),
              first=None, cutoff='2011-12-31'):
    root.mkdir()
    (root / 'daily').mkdir()
    pq.write_table(pa.Table.from_pylist(events, schema=DAILY_SCHEMA), root / 'daily' / 'data.parquet')
    first = first or {CELL: START, NEIGHBOR: START, FAR: START}
    cells = []
    neighbors = []
    for cell, initial in first.items():
        lat, lon = h3.cell_to_latlng(cell)
        cells.append(dict(cell_id=cell, latitude=lat, longitude=lon,
                          first_detection_date=initial, parent4_id=h3.cell_to_parent(cell, 4),
                          parent3_id=h3.cell_to_parent(cell, 3)))
        neighbors.extend(dict(cell_id=cell, neighbor_id=other) for other in h3.grid_ring(cell, 1))
    pq.write_table(pa.Table.from_pylist(cells), root / 'cells.parquet')
    pq.write_table(pa.Table.from_pylist(neighbors), root / 'neighbors.parquet')
    platforms = []
    for offset in range((END - START).days + 1):
        day = START + timedelta(days=offset)
        if day not in missing:
            platforms.extend(dict(sensor=sensor, platform='P-' + sensor,
                                  acquisition_date=day, detections=1) for sensor in sensors)
    pq.write_table(pa.Table.from_pylist(platforms), root / 'platform_days.parquet')
    manifest = dict(schema='global-cell-activity/1.0.0', h3_resolution=5,
                    acquisition_range={'min': START.isoformat(), 'max': END.isoformat()},
                    mask={'manifest_sha256': 'a' * 64, 'parameters': {'reference_end': cutoff}})
    manifest['outputs'] = _outputs(root)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def _keys(path, pairs):
    pq.write_table(pa.Table.from_pylist(
        [dict(cell_id=cell, issue_date=day) for cell, day in pairs],
        schema=pa.schema([('cell_id', pa.string()), ('issue_date', pa.date32())])), path)
    return path


def _build(root, keys, target, labels=False):
    build(root, keys, target, labels=labels, memory_limit='128MB', threads=1)
    return pq.read_table(target / 'features.parquet')


def test_subset_invariance_uses_full_ring_and_parents(tmp_path):
    root = _activity(tmp_path / 'activity', [_event(NEIGHBOR, ISSUE), _event(CELL, ISSUE)])
    subset = _keys(tmp_path / 'subset.parquet', [(CELL, ISSUE)])
    full = _keys(tmp_path / 'full.parquet', [(NEIGHBOR, ISSUE), (CELL, ISSUE), (FAR, ISSUE)])
    one = _build(root, subset, tmp_path / 'one').to_pylist()[0]
    many = _build(root, full, tmp_path / 'many').to_pylist()
    assert one == next(row for row in many if row['cell_id'] == CELL)
    assert one['ring1_active_cell_days_7d'] == 1
    assert one['parent4_active_cell_days_7d'] == 2
    assert one['parent3_active_cell_days_30d'] == 2


def test_future_activity_cannot_change_any_feature(tmp_path):
    history = [_event(CELL, ISSUE - timedelta(days=6)), _event(NEIGHBOR, ISSUE)]
    before = _activity(tmp_path / 'before', history)
    after = _activity(tmp_path / 'after', history + [
        _event(CELL, ISSUE + timedelta(days=1), viirs=500, modis=20, frp=900.),
        _event(NEIGHBOR, ISSUE + timedelta(days=7)),
    ])
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE)])
    a = _build(before, keys, tmp_path / 'a', labels=True).to_pylist()[0]
    b = _build(after, keys, tmp_path / 'b', labels=True).to_pylist()[0]
    assert {name: a[name] for name in FEATURE_NAMES} == {name: b[name] for name in FEATURE_NAMES}
    assert a['label'] == 0
    assert b['label'] == 1
    assert b['target_active_days'] == 1


def test_inclusive_windows_and_separate_label_boundaries(tmp_path):
    events = [_event(CELL, ISSUE + timedelta(days=offset), frp=10.)
              for offset in (-30, -29, -7, -6, 0, 1, 7, 8)]
    root = _activity(tmp_path / 'activity', events)
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE)])
    table = _build(root, keys, tmp_path / 'features', labels=True)
    row = table.to_pylist()[0]
    assert row['cell_active_days_1d'] == 1
    assert row['cell_active_days_7d'] == 2
    assert row['cell_active_days_30d'] == 4
    assert row['cell_active_days_90d'] == 5
    assert row['cell_viirs_detections_7d'] == 2
    assert row['cell_modis_detections_7d'] == 0
    assert row['cell_viirs_frp_mean_7d_mw'] == 10
    assert row['cell_modis_frp_mean_7d_mw'] is None
    assert row['days_since_last_detection'] == 0
    assert row['days_since_first_detection'] == (ISSUE - START).days
    assert row['target_start_date'] == ISSUE + timedelta(days=1)
    assert row['target_end_date'] == ISSUE + timedelta(days=7)
    assert row['target_active_days'] == 2
    assert row['label'] == 1
    assert table.column_names == [
        'cell_id', 'issue_date', 'target_start_date', 'target_end_date', 'horizon_days',
        *FEATURE_NAMES, 'label', 'target_active_days',
    ]
    assert len(FEATURE_NAMES) == 39
    assert all(table.schema.field(name).type == pa.float64() for name in FEATURE_NAMES)
    inference = _build(root, keys, tmp_path / 'inference')
    assert 'label' not in inference.column_names
    assert 'target_active_days' not in inference.column_names


def test_reporting_gaps_and_missing_frp_are_not_zero_filled(tmp_path):
    root = _activity(tmp_path / 'activity', [_event(CELL, ISSUE)],
                     missing=(ISSUE - timedelta(days=2),), sensors=('VIIRS',))
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE), (FAR, ISSUE)])
    rows = _build(root, keys, tmp_path / 'features').to_pylist()
    focal = next(row for row in rows if row['cell_id'] == CELL)
    quiet = next(row for row in rows if row['cell_id'] == FAR)
    assert focal['cell_active_days_1d'] == 1
    assert quiet['cell_active_days_1d'] == 0
    assert focal['cell_active_days_7d'] is None
    assert focal['ring1_active_cell_days_7d'] is None
    assert focal['cell_viirs_detections_7d'] is None
    assert focal['cell_modis_detections_1d'] is None
    assert focal['cell_viirs_frp_mean_7d_mw'] is None
    assert focal['modis_prior_month_supported_days'] == 0
    assert focal['modis_prior_month_active_fraction'] is None
    assert focal['modis_prior_active_years'] is None
    assert quiet['days_since_last_detection'] is None


def test_prior_calendar_year_month_and_reporting_platform_counts(tmp_path):
    root = _activity(tmp_path / 'activity', [
        _event(CELL, date(2012, 1, 31), viirs=0, modis=1),
        _event(CELL, date(2013, 1, 1), viirs=1, modis=1),
        _event(CELL, date(2013, 2, 1), viirs=1, modis=1),
        _event(CELL, date(2014, 1, 1), viirs=1, modis=1),
    ])
    platform_path = root / 'platform_days.parquet'
    platforms = pq.read_table(platform_path).to_pylist()
    platforms += [dict(sensor='VIIRS', platform='second', acquisition_date=ISSUE, detections=1),
                  dict(sensor='VIIRS', platform='future', acquisition_date=ISSUE + timedelta(days=1), detections=1)]
    pq.write_table(pa.Table.from_pylist(platforms), platform_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['outputs'] = _outputs(root)
    (root / 'manifest.json').write_text(json.dumps(manifest))
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE)])
    row = _build(root, keys, tmp_path / 'features').to_pylist()[0]
    assert row['modis_prior_month_supported_days'] == 62
    assert row['modis_prior_month_active_fraction'] == pytest.approx(2 / 62)
    assert row['viirs_prior_month_active_fraction'] == pytest.approx(1 / 62)
    assert row['modis_prior_active_years'] == 2
    assert row['modis_prior_supported_years'] == 2
    assert row['viirs_platforms_7d'] == 2
    assert row['modis_platforms_7d'] == 1


def test_sample_is_deterministic_and_candidate_eligibility_is_causal(tmp_path):
    future = ISSUE + timedelta(days=2)
    root = _activity(tmp_path / 'activity', [_event(CELL, ISSUE), _event(FAR, future)],
                     first={CELL: ISSUE, FAR: future})
    kwargs = dict(start=ISSUE, end=future, rate=1., seed='stable', memory_limit='128MB', threads=1)
    sample(root, tmp_path / 'all', **kwargs)
    pairs = {(row['cell_id'], row['issue_date']) for row in pq.read_table(tmp_path / 'all' / 'keys.parquet').to_pylist()}
    assert pairs == {(CELL, ISSUE), (CELL, ISSUE + timedelta(days=1)), (CELL, future), (FAR, future)}
    kwargs.update(rate=.5, start=START, end=END)
    sample(root, tmp_path / 'a', **kwargs)
    sample(root, tmp_path / 'b', **kwargs)
    assert pq.read_table(tmp_path / 'a' / 'keys.parquet').equals(pq.read_table(tmp_path / 'b' / 'keys.parquet'))
    keys = _keys(tmp_path / 'invalid.parquet', [(FAR, ISSUE)])
    with pytest.raises(ValueError, match='causally eligible'):
        _build(root, keys, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('pairs', [[(CELL, ISSUE), (CELL, ISSUE)], [('not-in-domain', ISSUE)], [(CELL, None)]])
def test_invalid_keys_are_rejected_atomically(tmp_path, pairs):
    root = _activity(tmp_path / 'activity', [])
    keys = _keys(tmp_path / 'keys.parquet', pairs)
    with pytest.raises(ValueError, match='unique|nonnull|eligible'):
        _build(root, keys, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.output-*'))


def test_future_mask_and_incomplete_horizons_are_rejected(tmp_path):
    root = _activity(tmp_path / 'future-mask', [], cutoff=(ISSUE + timedelta(days=1)).isoformat())
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE)])
    with pytest.raises(ValueError, match='reference_end'):
        _build(root, keys, tmp_path / 'mask-output')
    root = _activity(tmp_path / 'gap', [], missing=(ISSUE + timedelta(days=3),))
    with pytest.raises(ValueError, match='horizon'):
        _build(root, keys, tmp_path / 'gap-output', labels=True)
    late_keys = _keys(tmp_path / 'late.parquet', [(CELL, END - timedelta(days=6))])
    with pytest.raises(ValueError, match='horizon'):
        _build(root, late_keys, tmp_path / 'late-output', labels=True)
    assert _build(root, late_keys, tmp_path / 'late-inference').num_rows == 1


def test_initial_windows_are_null_and_manifest_records_contract(tmp_path):
    root = _activity(tmp_path / 'activity', [_event(CELL, START)])
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, START)])
    output = tmp_path / 'output'
    row = _build(root, keys, output).to_pylist()[0]
    assert row['cell_active_days_1d'] == 1
    assert row['cell_active_days_7d'] is None
    assert row['cell_active_days_365d'] is None
    assert row['viirs_prior_month_active_fraction'] is None
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['feature_names'] == list(FEATURE_NAMES)
    assert manifest['schema'] == 'global-weekly-features/1.0.0'
    assert manifest['positives'] is None
    assert manifest['labels'] is False
    assert manifest['rows'] == 1
    assert manifest['issue_range'] == {'min': START.isoformat(), 'max': START.isoformat()}
    assert manifest['outputs'][0]['size_bytes'] == (output / 'features.parquet').stat().st_size
    assert not (output / 'work.duckdb').exists()
    with pytest.raises(FileExistsError):
        build(root, keys, output, labels=False)


@pytest.mark.parametrize('change', ['bytes', 'extra', 'duplicate', 'missing'])
def test_activity_integrity_is_required_before_publication(tmp_path, change):
    root = _activity(tmp_path / 'activity', [_event(CELL, ISSUE)])
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if change == 'bytes':
        with (root / 'cells.parquet').open('ab') as stream:
            stream.write(b'corruption')
    elif change == 'extra':
        pq.write_table(pa.Table.from_pylist([_event(CELL, ISSUE)], schema=DAILY_SCHEMA),
                       root / 'daily' / 'undeclared.parquet')
    elif change == 'duplicate':
        manifest['outputs'].append(manifest['outputs'][0])
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest.pop('outputs')
        manifest_path.write_text(json.dumps(manifest))
    keys = _keys(tmp_path / 'keys.parquet', [(CELL, ISSUE)])
    with pytest.raises(ValueError, match='integrity|declared|unique|hashed'):
        _build(root, keys, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.output-*'))
