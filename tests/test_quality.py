from datetime import date, timedelta
import json

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from fire_atlas_data import quality as q
from fire_atlas_data import ranking as r


def _row(cell='a', label=0, split='train', day=date(2026, 9, 15)):
    return dict(cell_id=cell, issue_date=day, target_date=day + timedelta(days=1),
                region='fixture', label=label, split=split,
                **{name: float(label or 0) for name in q.ENHANCED_FEATURES})




def test_sampling_is_region_date_local_and_all_keeps_no_positive_groups():
    rows = [_row(f'negative{i}') for i in range(30)] + [_row('positive', 1)]
    empty = _row('no-positive-day', day=date(2026, 9, 16))
    other = _row('no-positive-region')
    other['region'] = 'other'
    rows += [empty, other, _row('validation', 1, 'validation')]
    sampled = q._sample(rows, 10)
    assert len(sampled) == 11 and sum(x['label'] for x in sampled) == 1
    assert empty not in sampled and other not in sampled
    assert q._sample(list(reversed(rows)), 10) == sampled
    all_rows = q._sample(rows, 'ALL')
    assert len(all_rows) == 33 and empty in all_rows and other in all_rows
    assert all(x['split'] == 'train' for x in all_rows)


def test_no_positive_folds_and_original_tie_retention():
    no_positives = q._summary([_row()], [1.], 10)
    assert no_positives['daily'][0]['recall'] is None
    assert no_positives['macro']['positive_days'] == 0
    assert no_positives['micro']['recall'] is None
    with pytest.raises(ValueError, match='no observed positives'):
        q._select([dict(name='original', metrics=no_positives)])
    results = [dict(name=name, metrics={'macro': {'recall': score}})
               for name, score in [('original', .5), ('simple', .5), ('complex', .5)]]
    assert q._select(results) == 'original'
    results[-1]['metrics']['macro']['recall'] = .6
    assert q._select(results) == 'complex'
    # This API accepts only validation metrics: arbitrary TEST information is ignored.
    results[0]['test'] = {'recall': 1.0}
    results[-1]['test'] = {'recall': 0.0}
    assert q._select(results) == 'complex'


def test_matrix_rejects_wrong_order_and_preserves_missing_values():
    with pytest.raises(ValueError, match='names/order'):
        q._matrix([_row()], list(reversed(q.ENHANCED_FEATURES)))
    row = _row()
    row['days_since_last'] = None
    assert np.isnan(q._matrix([row], r.FULL)[0, r.FULL.index('days_since_last')])


def _features(tmp_path, test_flip=False):
    directory = tmp_path / 'features'
    directory.mkdir(parents=True)
    rows = []
    cells = sorted(h3.grid_disk(h3.latlng_to_cell(-16., -47., 5), 2))[:15]
    for offset, split in enumerate(('train', 'validation', 'test', 'publish')):
        for i, cell in enumerate(cells):
            label = int(i < 3)
            row = _row(cell, label, split, date(2026, 9, 15) + timedelta(days=offset))
            if split == 'test' and test_flip:
                row['label'] = 1 - label
            if split == 'publish':
                row['label'] = None
            rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), directory / 'features.parquet')
    (directory / 'manifest.json').write_text(json.dumps(dict(method_version='1.1.0', config=dict(
        enhanced_features=True, minimum_history_days=14, bbox=[-48,-17,-46,-15], h3_resolution=5,
        region='fixture', sources=['fixture'], splits={'train': ['2026-09-15', '2026-09-15']}))))
    return directory, rows


def _baseline(tmp_path, rows):
    directory = tmp_path / 'baseline'
    directory.mkdir()
    train = [x for x in rows if x['split'] == 'train']
    matrix = xgb.DMatrix(q._matrix(train, r.FULL), label=[x['label'] for x in train])
    model = xgb.train(dict(objective='binary:logistic', max_depth=2, nthread=2, seed=42), matrix, num_boost_round=2)
    model.save_model(directory / 'full.json')
    (directory / 'manifest.json').write_text(json.dumps(dict(K=10, config={}, models=dict(full=dict(
        file='full.json', features=r.FULL, sha256=r._hash(directory / 'full.json'))))))
    return directory


def _small_search(monkeypatch):
    profiles = q._profiles()
    # Exercise actual fitting/reload, but keep the integration fixture inexpensive.
    monkeypatch.setattr(q, '_profiles', lambda snpp_history=False: [dict(s, params=dict(s['params'], n_estimators=2))
                                              for s in (profiles[0], profiles[-1])])


def test_training_rejects_original_baseline_for_changed_target(tmp_path):
    features, rows = _features(tmp_path)
    baseline = _baseline(tmp_path, rows)
    path = features / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['config']['target_mode'] = 'any_activity'
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'wrong-target-model'
    with pytest.raises(ValueError, match='baseline target differs'):
        q.train(features, output, baseline)
    assert not output.exists()


def test_train_reload_publish_and_test_never_guides_selection(tmp_path, monkeypatch):
    features, rows = _features(tmp_path)
    baseline = _baseline(tmp_path, rows)
    _small_search(monkeypatch)
    original_predict = q._predict
    seen_test_calls = []

    def guarded_predict(directory, spec, subset):
        if subset[0]['split'] == 'test':
            frozen = json.loads((directory / 'selection.json').read_text())
            assert frozen['selected_model']
            seen_test_calls.append(spec['file'])
        return original_predict(directory, spec, subset)

    monkeypatch.setattr(q, '_predict', guarded_predict)
    output = tmp_path / 'model'
    manifest = q.train(features, output, baseline)
    assert len(seen_test_calls) == (1 if manifest['selected_model'] == 'original' else 2)
    comparison = json.loads((output / 'comparison.json').read_text())
    assert comparison['test_reused_diagnostic'] is True
    assert set(comparison['recurrence_references']['test']) == {'neighbors_7d', 'cell_active_days_30d'}
    predictions = pq.read_table(output / 'test_predictions.parquet').to_pylist()
    expected = {(x['cell_id'], x['issue_date'], x['label']) for x in rows if x['split'] == 'test'}
    for name in comparison['test']:
        actual = [x for x in predictions if x['model_name'] == name]
        assert {(x['cell_id'], x['issue_date'], x['label']) for x in actual} == expected
        assert sum(x['selected_top_k'] for x in actual) == 10
        assert sum(x['label'] for x in actual if x['selected_top_k']) == comparison['test'][name]['micro']['hits']
    assert manifest['outputs']['test_predictions.parquet']['sha256'] == r._hash(output / 'test_predictions.parquet')
    q.publish(features, output, tmp_path / 'published')
    published = pq.read_table(tmp_path / 'published' / 'rankings.parquet').to_pylist()
    assert len(published) == 10
    assert all(x['target_date_utc'] == date(2026, 9, 19) for x in published)
    # Change only TEST labels and retrain: same validation metrics and same frozen winner.
    other_features, _ = _features(tmp_path / 'other', test_flip=True)
    other = q.train(other_features, tmp_path / 'other-model', baseline)
    other_comparison = json.loads((tmp_path / 'other-model' / 'comparison.json').read_text())
    assert other['selected_model'] == manifest['selected_model']
    assert other_comparison['validation'] == comparison['validation']
    # Native order, artifact hash and publication feature types are all checked.
    bad_spec = dict(manifest['models'][manifest['selected_model']], features=list(reversed(r.FULL)))
    with pytest.raises(ValueError, match='names/order'):
        original_predict(output, bad_spec, [x for x in rows if x['split'] == 'publish'])
    manifest['feature_contract']['feature_types']['neighbors_3d'] = 'wrong'
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contract differs'):
        q.publish(features, output, tmp_path / 'bad-schema')
    manifest['feature_contract']['feature_types']['neighbors_3d'] = 'double'
    (output / 'manifest.json').write_text(json.dumps(manifest))
    (output / 'comparison.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact hash'):
        q.publish(features, output, tmp_path / 'bad-artifact')


def test_contract_allows_dates_and_sources_not_geometry(tmp_path):
    features, _ = _features(tmp_path)
    _, _, types = q._load(features)
    first, _ = q._contract(features, types)
    metadata = features / 'manifest.json'
    obj = json.loads(metadata.read_text())
    obj['config'].update(sources=['new-source'], splits={}, warmup_start='2026-10-01', publish_date='2026-11-01')
    metadata.write_text(json.dumps(obj))
    assert q._contract(features, types)[0] == first
    obj['config']['h3_resolution'] = 6
    metadata.write_text(json.dumps(obj))
    assert q._contract(features, types)[0] != first


def test_publish_labels_and_existing_output_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        q.train('unused', tmp_path)
    with pytest.raises(FileExistsError):
        q.publish('unused', 'unused', tmp_path)
    features, rows = _features(tmp_path)
    rows[-1]['label'] = 0
    pq.write_table(pa.Table.from_pylist(rows), features / 'features.parquet')
    with pytest.raises(ValueError, match='Publish labels must be null'):
        q._load(features)


def _frozen_evaluation_fixture(tmp_path, regressed=True):
    features, rows = _features(tmp_path)
    model_dir = _baseline(tmp_path, rows)
    _, files, types = q._load(features)
    contract, provenance = q._contract(features, types)
    spec = dict(file='full.json', sha256=r._hash(model_dir / 'full.json'),
                features=r.FULL, original_unnamed=True)
    manifest = dict(selected_model='winner', model_version='frozen-fixture', K=10,
                    models={'original': spec, 'winner': spec},
                    feature_contract=contract, feature_manifest=provenance,
                    ranking_code_sha256=r._hash(r.__file__),
                    quality_code_sha256='historical-code-hash',
                    inputs=[dict(path=str(f), sha256=r._hash(f)) for f in files])
    r._json(model_dir / 'selection.json', dict(selected_model='winner', model_version='frozen-fixture'))
    r._json(model_dir / 'comparison.json', dict(
        diagnostic_test_regression=regressed,
        test={'original': {'daily': [{'issue_date': '2026-09-17'}]}}))
    manifest['outputs'] = {name: dict(sha256=r._hash(model_dir / name))
                           for name in ('selection.json', 'comparison.json')}
    r._json(model_dir / 'manifest.json', manifest)
    new_features, new_rows = _features(tmp_path / 'new')
    for row in new_rows:
        row['issue_date'] += timedelta(days=60)
        row['target_date'] += timedelta(days=60)
    pq.write_table(pa.Table.from_pylist(new_rows), new_features / 'features.parquet')
    return new_features, model_dir, new_rows


def test_independent_evaluate_never_fits_selects_or_unlocks(tmp_path, monkeypatch):
    features, model_dir, rows = _frozen_evaluation_fixture(tmp_path)
    frozen_manifest_hash = r._hash(model_dir / 'manifest.json')
    frozen_selection_hash = r._hash(model_dir / 'selection.json')

    def prohibited(*args, **kwargs):
        raise AssertionError('Evaluation must never fit/search/select')

    for name in ('_fit', '_select', '_profiles'):
        monkeypatch.setattr(q, name, prohibited)
    output = tmp_path / 'independent'
    report = q.evaluate(features, model_dir, output)
    assert report['evaluation_type'] == 'independent_evaluation'
    assert report['promotion_unlocked'] is False
    assert report['original_diagnostic_test_regression'] is True
    assert report['frozen_quality_code_sha256'] == 'historical-code-hash'
    assert report['evaluator_code_sha256'] == r._hash(q.__file__)
    assert r._hash(model_dir / 'manifest.json') == frozen_manifest_hash
    assert r._hash(model_dir / 'selection.json') == frozen_selection_hash
    predictions = pq.read_table(output / 'test_predictions.parquet').to_pylist()
    test = [row for row in rows if row['split'] == 'test']
    expected = {(row['cell_id'], row['issue_date'], row['label']) for row in test}
    assert set(report['test']) == {'original', 'winner'}
    for name, metrics in report['test'].items():
        actual = [row for row in predictions if row['model_name'] == name]
        assert {(row['cell_id'], row['issue_date'], row['label']) for row in actual} == expected
        assert metrics['micro']['positives'] == sum(row['label'] for row in test)
        assert metrics['micro']['hits'] == sum(row['label'] for row in actual if row['selected_top_k'])
        assert metrics['micro']['selected'] == 10
    evidence = json.loads((output / 'independent_evaluation.json').read_text())
    assert evidence['artifacts']['comparison.json']['sha256'] == r._hash(output / 'comparison.json')
    assert report['artifacts']['test_predictions.csv']['sha256'] == r._hash(output / 'test_predictions.csv')
    with pytest.raises(ValueError, match='Publication blocked.*regressed'):
        q.publish(features, model_dir, tmp_path / 'blocked')
    with pytest.raises(FileExistsError):
        q.evaluate(features, model_dir, output)


def test_independent_evaluate_rejects_old_dates_contract_and_corruption(tmp_path):
    features, model_dir, rows = _frozen_evaluation_fixture(tmp_path)
    # An ordinary diagnostic dataset cannot be relabeled as independent evidence.
    with pytest.raises(ValueError, match='must follow'):
        q.evaluate(tmp_path / 'features', model_dir, tmp_path / 'old')
    metadata = features / 'manifest.json'
    original = json.loads(metadata.read_text())
    changed = json.loads(metadata.read_text())
    changed['config']['h3_resolution'] = 6
    metadata.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='contract differs'):
        q.evaluate(features, model_dir, tmp_path / 'wrong-grid')
    metadata.write_text(json.dumps(original))
    with (model_dir / 'full.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='Model artifact hash'):
        q.evaluate(features, model_dir, tmp_path / 'corrupt-model')


def test_independent_evaluate_without_positives_is_insufficient_not_failure(tmp_path, monkeypatch):
    features, model_dir, rows = _frozen_evaluation_fixture(tmp_path)
    for row in rows:
        if row['split'] == 'test':
            row['label'] = 0
    pq.write_table(pa.Table.from_pylist(rows), features / 'features.parquet')

    def prohibited(*args, **kwargs):
        raise AssertionError('Zero-positive evidence must not trigger fitting or reselection')

    monkeypatch.setattr(q, '_fit', prohibited)
    monkeypatch.setattr(q, '_select', prohibited)
    report = q.evaluate(features, model_dir, tmp_path / 'zero-positive-evidence')
    assert report['observed_positive_count'] == 0
    assert report['gain_measurement_supported'] is False
    assert report['promotion_unlocked'] is False
    assert 'Insufficient' in report['conclusion']
    for metrics in report['test'].values():
        assert metrics['micro']['recall'] is None
        assert metrics['macro']['recall'] is None
        assert metrics['micro']['hits'] == metrics['micro']['positives'] == 0
        assert all(day['recall'] is None for day in metrics['daily'])


def test_input_coverage_records_gap_without_certifying_availability(tmp_path):
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([
        dict(platform='NOAA-20', sensor='VIIRS', acquisition_date=day)
        for day in (date(2026, 10, 22), date(2026, 11, 5), date(2026, 11, 6))
    ]), source)
    coverage = q._input_coverage({'sources': [dict(path=str(source), sha256=r._hash(source))]})
    assert coverage['dates_without_supplied_noaa20_detections'] == [
        dict(start='2026-10-23', end='2026-11-04')]
    assert coverage['continuous_observation_coverage_certified'] is False
    assert 'calendar age' in coverage['interpretation']
    assert 'not a clean' in coverage['covariate_shift']


def test_history_profiles_preserve_search_and_native_contract(tmp_path):
    legacy, history = q._profiles(), q._profiles(True)
    assert len(history) == len(legacy) == 11
    assert history[0] == legacy[0]
    for old, new in zip(legacy[1:], history[1:]):
        assert old['params'] == new['params'] and old['policy'] == new['policy']
        assert new['features'] == old['features'] + q.HISTORY_FEATURES[len(q.ENHANCED_FEATURES):]
    rows = [_row(str(i), i % 2) for i in range(12)]
    for row in rows:
        row.update({name: float(row['label']) for name in q.HISTORY_FEATURES[len(q.ENHANCED_FEATURES):]})
    rows[0]['snpp_cell_frp_mean'] = None
    spec = dict(history[1], params=dict(history[1]['params'], n_estimators=2))
    model = q._fit(rows, spec)
    path = tmp_path/'native.json'
    model.save_model(path)
    saved = dict(spec, file=path.name, sha256=r._hash(path))
    assert model.feature_names == q.HISTORY_FEATURES
    np.testing.assert_array_equal(q._predict(tmp_path, saved, rows),
        model.predict(xgb.DMatrix(q._matrix(rows, spec['features']), feature_names=spec['features'])))
    with pytest.raises(ValueError, match='names/order'):
        q._matrix(rows, list(reversed(q.HISTORY_FEATURES)))
    assert q._matrix(rows, r.FULL).shape == (12, 9)
