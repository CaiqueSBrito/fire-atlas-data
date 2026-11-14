"""Technical synthetic fixtures; no scientific fitting or performance claims."""
from datetime import date, timedelta
import json
from pathlib import Path

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data import global_training as training
from fire_atlas_data.global_features import FEATURE_NAMES, METHOD_VERSION
from fire_atlas_data.normalization import sha256
from fire_atlas_data.ranking import _json

DAYS = ('2019-01-20', '2019-01-21', '2019-02-01', '2019-02-02',
        '2019-03-01', '2019-03-02', '2019-04-01', '2019-04-02',
        '2019-05-01', '2019-05-02')


class FakeBooster:
    """Deterministic test double isolates temporal orchestration from training."""

    def __init__(self, base=.4, rounds=2):
        self.base = base
        self.rounds = rounds
        self.feature_names = list(FEATURE_NAMES)

    def inplace_predict(self, values, *, missing, validate_features):
        return np.clip(self.base + values[:, 0] * .1, 0, 1).astype(np.float32)

    def save_model(self, path):
        _json(path, dict(base=self.base, rounds=self.rounds, names=self.feature_names))

    def load_model(self, path):
        stored = json.loads(Path(path).read_text())
        self.base = stored['base']
        self.rounds = stored['rounds']
        self.feature_names = stored['names']

    def set_param(self, parameters):
        pass

    def save_config(self):
        return json.dumps({'learner': {'objective': {'name': 'binary:logistic'}}})

    def num_boosted_rounds(self):
        return self.rounds


def _fake_fit(data, config):
    return FakeBooster(base=float(data['label'].mean()) / 2, rounds=config['rounds'])


def _configuration(tmp_path, **changes):
    config = dict(schema='global-weekly-training/1.0.0', horizon_days=7, purge_days=7,
        train_start='2019-01-20',
        validation_folds=[dict(name='validation', train_end='2019-01-21', start='2019-02-01', end='2019-02-02')],
        final_train_end='2019-02-02', calibration=dict(name='calibration', start='2019-03-01', end='2019-03-02'),
        tests=[dict(name='test', start='2019-04-01', end='2019-04-02')],
        diagnostics=[dict(name='diagnostic', start='2019-05-01', end='2019-05-02')],
        engineering_fixture=True, rounds=2, nthread=1, max_matrix_rows=100, memory_limit='256MB')
    config.update(changes)
    path = tmp_path / 'config.json'
    _json(path, config)
    return path


def _features(tmp_path, *, single_class_test=False):
    root = tmp_path / 'features'
    root.mkdir()
    cells = sorted([h3.latlng_to_cell(-10, -45, 5), h3.latlng_to_cell(20, 30, 5)])
    columns = {name: [] for name in (*training.IDENTIFIERS, *FEATURE_NAMES, 'label', 'target_active_days')}
    for index, day in enumerate(DAYS):
        for cell_index, cell in enumerate(cells):
            issue = date.fromisoformat(day)
            latitude, longitude = h3.cell_to_latlng(cell)
            values = dict.fromkeys(FEATURE_NAMES, 0.)
            values.update(cell_active_days_1d=float(cell_index), cell_active_days_7d=float(cell_index + 1),
                          viirs_prior_month_active_fraction=None,
                          modis_prior_month_active_fraction=.1 + cell_index * .5,
                          cell_viirs_frp_mean_30d_mw=None, latitude=latitude, longitude=longitude)
            label = (index + cell_index) % 2
            if single_class_test and day.startswith('2019-04'):
                label = 0
            values.update(cell_id=cell, issue_date=issue, target_start_date=issue + timedelta(days=1),
                          target_end_date=issue + timedelta(days=7), horizon_days=7,
                          label=label, target_active_days=label * 2)
            for name in columns:
                columns[name].append(values[name])
    types = {'cell_id': pa.string(), 'issue_date': pa.date32(), 'target_start_date': pa.date32(),
             'target_end_date': pa.date32(), 'horizon_days': pa.int32(), 'label': pa.int32(),
             'target_active_days': pa.int32()}
    table = pa.table({name: pa.array(values, type=types.get(name, pa.float64())) for name, values in columns.items()})
    pq.write_table(table, root / 'features.parquet')
    manifest = dict(schema='global-weekly-features/1.0.0', feature_names=list(FEATURE_NAMES),
        feature_method_version=METHOD_VERSION, labels=True, rows=table.num_rows,
        positives=sum(columns['label']), issue_range={'min': DAYS[0], 'max': DAYS[-1]},
        keys_sha256='a' * 64,
        activity=dict(manifest_sha256='b' * 64, mask_manifest_sha256='c' * 64,
                      acquisition_range={'min': '2013-01-01', 'max': '2019-05-09'}))
    _refresh_manifest(root, manifest)
    return root


def _refresh_manifest(root, manifest=None):
    if manifest is None:
        manifest = json.loads((root / 'manifest.json').read_text())
    path = root / 'features.parquet'
    manifest['outputs'] = [dict(path=path.name, size_bytes=path.stat().st_size, sha256=sha256(path))]
    _json(root / 'manifest.json', manifest)


def _set_column(root, name, values):
    path = root / 'features.parquet'
    table = pq.read_table(path)
    index = table.schema.get_field_index(name)
    table = table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))
    pq.write_table(table, path)
    _refresh_manifest(root)


def _validation(scores):
    return {fold: {name: dict(average_precision=values[index]) for name, values in scores.items()}
            for index, fold in enumerate(('a', 'b', 'c'))}


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(training, '_fit', _fake_fit)
    monkeypatch.setattr(training.xgb, 'Booster', FakeBooster)


def test_scientific_configuration_fixes_expanding_years_and_gaps():
    config = training._config(Path(__file__).parents[1] / 'configs' / 'global-weekly.json')
    assert config['train_start'] == date(2013, 1, 20)
    assert [(fold['train_end'], fold['start'], fold['end']) for fold in config['validation_folds']] == [
        (date(year - 1, 12, 24), date(year, 1, 1), date(year, 12, 24)) for year in (2020, 2021, 2022)]
    assert config['final_train_end'] == date(2022, 12, 24)
    assert (config['calibration']['start'], config['calibration']['end']) == (date(2023, 1, 1), date(2023, 12, 24))
    assert [(part['start'], part['end']) for part in config['tests']] == [
        (date(year, 1, 1), date(year, 12, 24)) for year in (2024, 2025)]
    assert [(part['start'], part['end']) for part in config['diagnostics']] == [
        (date(2026, 1, 1), date(2026, 7, 24)), (date(2026, 8, 1), date(2026, 11, 6))]
    assert (config['rounds'], config['nthread']) == (300, 2)


@pytest.mark.parametrize('last_train', ['2019-01-25', '2019-02-01'])
def test_purge_rejects_targets_touching_evaluation_issue(tmp_path, last_train):
    path = _configuration(tmp_path, validation_folds=[dict(name='validation', train_end=last_train,
                                                           start='2019-02-01', end='2019-02-02')])
    with pytest.raises(ValueError, match='purge'):
        training._config(path)


def test_config_rejects_calibration_test_overlap_and_hidden_parameter_search(tmp_path):
    path = _configuration(tmp_path, tests=[dict(name='test', start='2019-03-09', end='2019-04-02')])
    with pytest.raises(ValueError, match='purge'):
        training._config(path)
    path = _configuration(tmp_path, engineering_fixture=False)
    with pytest.raises(ValueError, match='engineering_fixture'):
        training._config(path)
    path = _configuration(tmp_path, max_depth=7)
    with pytest.raises(ValueError, match='configuration fields'):
        training._config(path)


def test_baseline_selection_freezes_mean_winner_and_requires_every_strict_gain():
    config = dict(validation_folds=[{'name': name} for name in ('a', 'b', 'c')], nthread=2,
                  rounds=300, engineering_fixture=False)
    scores = _validation(dict(xgboost=[.7, .5, .9], persistence=[.2, .2, .9],
                              recurrence=[.5, .5, .5], climatology=[.4, .4, .4]))
    decision = training._select(scores, config)
    assert decision['reference_baseline'] == 'recurrence'
    assert decision['fold_ap_gains']['b'] == 0
    assert not decision['sustained_improvement']
    scores['b']['xgboost']['average_precision'] = .6
    assert training._select(scores, config)['sustained_improvement']
    scores['c']['xgboost']['average_precision'] = None
    assert not training._select(scores, config)['sustained_improvement']


def test_baselines_match_feature_scores_and_climatology_fallback():
    values = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    values[:, FEATURE_NAMES.index('cell_active_days_1d')] = [0, 1, 1]
    values[:, FEATURE_NAMES.index('cell_active_days_7d')] = [2, 5, 7]
    values[:, FEATURE_NAMES.index('viirs_prior_month_active_fraction')] = [.8, np.nan, np.nan]
    values[:, FEATURE_NAMES.index('modis_prior_month_active_fraction')] = [.1, .3, np.nan]
    result = training._baselines(dict(values=values))
    np.testing.assert_array_equal(result['persistence'], [0, 1, 1])
    np.testing.assert_array_equal(result['recurrence'], [2, 5, 7])
    np.testing.assert_allclose(result['climatology'], [.8, .3, 0])


def test_single_class_metrics_are_null_ap_with_finite_probability_scores():
    result = training._probability_metrics(np.zeros(3, dtype=np.int32), np.array([.1, .4, .8]),
                                           np.array([0., .5, 1.]))
    assert result['average_precision'] is None
    assert result['raw_average_precision'] is None
    assert result['brier'] == pytest.approx(1.25 / 3)
    assert np.isfinite(result['log_loss'])
    assert sum(item['rows'] for item in result['reliability']) == 3
    assert result['reliability'][-1]['rows'] == 1


def test_isotonic_maps_clamp_and_survive_json(tmp_path):
    mapping = training._isotonic(np.array([0., .2, .4, .6, .8, 1.]), np.array([0, 1, 0, 1, 1, 1]))
    path = tmp_path / 'calibrator.json'
    _json(path, mapping)
    restored = json.loads(path.read_text())
    probes = np.array([-1, 0, .3, .9, 1, 2])
    expected = training._calibrate(probes, mapping)
    np.testing.assert_array_equal(expected, training._calibrate(probes, restored))
    assert np.all(np.diff(expected) >= 0)
    assert expected[0] == expected[1] and expected[-1] == expected[-2]


def test_feature_contract_rejects_order_hash_and_method_changes(tmp_path):
    root = _features(tmp_path)
    path = root / 'manifest.json'
    original = json.loads(path.read_text())
    for key, value, message in [('feature_names', list(reversed(FEATURE_NAMES)), 'order'),
                                ('feature_method_version', 'future', 'version'),
                                ('labels', False, 'labeled')]:
        _json(path, dict(original, **{key: value}))
        with pytest.raises(ValueError, match=message):
            training._feature_contract(root)
    _json(path, original)
    with (root / 'features.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        training._feature_contract(root)


def test_feature_schema_rejects_permuted_parquet_even_with_matching_hash(tmp_path):
    root = _features(tmp_path)
    path = root / 'features.parquet'
    table = pq.read_table(path)
    pq.write_table(table.select(list(reversed(table.column_names))), path)
    _refresh_manifest(root)
    with pytest.raises(ValueError, match='schema/order'):
        training._feature_contract(root)


@pytest.mark.parametrize('corruption', ['duplicate', 'wrong_h3', 'future_window', 'unsorted'])
def test_keys_and_target_windows_are_enforced(tmp_path, corruption):
    root = _features(tmp_path)
    table = pq.read_table(root / 'features.parquet')
    if corruption == 'duplicate':
        values = table['cell_id'].to_pylist()
        values[1] = values[0]
        _set_column(root, 'cell_id', values)
    elif corruption == 'wrong_h3':
        values = table['cell_id'].to_pylist()
        values[0] = h3.cell_to_parent(values[0], 4)
        _set_column(root, 'cell_id', values)
    elif corruption == 'future_window':
        values = table['target_end_date'].to_pylist()
        values[0] += timedelta(days=1)
        _set_column(root, 'target_end_date', values)
    else:
        pq.write_table(table.take(pa.array([1, 0, *range(2, table.num_rows)])), root / 'features.parquet')
        _refresh_manifest(root)
    path, manifest = training._feature_contract(root)
    with duckdb.connect() as con, pytest.raises(ValueError):
        training._validate_keys(con, path, manifest)


@pytest.mark.parametrize('column,value', [('label', 2), ('target_active_days', 8),
                                         ('cell_active_days_30d', float('inf'))])
def test_partition_rejects_invalid_outcomes_and_nonfinite_features(tmp_path, column, value):
    root = _features(tmp_path)
    table = pq.read_table(root / 'features.parquet')
    values = table[column].to_pylist()
    values[0] = value
    _set_column(root, column, values)
    config = training._config(_configuration(tmp_path))
    with duckdb.connect() as con, pytest.raises(ValueError, match='Invalid labels'):
        training._load_partition(con, root / 'features.parquet', date(2019, 1, 20), date(2019, 1, 21), config)


def test_partition_streams_float32_preserving_nulls_and_refuses_truncation(tmp_path):
    root = _features(tmp_path)
    config = training._config(_configuration(tmp_path))
    with duckdb.connect() as con:
        data = training._load_partition(con, root / 'features.parquet', date(2019, 1, 20), date(2019, 1, 21), config)
        assert data['values'].dtype == np.float32
        assert data['values'].shape == (4, len(FEATURE_NAMES))
        assert np.isnan(data['values'][:, FEATURE_NAMES.index('cell_viirs_frp_mean_30d_mw')]).all()
        config['max_matrix_rows'] = 3
        with pytest.raises(ValueError, match='never truncated'):
            training._load_partition(con, root / 'features.parquet', date(2019, 1, 20), date(2019, 1, 21), config)


def test_training_freezes_before_any_heldout_outcome_reads(tmp_path, fake_backend, monkeypatch):
    root = _features(tmp_path, single_class_test=True)
    config_path = _configuration(tmp_path)
    original_load = training._load_partition
    original_validate = training._validate_all_outcomes
    calls = []

    def guarded_load(con, path, start, end, config):
        if start >= date(2019, 3, 1):
            stage = next(tmp_path.glob('.model-*'))
            assert (stage / 'selection.json').is_file()
        calls.append((start, end))
        return original_load(con, path, start, end, config)

    def guarded_validate(con, path, manifest):
        stage = next(tmp_path.glob('.model-*'))
        assert (stage / 'selection.json').is_file()
        return original_validate(con, path, manifest)

    monkeypatch.setattr(training, '_load_partition', guarded_load)
    monkeypatch.setattr(training, '_validate_all_outcomes', guarded_validate)
    result = training.train(root, config_path, tmp_path / 'model')
    assert calls[:2] == [(date(2019, 1, 20), date(2019, 1, 21)), (date(2019, 2, 1), date(2019, 2, 2))]
    assert result['data_cutoff_date'] == '2019-03-09'
    assert result['reload_prediction_equality']
    assert result['feature_method_version'] == METHOD_VERSION
    outputs = {item['path']: item for item in result['outputs']}
    assert {'model.json', 'calibrator.json', 'baseline_calibrators.json', 'selection.json',
            'metrics.json', 'assignments.parquet', 'predictions.parquet'} == set(outputs)
    for name, artifact in outputs.items():
        assert artifact['sha256'] == sha256(tmp_path / 'model' / name)
    predictions = pq.read_table(tmp_path / 'model' / 'predictions.parquet').to_pylist()
    assert {row['partition'] for row in predictions} == {'validation', 'calibration', 'test', 'diagnostic'}
    assert len(predictions) == 4 * 4 * 4
    assert all(0 <= row['probability'] <= 1 for row in predictions if row['calibrated'])
    metrics = json.loads((tmp_path / 'model' / 'metrics.json').read_text())
    assert metrics['tests']['test']['xgboost']['average_precision'] is None
    assert metrics['calibration']['in_sample']
    assert metrics['tests']['test']['xgboost']['latitude_month_strata']
    assert 'reference_skill' in metrics['tests']['test']['xgboost']
    assignments = pq.read_table(tmp_path / 'model' / 'assignments.parquet').to_pylist()
    final_train = [row for row in assignments if row['fold'] == 'final' and row['partition'] == 'train']
    assert max(row['issue_date'] for row in final_train) == date(2019, 2, 2)
    assert not list(tmp_path.glob('.model-*'))


def test_changed_heldout_labels_cannot_change_selection_or_model(tmp_path, fake_backend):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    first_features = _features(first)
    second_features = _features(second, single_class_test=True)
    training.train(first_features, _configuration(first), first / 'model')
    training.train(second_features, _configuration(second), second / 'model')
    for filename in ('model.json', 'calibrator.json', 'baseline_calibrators.json', 'selection.json'):
        assert (first / 'model' / filename).read_bytes() == (second / 'model' / filename).read_bytes()


def test_training_failure_is_atomic_and_existing_output_untouched(tmp_path, fake_backend):
    root = _features(tmp_path)
    config_path = _configuration(tmp_path, max_matrix_rows=3)
    output = tmp_path / 'model'
    with pytest.raises(ValueError, match='never truncated'):
        training.train(root, config_path, output)
    assert not output.exists() and not list(tmp_path.glob('.model-*'))
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('existing result')
    with pytest.raises(FileExistsError):
        training.train(root, config_path, output)
    assert sentinel.read_text() == 'existing result'


def test_native_reload_rejects_changed_predictions(tmp_path, monkeypatch):
    class CorruptBooster(FakeBooster):
        def load_model(self, path):
            super().load_model(path)
            self.base += .1

    monkeypatch.setattr(training.xgb, 'Booster', CorruptBooster)
    data = dict(values=np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32), label=np.array([0, 1]))
    with pytest.raises(ValueError, match='predictions differ'):
        training._reload(tmp_path, FakeBooster(), data, dict(nthread=1, rounds=2))


def test_unsupported_baseline_blocks_without_zero_filling_or_row_exclusion():
    values = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    values[0, FEATURE_NAMES.index('cell_active_days_7d')] = np.nan
    with pytest.raises(ValueError, match='missing baseline support'):
        training._baselines(dict(values=values))
    assert np.isnan(values[0, FEATURE_NAMES.index('cell_active_days_7d')])


def test_fixed_hist_fit_parameters_without_real_training(monkeypatch):
    observed = {}

    def quantile(values, **parameters):
        observed['values'] = values
        observed['matrix'] = parameters
        return 'quantile'

    def fit(parameters, matrix, *, num_boost_round):
        observed.update(parameters=parameters, rounds=num_boost_round, matrix_kind=matrix)
        return 'trained'

    monkeypatch.setattr(training.xgb, 'QuantileDMatrix', quantile)
    monkeypatch.setattr(training.xgb, 'train', fit)
    data = dict(values=np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32), label=np.array([0, 1]))
    assert training._fit(data, dict(nthread=2, rounds=300)) == 'trained'
    assert observed['values'] is data['values']
    assert observed['matrix']['feature_names'] == list(FEATURE_NAMES)
    assert observed['matrix']['label'] is data['label']
    assert observed['parameters'] == dict(objective='binary:logistic', tree_method='hist', max_depth=6,
                                          eta=.1, min_child_weight=20, reg_lambda=10, seed=42, nthread=2)
    assert observed['rounds'] == 300 and observed['matrix_kind'] == 'quantile'
