"""Fixed, temporally isolated global weekly probability experiment."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
import xgboost as xgb

from .global_features import FEATURE_NAMES, HORIZON_DAYS, METHOD_VERSION as FEATURE_METHOD_VERSION
from .normalization import identifier, literal, sha256
from .ranking import _json

BASELINES = ('persistence', 'recurrence', 'climatology')
MODEL_NAMES = ('xgboost', *BASELINES)
IDENTIFIERS = ('cell_id', 'issue_date', 'target_start_date', 'target_end_date', 'horizon_days')
PARAMS = dict(objective='binary:logistic', tree_method='hist', max_depth=6, eta=.1,
              min_child_weight=20, reg_lambda=10, seed=42, nthread=2)
BATCH_ROWS = 65536
LIMITATIONS = [
    'The target is supplied non-static detections, not verified physical fire occurrence.',
    'Unknown FIRMS types outside the frozen mask may include unrecognized static sources.',
    'Candidate cells have prior accepted detections; this is not an exhaustive land grid.',
    'Overlapping target windows and repeated cells are not independent events.',
    'Sensor reporting and coverage vary; a missing observation is not verified absence.',
    'Calibration reflects the supplied key sample, not an independently measured population.',
    'Missing persistence/recurrence support blocks evaluation; no row exclusion or zero imputation is applied.',
    '2024-2025 previously inspected data are held-out diagnostics, not independent proof.',
    'No operational or geographic-transfer claim follows from this experiment.',
]
PREDICTION_SCHEMA = pa.schema([
    ('cell_id', pa.string()), ('issue_date', pa.date32()),
    ('target_start_date', pa.date32()), ('target_end_date', pa.date32()),
    ('horizon_days', pa.int32()), ('label', pa.int32()),
    ('fold', pa.string()), ('partition', pa.string()), ('period', pa.string()),
    ('model', pa.string()), ('raw_score', pa.float64()),
    ('probability', pa.float64()), ('calibrated', pa.bool_()),
])


def _day(value):
    if not isinstance(value, str):
        raise ValueError('Protocol dates must be ISO calendar dates')
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError('Protocol dates must be YYYY-MM-DD')
    return result


def _interval(value, *, training=False):
    fields = {'name', 'start', 'end'} | ({'train_end'} if training else set())
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f'Interval requires exactly {sorted(fields)}')
    if not isinstance(value['name'], str) or not value['name'] or value['name'] == 'final':
        raise ValueError('Intervals require nonempty names other than final')
    result = {key: value[key] if key == 'name' else _day(value[key]) for key in fields}
    if result['start'] > result['end']:
        raise ValueError('Interval start exceeds end')
    return result


def _purged(end, start, purge):
    if end + timedelta(days=purge) >= start:
        raise ValueError('Partitions overlap or violate the seven-day outcome purge')


def _config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'schema', 'train_start', 'validation_folds', 'final_train_end',
                'calibration', 'tests', 'diagnostics', 'horizon_days', 'purge_days'}
    optional = {'rounds', 'nthread', 'engineering_fixture', 'memory_limit', 'max_matrix_rows'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError('Invalid global weekly training configuration fields')
    if config['schema'] != 'global-weekly-training/1.0.0':
        raise ValueError('Require global-weekly-training/1.0.0 configuration')
    if config['horizon_days'] != HORIZON_DAYS or config['purge_days'] != HORIZON_DAYS:
        raise ValueError('Global weekly training requires horizon and purge of seven days')
    config.setdefault('rounds', 300)
    config.setdefault('nthread', 2)
    config.setdefault('engineering_fixture', False)
    config.setdefault('memory_limit', '2GB')
    config.setdefault('max_matrix_rows', 2000000)
    for key in ('rounds', 'nthread', 'max_matrix_rows'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['engineering_fixture']) is not bool:
        raise ValueError('engineering_fixture must be boolean')
    if not config['engineering_fixture'] and (config['rounds'] != 300 or config['nthread'] != 2):
        raise ValueError('Nondefault rounds/nthread require explicit engineering_fixture=true')
    if not isinstance(config['memory_limit'], str) or not config['memory_limit']:
        raise ValueError('memory_limit must be a nonempty DuckDB memory limit')
    config['train_start'] = _day(config['train_start'])
    config['final_train_end'] = _day(config['final_train_end'])
    for key in ('validation_folds', 'tests', 'diagnostics'):
        if not isinstance(config[key], list):
            raise ValueError(f'{key} must be a list')
    if not config['validation_folds'] or not config['tests']:
        raise ValueError('Require nonempty validation folds and test periods')
    config['validation_folds'] = [_interval(fold, training=True) for fold in config['validation_folds']]
    config['calibration'] = _interval(config['calibration'])
    config['tests'] = [_interval(item) for item in config['tests']]
    config['diagnostics'] = [_interval(item) for item in config['diagnostics']]
    periods = [*config['validation_folds'], config['calibration'], *config['tests'], *config['diagnostics']]
    if len({item['name'] for item in periods}) != len(periods):
        raise ValueError('Period names must be unique')
    folds = config['validation_folds']
    for fold in folds:
        if fold['train_end'] < config['train_start']:
            raise ValueError('Training interval is empty')
        _purged(fold['train_end'], fold['start'], config['purge_days'])
    for previous, following in zip(folds, folds[1:]):
        if following['train_end'] < previous['end']:
            raise ValueError('Validation folds must use expanding training intervals')
    if config['final_train_end'] < folds[-1]['end']:
        raise ValueError('Final training must include all validation years')
    _purged(config['final_train_end'], config['calibration']['start'], config['purge_days'])
    for previous, following in zip(periods, periods[1:]):
        _purged(previous['end'], following['start'], config['purge_days'])
    return config


def _serializable_config(config):
    return json.loads(json.dumps(config, default=str))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def _feature_contract(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'global-weekly-features/1.0.0' or manifest.get('labels') is not True:
        raise ValueError('Require labeled global-weekly-features/1.0.0')
    if manifest.get('feature_method_version') != FEATURE_METHOD_VERSION:
        raise ValueError('Feature transformation version mismatch')
    if manifest.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Feature names/order mismatch')
    path = directory / 'features.parquet'
    outputs = manifest.get('outputs', [])
    matches = [item for item in outputs if item.get('path') == path.name]
    if len(matches) != 1 or matches[0].get('sha256') != sha256(path):
        raise ValueError('Feature artifact hash mismatch')
    if matches[0].get('size_bytes') != path.stat().st_size:
        raise ValueError('Feature artifact size mismatch')
    expected = pa.schema([('cell_id', pa.string()), ('issue_date', pa.date32()),
                          ('target_start_date', pa.date32()), ('target_end_date', pa.date32()),
                          ('horizon_days', pa.int32()),
                          *((name, pa.float64()) for name in FEATURE_NAMES),
                          ('label', pa.int32()), ('target_active_days', pa.int32())])
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise ValueError('Feature Parquet schema/order mismatch')
    activity = manifest.get('activity', {})
    if not all(_digest(value) for value in (activity.get('manifest_sha256'),
                                          activity.get('mask_manifest_sha256'), manifest.get('keys_sha256'))):
        raise ValueError('Require activity, mask and key provenance hashes')
    acquisition = activity.get('acquisition_range', {})
    if _day(acquisition.get('min')) > _day(acquisition.get('max')):
        raise ValueError('Invalid acquisition range')
    return path, manifest


def _validate_keys(con, path, manifest):
    rows, first, last, invalid = con.execute('''SELECT count(*), min(issue_date), max(issue_date),
        count(*) FILTER (WHERE cell_id IS NULL OR issue_date IS NULL OR target_start_date IS NULL
        OR target_end_date IS NULL OR horizon_days IS NULL OR horizon_days != 7
        OR target_start_date != issue_date + 1 OR target_end_date != issue_date + 7)
        FROM read_parquet(?)''', [str(path)]).fetchone()
    if not rows or invalid:
        raise ValueError('Empty features or invalid keys/target windows')
    if type(manifest.get('rows')) is not int or rows != manifest['rows']:
        raise ValueError('Feature manifest row count mismatch')
    if manifest.get('issue_range') != {'min': str(first), 'max': str(last)}:
        raise ValueError('Feature manifest issue range mismatch')
    acquisition_end = _day(manifest['activity']['acquisition_range']['max'])
    if last + timedelta(days=HORIZON_DAYS) > acquisition_end:
        raise ValueError('Labels extend beyond supplied acquisition support')
    previous = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=['issue_date', 'cell_id']):
        days = batch.column(0).to_numpy(zero_copy_only=False).astype('datetime64[D]')
        cells = batch.column(1).to_pylist()
        for cell in set(cells):
            if not h3.is_valid_cell(cell) or h3.get_resolution(cell) != 5:
                raise ValueError('Feature keys must be valid H3 resolution-five cells')
        for day, cell in zip(days, cells):
            current = (day, cell)
            if previous is not None and current <= previous:
                raise ValueError('Feature keys must be unique and sorted by issue_date, cell_id')
            previous = current


def _bad_data_sql():
    invalid_features = ' OR '.join(
        f'({identifier(name)} IS NOT NULL AND (NOT isfinite({identifier(name)}) '
        f'OR abs({identifier(name)}) > {float(np.finfo(np.float32).max)}))' for name in FEATURE_NAMES)
    return f'''label IS NULL OR label NOT IN (0,1) OR target_active_days IS NULL
        OR target_active_days NOT BETWEEN 0 AND 7 OR label != (target_active_days > 0)::INTEGER
        OR latitude IS NULL OR latitude NOT BETWEEN -90 AND 90
        OR longitude IS NULL OR longitude NOT BETWEEN -180 AND 180 OR {invalid_features}'''


def _validate_all_outcomes(con, path, manifest):
    # Deliberately called only after selection.json has been durably closed.
    positives, invalid = con.execute(f'''SELECT sum(label), count(*) FILTER (WHERE {_bad_data_sql()})
        FROM read_parquet(?)''', [str(path)]).fetchone()
    if invalid:
        raise ValueError('Invalid labels, active-day outcomes or finite float32 features')
    if type(manifest.get('positives')) is not int or positives != manifest['positives']:
        raise ValueError('Feature manifest positive count mismatch')


def _partitions(config):
    result = []
    for fold in config['validation_folds']:
        result.extend([
            dict(fold=fold['name'], partition='train', period=fold['name'],
                 start=config['train_start'], end=fold['train_end']),
            dict(fold=fold['name'], partition='validation', period=fold['name'],
                 start=fold['start'], end=fold['end']),
        ])
    result.append(dict(fold='final', partition='train', period='final',
                       start=config['train_start'], end=config['final_train_end']))
    for kind, intervals in [('calibration', [config['calibration']]),
                             ('test', config['tests']), ('diagnostic', config['diagnostics'])]:
        result.extend(dict(fold='final', partition=kind, period=item['name'],
                           start=item['start'], end=item['end']) for item in intervals)
    return result


def _assignments(con, path, stage, config):
    selections = []
    for part in _partitions(config):
        selections.append(f'''SELECT cell_id, issue_date, target_start_date, target_end_date,
            {literal(part['fold'])} AS fold, {literal(part['partition'])} AS partition,
            {literal(part['period'])} AS period FROM read_parquet({literal(str(path))})
            WHERE issue_date BETWEEN DATE {literal(str(part['start']))} AND DATE {literal(str(part['end']))}''')
    con.execute(f'''COPY ({' UNION ALL '.join(selections)} ORDER BY fold, partition, period, issue_date, cell_id)
        TO {literal(str(stage / 'assignments.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)''')


def _load_partition(con, path, start, end, config):
    rows, invalid = con.execute(f'''SELECT count(*), count(*) FILTER (WHERE {_bad_data_sql()})
        FROM read_parquet(?) WHERE issue_date BETWEEN ? AND ?''', [str(path), start, end]).fetchall()[0]
    if not rows:
        raise ValueError(f'Empty requested partition: {start} through {end}')
    if rows > config['max_matrix_rows']:
        minimum = rows * len(FEATURE_NAMES) * np.dtype(np.float32).itemsize
        raise ValueError(f'Partition has {rows} rows, above max_matrix_rows={config["max_matrix_rows"]}; '
                         f'the float32 matrix alone requires {minimum} bytes plus XGBoost/metadata overhead. '
                         'Provide sufficient RAM and explicitly raise max_matrix_rows; rows are never truncated.')
    if invalid:
        raise ValueError('Invalid labels, active-day outcomes or finite float32 features')
    data = dict(values=np.empty((rows, len(FEATURE_NAMES)), dtype=np.float32),
                label=np.empty(rows, dtype=np.int32), cell_id=np.empty(rows, dtype=object),
                issue_date=np.empty(rows, dtype='datetime64[D]'))
    columns = ['cell_id', 'issue_date', 'label', *FEATURE_NAMES]
    reader = con.sql(f'''SELECT {','.join(identifier(name) for name in columns)}
        FROM read_parquet(?) WHERE issue_date BETWEEN ? AND ? ORDER BY issue_date, cell_id''',
        params=[str(path), start, end]).to_arrow_reader(BATCH_ROWS)
    offset = 0
    for batch in reader:
        stop = offset + batch.num_rows
        data['cell_id'][offset:stop] = batch.column(0).to_pylist()
        data['issue_date'][offset:stop] = batch.column(1).to_numpy(zero_copy_only=False).astype('datetime64[D]')
        data['label'][offset:stop] = batch.column(2).to_numpy(zero_copy_only=False)
        for index in range(len(FEATURE_NAMES)):
            data['values'][offset:stop, index] = batch.column(index + 3).to_numpy(zero_copy_only=False)
        offset = stop
    if offset != rows:
        raise ValueError('Partition changed during streaming read')
    return data


def _matrix(data, config):
    return xgb.QuantileDMatrix(data['values'], label=data['label'], feature_names=list(FEATURE_NAMES),
                               missing=np.nan, nthread=config['nthread'])


def _fit(data, config):
    if len(np.unique(data['label'])) != 2:
        raise ValueError('Training requires both binary outcome classes')
    return xgb.train(dict(PARAMS, nthread=config['nthread']), _matrix(data, config),
                     num_boost_round=config['rounds'])


def _predict(model, data, config):
    values = data['values']
    if model.feature_names != list(FEATURE_NAMES) or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Model feature names/order mismatch')
    if values.dtype != np.float32 or not values.flags.c_contiguous:
        raise ValueError('Scoring requires the bounded contiguous float32 matrix')
    result = model.inplace_predict(values, missing=np.nan, validate_features=True)
    if result.shape != data['label'].shape or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Model must produce one finite probability per row')
    return result


def _baselines(data):
    values = data['values']
    persistence = values[:, FEATURE_NAMES.index('cell_active_days_1d')]
    recurrence = values[:, FEATURE_NAMES.index('cell_active_days_7d')]
    viirs = values[:, FEATURE_NAMES.index('viirs_prior_month_active_fraction')]
    modis = values[:, FEATURE_NAMES.index('modis_prior_month_active_fraction')]
    if not np.isfinite(persistence).all() or not np.isfinite(recurrence).all():
        raise ValueError('Persistence/recurrence require supported activity windows; missing baseline support '
                         'blocks this protocol. Prepare a supported sample or revise the protocol explicitly; '
                         'rows are never silently excluded or zero-imputed.')
    return dict(persistence=persistence, recurrence=recurrence,
                climatology=np.where(np.isnan(viirs), np.where(np.isnan(modis), 0, modis), viirs))


def _ap(labels, scores):
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def _select(validation, config):
    fold_names = [fold['name'] for fold in config['validation_folds']]
    means = {}
    for name in MODEL_NAMES:
        scores = [validation[fold][name]['average_precision'] for fold in fold_names]
        defined = [value for value in scores if value is not None]
        means[name] = float(np.mean(defined)) if defined else None
    reference = max(BASELINES, key=lambda name: means[name] if means[name] is not None else -1)
    gains = {}
    for fold in fold_names:
        model = validation[fold]['xgboost']['average_precision']
        baseline = validation[fold][reference]['average_precision']
        gains[fold] = model - baseline if model is not None and baseline is not None else None
    return dict(schema='global-weekly-selection/1.0.0', model='xgboost', primary_metric='average_precision',
                mean_validation_ap=means, reference_baseline=reference, fold_ap_gains=gains,
                sustained_improvement=all(value is not None and value > 0 for value in gains.values()),
                baseline_tie_order=list(BASELINES), validation_folds=validation,
                frozen_before_calibration_and_test_outcomes=True,
                parameters=dict(PARAMS, nthread=config['nthread']), rounds=config['rounds'],
                engineering_fixture=config['engineering_fixture'])


def _isotonic(raw, labels):
    fit = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(raw, labels)
    return dict(method='isotonic', x=fit.X_thresholds_.tolist(), y=fit.y_thresholds_.tolist())


def _calibrate(raw, mapping):
    return np.interp(raw, mapping['x'], mapping['y'])


def _probability_metrics(labels, raw, probabilities):
    result = dict(rows=int(len(labels)), positives=int(labels.sum()),
                  average_precision=_ap(labels, probabilities if probabilities is not None else raw),
                  raw_average_precision=_ap(labels, raw), brier=None, log_loss=None, reliability=[])
    if probabilities is None:
        return result
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid calibrated probability')
    result['brier'] = float(np.mean((probabilities - labels) ** 2))
    clipped = np.clip(probabilities, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    result['log_loss'] = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
    bins = np.minimum((probabilities * 10).astype(np.int32), 9)
    for index in range(10):
        selected = bins == index
        count = int(selected.sum())
        result['reliability'].append(dict(lower=index / 10, upper=(index + 1) / 10,
            upper_inclusive=index == 9, rows=count,
            mean_probability=float(probabilities[selected].mean()) if count else None,
            observed_fraction=float(labels[selected].mean()) if count else None))
    return result


def _reference_skill(reports, reference):
    for report in reports.values():
        report['reference_skill'] = dict(reference=reference)
        for metric in ('brier', 'log_loss'):
            denominator = reports[reference][metric]
            report['reference_skill'][metric] = (1 - report[metric] / denominator
                if denominator is not None and denominator > 0 and report[metric] is not None else None)
        ap, reference_ap = report['average_precision'], reports[reference]['average_precision']
        report['reference_skill']['average_precision_gain'] = (
            ap - reference_ap if ap is not None and reference_ap is not None else None)


def _reports(data, raw, probabilities, reference=None):
    reports = {name: _probability_metrics(data['label'], raw[name], probabilities.get(name)) for name in MODEL_NAMES}
    if reference is not None:
        _reference_skill(reports, reference)
    latitudes = data['values'][:, FEATURE_NAMES.index('latitude')]
    bands = np.minimum(np.floor((latitudes + 90) / 30).astype(np.int32), 5)
    months = data['issue_date'].astype('datetime64[M]').astype(np.int64) % 12 + 1
    groups = bands * 12 + months - 1
    for report in reports.values():
        report['latitude_month_strata'] = []
    for group in np.unique(groups):
        selected = groups == group
        stratum = {name: _probability_metrics(data['label'][selected], raw[name][selected],
                   probabilities[name][selected] if name in probabilities else None) for name in MODEL_NAMES}
        if reference is not None:
            _reference_skill(stratum, reference)
        for name in MODEL_NAMES:
            stratum[name].update(latitude_min=int(group // 12) * 30 - 90,
                                 latitude_max=int(group // 12) * 30 - 60, month=int(group % 12) + 1)
            reports[name]['latitude_month_strata'].append(stratum[name])
    return reports


def _export_predictions(writer, data, raw, probabilities, *, fold, partition, period, calibrated):
    for offset in range(0, len(data['label']), BATCH_ROWS):
        stop = min(offset + BATCH_ROWS, len(data['label']))
        days = data['issue_date'][offset:stop]
        rows = stop - offset
        common = dict(cell_id=pa.array(data['cell_id'][offset:stop], type=pa.string()),
                      issue_date=pa.array(days, type=pa.date32()),
                      target_start_date=pa.array(days + np.timedelta64(1, 'D'), type=pa.date32()),
                      target_end_date=pa.array(days + np.timedelta64(7, 'D'), type=pa.date32()),
                      horizon_days=pa.array(np.full(rows, 7, dtype=np.int32)),
                      label=pa.array(data['label'][offset:stop]), fold=pa.array([fold] * rows),
                      partition=pa.array([partition] * rows), period=pa.array([period] * rows),
                      calibrated=pa.array(np.full(rows, calibrated)))
        for name in MODEL_NAMES:
            columns = dict(common, model=pa.array([name] * rows),
                           raw_score=pa.array(raw[name][offset:stop], type=pa.float64()),
                           probability=pa.array(probabilities[name][offset:stop], type=pa.float64())
                           if name in probabilities else pa.nulls(rows, type=pa.float64()))
            writer.write_table(pa.table(columns, schema=PREDICTION_SCHEMA))


def _reload(stage, model, calibration, config):
    model.save_model(stage / 'model.json')
    restored = xgb.Booster()
    restored.load_model(stage / 'model.json')
    restored.set_param({'nthread': config['nthread']})
    objective = json.loads(restored.save_config())['learner']['objective']['name']
    if (restored.feature_names != list(FEATURE_NAMES) or objective != 'binary:logistic'
            or restored.num_boosted_rounds() != config['rounds']):
        raise ValueError('Native JSON model contract mismatch')
    raw = _predict(restored, calibration, config)
    if not np.array_equal(raw, _predict(model, calibration, config)):
        raise ValueError('Reloaded native JSON model predictions differ')
    return restored, raw


def _run(con, path, feature_manifest, config, stage):
    _validate_keys(con, path, feature_manifest)
    _assignments(con, path, stage, config)
    validation = {}
    metrics = dict(validation=validation, calibration={}, tests={}, diagnostics={})
    with pq.ParquetWriter(stage / 'predictions.parquet', PREDICTION_SCHEMA, compression='zstd') as writer:
        for fold in config['validation_folds']:
            training = _load_partition(con, path, config['train_start'], fold['train_end'], config)
            model = _fit(training, config)
            del training
            evaluation = _load_partition(con, path, fold['start'], fold['end'], config)
            raw = dict(xgboost=_predict(model, evaluation, config), **_baselines(evaluation))
            probabilities = {'xgboost': raw['xgboost']}
            validation[fold['name']] = _reports(evaluation, raw, probabilities)
            _export_predictions(writer, evaluation, raw, probabilities, fold=fold['name'],
                                partition='validation', period=fold['name'], calibrated=False)
            del evaluation, model, raw, probabilities
        selection = _select(validation, config)
        _json(stage / 'selection.json', selection)
        # No calibration/test outcome query, matrix or metrics exists above this barrier.
        _validate_all_outcomes(con, path, feature_manifest)
        training = _load_partition(con, path, config['train_start'], config['final_train_end'], config)
        model = _fit(training, config)
        del training
        interval = config['calibration']
        calibration = _load_partition(con, path, interval['start'], interval['end'], config)
        model, model_raw = _reload(stage, model, calibration, config)
        raw = dict(xgboost=model_raw, **_baselines(calibration))
        maps = {name: _isotonic(raw[name], calibration['label']) for name in MODEL_NAMES}
        _json(stage / 'calibrator.json', maps['xgboost'])
        _json(stage / 'baseline_calibrators.json', {name: maps[name] for name in BASELINES})
        restored_maps = dict(xgboost=json.loads((stage / 'calibrator.json').read_text()),
                             **json.loads((stage / 'baseline_calibrators.json').read_text()))
        probabilities = {name: _calibrate(raw[name], restored_maps[name]) for name in MODEL_NAMES}
        if any(not np.array_equal(probabilities[name], _calibrate(raw[name], maps[name])) for name in MODEL_NAMES):
            raise ValueError('Reloaded isotonic predictions differ')
        metrics['calibration'] = dict(in_sample=True, period=interval['name'],
            models=_reports(calibration, raw, probabilities, selection['reference_baseline']))
        _export_predictions(writer, calibration, raw, probabilities, fold='final', partition='calibration',
                            period=interval['name'], calibrated=True)
        del calibration, model_raw, raw, probabilities
        for category, partition in [('tests', 'test'), ('diagnostics', 'diagnostic')]:
            for interval in config[category]:
                evaluation = _load_partition(con, path, interval['start'], interval['end'], config)
                raw = dict(xgboost=_predict(model, evaluation, config), **_baselines(evaluation))
                probabilities = {name: _calibrate(raw[name], restored_maps[name]) for name in MODEL_NAMES}
                metrics[category][interval['name']] = _reports(
                    evaluation, raw, probabilities, selection['reference_baseline'])
                _export_predictions(writer, evaluation, raw, probabilities, fold='final',
                                    partition=partition, period=interval['name'], calibrated=True)
                del evaluation, raw, probabilities
    _json(stage / 'metrics.json', metrics)
    return selection


def train(features_dir, config_path, output_dir):
    """Fit fixed expanding folds, freeze selection, then calibrate and evaluate."""
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(f'Output must not already exist: {target}')
    config = _config(config_path)
    path, feature_manifest = _feature_contract(features_dir)
    source_hashes = dict(features=sha256(path), manifest=sha256(path.parent / 'manifest.json'),
                         config=sha256(config_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}-', dir=target.parent))
    try:
        with duckdb.connect(str(stage / 'work.duckdb')) as con:
            con.execute(f'SET memory_limit={literal(config["memory_limit"])}')
            con.execute(f'SET threads={config["nthread"]}')
            con.execute(f'SET temp_directory={literal(str(stage / "spill"))}')
            selection = _run(con, path, feature_manifest, config, stage)
        (stage / 'work.duckdb').unlink()
        shutil.rmtree(stage / 'spill', ignore_errors=True)
        if source_hashes != dict(features=sha256(path), manifest=sha256(path.parent / 'manifest.json'),
                                 config=sha256(config_path)):
            raise ValueError('Training inputs changed during processing')
        manifest = dict(schema='global-weekly-model/1.0.0', feature_names=list(FEATURE_NAMES),
            horizon_days=HORIZON_DAYS, activity=feature_manifest['activity'],
            feature_manifest_sha256=source_hashes['manifest'], features_sha256=source_hashes['features'],
            config_sha256=source_hashes['config'], configuration=_serializable_config(config),
            parameters=dict(PARAMS, nthread=config['nthread']), rounds=config['rounds'],
            engineering_fixture=config['engineering_fixture'],
            train_start=str(config['train_start']), train_end=str(config['final_train_end']),
            calibration=_serializable_config(config)['calibration'],
            data_cutoff_date=str(config['calibration']['end'] + timedelta(days=HORIZON_DAYS)),
            reference_baseline=selection['reference_baseline'],
            sustained_improvement=selection['sustained_improvement'],
            reload_prediction_equality=True,
            code_sha256={name: sha256(Path(__file__).with_name(name))
                         for name in ('global_training.py', 'global_features.py')},
            feature_method_version=feature_manifest['feature_method_version'],
            versions=dict(duckdb=duckdb.__version__, numpy=np.__version__, pyarrow=pa.__version__,
                          xgboost=xgb.__version__),
            outputs=[dict(path=item.name, size_bytes=item.stat().st_size, sha256=sha256(item))
                     for item in sorted(stage.iterdir()) if item.is_file()], limitations=LIMITATIONS)
        _json(stage / 'manifest.json', manifest)
        if target.exists():
            raise FileExistsError(f'Output appeared during processing: {target}')
        os.rename(stage, target)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.features_dir, args.config, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
