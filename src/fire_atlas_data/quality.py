"""Validation-only bounded ranking search; reused TEST is diagnostic, not independent proof."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from . import ranking as r
from .features import ENHANCED_FEATURES, HISTORY_FEATURES

SPATIAL_FEATURES = ['neighbors_3d', 'neighbors_7d', 'days_since_last'] + [
    name for name in ENHANCED_FEATURES if name not in r.FULL]
SPATIAL_HISTORY_FEATURES = SPATIAL_FEATURES + HISTORY_FEATURES[len(ENHANCED_FEATURES):]

# Frozen before examining TEST: ordered from simpler to more complex for ties.
PROFILES = (
    dict(max_depth=2, n_estimators=80, learning_rate=.03, min_child_weight=5, reg_lambda=10),
    dict(max_depth=2, n_estimators=160, learning_rate=.08, min_child_weight=1, reg_lambda=1),
    dict(max_depth=3, n_estimators=80, learning_rate=.08, min_child_weight=5, reg_lambda=10),
    dict(max_depth=3, n_estimators=160, learning_rate=.03, min_child_weight=1, reg_lambda=10),
    dict(max_depth=4, n_estimators=160, learning_rate=.08, min_child_weight=5, reg_lambda=10),
)
ORIGINAL_PARAMS = dict(max_depth=3, n_estimators=120, learning_rate=.05,
                       min_child_weight=1, reg_lambda=1)
IDENTIFIERS = ['cell_id', 'issue_date', 'target_date', 'region', 'label', 'split']
VARIABLE_CONFIG = {'sources', 'splits', 'publish_date', 'warmup_start', 'memory_limit', 'threads'}


def _profiles(snpp_history=False):
    enhanced = HISTORY_FEATURES if snpp_history else ENHANCED_FEATURES
    spatial = SPATIAL_HISTORY_FEATURES if snpp_history else SPATIAL_FEATURES
    suffix = '_history' if snpp_history else ''
    return [dict(name='expanded_original', features=list(r.FULL), policy=10, params=ORIGINAL_PARAMS)] + [
        dict(name=f'{"spatial" if i >= 3 else "enhanced"}{suffix}_{i}_{policy}',
             features=list(spatial if i >= 3 else enhanced),
             policy=policy, params=params)
        for i, params in enumerate(PROFILES) for policy in (10, 'ALL')]


def _load(path, publication=False):
    files = r._files(path)
    metadata = (Path(path) if Path(path).is_dir() else Path(path).parent) / 'manifest.json'
    provenance = json.loads(metadata.read_text())
    feature_order = HISTORY_FEATURES if provenance['config'].get('snpp_history', False) else ENHANCED_FEATURES
    columns = IDENTIFIERS + list(feature_order)
    with duckdb.connect() as con:
        relation = con.read_parquet([str(f) for f in files], union_by_name=True)
        missing = set(columns) - set(relation.columns)
        if missing:
            raise ValueError(f'Missing feature columns: {sorted(missing)}')
        table = relation.project(','.join(columns)).order('issue_date,region,cell_id').to_arrow_table()
    rows = table.to_pylist()
    if not rows:
        raise ValueError('Empty feature dataset')
    seen = set()
    for row in rows:
        key = (row['issue_date'], row['cell_id'])
        if key in seen:
            raise ValueError(f'Duplicate issue/cell candidate: {key}')
        seen.add(key)
        if row['split'] not in {'train', 'validation', 'test', 'publish'}:
            raise ValueError('Invalid split')
        if row['split'] == 'publish':
            if row['label'] is not None:
                raise ValueError('Publish labels must be null; D+1 is not consumed')
        elif row['label'] not in (0, 1):
            raise ValueError('Invalid binary observation label')
        if row['target_date'] != row['issue_date'] + timedelta(days=1):
            raise ValueError('Target must be day after issue date')
    dates = {s: {x['issue_date'] for x in rows if x['split'] == s} for s in ('train', 'validation', 'test')}
    if not publication and (any(not d for d in dates.values()) or not (
            max(dates['train']) < min(dates['validation']) <= max(dates['validation']) < min(dates['test']))):
        raise ValueError('Require nonempty chronological train/validation/test splits')
    types = {n: str(table.schema.field(n).type) for n in feature_order}
    return rows, files, types


def _contract(path, types):
    path = Path(path)
    metadata = (path if path.is_dir() else path.parent) / 'manifest.json'
    if not metadata.is_file():
        raise ValueError('Feature manifest is required')
    provenance = json.loads(metadata.read_text())
    config = {k: v for k, v in provenance['config'].items() if k not in VARIABLE_CONFIG}
    if not (config.get('enhanced_features') or config.get('snpp_history')) or config.get('minimum_history_days') != 14:
        raise ValueError('Require enhanced_features=true or snpp_history=true and minimum_history_days=14')
    feature_order = HISTORY_FEATURES if config.get('snpp_history', False) else ENHANCED_FEATURES
    if list(types) != list(feature_order):
        raise ValueError('Feature types/order differ from configured feature mode')
    return dict(feature_order=list(feature_order), feature_types=types, config=config,
                method_version=provenance['method_version'],
                features_code_sha256=r._hash(Path(__file__).with_name('features.py'))), provenance


def _matrix(rows, names):
    if names not in (list(r.FULL), list(ENHANCED_FEATURES), list(SPATIAL_FEATURES),
                     list(HISTORY_FEATURES), list(SPATIAL_HISTORY_FEATURES)):
        raise ValueError('Model feature names/order mismatch')
    return np.asarray([[np.nan if row[n] is None else float(row[n]) for n in names]
                       for row in rows], dtype=np.float32)


def _sample(rows, policy):
    if policy == 10:
        return r._sample_train(rows, seed=42, ratio=10)
    if policy == 'ALL':
        return sorted((row for row in rows if row['split'] == 'train'),
                      key=lambda row: (row['issue_date'], row['region'], row['cell_id']))
    raise ValueError('Unknown predeclared sampling policy')


def _counts(rows):
    return dict(rows=len(rows), positives=sum(x['label'] == 1 for x in rows),
                controls=sum(x['label'] == 0 for x in rows),
                groups=len({(x['region'], x['issue_date']) for x in rows}))


def _summary(rows, scores, k):
    daily = r._metrics(rows, scores, k)
    hits = sum(x['hits'] for x in daily)
    positives = sum(x['n_positives'] for x in daily)
    selected = sum(x['n_selected'] for x in daily)
    recalls = [x['recall'] for x in daily if x['recall'] is not None]
    precisions = [x['precision'] for x in daily if x['precision'] is not None]
    return dict(daily=daily, micro=dict(hits=hits, positives=positives, selected=selected,
                recall=hits / positives if positives else None,
                precision=hits / selected if selected else None),
                macro=dict(recall=sum(recalls) / len(recalls) if recalls else None,
                           precision=sum(precisions) / len(precisions) if precisions else None,
                           positive_days=len(recalls), days=len(daily)))


def _select(validation_results):
    # Strict improvement only; initial frozen original wins equal objectives.
    if not validation_results or any(x['metrics']['macro']['recall'] is None for x in validation_results):
        raise ValueError('Validation has no observed positives; selection unsupported')
    return max(validation_results, key=lambda x: x['metrics']['macro']['recall'])['name']


def _fit(rows, spec):
    names = spec['features']
    params = dict(spec['params'])
    rounds = params.pop('n_estimators')
    params['eta'] = params.pop('learning_rate')
    params['lambda'] = params.pop('reg_lambda')
    params.update(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                  subsample=.8, colsample_bytree=1., seed=42, nthread=2)
    data = xgb.DMatrix(_matrix(rows, names), label=[x['label'] for x in rows], feature_names=names)
    return xgb.train(params, data, num_boost_round=rounds)


def _predict(model_dir, spec, rows):
    path = Path(model_dir) / spec['file']
    if r._hash(path) != spec['sha256']:
        raise ValueError('Model artifact hash mismatch')
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({'nthread': 2})
    names = spec['features']
    matrix = _matrix(rows, names)
    if model.num_features() != len(names):
        raise ValueError('Model feature count mismatch')
    if model.feature_names is not None and model.feature_names != names:
        raise ValueError('Native model feature names/order mismatch')
    if model.feature_names is None and not spec.get('original_unnamed'):
        raise ValueError('Native model feature names missing')
    return model.predict(xgb.DMatrix(matrix, feature_names=names if model.feature_names else None))


def _export(output, rows, scores_by_model, k, version):
    predictions = []
    for name, scores in scores_by_model.items():
        groups = {}
        for row, score in zip(rows, scores):
            groups.setdefault((row['region'], row['issue_date']), []).append((row, score))
        for group in groups.values():
            for rank, (row, score) in enumerate(r._ordered([x[0] for x in group], [x[1] for x in group]), 1):
                predictions.append({n: row[n] for n in IDENTIFIERS} | dict(
                    score=float(score), rank=rank, selected_top_k=rank <= k,
                    model_name=name, model_version=version))
    pq.write_table(pa.Table.from_pylist(predictions), output / 'test_predictions.parquet')
    with (output / 'test_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return {f: dict(sha256=r._hash(output / f), rows=len(predictions))
            for f in ('test_predictions.parquet', 'test_predictions.csv')}


def train(features, output_dir, baseline_model_dir='output/rf10-prototype/model-heldout', k=10):
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    if k != 10:
        raise ValueError('This predeclared experiment fixes K=10')
    rows, files, types = _load(features)
    contract, provenance = _contract(features, types)
    validation = [x for x in rows if x['split'] == 'validation']
    # Reject unsupported validation before training or touching TEST scores.
    r._objective(r._metrics(validation, np.zeros(len(validation)), k))
    baseline_dir = Path(baseline_model_dir)
    baseline_path = baseline_dir / 'manifest.json'
    baseline = json.loads(baseline_path.read_text())
    baseline_spec = baseline['models']['full']
    if baseline_spec['features'] != r.FULL:
        raise ValueError('Original full model feature names/order mismatch')
    if baseline.get('K') != k:
        raise ValueError('Original baseline K differs')
    baseline_contract = baseline.get('feature_contract', {})
    baseline_config = baseline_contract.get('config') or {}
    if baseline_config.get('target_mode', 'novel_activity') != contract['config'].get('target_mode', 'novel_activity'):
        raise ValueError('Original baseline target differs; do not compare or select across target definitions')
    for key in ('bbox', 'h3_resolution', 'region'):
        if key in baseline_config and baseline_config[key] != contract['config'].get(key):
            raise ValueError('Original baseline geometry differs')
    source = baseline_dir / baseline_spec['file']
    if r._hash(source) != baseline_spec['sha256']:
        raise ValueError('Original model artifact hash mismatch')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'original.json')
    original = dict(file='original.json', sha256=r._hash(source), features=list(r.FULL),
                    original_unnamed=True, policy='frozen original', params=baseline.get('config'))
    models = {'original': original}
    results = [dict(name='original', metrics=_summary(validation, _predict(output, original, validation), k),
                    training_counts=None, training_counts_note='Frozen original; see baseline manifest')]
    for spec in _profiles(contract['config'].get('snpp_history', False)):
        sampled = _sample(rows, spec['policy'])
        if {x['label'] for x in sampled} != {0, 1}:
            raise ValueError('Sampled training requires positives and controls')
        model = _fit(sampled, spec)
        path = output / f"{spec['name']}.json"
        model.save_model(path)
        saved = dict(spec, file=path.name, sha256=r._hash(path))
        models[spec['name']] = saved
        # All evaluation runs through reloaded native artifacts.
        scores = _predict(output, saved, validation)
        direct = model.predict(xgb.DMatrix(_matrix(validation, spec['features']), feature_names=spec['features']))
        if not np.array_equal(scores, direct):
            raise ValueError('Reload inference differs from fitted model')
        results.append(dict(name=spec['name'], metrics=_summary(validation, scores, k),
                            training_counts=_counts(sampled)))
    winner = _select(results)
    manifest = dict(schema_version=1, K=k, seed=42, n_jobs=2, selected_model=winner,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), models=models,
                    feature_contract=contract, feature_manifest=provenance,
                    quality_code_sha256=r._hash(__file__), ranking_code_sha256=r._hash(r.__file__),
                    grid=_profiles(contract['config'].get('snpp_history', False)), selection='validation macro daily recall@10; no-positive days omitted; strict improvement over original; ties favor listed simpler profiles',
                    baseline_manifest=baseline, baseline_manifest_sha256=r._hash(baseline_path),
                    inputs=[dict(path=str(f), sha256=r._hash(f)) for f in files],
                    training_counts=_counts([x for x in rows if x['split'] == 'train']),
                    versions={p: importlib.metadata.version(p) for p in ('xgboost', 'numpy', 'duckdb', 'h3', 'pyarrow')},
                    interpretation='Observation-conditioned retrospective ranking scores, not calibrated probabilities. TEST reused: diagnostic, not independent proof.')
    manifest['model_version'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    # Freeze decision on disk BEFORE any TEST predictions. No fitting or selection below.
    r._json(output / 'selection.json', dict(selected_model=winner, validation=results,
                                           model_version=manifest['model_version']))
    test = [x for x in rows if x['split'] == 'test']
    scores_by_model = {name: _predict(output, models[name], test) for name in dict.fromkeys(('original', winner))}
    comparison = dict(selected_model=winner, validation=results,
                      test={name: _summary(test, scores, k) for name, scores in scores_by_model.items()},
                      training_counts=manifest['training_counts'], test_reused_diagnostic=True,
                      selection_metric=manifest['selection'],
                      limitations=['TEST was previously inspected and is diagnostic, not independent proof.',
                                   'No weather inputs; observation labels do not establish absence of fire.',
                                   '30-day histories are left-censored; 14-day initial warmup only.',
                                   'Nominal historical emission is not measured operational availability.'])
    comparison['recurrence_references'] = {
        split: {name: _summary(subset, [float(row[name] or 0) for row in subset], k)
                for name in ('neighbors_7d', 'cell_active_days_30d')}
        for split, subset in (('validation', validation), ('test', test))}
    original_recall = comparison['test']['original']['macro']['recall']
    winner_recall = comparison['test'][winner]['macro']['recall']
    comparison['diagnostic_test_regression'] = (
        winner_recall < original_recall
        if winner_recall is not None and original_recall is not None else None)
    comparison['diagnostic_test_macro_recall_delta'] = (
        winner_recall - original_recall
        if winner_recall is not None and original_recall is not None else None)
    comparison['sampling_policies'] = {
        '10': 'Within each training region/date retain all positives and up to 10 controls per positive; no-positive groups excluded.',
        'ALL': 'All training positives and controls, including all controls in no-positive region/date groups.'}
    manifest['outputs'] = _export(output, test, scores_by_model, k, manifest['model_version'])
    r._json(output / 'comparison.json', comparison)
    for name in ('selection.json', 'comparison.json'):
        manifest['outputs'][name] = dict(sha256=r._hash(output / name))
    r._json(output / 'manifest.json', manifest)
    return manifest


def _frozen_artifacts(model_dir, manifest):
    for filename, spec in manifest['outputs'].items():
        if r._hash(Path(model_dir) / filename) != spec['sha256']:
            raise ValueError('Output artifact hash mismatch')
    selection = json.loads((Path(model_dir) / 'selection.json').read_text())
    if selection['selected_model'] != manifest['selected_model'] or selection['model_version'] != manifest['model_version']:
        raise ValueError('Frozen selection differs from model manifest')
    return json.loads((Path(model_dir) / 'comparison.json').read_text())


def _input_coverage(provenance):
    sources = provenance.get('sources', [])
    paths = [item['path'] for item in sources]
    for item in sources:
        if item.get('sha256') and r._hash(item['path']) != item['sha256']:
            raise ValueError('Evaluation source artifact hash mismatch')
    gaps = []
    if paths:
        with duckdb.connect() as con:
            days = [row[0] for row in con.read_parquet(paths, union_by_name=True)
                    .filter("platform = 'NOAA-20' AND sensor = 'VIIRS'")
                    .project('acquisition_date').distinct().order('acquisition_date').fetchall()]
        for before, after in zip(days, days[1:]):
            if (after - before).days > 1:
                gaps.append(dict(start=str(before + timedelta(days=1)),
                                 end=str(after - timedelta(days=1))))
    return dict(continuous_observation_coverage_certified=False,
                dates_without_supplied_noaa20_detections=gaps,
                interpretation='Dates without supplied detections are input gaps or unobserved activity, not confirmed fire absence. history_days is calendar age, not data availability or continuous 14-day coverage.',
                covariate_shift='New source/date batch and any input gaps can shift recurrence/elapsed features; this is not a clean matched-coverage or operational evaluation.')


def evaluate(features, model_dir, output_dir):
    """Evaluate a frozen decision on later labeled dates; never fit or promote."""
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    model_dir = Path(model_dir)
    manifest_path = model_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    comparison = _frozen_artifacts(model_dir, manifest)
    rows, files, types = _load(features, publication=True)
    contract, provenance = _contract(features, types)
    if contract != manifest['feature_contract']:
        raise ValueError('Feature schema/geometry/version contract differs from training')
    if r._hash(r.__file__) != manifest['ranking_code_sha256']:
        raise ValueError('Ranking scoring code differs from training')
    rows = [row for row in rows if row['split'] == 'test']
    if not rows:
        raise ValueError('Independent evaluation requires labeled test rows')
    previous_dates = [item['issue_date'] for item in comparison['test']['original']['daily']]
    if not previous_dates or min(str(row['issue_date']) for row in rows) <= max(previous_dates):
        raise ValueError('Independent evaluation dates must follow the original diagnostic TEST')
    inputs = [dict(path=str(file), sha256=r._hash(file)) for file in files]
    if {item['sha256'] for item in inputs} & {item['sha256'] for item in manifest['inputs']}:
        raise ValueError('Independent evaluation must use a new feature batch')
    coverage = _input_coverage(provenance)
    # Only the frozen original and selected winner; no candidate search or fitting.
    names = list(dict.fromkeys(('original', manifest['selected_model'])))
    scores = {name: _predict(model_dir, manifest['models'][name], rows) for name in names}
    metrics = {name: _summary(rows, values, manifest['K']) for name, values in scores.items()}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = _export(output, rows, scores, manifest['K'], manifest['model_version'])
    report = dict(evaluation_type='independent_evaluation',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  selected_model=manifest['selected_model'], model_version=manifest['model_version'],
                  K=manifest['K'], test=metrics, inputs=inputs, feature_manifest=provenance,
                  feature_manifest_sha256=r._hash((Path(features) if Path(features).is_dir() else Path(features).parent) / 'manifest.json'),
                  frozen_manifest_sha256=r._hash(manifest_path),
                  frozen_quality_code_sha256=manifest['quality_code_sha256'],
                  evaluator_code_sha256=r._hash(__file__),
                  evaluator_code_differs_from_training=r._hash(__file__) != manifest['quality_code_sha256'],
                  evaluator_code_change_note='Additive frozen-evaluation tooling; original training code hash retained, no retraining or manifest rewriting.',
                  coverage=coverage,
                  scoring_method='Frozen native JSONs; exact feature order/types and native names; no fit/search/selection.',
                  models={name: manifest['models'][name] for name in names},
                  artifacts=artifacts, promotion_unlocked=False,
                  observed_positive_count=sum(row['label'] for row in rows),
                  gain_measurement_supported=any(row['label'] == 1 for row in rows),
                  conclusion=('Insufficient observed positives to measure recall gain; null recall is not evidence of fire absence.'
                              if not any(row['label'] == 1 for row in rows)
                              else 'Frozen comparison only; few dates and coverage shift preclude an operational or automatic-promotion claim.'),
                  original_diagnostic_test_regression=comparison.get('diagnostic_test_regression'),
                  limitations=['New public source/date batch; only a few observation-conditioned days.',
                               'Retrospective labels and nominal emission do not validate operational availability.',
                               'No automatic promotion or retuning; previous diagnostic regression remains blocking.',
                               'Scores are not calibrated probabilities; controls are not verified absence of fire.'])
    r._json(output / 'comparison.json', report)
    r._json(output / 'independent_evaluation.json', dict(
        evaluation_type='independent_evaluation', model_version=manifest['model_version'],
        promotion_unlocked=False, artifacts=artifacts | {
            'comparison.json': dict(sha256=r._hash(output / 'comparison.json'))}))
    return report


def publish(features, model_dir, output_dir):
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    rows, _, types = _load(features, publication=True)
    manifest = json.loads((Path(model_dir) / 'manifest.json').read_text())
    comparison = _frozen_artifacts(model_dir, manifest)
    if comparison.get('diagnostic_test_regression'):
        raise ValueError('Publication blocked: selected model regressed on diagnostic TEST; independent evaluation cannot automatically unlock promotion')
    contract, _ = _contract(features, types)
    if contract != manifest['feature_contract']:
        raise ValueError('Feature schema/geometry/version contract differs from training')
    if r._hash(__file__) != manifest['quality_code_sha256'] or r._hash(r.__file__) != manifest['ranking_code_sha256']:
        raise ValueError('Scoring code differs from training')
    # Frozen artifacts, including the diagnostic promotion gate, verified above.
    # Publication never falls back to labeled TEST.
    rows = [x for x in rows if x['split'] == 'publish']
    if not rows:
        raise ValueError('No unlabeled publish candidates')
    scores = _predict(model_dir, manifest['models'][manifest['selected_model']], rows)
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault((row['region'], row['issue_date']), []).append((row, score))
    ranked = []
    for (region, day), group in sorted(grouped.items()):
        ordered = r._ordered([x[0] for x in group], [x[1] for x in group])[:manifest['K']]
        for rank, (row, score) in enumerate(ordered, 1):
            ranked.append(dict(region=region, cell_id=row['cell_id'],
                               issued_at_utc=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=23, minute=59, second=59),
                               target_date_utc=row['target_date'], score=float(score), rank=rank,
                               model_version=manifest['model_version']))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(ranked), output / 'rankings.parquet')
    generated = datetime.now(timezone.utc).isoformat()
    r._map(output / 'map.html', ranked, generated, 'publish')
    r._json(output / 'publication.json', dict(generated_at_utc=generated,
            model_version=manifest['model_version'], historical_not_live=True, split='publish',
            K=manifest['K'], rows=len(ranked), label=None,
            artifacts={name: r._hash(output / name) for name in ('rankings.parquet', 'map.html')}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--features', required=True)
    training.add_argument('--output-dir', required=True)
    training.add_argument('--baseline-model-dir', default='output/rf10-prototype/model-heldout')
    training.add_argument('--k', type=int, default=10)
    publishing = commands.add_parser('publish')
    publishing.add_argument('--features', required=True)
    publishing.add_argument('--model-dir', required=True)
    publishing.add_argument('--output-dir', required=True)
    evaluating = commands.add_parser('evaluate')
    evaluating.add_argument('--features', required=True)
    evaluating.add_argument('--model-dir', required=True)
    evaluating.add_argument('--output-dir', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    {'train': train, 'publish': publish, 'evaluate': evaluate}[command](**args)


if __name__ == '__main__':
    main()
