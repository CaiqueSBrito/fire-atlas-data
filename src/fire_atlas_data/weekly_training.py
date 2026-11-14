"""Finite weekly ranking search; previously seen final years remain diagnostic."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from .normalization import sha256
from .ranking import _json
from .weekly_features import FEATURE_NAMES
from .weekly_metrics import ranks_and_selection, summarize

ORIGINAL_FEATURES = ['detection_rows_on_issue', 'cell_detection_rows_7d',
                     'regional_detection_rows_7d']
IDENTIFIERS = ['cell_id', 'issue_date', 'target_start_date', 'target_end_date', 'horizon_days']
FOLDS = ('2021', '2022', '2023')
BASELINES = ('original', 'persistence', 'recurrence', 'climatology')
GRID_SHA256 = 'fa93dc8bbed57263784a47d3ea787f5310d7162a5482006ca9b955402efe819e'
ASSIGNMENT_SHA256 = '73a9d70c67f2fe02ec70e18c2854f36578bc4dab3865d87a8f244c26f1ab479d'
LIMITATIONS = [
    '2024-2025 already inspected: reused diagnostic, not independent validation.',
    'NOAA-20 SP retrospective acquisition cutoff; historical available_at unknown.',
    'Zero means no supplied detection, not verified absence of physical fire.',
    'Overlapping windows; no independent-event or geographic-transfer claim.',
    'Scores are rankings, not calibrated probabilities. No operational promotion.',
]


def _profiles():
    common = dict(tree_method='hist', gamma=0, reg_alpha=0, subsample=1,
                  colsample_bytree=1, seed=42, nthread=2)
    original = dict(name='original', features=ORIGINAL_FEATURES, rounds=120,
                    weighting='all', params=dict(common, objective='binary:logistic',
                    max_depth=3, eta=.05, min_child_weight=1, reg_lambda=1, scale_pos_weight=1))
    profiles = [original]
    logistic = ((2, 240, .05), (3, 240, .05), (5, 400, .03), (3, 600, .03))
    for number, (depth, rounds, eta) in enumerate(logistic, 1):
        profiles.append(dict(name=f'logistic_{number}', features=list(FEATURE_NAMES),
            rounds=rounds, weighting='all', params=dict(common, objective='binary:logistic',
            max_depth=depth, eta=eta, min_child_weight=10, reg_lambda=10, scale_pos_weight=1)))
    for objective in ('rank:ndcg', 'rank:pairwise'):
        for number, (depth, rounds, eta) in enumerate(logistic[1:3], 1):
            rank_params = dict(lambdarank_pair_method='topk' if objective == 'rank:ndcg' else 'mean',
                               lambdarank_num_pair_per_sample=12 if objective == 'rank:ndcg' else 4)
            profiles.append(dict(name=f'{objective.split(":")[1]}_{number}',
                features=list(FEATURE_NAMES), rounds=rounds, weighting='all',
                params=dict(common, objective=objective, max_depth=depth, eta=eta,
                            min_child_weight=10, reg_lambda=10, **rank_params)))
    for number, (depth, rounds, eta) in enumerate(logistic[1:3], 1):
        profiles.append(dict(name=f'macro_weighted_{number}', features=list(FEATURE_NAMES),
            rounds=rounds, weighting='inverse_daily_positives',
            params=dict(common, objective='binary:logistic', max_depth=depth, eta=eta,
                        min_child_weight=10, reg_lambda=10, scale_pos_weight=1)))
    return profiles


def _code_hashes():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in
            ('weekly_training.py', 'weekly_features.py', 'weekly_metrics.py')}


def _feature_contract(path):
    path = Path(path)
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    if manifest.get('schema') != 'weekly-features/1.0.0':
        raise ValueError('Require weekly feature manifest')
    if manifest.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Feature names/order mismatch')
    outputs = {item['path']: item['sha256'] for item in manifest['outputs']}
    if outputs.get(path.name) != sha256(path):
        raise ValueError('Feature artifact hash mismatch')
    if manifest['code_sha256'] != sha256(Path(__file__).with_name('weekly_features.py')):
        raise ValueError('Feature transformation code changed')
    schema = pq.read_schema(path)
    types = {name: str(schema.field(name).type) for name in FEATURE_NAMES}
    if manifest['feature_types'] != types:
        raise ValueError('Feature types mismatch')
    return manifest, dict(feature_names=list(FEATURE_NAMES), feature_types=types,
                          feature_units=manifest['feature_units'], cell_ids=manifest['cell_ids'],
                          feature_code_sha256=manifest['code_sha256'])


def _load_partition(con, features, assignments, fold, partition):
    columns = IDENTIFIERS + ['label'] + list(FEATURE_NAMES)
    projection = ','.join('f.' + name for name in columns)
    return con.execute(f'''SELECT {projection} FROM read_parquet(?) f
        JOIN read_parquet(?) a USING(cell_id,issue_date)
        WHERE a.fold=? AND a.partition=? ORDER BY f.issue_date,f.cell_id''',
        [str(features), str(assignments), fold, partition]).fetchnumpy()


def _matrix(data, names, training=False):
    values = np.column_stack([np.ma.filled(data[name].astype(np.float32), np.nan)
                              for name in names])
    if np.isinf(values).any():
        raise ValueError('Infinite predictor values')
    return xgb.DMatrix(values, feature_names=names,
                       label=data['label'] if training else None, nthread=2)


def _daily_weights(data):
    _, inverse = np.unique(data['issue_date'], return_inverse=True)
    positive_counts = np.bincount(inverse, weights=data['label'])
    weights = 1.0 / np.maximum(positive_counts[inverse], 1)
    return weights / weights.mean()


def _check_partition(train, evaluation):
    if not len(train['label']) or not len(evaluation['label']):
        raise ValueError('Empty training/evaluation partition')
    if train['target_end_date'].max() >= evaluation['issue_date'].min():
        raise ValueError('Training outcomes cross the evaluation cutoff')
    if not np.isin(train['label'], [0, 1]).all() or not np.isin(evaluation['label'], [0, 1]).all():
        raise ValueError('Require binary supplied-detection labels')
    if len(np.unique(train['label'])) != 2:
        raise ValueError('Training requires both supplied-detection classes')


def _fit(data, spec, directory, prefix):
    matrix = _matrix(data, spec['features'], training=True)
    if spec['weighting'] == 'inverse_daily_positives':
        matrix.set_weight(_daily_weights(data))
    if spec['params']['objective'].startswith('rank:'):
        _, counts = np.unique(data['issue_date'], return_counts=True)
        matrix.set_group(counts)
    model = xgb.train(spec['params'], matrix, num_boost_round=spec['rounds'])
    name = f'{prefix}-{spec["name"]}.json'
    model.save_model(directory / name)
    _json(directory / f'{prefix}-{spec["name"]}-config.json', json.loads(model.save_config()))
    return model, dict(file=name, sha256=sha256(directory / name), features=spec['features'],
                       rounds=spec['rounds'], name=spec['name'])


def _predict(directory, artifact, data, expected=None):
    path = directory / artifact['file']
    if sha256(path) != artifact['sha256']:
        raise ValueError('Native model hash mismatch')
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({'nthread': 2})
    if model.feature_names != artifact['features'] or model.num_boosted_rounds() != artifact['rounds']:
        raise ValueError('Native model feature/round contract mismatch')
    matrix = _matrix(data, artifact['features'])
    scores = model.predict(matrix)
    if expected is not None and not np.array_equal(scores, expected.predict(matrix)):
        raise ValueError('Reloaded model predictions differ')
    return scores


def _climatology(data):
    groups = {}
    months = data['issue_date'].astype('datetime64[M]').astype(str)
    for cell, month, label in zip(data['cell_id'], months, data['label']):
        key = cell + ':' + month[-2:]
        total, count = groups.get(key, (0, 0))
        groups[key] = total + int(label), count + 1
    return dict(groups={key: total / count for key, (total, count) in groups.items()},
                fallback=float(data['label'].mean()))


def _reference_scores(data, climate):
    months = data['issue_date'].astype('datetime64[M]').astype(str)
    return dict(persistence=data[ORIGINAL_FEATURES[0]].astype(float),
                recurrence=data[ORIGINAL_FEATURES[1]].astype(float),
                climatology=np.asarray([climate['groups'].get(cell + ':' + month[-2:], climate['fallback'])
                    for cell, month in zip(data['cell_id'], months)]))


def _export(data, scores, directory, prefix):
    reports = {}
    path = directory / f'{prefix}-predictions.parquet'
    # Stream one model at a time instead of holding every model's rows in memory.
    writer = None
    try:
        for name, values in scores.items():
            reports[name] = summarize(data, values)
            rank, selected = ranks_and_selection(data['cell_id'], data['issue_date'], values)
            columns = {key: data[key] for key in IDENTIFIERS + ['label']}
            columns.update(model=np.full(len(values), name, dtype=object),
                           score=values.astype(np.float64), rank=rank, selected_top_k=selected)
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    _json(directory / f'{prefix}-metrics.json', reports)
    return reports


def _select(validation):
    names = list(validation[FOLDS[0]])
    if any(set(validation[fold]) != set(names) for fold in FOLDS):
        raise ValueError('Candidates differ between folds')
    means = {}
    for name in names:
        summaries = [validation[fold][name]['summary'] for fold in FOLDS]
        if any(item['macro_recall'] is None for item in summaries):
            raise ValueError('Positive days required in every validation fold')
        means[name] = dict(precision=float(np.mean([item['precision'] for item in summaries])),
                          macro_recall=float(np.mean([item['macro_recall'] for item in summaries])))
    winner = 'original'
    gates = {}
    for name in names:
        precision_safe = all(validation[f][name]['summary']['precision'] >=
                             validation[f]['original']['summary']['precision'] for f in FOLDS)
        recall_safe = means[name]['macro_recall'] >= means['original']['macro_recall']
        gates[name] = dict(no_fold_precision_regression=precision_safe,
                           no_mean_recall_regression=recall_safe)
        if precision_safe and recall_safe and means[name]['precision'] > means[winner]['precision']:
            winner = name
    return dict(selected_model=winner, validation_means=means, candidate_gates=gates,
                primary_metric='precision_at_10', recall_guard='mean_daily_macro_recall',
                frozen_before_new_diagnostic_scores=True, independent_validation=False,
                promotion_unlocked=False)


def _counts(data):
    return dict(rows=len(data['label']), positives=int(data['label'].sum()),
                controls=int((data['label'] == 0).sum()),
                days=len(np.unique(data['issue_date'])), all_rows_retained=True)


def _load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'weekly-ranking-experiment/2.0.0':
        raise ValueError('Require completed weekly v2 model')
    for filename, digest in manifest['outputs'].items():
        if sha256(directory / filename) != digest:
            raise ValueError('Model bundle artifact hash mismatch')
    if manifest['code_sha256'] != _code_hashes():
        raise ValueError('Weekly scoring code changed')
    return manifest, json.loads((directory / 'final-freeze.json').read_text())


def train(features, assignments, protocol, output_dir):
    """Select only with 2021-2023; freeze before any new final-period scores."""
    started = time.monotonic()
    features, assignments, protocol = Path(features), Path(assignments), Path(protocol)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    provenance, contract = _feature_contract(features)
    if provenance['grid_sha256'] != GRID_SHA256 or sha256(assignments) != ASSIGNMENT_SHA256:
        raise ValueError('Weekly experiment requires the frozen grid and partitions')
    output.mkdir(parents=True)
    profiles = _profiles()
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'duckdb', 'pyarrow', 'xgboost', 'scikit-learn', 'h3')}
    _json(output / 'training-plan.json', dict(profiles=profiles, features=contract,
        feature_manifest_sha256=sha256(features.parent / 'manifest.json'),
        features_sha256=sha256(features), assignments_sha256=sha256(assignments),
        protocol_sha256=sha256(protocol), code_sha256=_code_hashes(), versions=versions,
        python=platform.python_version(), platform=platform.platform(), top_k=10,
        validation_folds=list(FOLDS), diagnostic_years=[2024, 2025], limitations=LIMITATIONS))
    with duckdb.connect() as con:
        con.execute("SET memory_limit='512MB'")
        con.execute('SET threads=2')
        validation = {}
        artifacts = {}
        for fold in FOLDS:
            training = _load_partition(con, features, assignments, fold, 'train')
            evaluation = _load_partition(con, features, assignments, fold, 'validation')
            _check_partition(training, evaluation)
            _json(output / f'{fold}-training-counts.json', _counts(training))
            climate = _climatology(training)
            _json(output / f'{fold}-climatology.json', climate)
            original, artifact = _fit(training, profiles[0], output, fold)
            scores = {'original': _predict(output, artifact, evaluation, original)}
            artifacts[f'{fold}-original'] = artifact
            scores.update(_reference_scores(evaluation, climate))
            for spec in profiles[1:]:
                model, artifact = _fit(training, spec, output, fold)
                scores[spec['name']] = _predict(output, artifact, evaluation, model)
                artifacts[f'{fold}-{spec["name"]}'] = artifact
                print(json.dumps(dict(fold=fold, candidate=spec['name'], fit_complete=True)), flush=True)
            validation[fold] = _export(evaluation, scores, output, fold)
            print(json.dumps(dict(fold=fold, validation={name: report['summary']['precision']
                  for name, report in validation[fold].items()})), flush=True)
        selection = _select(validation)
        selection['fold_models'] = artifacts
        _json(output / 'selection.json', selection)
        selection_hash = sha256(output / 'selection.json')
        winner = selection['selected_model']
        training = _load_partition(con, features, assignments, 'final', 'train')
        _json(output / 'final-training-counts.json', _counts(training))
        climate = _climatology(training)
        _json(output / 'final-climatology.json', climate)
        original, original_artifact = _fit(training, profiles[0], output, 'final')
        selected_model, selected_artifact = original, original_artifact
        if winner not in BASELINES:
            spec = next(spec for spec in profiles if spec['name'] == winner)
            selected_model, selected_artifact = _fit(training, spec, output, 'final')
        elif winner != 'original':
            selected_model, selected_artifact = None, None
        freeze = dict(selected_model=winner, selected_artifact=selected_artifact,
                      original_artifact=original_artifact, selection_sha256=selection_hash,
                      climatology_sha256=sha256(output / 'final-climatology.json'),
                      contract=contract, training_last_target=str(training['target_end_date'].max()),
                      independent_validation=False, promotion_unlocked=False)
        _json(output / 'final-freeze.json', freeze)
        # This is the first model scoring of the previously observed diagnostic years.
        diagnostic = _load_partition(con, features, assignments, 'final', 'reserved_final')
        _check_partition(training, diagnostic)
        scores = {'original': _predict(output, original_artifact, diagnostic, original)}
        scores.update(_reference_scores(diagnostic, climate))
        if winner not in BASELINES:
            scores[winner] = _predict(output, selected_artifact, diagnostic, selected_model)
        final = {}
        years = diagnostic['issue_date'].astype('datetime64[Y]').astype(str)
        for year in ('2024', '2025'):
            mask = years == year
            final[year] = _export({key: value[mask] for key, value in diagnostic.items()},
                {key: value[mask] for key, value in scores.items()}, output, f'diagnostic-{year}')
        aggregate = {name: summarize(diagnostic, values)['summary'] for name, values in scores.items()}
    if sha256(output / 'selection.json') != selection_hash:
        raise ValueError('Selection changed after diagnostic scoring')
    comparison = dict(selected_model=winner, validation_means=selection['validation_means'],
        diagnostic={year: {name: report['summary'] for name, report in reports.items()}
                    for year, reports in final.items()}, diagnostic_aggregate=aggregate,
        diagnostic_deltas={year: dict(precision=reports[winner]['summary']['precision'] -
                                    reports['original']['summary']['precision'],
                                    macro_recall=reports[winner]['summary']['macro_recall'] -
                                    reports['original']['summary']['macro_recall'])
                           for year, reports in final.items()},
        target_precision=.8, target_met_all_diagnostic_years=all(
            reports[winner]['summary']['precision'] >= .8 for reports in final.values()),
        independent_validation=False, promotion_unlocked=False, reload_exact=True,
        limitations=LIMITATIONS, elapsed_seconds=time.monotonic() - started)
    _json(output / 'comparison.json', comparison)
    _json(output / 'manifest.json', dict(schema='weekly-ranking-experiment/2.0.0',
        selected_model=winner, contract=contract, versions=versions, code_sha256=_code_hashes(),
        grid_sha256=GRID_SHA256, assignment_sha256=ASSIGNMENT_SHA256,
        feature_provenance=provenance, protocol_sha256=sha256(protocol),
        outputs={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        limitations=LIMITATIONS, independent_validation=False, promotion_unlocked=False))
    return comparison


def score(features, model_dir, output_dir, issue_date):
    """Historical inference projects predictors only; labels are neither required nor read."""
    features, directory, output = Path(features), Path(model_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    _, freeze = _load_bundle(directory)
    _, contract = _feature_contract(features)
    if contract != freeze['contract']:
        raise ValueError('Inference feature contract mismatch')
    with duckdb.connect() as con:
        columns = ','.join(IDENTIFIERS + list(FEATURE_NAMES))
        data = con.execute(f'''SELECT {columns} FROM read_parquet(?)
            WHERE issue_date=?::DATE ORDER BY cell_id''', [str(features), issue_date]).fetchnumpy()
    if len(data['cell_id']) != 161 or data['cell_id'].tolist() != contract['cell_ids']:
        raise ValueError('Require all 161 fixed-geography cells for the issue date')
    days = data['issue_date'].astype('datetime64[D]')
    if (not np.all(data['horizon_days'] == 7)
            or not np.all(data['target_start_date'].astype('datetime64[D]') == days + np.timedelta64(1, 'D'))
            or not np.all(data['target_end_date'].astype('datetime64[D]') == days + np.timedelta64(7, 'D'))):
        raise ValueError('Inference requires the same D+1..D+7 target')
    if days.min() <= np.datetime64(freeze['training_last_target']).astype('datetime64[D]'):
        raise ValueError('Inference issue must follow the training outcome cutoff')
    winner = freeze['selected_model']
    if freeze['selected_artifact'] is not None:
        scores = _predict(directory, freeze['selected_artifact'], data)
    else:
        climate = json.loads((directory / 'final-climatology.json').read_text())
        scores = _reference_scores(data, climate)[winner]
    rank, selected = ranks_and_selection(data['cell_id'], data['issue_date'], scores)
    output.mkdir(parents=True)
    columns = {key: data[key] for key in IDENTIFIERS}
    columns.update(score=scores.astype(np.float64), rank=rank, selected_top_k=selected)
    pq.write_table(pa.table(columns), output / 'rankings.parquet', compression='zstd')
    result = dict(schema='weekly-ranking-inference/1.0.0', selected_model=winner,
        issue_date=issue_date, rows=len(scores), highlights=int(selected.sum()),
        labels_read=False, model_manifest_sha256=sha256(directory / 'manifest.json'),
        features_sha256=sha256(features), rankings_sha256=sha256(output / 'rankings.parquet'),
        retrospective=True, promotion_unlocked=False, limitations=LIMITATIONS)
    _json(output / 'manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    training = subparsers.add_parser('train')
    training.add_argument('--features', required=True)
    training.add_argument('--assignments', required=True)
    training.add_argument('--protocol', required=True)
    training.add_argument('--output-dir', required=True)
    inference = subparsers.add_parser('score')
    inference.add_argument('--features', required=True)
    inference.add_argument('--model-dir', required=True)
    inference.add_argument('--output-dir', required=True)
    inference.add_argument('--issue-date', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    print(json.dumps(train(**args) if command == 'train' else score(**args), indent=2))


if __name__ == '__main__':
    main()
