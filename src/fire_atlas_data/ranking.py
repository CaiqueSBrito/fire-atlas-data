"""Retrospective, observation-conditioned cell rankings; never calibrated probabilities."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBClassifier

SEASONAL = ["seasonal_modis", "seasonal_viirs"]
FULL = SEASONAL + ["doy_sin", "doy_cos", "neighbors_3d", "neighbors_7d", "days_since_last", "regional_7d", "regional_30d"]
COLUMNS = ["cell_id", "issue_date", "target_date", "region", *FULL, "label", "split"]


def _files(path):
    p = Path(path)
    files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    if not files or any(not f.is_file() for f in files):
        raise ValueError(f"No Parquet inputs: {path}")
    return files


def _load(path, publication=False):
    files = _files(path)
    with duckdb.connect() as con:
        table = con.read_parquet([str(f) for f in files], union_by_name=True)
        missing = set(COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        rows = table.project(','.join(COLUMNS)).order('issue_date,cell_id').fetchall()
    if not rows:
        raise ValueError("Empty feature dataset")
    records = [dict(zip(COLUMNS, row)) for row in rows]
    seen = set()
    for r in records:
        key = (r['issue_date'], r['cell_id'])
        if key in seen:
            raise ValueError(f"Duplicate issue/cell candidate: {key}")
        seen.add(key)
        allowed = {'train', 'validation', 'test', 'publish'}
        if r['split'] not in allowed or (r['split'] != 'publish' and r['label'] not in (0, 1)):
            raise ValueError("Invalid split or binary observation label")
        if r['target_date'] != r['issue_date'] + timedelta(days=1):
            raise ValueError("Target must be day after issue date")
    dates = {s: {r['issue_date'] for r in records if r['split'] == s} for s in ('train','validation','test')}
    if not publication and any(not d for d in dates.values()):
        raise ValueError("All three temporal splits must be nonempty")
    if not publication and not (max(dates['train']) < min(dates['validation']) and max(dates['validation']) < min(dates['test'])):
        raise ValueError("Splits must be disjoint chronological train/validation/test")
    return records, files


def _matrix(rows, names):
    if names not in (SEASONAL, FULL):
        raise ValueError("Model feature names/order do not match supported schema")
    return np.asarray([[np.nan if r[n] is None else float(r[n]) for n in names] for r in rows], dtype=np.float32)


def _sample_train(rows, seed=42, ratio=10):
    groups = {}
    for r in rows:
        if r['split'] == 'train':
            groups.setdefault((r['issue_date'], r['region']), []).append(r)
    selected = []
    for key in sorted(groups):
        group = groups[key]
        positives = sorted((r for r in group if r['label'] == 1), key=lambda r: r['cell_id'])
        negatives = [r for r in group if r['label'] == 0]
        negatives.sort(key=lambda r: hashlib.sha256(f"{seed}|{key}|{r['cell_id']}".encode()).digest())
        selected.extend(positives + negatives[:ratio * len(positives)])
    return selected


def _ordered(rows, scores):
    if len(rows) != len(scores):
        raise ValueError("Score count differs from candidate count")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite ranking score")
    return sorted(zip(rows, scores), key=lambda item: (-float(item[1]), item[0]['cell_id']))


def _metrics(rows, scores, k):
    groups = {}
    for r, score in zip(rows, scores):
        groups.setdefault((r['region'], r['issue_date']), []).append((r, score))
    result = []
    for (region, day), group in sorted(groups.items()):
        ordered = _ordered([r for r, _ in group], [s for _, s in group])
        selected = ordered[:k]
        positives = sum(r['label'] for r, _ in group)
        hits = sum(r['label'] for r, _ in selected)
        result.append({'region': region, 'month': day.strftime('%Y-%m'), 'issue_date': str(day), 'n_candidates': len(group), 'n_positives': positives,
                       'hits': hits, 'K': k, 'n_selected': len(selected),
                       'precision': hits / len(selected) if selected else None,
                       'recall': hits / positives if positives else None})
    return result


def _objective(metrics):
    values = [m['recall'] for m in metrics if m['recall'] is not None]
    if not values:
        raise ValueError("Validation has no observed positives; model selection is unsupported")
    return sum(values) / len(values)


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _feature_contract(features):
    path = Path(features)
    manifest_path = (path if path.is_dir() else path.parent) / 'manifest.json'
    provenance = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    # Acquisition/split dates and sources may change when scoring a new issue date.
    variable = {'sources', 'splits', 'publish_date', 'warmup_start', 'memory_limit', 'threads'}
    config = {k: v for k, v in provenance['config'].items() if k not in variable} if provenance else None
    contract = {'feature_order': FULL, 'config': config,
                'method_version': provenance.get('method_version') if provenance else None,
                'features_code_sha256': _hash(Path(__file__).with_name('features.py'))}
    return contract, provenance


def _backtest(output, rows, scores, manifest):
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault((row['region'], row['issue_date']), []).append((row, score))
    predictions = []
    for group in grouped.values():
        ordered = _ordered([row for row, _ in group], [score for _, score in group])
        for rank, (row, score) in enumerate(ordered, 1):
            predictions.append({key: row[key] for key in ('cell_id', 'region', 'issue_date', 'target_date', 'label')} |
                               {'score': float(score), 'rank': rank, 'selected_top_k': rank <= manifest['K'],
                                'model_version': manifest['model_version'], 'model_name': manifest['selected_model']})
    path = output / 'backtest_predictions.parquet'
    pq.write_table(pa.Table.from_pylist(predictions), path)
    return {'path': path.name, 'sha256': _hash(path), 'rows': len(predictions),
            'split': 'test', 'label_interpretation': '1 observed new detection; 0 unlabeled, not confirmed fire absence'}


def _predict(model_dir, manifest, name, rows):
    spec = manifest['models'][name]
    path = Path(model_dir) / spec['file']
    if _hash(path) != spec['sha256']:
        raise ValueError("Model hash mismatch")
    model = XGBClassifier()
    model.load_model(path)
    names = spec['features']
    if model.n_features_in_ != len(names):
        raise ValueError("Model feature count mismatch")
    return model.predict_proba(_matrix(rows, names))[:, 1]


def _map(path, ranked, generated, split):
    latest = max(r['target_date_utc'] for r in ranked)
    rows = [r for r in ranked if r['target_date_utc'] == latest]
    polygons = []
    points = []
    for row in rows:
        boundary = [(lon, -lat) for lat, lon in h3.cell_to_boundary(row['cell_id'])]
        points.extend(boundary)
        polygons.append((row, boundary))
    xs, ys = zip(*points)
    width, height = max(max(xs)-min(xs), .01), max(max(ys)-min(ys), .01)
    shapes = []
    for row, boundary in polygons:
        coords = ' '.join(f'{x},{y}' for x, y in boundary)
        title = html.escape(f"{row['cell_id']} | posição {row['rank']} | escore {row['score']:.6g}")
        shapes.append(f'<polygon points="{coords}" fill="#e4572e" fill-opacity="0.65" stroke="#333" stroke-width="{width/2000}"><title>{title}</title></polygon>')
    payload = json.dumps(rows, default=str).replace('<', '\\u003c')
    Path(path).write_text(f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Ranking retrospectivo de focos observados</title>
<style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:1em}}svg{{width:100%;height:65vh;background:#eef3f5}}polygon:hover{{fill:#ffdf00}}</style>
<h1>Ranking retrospectivo — não operacional</h1><p>Data-alvo: {latest}. Partição: {html.escape(split)}. Emissão nominal histórica: {html.escape(str(rows[0]['issued_at_utc']))}. Atualização deste artefato (UTC): {generated}.</p>
<p>Ordenação histórica, não previsão atual nem emitida ao vivo. Partição test: alvos observados reservados; partição publish: rótulos desconhecidos, sem avaliação. Escores não são probabilidades calibradas. Controles sem detecção não comprovam ausência de fogo. Sem histórico anterior, a referência sazonal é não suportada.</p>
<p>Polígonos H3 em coordenadas geográficas (sem mapa-base); norte para cima. Passe o cursor para posição e escore.</p>
<svg role="img" aria-label="Células H3 ordenadas" viewBox="{min(xs)-width*.03} {min(ys)-height*.03} {width*1.06} {height*1.06}" preserveAspectRatio="xMidYMid meet">{''.join(shapes)}</svg>
<script type="application/json" id="rankings">{payload}</script></html>''')


def train(features, output_dir, k=100, seed=42):
    if k < 1:
        raise ValueError("K must be positive")
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    rows, files = _load(features)
    training = _sample_train(rows, seed)
    if {r['label'] for r in training} != {0, 1}:
        raise ValueError("Sampled training requires observed positives and unlabeled controls")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(n_estimators=120, max_depth=3, learning_rate=.05, subsample=.8,
                  colsample_bytree=1., n_jobs=2, tree_method='hist', random_state=seed,
                  objective='binary:logistic', eval_metric='logloss')
    manifest = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'K': k, 'seed': seed, 'controls_per_positive': 10, 'config': config,
                'versions': {p: importlib.metadata.version(p) for p in ('xgboost','numpy','duckdb','h3')},
                'inputs': [{'path': str(f), 'sha256': _hash(f)} for f in files],
                'models': {}, 'interpretation': 'Retrospective ranking of observed new detections; not calibrated probabilities.',
                'seasonal_reference': 'Unsupported previous-years baseline: no historical years available.'}
    contract, provenance = _feature_contract(features)
    manifest['feature_contract'] = contract
    manifest['feature_manifest'] = provenance
    manifest['ranking_code_sha256'] = _hash(__file__)
    validation = [r for r in rows if r['split'] == 'validation']
    test = [r for r in rows if r['split'] == 'test']
    comparison = {'seasonal_reference_supported': False, 'seasonal_reference_note': manifest['seasonal_reference'], 'validation': {}, 'test': {}}
    for name, names in [('seasonal_only', SEASONAL), ('full', FULL)]:
        model = XGBClassifier(**config)
        model.fit(_matrix(training, names), np.asarray([r['label'] for r in training]))
        path = output / f'{name}.json'
        model.save_model(path)
        manifest['models'][name] = {'file': path.name, 'features': names, 'sha256': _hash(path)}
        comparison['validation'][name] = _metrics(validation, _predict(output, manifest, name, validation), k)
    # Ties deliberately favor the simpler seasonal-only model; test never selects.
    manifest['selected_model'] = max(('seasonal_only','full'), key=lambda n: _objective(comparison['validation'][n]))
    for name in manifest['models']:
        comparison['test'][name] = _metrics(test, _predict(output, manifest, name, test), k)
    for split, subset in [('validation', validation), ('test', test)]:
        comparison[split]['seasonal_reference'] = _metrics(subset, [float(r['seasonal_modis'] or 0) + float(r['seasonal_viirs'] or 0) for r in subset], k)
    comparison['selection_metric'] = 'macro daily recall@K on validation observed detections; no-positive days omitted'
    comparison['selected_model'] = manifest['selected_model']
    comparison['limitations'] = ['Labels are observed detections, not verified fire absence.', 'available_at is unavailable; nominal day-end timing is not operational latency validation.', 'No meaningful previous-years seasonal comparison or scientific generalization claim is supported.']
    manifest['model_version'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    selected_scores = _predict(output, manifest, manifest['selected_model'], test)
    manifest['backtest_output'] = _backtest(output, test, selected_scores, manifest)
    _json(output / 'manifest.json', manifest)
    _json(output / 'comparison.json', comparison)
    return manifest


def publish(features, model_dir, output_dir):
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    rows, _ = _load(features, publication=True)
    manifest = json.loads((Path(model_dir) / 'manifest.json').read_text())
    contract, _ = _feature_contract(features)
    if contract != manifest['feature_contract']:
        raise ValueError("Feature geometry/version contract differs from training")
    if _hash(__file__) != manifest['ranking_code_sha256']:
        raise ValueError("Ranking code differs from training")
    split = 'publish' if any(r['split'] == 'publish' for r in rows) else 'test'
    rows = [r for r in rows if r['split'] == split]
    if not rows:
        raise ValueError("No publish or held-out test candidates")
    scores = _predict(model_dir, manifest, manifest['selected_model'], rows)
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault((row['region'], row['issue_date']), []).append((row, score))
    ranked = []
    for (region, day), group in sorted(grouped.items()):
        ordered = _ordered([r for r, _ in group], [s for _, s in group])[:manifest['K']]
        for rank, (row, score) in enumerate(ordered, 1):
            ranked.append({'region': region, 'cell_id': row['cell_id'], 'issued_at_utc': datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=23, minute=59, second=59),
                           'target_date_utc': row['target_date'], 'score': float(score), 'rank': rank, 'model_version': manifest['model_version']})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([('region', pa.string()), ('cell_id', pa.string()), ('issued_at_utc', pa.timestamp('us', tz='UTC')), ('target_date_utc', pa.date32()), ('score', pa.float64()), ('rank', pa.int32()), ('model_version', pa.string())])
    pq.write_table(pa.Table.from_pylist(ranked, schema=schema), output / 'rankings.parquet')
    generated = datetime.now(timezone.utc).isoformat()
    _map(output / 'map.html', ranked, generated, split)
    _json(output / 'publication.json', {'generated_at_utc': generated, 'model_version': manifest['model_version'], 'historical_not_live': True, 'split': split, 'K': manifest['K'], 'rows': len(ranked)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--features', required=True)
    training.add_argument('--output-dir', required=True)
    training.add_argument('--k', type=int, default=100)
    training.add_argument('--seed', type=int, default=42)
    publishing = commands.add_parser('publish')
    publishing.add_argument('--features', required=True)
    publishing.add_argument('--model-dir', required=True)
    publishing.add_argument('--output-dir', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    (train if command == 'train' else publish)(**args)


if __name__ == '__main__':
    main()
