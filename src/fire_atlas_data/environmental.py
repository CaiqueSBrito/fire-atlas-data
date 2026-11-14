"""Paired retrospective environmental ablations; no operational promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from . import quality as q
from . import ranking as r
from .features import HISTORY_FEATURES
from .normalization import literal, sha256
from .weather import ERA5_FEATURES, WEATHER_FEATURES
from .vegetation import VEGETATION_FEATURES


ENVIRONMENT_FEATURES = list(HISTORY_FEATURES) + list(WEATHER_FEATURES) + list(ERA5_FEATURES) + list(VEGETATION_FEATURES)
FIXED_PARAMS = dict(max_depth=3, n_estimators=120, learning_rate=.05,
                    min_child_weight=1, reg_lambda=1)
PROFILES = {
    'weather': list(HISTORY_FEATURES) + list(WEATHER_FEATURES),
    'historical_weather': list(HISTORY_FEATURES) + list(ERA5_FEATURES),
    'vegetation': list(HISTORY_FEATURES) + list(VEGETATION_FEATURES),
    'environment': ENVIRONMENT_FEATURES,
}


def _json(path):
    return json.loads(Path(path).read_text())


def _identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def _verify_identity(item):
    if sha256(Path(item['path'])) != item['sha256']:
        raise ValueError('Input artifact hash mismatch')


def _source(directory, expected):
    directory = Path(directory).resolve(strict=True)
    manifest = _json(directory / 'manifest.json')
    path = directory / 'covariates.parquet'
    # All producers identify the physical covariate output by its content hash.
    outputs = manifest.get('outputs', [])
    if isinstance(outputs, dict):
        entry = outputs.get('covariates.parquet')
    else:
        entry = next((x for x in outputs if x.get('path') == 'covariates.parquet'), None)
    if not entry or entry.get('sha256') != sha256(path):
        raise ValueError('Covariate artifact absent from manifest or hash mismatch')
    return path, dict(manifest=_identity(directory/'manifest.json'), provenance=manifest,
                      covariates=_identity(path), feature_order=list(expected))


def _validate_covariates(con, view, expected):
    schema = dict(row[:2] for row in con.execute(f'DESCRIBE {view}').fetchall())
    if set(schema) != {'cell_id', 'issue_date', *expected}:
        raise ValueError('Covariate columns differ from declared contract')
    if schema['cell_id'] != 'VARCHAR' or schema['issue_date'] != 'DATE':
        raise ValueError('Covariate keys must be VARCHAR/DATE')
    if any(schema[name] != 'DOUBLE' for name in expected):
        raise ValueError('Covariates must be DOUBLE with nulls preserved')
    if con.execute(f'''SELECT count(*) FROM (SELECT cell_id,issue_date,count(*) n
                       FROM {view} GROUP BY ALL HAVING n!=1 OR cell_id IS NULL OR issue_date IS NULL)''').fetchall()[0][0]:
        raise ValueError('Duplicate or null covariate keys')
    if con.execute(f'''SELECT count(*) FROM ((SELECT cell_id,issue_date FROM {view}
                       EXCEPT SELECT cell_id,issue_date FROM base) UNION ALL
                       (SELECT cell_id,issue_date FROM base EXCEPT SELECT cell_id,issue_date FROM {view}))''').fetchall()[0][0]:
        raise ValueError('Covariate key set differs from base feature population')
    nonfinite = ' OR '.join(f'({name} IS NOT NULL AND NOT isfinite({name}))' for name in expected)
    if con.execute(f'SELECT count(*) FROM {view} WHERE {nonfinite}').fetchall()[0][0]:
        raise ValueError('Non-finite environmental covariate')


def _matrix(rows, names):
    if names not in (list(HISTORY_FEATURES), *PROFILES.values()):
        raise ValueError('Unknown feature order')
    return np.asarray([[np.nan if row[name] is None else float(row[name]) for name in names]
                       for row in rows], dtype=np.float32)


def _load(path):
    path = Path(path).resolve(strict=True)
    manifest = _json(path.parent/'manifest.json')
    if manifest.get('method_version') != 'environment-1.0.0' or manifest.get('feature_columns') != ENVIRONMENT_FEATURES:
        raise ValueError('Environmental feature contract differs')
    if manifest['base_manifest']['config'].get('target_mode') != 'any_activity':
        raise ValueError('Require any_activity target')
    entry = next(x for x in manifest['outputs'] if x['path'] == 'features.parquet')
    if sha256(path) != entry['sha256']:
        raise ValueError('Enriched feature hash mismatch')
    table = pq.read_table(path)
    for name in ENVIRONMENT_FEATURES:
        if str(table.schema.field(name).type) not in ('double','int32'):
            raise ValueError('Unexpected feature type')
    rows = table.to_pylist()
    if not rows or len({(x['cell_id'],x['issue_date']) for x in rows}) != len(rows):
        raise ValueError('Empty or duplicate feature population')
    for row in rows:
        if row['split'] == 'publish':
            if row['label'] is not None:
                raise ValueError('Publish labels must be null')
        elif row['split'] not in ('train','validation','test') or row['label'] not in (0,1):
            raise ValueError('Invalid split/label')
    return rows, manifest


def _predict(directory, spec, rows):
    path = Path(directory)/spec['file']
    if sha256(path) != spec['sha256']:
        raise ValueError('Model artifact hash mismatch')
    model=xgb.Booster(); model.load_model(path); model.set_param({'nthread':2})
    names=spec['features']
    if model.feature_names != names or model.num_features() != len(names):
        raise ValueError('Native model feature names/order differ')
    return model.predict(xgb.DMatrix(_matrix(rows,names),feature_names=names))


def _fit(rows, names):
    params=dict(FIXED_PARAMS)
    rounds=params.pop('n_estimators')
    params['eta']=params.pop('learning_rate'); params['lambda']=params.pop('reg_lambda')
    params.update(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                  seed=42,nthread=2,subsample=.8,colsample_bytree=1.)
    return xgb.train(params,xgb.DMatrix(_matrix(rows,names),label=[x['label'] for x in rows],
                                     feature_names=names),num_boost_round=rounds)


def _discrimination(rows, scores):
    labels=np.asarray([x['label'] for x in rows])
    # Pairwise AUC without a positive-by-negative allocation.
    order=np.argsort(scores,kind='stable')
    sorted_scores=np.asarray(scores)[order]; sorted_labels=labels[order]
    positives=int(labels.sum()); negatives=len(labels)-positives
    if not positives or not negatives:
        return dict(roc_auc=None,average_precision=None)
    _, starts, counts=np.unique(sorted_scores,return_index=True,return_counts=True)
    auc_pairs=0.; negatives_below=0
    for start,count in zip(starts,counts):
        pos=int(sorted_labels[start:start+count].sum()); neg=count-pos
        auc_pairs += pos*(negatives_below+.5*neg); negatives_below += neg
    hits=seen=0; ap=0.
    for start,count in reversed(list(zip(starts,counts))):
        pos=int(sorted_labels[start:start+count].sum()); hits+=pos; seen+=count
        ap += pos/positives*hits/seen
    return dict(roc_auc=auc_pairs/(positives*negatives),average_precision=ap)


def _summary(rows,scores):
    return dict(**q._summary(rows,scores,10), **_discrimination(rows,scores))


def _contract(metadata):
    return dict(feature_order=metadata['feature_columns'],
                base_config={k:v for k,v in metadata['base_manifest']['config'].items() if k not in q.VARIABLE_CONFIG},
                source_contracts={key:{field:value for field,value in item['provenance'].items()
                                      if field in ('method','method_version','feature_columns','columns','units','column_units','availability','limitations','config')}
                                  for key,item in metadata['environment_sources'].items()})


def enrich(features,weather,era5,vegetation,output_dir):
    """Join exact populations, preserving original labels and explicit unknown observations."""
    features=Path(features).resolve(strict=True); target=Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    base=_json(features.parent/'manifest.json')
    if base['config'].get('target_mode') != 'any_activity' or base.get('feature_columns') != list(HISTORY_FEATURES):
        raise ValueError('Require any_activity HISTORY_FEATURES base')
    entry=next(x for x in base['outputs'] if x['path']=='features.parquet')
    if sha256(features) != entry['sha256']:
        raise ValueError('Base feature artifact hash mismatch')
    wx,wx_metadata=_source(weather,WEATHER_FEATURES)
    veg,veg_metadata=_source(vegetation,VEGETATION_FEATURES)
    historical,historical_metadata=_source(era5,ERA5_FEATURES)
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{target.name}-',dir=target.parent))
    try:
        with duckdb.connect() as con:
            con.execute("SET memory_limit='512MB'"); con.execute('SET threads=2')
            con.read_parquet(str(features)).create_view('base')
            if con.execute('SELECT count(*) FROM (SELECT cell_id,issue_date,count(*) n FROM base GROUP BY ALL HAVING n!=1)').fetchall()[0][0]:
                raise ValueError('Duplicate base keys')
            con.read_parquet(str(wx)).create_view('weather')
            con.read_parquet(str(veg)).create_view('vegetation')
            _validate_covariates(con,'weather',WEATHER_FEATURES)
            _validate_covariates(con,'vegetation',VEGETATION_FEATURES)
            con.read_parquet(str(historical)).create_view('era5')
            _validate_covariates(con,'era5',ERA5_FEATURES)
            wx_columns=','.join(f'w.{name}' for name in WEATHER_FEATURES)
            veg_columns=','.join(f'v.{name}' for name in VEGETATION_FEATURES)
            historical_columns=','.join(f'h.{name}' for name in ERA5_FEATURES)
            con.execute(f'''COPY (SELECT b.*,{wx_columns},{historical_columns},{veg_columns},
                CASE WHEN b.label=1 THEN 'detected' ELSE 'unknown' END fire_observation_status
                FROM base b JOIN weather w USING(cell_id,issue_date)
                JOIN era5 h USING(cell_id,issue_date)
                JOIN vegetation v USING(cell_id,issue_date) ORDER BY b.issue_date,b.cell_id)
                TO {literal(str(stage/'features.parquet'))} (FORMAT PARQUET,COMPRESSION ZSTD)''')
            counts=con.execute('SELECT split,count(*),sum(label) FROM base GROUP BY split ORDER BY split').fetchall()
        metadata=dict(method_version='environment-1.0.0',feature_columns=ENVIRONMENT_FEATURES,
                      base_manifest=base,base_manifest_identity=_identity(features.parent/'manifest.json'),
                      base_features=_identity(features),environment_sources=dict(weather=wx_metadata,era5=historical_metadata,vegetation=veg_metadata),
                      split_counts={s:dict(rows=n,positives=p) for s,n,p in counts},
                      code_sha256=sha256(Path(__file__)),
                      observation_semantics='label=0 is no supplied detection, not confirmed fire absence; no fire cloud mask available; fire_observation_status=unknown. Vegetation QA is not fire observability.',
                      limitations=['Retrospective only: historical API availability is unknown; no operational promotion.',
                                   'Known September/October periods reused; reserved independent period untouched.'],
                      outputs=[dict(path='features.parquet',sha256=sha256(stage/'features.parquet'))])
        for item in (metadata['base_features'],metadata['base_manifest_identity'],wx_metadata['manifest'],wx_metadata['covariates'],historical_metadata['manifest'],historical_metadata['covariates'],veg_metadata['manifest'],veg_metadata['covariates']):
            _verify_identity(item)
        r._json(stage/'manifest.json',metadata)
        stage.rename(target)
        return metadata
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def train(features,baseline_model_dir,output_dir):
    """Three predeclared ablations against an immutable compatible baseline."""
    target=Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    rows,metadata=_load(features)
    splits={s:[row for row in rows if row['split']==s] for s in ('train','validation','test','publish')}
    dates={s:sorted({row['issue_date'] for row in subset}) for s,subset in splits.items()}
    if any(not dates[s] for s in ('train','validation','test')) or not (max(dates['train'])<min(dates['validation'])<=max(dates['validation'])<min(dates['test'])):
        raise ValueError('Require nonempty chronological splits')
    baseline_dir=Path(baseline_model_dir).resolve(strict=True)
    baseline=_json(baseline_dir/'manifest.json')
    if baseline.get('training_plan',{}).get('profile',{}).get('params') != FIXED_PARAMS:
        raise ValueError('Baseline fixed parameters differ')
    if baseline.get('feature_contract',{}).get('config') != {k:v for k,v in metadata['base_manifest']['config'].items() if k not in q.VARIABLE_CONFIG}:
        raise ValueError('Baseline target/geography/feature contract differs')
    original=baseline['models']['activity_fixed']
    if original['features']!=list(HISTORY_FEATURES) or sha256(baseline_dir/original['file']) != original['sha256']:
        raise ValueError('Baseline feature/model artifact mismatch')
    if baseline['feature_manifest'] != metadata['base_manifest']:
        raise ValueError('Paired experiment requires exact original feature population')
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{target.name}-',dir=target.parent))
    try:
        plan=dict(profiles=PROFILES,fixed_params=FIXED_PARAMS,seed=42,nthread=2,top_k=10,
                  selection='Strict validation macro daily recall improvement over baseline; ties retain baseline/list order.',
                  scope='Retrospective paired ablation; no hyperparameter search or promotion; reused TEST diagnostic.')
        r._json(stage/'training-plan.json',plan)
        shutil.copyfile(baseline_dir/original['file'],stage/'baseline.json')
        models={'baseline':dict(original,file='baseline.json')}
        validations={'baseline':_predict(stage,models['baseline'],splits['validation'])}
        for name,names in PROFILES.items():
            model=_fit(splits['train'],names); model.save_model(stage/f'{name}.json')
            spec=dict(file=f'{name}.json',sha256=sha256(stage/f'{name}.json'),features=names)
            models[name]=spec
            values=_predict(stage,spec,splits['validation'])
            direct=model.predict(xgb.DMatrix(_matrix(splits['validation'],names),feature_names=names))
            if not np.array_equal(values,direct):
                raise ValueError('Reload inference differs from fitted model')
            validations[name]=values
        validation_metrics={name:_summary(splits['validation'],scores) for name,scores in validations.items()}
        if any(item['macro']['recall'] is None for item in validation_metrics.values()):
            raise ValueError('Validation has no observed positives')
        winner=max(validation_metrics,key=lambda name:validation_metrics[name]['macro']['recall'])
        r._json(stage/'selection.json',dict(selected_model=winner,validation=validation_metrics,selection=plan['selection']))
        # No fit or selection after this point. Score every predeclared ablation as diagnostic.
        test_scores={name:_predict(stage,spec,splits['test']) for name,spec in models.items()}
        comparison=dict(selected_model=winner,validation=validation_metrics,
                        test={name:_summary(splits['test'],scores) for name,scores in test_scores.items()},
                        training_counts=q._counts(splits['train']),test_reused_diagnostic=True,promotion_unlocked=False)
        for split in ('validation','test'):
            for ref in ('persistence','cell_active_days_30d','neighbors_7d'):
                scores=[int(row['days_since_last']==0) if ref=='persistence' else row[ref] for row in splits[split]]
                comparison[split][ref]=_summary(splits[split],scores)
        version=sha256(stage/models[winner]['file'])[:16]
        outputs=q._export(stage,splits['test'],test_scores,10,version)
        r._json(stage/'comparison.json',comparison)
        for name in ('training-plan.json','selection.json','comparison.json'):
            outputs[name]=dict(sha256=sha256(stage/name))
        manifest=dict(schema_version=1,model_version=version,selected_model=winner,models=models,
                      training_plan=plan,feature_contract=_contract(metadata),feature_manifest=_identity(Path(features).parent/'manifest.json'),
                      feature_input=_identity(features),baseline_manifest=_identity(baseline_dir/'manifest.json'),
                      code_sha256=sha256(Path(__file__)),
                      library_versions={name:importlib.metadata.version(name) for name in ('xgboost','numpy','duckdb','pyarrow')},
                      generated_at_utc=datetime.now(timezone.utc).isoformat(),outputs=outputs,promotion_unlocked=False)
        r._json(stage/'manifest.json',manifest)
        stage.rename(target)
        return comparison
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise


def score(features,model_dir,output_dir):
    """Exercise null-label inference, never publish an operational forecast."""
    target=Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    rows,metadata=_load(features); model_dir=Path(model_dir)
    manifest=_json(model_dir/'manifest.json')
    if manifest['code_sha256']!=sha256(Path(__file__)) or manifest['feature_contract']!=_contract(metadata):
        raise ValueError('Model code or feature contract differs')
    for filename,item in manifest['outputs'].items():
        if sha256(model_dir/filename)!=item['sha256']:
            raise ValueError('Frozen evaluation artifact hash mismatch')
    subset=[row for row in rows if row['split']=='publish']
    if not subset:
        raise ValueError('Require null-label publish partition')
    scores=_predict(model_dir,manifest['models'][manifest['selected_model']],subset)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite inference scores')
    target.mkdir(parents=True)
    groups={}
    for row,value in zip(subset,scores):
        groups.setdefault((row['region'],row['issue_date']),[]).append((row,value))
    result=[]
    for group in groups.values():
        for rank,(row,value) in enumerate(r._ordered([x[0] for x in group],[x[1] for x in group]),1):
            result.append(dict(cell_id=row['cell_id'],issue_date=row['issue_date'],target_date=row['target_date'],
                               score=float(value),rank=rank,selected_top_k=rank<=10,model_version=manifest['model_version']))
    pq.write_table(pa.Table.from_pylist(result),target/'scores.parquet')
    report=dict(rows=len(result),source_feature_manifest=_identity(Path(features).parent/'manifest.json'),
                model_manifest=_identity(model_dir/'manifest.json'),scores_sha256=sha256(target/'scores.parquet'),
                retrospective=True,promotion_unlocked=False,operational_publication=False,
                note='Unlabeled historical inference only; scores are not calibrated fire probabilities.')
    r._json(target/'inference.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('enrich'); p.add_argument('--features',required=True); p.add_argument('--weather',required=True); p.add_argument('--era5',required=True); p.add_argument('--vegetation',required=True); p.add_argument('--output-dir',required=True)
    p=commands.add_parser('train'); p.add_argument('--features',required=True); p.add_argument('--baseline-model-dir',required=True); p.add_argument('--output-dir',required=True)
    p=commands.add_parser('score'); p.add_argument('--features',required=True); p.add_argument('--model-dir',required=True); p.add_argument('--output-dir',required=True)
    args=vars(parser.parse_args(argv)); command=args.pop('command')
    result={'enrich':enrich,'train':train,'score':score}[command](**args)
    if command=='train':
        print(json.dumps({split:{name:item['micro'] for name,item in result[split].items()} for split in ('validation','test')},indent=2))
        print('Selected on validation:',result['selected_model'],'; no promotion')
    else:
        print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':
    main()
