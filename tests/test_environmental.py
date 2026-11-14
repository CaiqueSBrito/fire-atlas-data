"""Synthetic contract regressions, not scientific validation."""
from datetime import date, timedelta
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data import environmental as e
from fire_atlas_data import quality as q
from fire_atlas_data import ranking as r


def _write_manifest(directory, rows, columns):
    directory.mkdir()
    schema=pa.schema([('cell_id',pa.string()),('issue_date',pa.date32()),
                      *((name,pa.float64()) for name in columns)])
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),directory/'covariates.parquet')
    r._json(directory/'manifest.json',dict(method_version='fixture',columns=list(columns),
        units={name:'fixture' for name in columns},outputs=[dict(path='covariates.parquet',sha256=r._hash(directory/'covariates.parquet'))]))
    return directory


def _inputs(tmp_path):
    rows=[]
    for offset,split in enumerate(('train','validation','test','publish')):
        for i in range(15):
            label=None if split=='publish' else int(i<3)
            row=dict(cell_id=f'cell-{i:02}',issue_date=date(2026,10,1)+timedelta(days=offset),
                     target_date=date(2026,10,2)+timedelta(days=offset),region='fixture',split=split,label=label)
            row.update({name:float(i%4) for name in e.HISTORY_FEATURES})
            rows.append(row)
    base=tmp_path/'base';base.mkdir()
    pq.write_table(pa.Table.from_pylist(rows),base/'features.parquet')
    config=dict(target_mode='any_activity',snpp_history=True,enhanced_features=True,
                minimum_history_days=14,bbox=[-48,-17,-46,-15],h3_resolution=5,region='fixture')
    metadata=dict(method_version='1.3.0',config=config,feature_columns=list(e.HISTORY_FEATURES),
                  outputs=[dict(path='features.parquet',sha256=r._hash(base/'features.parquet'))])
    r._json(base/'manifest.json',metadata)
    sources=[]
    for folder,columns in [('wx',e.WEATHER_FEATURES),('era5',e.ERA5_FEATURES),('veg',e.VEGETATION_FEATURES)]:
        values=[dict(cell_id=row['cell_id'],issue_date=row['issue_date'],
                     **{name:(None if row['cell_id']=='cell-14' else float(row['label'] or 0)) for name in columns}) for row in rows]
        sources.append(_write_manifest(tmp_path/folder,values,columns))
    return base,rows,metadata,sources


def _enrich(tmp_path, base, sources):
    output=tmp_path/'enriched'
    e.enrich(base/'features.parquet',sources[0],sources[1],sources[2],output)
    return output


def _baseline(tmp_path,rows,metadata):
    directory=tmp_path/'baseline';directory.mkdir()
    spec=dict(features=list(e.HISTORY_FEATURES),params=e.FIXED_PARAMS)
    model=q._fit([x for x in rows if x['split']=='train'],spec)
    model.save_model(directory/'activity_fixed.json')
    saved=dict(file='activity_fixed.json',features=list(e.HISTORY_FEATURES),sha256=r._hash(directory/'activity_fixed.json'))
    r._json(directory/'manifest.json',dict(training_plan=dict(profile=spec),feature_contract=dict(config=metadata['config']),
                                         feature_manifest=metadata,models=dict(activity_fixed=saved)))
    return directory


def test_enrichment_preserves_population_labels_and_unknowns(tmp_path):
    base,rows,_,sources=_inputs(tmp_path)
    output=_enrich(tmp_path,base,sources)
    actual=pq.read_table(output/'features.parquet').to_pylist()
    original={(x['cell_id'],x['issue_date']):x for x in rows}
    assert len(actual)==len(rows)
    for row in actual:
        assert {name:row[name] for name in rows[0]}==original[row['cell_id'],row['issue_date']]
        assert row['fire_observation_status']==('detected' if row['label']==1 else 'unknown')
        if row['cell_id']=='cell-14':
            assert all(row[name] is None for name in e.WEATHER_FEATURES)
    with pytest.raises(FileExistsError):
        _enrich(tmp_path,base,sources)


@pytest.mark.parametrize('defect',['duplicate','future_key','missing','infinite'])
def test_covariate_population_mismatch_or_invalid_values_fail_atomically(tmp_path,defect):
    base,_,_,sources=_inputs(tmp_path)
    path=sources[0]/'covariates.parquet'
    table=pq.read_table(path);rows=table.to_pylist()
    if defect=='duplicate': rows.append(dict(rows[0]))
    elif defect=='future_key': rows[0]['issue_date']+=timedelta(days=10)
    elif defect=='missing': rows.pop()
    else: rows[0][e.WEATHER_FEATURES[0]]=float('inf')
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
    manifest=json.loads((sources[0]/'manifest.json').read_text());manifest['outputs'][0]['sha256']=r._hash(path)
    r._json(sources[0]/'manifest.json',manifest)
    with pytest.raises(ValueError): _enrich(tmp_path,base,sources)
    assert not (tmp_path/'enriched').exists()


def test_covariate_hash_tampering_rejected(tmp_path):
    base,_,_,sources=_inputs(tmp_path)
    path=sources[0]/'covariates.parquet'
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='hash mismatch'): _enrich(tmp_path,base,sources)


def test_train_reload_exports_consistent_metrics_and_null_label_inference(tmp_path):
    base,rows,metadata,sources=_inputs(tmp_path)
    enriched=_enrich(tmp_path,base,sources)
    baseline=_baseline(tmp_path,rows,metadata)
    model_dir=tmp_path/'model'
    result=e.train(enriched/'features.parquet',baseline,model_dir)
    predictions=pq.read_table(model_dir/'test_predictions.parquet').to_pylist()
    for name in ('baseline',*e.PROFILES):
        hits=sum(x['label'] for x in predictions if x['model_name']==name and x['selected_top_k'])
        assert result['test'][name]['micro']['hits']==hits
        assert result['test'][name]['micro']['positives']==3
    report=e.score(enriched/'features.parquet',model_dir,tmp_path/'inference')
    scored=pq.read_table(tmp_path/'inference/scores.parquet').to_pylist()
    assert {x['cell_id'] for x in scored}=={x['cell_id'] for x in rows if x['split']=='publish'}
    assert sorted(x['rank'] for x in scored)==list(range(1,16))
    assert report['operational_publication'] is False
    feature_manifest=enriched/'manifest.json'
    original_manifest=feature_manifest.read_text()
    changed=json.loads(original_manifest)
    changed['environment_sources']['weather']['provenance']['column_units']={e.WEATHER_FEATURES[3]:'km/h instead of m/s'}
    r._json(feature_manifest,changed)
    with pytest.raises(ValueError,match='feature contract differs'):
        e.score(enriched/'features.parquet',model_dir,tmp_path/'wrong-units')
    assert not (tmp_path/'wrong-units').exists()
    feature_manifest.write_text(original_manifest)
    manifest=json.loads((model_dir/'manifest.json').read_text())
    artifact=model_dir/manifest['models'][manifest['selected_model']]['file']
    artifact.write_bytes(artifact.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        e.score(enriched/'features.parquet',model_dir,tmp_path/'corrupt-inference')
    assert not (tmp_path/'corrupt-inference').exists()


@pytest.mark.parametrize('scores,auc,ap',[([1.,0.],1.,1.),([0.,1.],0.,.5),([.5,.5],.5,.5)])
def test_discrimination_ties_and_order(scores,auc,ap):
    result=e._discrimination([dict(label=1),dict(label=0)],np.asarray(scores))
    assert result==dict(roc_auc=auc,average_precision=ap)
