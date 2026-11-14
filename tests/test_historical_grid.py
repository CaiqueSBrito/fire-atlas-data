"""Daily target windows include future labels but never future predictor counts."""
from datetime import date
import json

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data.historical_grid import build
from fire_atlas_data.normalization import sha256
from fire_atlas_data.ranking import _json


def _history(tmp_path):
    history=tmp_path/'history';history.mkdir()
    a=h3.latlng_to_cell(-16,-47,5)
    b=next(cell for cell in h3.grid_disk(a,1) if cell!=a)
    candidates=tmp_path/'candidates.parquet'
    pq.write_table(pa.Table.from_pylist([dict(cell_id=cell) for cell in (a,b)]),candidates)
    rows=[dict(latitude=h3.cell_to_latlng(cell)[0],longitude=h3.cell_to_latlng(cell)[1],
               acquisition_date=day) for cell,day in [(a,date(2018,4,1)),(a,date(2018,4,2)),
                                                      (b,date(2018,4,2)),(a,date(2018,4,9))]]
    pq.write_table(pa.Table.from_pylist(rows),history/'observations.parquet')
    _json(history/'manifest.json',dict(schema='historical-observations/1.0.0',product_declared='VIIRS_NOAA20_SP',
                                      counts=dict(region=4),outputs=[dict(path='observations.parquet',sha256=sha256(history/'observations.parquet'))]))
    return history,candidates,a,b


def test_all_cells_labels_and_exact_past_seven_day_counts(tmp_path):
    history,candidates,a,b=_history(tmp_path)
    output=tmp_path/'grid'
    metadata=build(history,candidates,'2018-04-01','2018-04-08',output,horizon_days=1)
    assert metadata['rows']==16 and metadata['positives']==3
    rows=pq.read_table(output/'grid.parquet').to_pylist()
    indexed={(r['cell_id'],r['issue_date']):r for r in rows}
    first=indexed[a,date(2018,4,1)]
    assert first['label']==1 and first['cell_detection_rows_7d']==1
    assert first['regional_detection_rows_7d']==1
    assert indexed[b,date(2018,4,1)]['label']==1
    assert indexed[b,date(2018,4,1)]['cell_detection_rows_7d']==0
    assert indexed[b,date(2018,4,7)]['label']==0
    assert indexed[b,date(2018,4,7)]['fire_observation_status']=='unknown'
    # April1 falls out of D-6..D on April8; April9 only changes the label.
    last=indexed[a,date(2018,4,8)]
    assert last['cell_detection_rows_7d']==1 and last['label']==1
    assert last['detection_rows_on_issue']==0
    assert metadata['target_product']=='VIIRS_NOAA20_SP'


def test_historical_grid_refuses_future_or_unsupported_range(tmp_path):
    history,candidates,_,_=_history(tmp_path)
    with pytest.raises(ValueError,match='Historical issues'):
        build(history,candidates,'2026-01-01','2026-01-02',tmp_path/'bad',horizon_days=1)
    assert not (tmp_path/'bad').exists()


def test_historical_input_tampering_is_not_reused(tmp_path):
    history,candidates,_,_=_history(tmp_path)
    path=history/'observations.parquet';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        build(history,candidates,'2018-04-01','2018-04-02',tmp_path/'bad',horizon_days=1)
    assert not (tmp_path/'bad').exists()


def test_seven_day_target_includes_d_plus_1_and_d_plus_7_only(tmp_path):
    history,candidates,a,b=_history(tmp_path)
    cells=[a,b]+sorted(set(h3.grid_disk(a,1))-{a,b})[:3]
    c,d,e=cells[2:]
    pq.write_table(pa.Table.from_pylist([dict(cell_id=cell) for cell in cells]),candidates)
    events=[(a,date(2018,4,1)),(b,date(2018,4,2)),(c,date(2018,4,8)),
            (d,date(2018,4,9)),(e,date(2018,4,3)),(e,date(2018,4,5))]
    rows=[dict(latitude=h3.cell_to_latlng(cell)[0],longitude=h3.cell_to_latlng(cell)[1],
               acquisition_date=day) for cell,day in events]
    source=history/'observations.parquet'
    pq.write_table(pa.Table.from_pylist(rows),source)
    manifest=json.loads((history/'manifest.json').read_text())
    manifest['outputs'][0]['sha256']=sha256(source)
    _json(history/'manifest.json',manifest)
    output=tmp_path/'seven-days'
    result=build(history,candidates,'2018-04-01','2018-04-01',output)
    actual={row['cell_id']:row for row in pq.read_table(output/'grid.parquet').to_pylist()}
    assert {cell:row['label'] for cell,row in actual.items()}=={a:0,b:1,c:1,d:0,e:1}
    assert actual[a]['detection_rows_on_issue']==1
    assert actual[e]['target_active_days']==2 and actual[e]['target_detection_rows']==2
    assert actual[c]['target_start_date']==date(2018,4,2)
    assert actual[c]['target_end_date']==date(2018,4,8)
    assert all(row['horizon_days']==7 for row in actual.values())
    assert result['horizon_days']==7
    assert 'target_active_days' not in result['predictor_columns']


def test_future_window_changes_labels_without_changing_lags(tmp_path):
    history,candidates,a,b=_history(tmp_path)
    first=tmp_path/'first'
    build(history,candidates,'2018-04-01','2018-04-02',first)
    source=history/'observations.parquet'
    rows=pq.read_table(source).to_pylist()
    rows.append(dict(latitude=h3.cell_to_latlng(b)[0],longitude=h3.cell_to_latlng(b)[1],
                     acquisition_date=date(2018,4,8)))
    pq.write_table(pa.Table.from_pylist(rows),source)
    manifest=json.loads((history/'manifest.json').read_text())
    manifest['outputs'][0]['sha256']=sha256(source)
    _json(history/'manifest.json',manifest)
    second=tmp_path/'second'
    build(history,candidates,'2018-04-01','2018-04-02',second)
    before={(r['cell_id'],r['issue_date']):r for r in pq.read_table(first/'grid.parquet').to_pylist()}
    after={(r['cell_id'],r['issue_date']):r for r in pq.read_table(second/'grid.parquet').to_pylist()}
    assert before[b,date(2018,4,2)]['label']==0 and after[b,date(2018,4,2)]['label']==1
    for key,row in before.items():
        for column in ('detection_rows_on_issue','cell_detection_rows_7d','regional_detection_rows_7d'):
            assert row[column]==after[key][column]


def test_incomplete_seven_day_tail_is_refused(tmp_path):
    history,candidates,_,_=_history(tmp_path)
    with pytest.raises(ValueError,match='Historical issues'):
        build(history,candidates,'2025-12-25','2025-12-25',tmp_path/'censored')
    assert not (tmp_path/'censored').exists()
    output=tmp_path/'last-complete'
    manifest=build(history,candidates,'2025-12-24',None,output)
    assert manifest['issue_dates']==['2025-12-24','2025-12-24']
    assert manifest['target_end_dates']==['2025-12-31','2025-12-31']


@pytest.mark.parametrize('horizon',[True,0,2,7.0])
def test_unsupported_horizon_fails_before_creating_output(tmp_path,horizon):
    history,candidates,_,_=_history(tmp_path)
    with pytest.raises(ValueError,match='horizon_days'):
        build(history,candidates,'2018-04-01','2018-04-02',tmp_path/'bad',horizon_days=horizon)
    assert not (tmp_path/'bad').exists()
