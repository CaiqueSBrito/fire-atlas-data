from datetime import date, timedelta
import json

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fire_atlas_data import ranking as r


def _row(cell='a', label=0, split='train', day=date(2026, 10, 1)):
    return dict(cell_id=cell, issue_date=day, target_date=day + timedelta(days=1),
                region='fixture', label=label, split=split, **{n: 0. for n in r.FULL})


def test_sampling_no_positive_groups_contribute_no_controls():
    rows = [_row(str(i)) for i in range(25)]
    assert r._sample_train(rows) == []
    rows += [_row('positive', 1), _row('validation', 1, 'validation')]
    selected = r._sample_train(rows)
    assert len(selected) == 11
    assert sum(x['label'] for x in selected) == 1
    assert selected == r._sample_train(list(reversed(rows)))


def test_ranking_ties_and_observed_metrics():
    rows = [_row('b', 1), _row('a'), _row('c', 1)]
    assert [x['cell_id'] for x, _ in r._ordered(rows, [.5, .5, .2])] == ['a', 'b', 'c']
    metric = r._metrics(rows, [.5, .5, .2], 2)[0]
    assert metric['n_positives'] == 2
    assert metric['hits'] == 1
    assert metric['precision'] == metric['recall'] == .5
    assert r._metrics([_row()], [0], 100)[0]['recall'] is None
    with pytest.raises(ValueError, match='Nonfinite'):
        r._ordered(rows, [0, np.nan, 1])


def test_feature_order_is_a_consumer_contract():
    with pytest.raises(ValueError, match='feature names/order'):
        r._matrix([_row()], list(reversed(r.FULL)))
    row = _row()
    row['days_since_last'] = None
    assert np.isnan(r._matrix([row], r.FULL)[0, r.FULL.index('days_since_last')])


def test_train_reload_publish_and_unlabeled_boundary(tmp_path):
    cells = [h3.latlng_to_cell(-16 + i * .2, -47, 5) for i in range(5)]
    rows = []
    for offset, split in enumerate(('train', 'validation', 'test', 'publish')):
        for i, cell in enumerate(cells):
            row = _row(cell, int(i == 0), split, date(2026, 10, 1) + timedelta(days=offset))
            row['neighbors_3d'] = float(i == 0)
            if split == 'publish':
                row['label'] = None
            rows.append(row)
    features = tmp_path / 'features.parquet'
    pq.write_table(pa.Table.from_pylist(rows), features)
    manifest = r.train(features, tmp_path / 'model', k=2)
    comparison = json.loads((tmp_path / 'model' / 'comparison.json').read_text())
    assert comparison['seasonal_reference_supported'] is False
    assert all(len(values) == 1 for values in comparison['test'].values())
    backtest = pq.read_table(tmp_path / 'model' / 'backtest_predictions.parquet').to_pylist()
    expected = [row for row in rows if row['split'] == 'test']
    assert {(row['cell_id'], row['issue_date'], row['label']) for row in backtest} == {
        (row['cell_id'], row['issue_date'], row['label']) for row in expected}
    assert [row['rank'] for row in backtest] == list(range(1, len(expected) + 1))
    selected = [row for row in backtest if row['selected_top_k']]
    metric = comparison['test'][manifest['selected_model']][0]
    assert len(selected) == metric['n_selected'] == 2
    assert sum(row['label'] for row in selected) == metric['hits']
    assert sum(row['label'] for row in backtest) == metric['n_positives']
    assert all(row['model_version'] == manifest['model_version'] for row in backtest)
    assert manifest['backtest_output']['sha256'] == r._hash(tmp_path / 'model' / 'backtest_predictions.parquet')
    r.publish(features, tmp_path / 'model', tmp_path / 'published')
    ranked = pq.read_table(tmp_path / 'published' / 'rankings.parquet').to_pylist()
    assert len(ranked) == 2
    assert [row['rank'] for row in ranked] == [1, 2]
    assert all(row['model_version'] == manifest['model_version'] for row in ranked)
    assert all(row['target_date_utc'] == date(2026, 10, 5) for row in ranked)
    page = (tmp_path / 'published' / 'map.html').read_text()
    assert '<polygon ' in page and 'não operacional' in page
    assert 'https://' not in page
    # Scoring only needs a publish dataset, not historical labels or other splits.
    unlabeled = tmp_path / 'unlabeled.parquet'
    pq.write_table(pa.Table.from_pylist([row for row in rows if row['split'] == 'publish']), unlabeled)
    r.publish(unlabeled, tmp_path / 'model', tmp_path / 'unlabeled-output')
    manifest['models'][manifest['selected_model']]['features'] = ['wrong']
    (tmp_path / 'model' / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='feature count|feature names/order'):
        r.publish(unlabeled, tmp_path / 'model', tmp_path / 'bad')


def test_temporal_split_rejected(tmp_path):
    path = tmp_path / 'bad.parquet'
    pq.write_table(pa.Table.from_pylist([_row(split=s) for s in ('train', 'validation', 'test')]), path)
    with pytest.raises(ValueError, match='Duplicate|chronological'):
        r._load(path)


def test_feature_contract_allows_new_dates_not_new_grid(tmp_path):
    path = tmp_path / 'features.parquet'
    manifest = {'method_version': '1', 'config': {'bbox': [-48,-17,-46,-15], 'h3_resolution': 5, 'splits': {'train': ['2026-10-01','2026-10-10']}}}
    metadata = tmp_path / 'manifest.json'
    metadata.write_text(json.dumps(manifest))
    first, provenance = r._feature_contract(path)
    assert provenance == manifest
    manifest['config']['splits'] = {'publish': ['2026-11-01','2026-11-01']}
    metadata.write_text(json.dumps(manifest))
    assert r._feature_contract(path)[0] == first
    manifest['config']['h3_resolution'] = 6
    metadata.write_text(json.dumps(manifest))
    assert r._feature_contract(path)[0] != first


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        r.train('unused.parquet', tmp_path)
    with pytest.raises(FileExistsError):
        r.publish('unused.parquet', 'unused-model', tmp_path)


def test_metrics_separate_regions_and_include_month():
    a, b = _row('a', 1), _row('b', 0)
    b['region'] = 'other'
    metrics = r._metrics([a, b], [0, 1], 1)
    assert len(metrics) == 2
    assert all(m['month'] == '2026-10' and m['n_selected'] == 1 for m in metrics)
