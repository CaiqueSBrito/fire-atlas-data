from datetime import date
import json

import numpy as np
import pytest

from fire_atlas_data.weekly_metrics import ranks_and_selection, summarize


def _data(labels, dates=None):
    labels = np.asarray(labels)
    count = labels.shape[-1]
    if dates is None:
        dates = np.repeat(np.array(['2023-01-31', '2023-02-01'], dtype='datetime64[D]'), count)
    return dict(cell_id=np.tile([f'cell-{i:02d}' for i in range(count)], labels.size // count),
                issue_date=np.asarray(dates), label=labels.ravel(), irrelevant=np.zeros(labels.size))


def test_fixed_budget_oracle_and_all_rows_preserved():
    data = _data([[1] * 12, [1, 1] + [0] * 10])
    scores = np.tile(np.arange(12, dtype=float), 2)
    ranks, selected = ranks_and_selection(data['cell_id'], data['issue_date'], scores)
    result = summarize(data, scores)
    metric = result['summary']
    assert ranks.dtype == np.int32
    assert selected.dtype == np.bool_
    assert metric['rows'] == 24
    assert metric['days'] == 2
    assert metric['highlights'] == selected.sum() == 20
    assert metric['tp'] == 10
    assert metric['fp'] == 10
    assert metric['fn'] == 4
    assert metric['positives'] == 14
    assert metric['precision'] == metric['tp'] / metric['highlights'] == .5
    assert metric['oracle_tp'] == 12
    assert metric['oracle_precision'] == .6
    assert metric['oracle_tp'] >= metric['tp']
    assert metric['oracle_macro_recall'] == pytest.approx((10 / 12 + 1) / 2)
    assert metric['precision_gap_to_80'] == pytest.approx(.3)
    assert metric['top_k'] == 10
    assert metric['precision_lift'] == pytest.approx(.5 / (14 / 24))
    for day in result['daily']:
        assert day['highlights'] == 10
        assert day['oracle_tp'] == min(10, day['positives'])
        assert day['oracle_fp'] == 10 - day['oracle_tp']
        assert day['oracle_fn'] == day['positives'] - day['oracle_tp']
    json.dumps(result, allow_nan=False)


def test_exact_ties_are_cell_ordered_with_one_based_ranks():
    cells = np.array(['c', 'a', 'b', 'b', 'c', 'a'])
    dates = np.array([date(2023, 1, 1)] * 3 + [date(2023, 1, 2)] * 3)
    ranks, selected = ranks_and_selection(cells, dates, np.ones(6), k=2)
    np.testing.assert_array_equal(ranks, [3, 1, 2, 2, 3, 1])
    np.testing.assert_array_equal(selected, [False, True, True, True, False, True])


def test_macro_micro_and_monthly_use_identical_definitions():
    data = _data([[1, 0, 0, 0], [1, 1, 1, 1]])
    scores = np.tile([4, 3, 2, 1], 2)
    result = summarize(data, scores, k=2)
    metric = result['summary']
    assert metric['macro_recall'] == .75
    assert metric['micro_recall'] == .6
    assert metric['precision'] == .75
    assert metric['oracle_macro_recall'] == .75
    for month, indices in [('2023-01', slice(0, 4)), ('2023-02', slice(4, 8))]:
        subset = {key: value[indices] for key, value in data.items()}
        assert result['monthly'][month] == summarize(subset, scores[indices], k=2)['summary']


def test_zero_positive_days_keep_their_entire_precision_budget():
    data = _data([[1, 0, 0], [0, 0, 0]])
    result = summarize(data, np.tile([3, 2, 1], 2), k=2)
    metric = result['summary']
    assert metric['zero_positive_days'] == 1
    assert metric['highlights'] == 4
    assert metric['precision'] == .25
    assert metric['macro_recall'] == metric['micro_recall'] == 1
    assert result['daily'][1]['precision'] == 0
    assert result['daily'][1]['recall'] is None
    assert result['monthly']['2023-02']['highlights'] == 2
    assert result['monthly']['2023-02']['precision'] == 0


def test_all_zero_labels_have_null_recall_ap_and_lift():
    result = summarize(_data([[0, 0, 0], [0, 0, 0]]), np.ones(6), k=2)
    for metric in [result['summary'], *result['monthly'].values()]:
        assert metric['precision'] == metric['prevalence'] == 0
        assert metric['oracle_precision'] == metric['oracle_tp'] == 0
        for key in ['macro_recall', 'micro_recall', 'average_precision',
                    'precision_lift', 'oracle_macro_recall']:
            assert metric[key] is None
    json.dumps(result, allow_nan=False)


def test_average_precision_uses_score_threshold_ties_not_cell_ties():
    data = _data([[1, 0, 1], [0, 0, 0]])
    result = summarize(data, np.ones(6), k=2)
    assert result['summary']['average_precision'] == pytest.approx(2 / 6)


def test_shuffled_input_preserves_row_alignment_and_identical_reports():
    data = _data([[1, 0, 1, 0], [0, 1, 0, 1]])
    scores = np.array([1, 1, 0, 2, 3, 1, 1, 2], dtype=float)
    permutation = np.array([7, 1, 4, 2, 6, 0, 5, 3])
    shuffled = {key: value[permutation] for key, value in data.items()}
    ranks, selected = ranks_and_selection(data['cell_id'], data['issue_date'], scores, k=2)
    shuffled_ranks, shuffled_selected = ranks_and_selection(
        shuffled['cell_id'], shuffled['issue_date'], scores[permutation], k=2)
    np.testing.assert_array_equal(shuffled_ranks, ranks[permutation])
    np.testing.assert_array_equal(shuffled_selected, selected[permutation])
    assert summarize(shuffled, scores[permutation], k=2) == summarize(data, scores, k=2)


def test_duplicate_keys_rejected_even_with_different_scores():
    with pytest.raises(ValueError, match='duplicate'):
        ranks_and_selection(['a', 'b', 'a'], [date(2023, 1, 1)] * 3, [0, 1, 2], k=2)


def test_insufficient_daily_population_rejected_without_excluding_days():
    with pytest.raises(ValueError, match='at least k'):
        ranks_and_selection(['a', 'b', 'a'], ['2023-01-01', '2023-01-01', '2023-01-02'],
                            [1, 2, 3], k=2)


@pytest.mark.parametrize('invalid', [None, np.nan, np.inf, -1, 2, .5, '1'])
def test_null_and_nonbinary_labels_are_rejected(invalid):
    data = _data([[1, 0, 1], [0, 0, 0]])
    data['label'] = data['label'].astype(object)
    data['label'][0] = invalid
    with pytest.raises(ValueError, match='binary'):
        summarize(data, np.ones(6), k=2)


def test_masked_labels_are_not_silently_unmasked():
    data = _data([[1, 0, 1], [0, 0, 0]])
    data['label'] = np.ma.array(data['label'], mask=[True, False, False, False, False, False])
    with pytest.raises(ValueError, match='null'):
        summarize(data, np.ones(6), k=2)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf, None])
def test_nonfinite_scores_rejected(invalid):
    with pytest.raises(ValueError, match='finite'):
        ranks_and_selection(['a', 'b'], ['2023-01-01'] * 2, [1, invalid], k=2)


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_invalid_budget_rejected(k):
    with pytest.raises(ValueError, match='positive integer'):
        ranks_and_selection(['a'], ['2023-01-01'], [1], k=k)


def test_empty_and_mismatched_inputs_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        ranks_and_selection([], [], [], k=1)
    with pytest.raises(ValueError, match='equal lengths'):
        ranks_and_selection(['a'], ['2023-01-01'], [1, 2], k=1)
    data = _data([[1, 0, 1], [0, 0, 0]])
    data['label'] = np.array([1])
    with pytest.raises(ValueError, match='same length'):
        summarize(data, np.ones(6), k=2)


def test_nat_dates_rejected():
    with pytest.raises(ValueError, match='null dates'):
        ranks_and_selection(['a'], [np.datetime64('NaT', 'D')], [1], k=1)
