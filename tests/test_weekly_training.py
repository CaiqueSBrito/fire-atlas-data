"""Consumer-visible selection safety and temporal outcome boundaries."""
import numpy as np
import pytest

from fire_atlas_data.weekly_training import FOLDS, _check_partition, _daily_weights, _select


def _validation(candidates):
    return {fold: {name: dict(summary=dict(precision=precision[index], macro_recall=recall[index]))
                   for name, (precision, recall) in candidates.items()}
            for index, fold in enumerate(FOLDS)}


def _partition(issue, last_target):
    return dict(label=np.array([0, 1]), issue_date=np.array([issue, issue], dtype='datetime64[D]'),
                target_end_date=np.array([last_target, last_target], dtype='datetime64[D]'))


def test_selection_requires_improvement_without_hiding_a_bad_fold_or_recall_loss():
    validation = _validation(dict(
        original=([.3, .3, .3], [.2, .2, .2]),
        one_bad_year=([.29, .8, .8], [.4, .4, .4]),
        lost_recall=([.8, .8, .8], [.19, .19, .19]),
        safe=([.4, .4, .4], [.21, .21, .21]),
    ))
    decision = _select(validation)
    assert decision['selected_model'] == 'safe'
    assert not decision['candidate_gates']['one_bad_year']['no_fold_precision_regression']
    assert not decision['candidate_gates']['lost_recall']['no_mean_recall_regression']


def test_selection_ties_keep_original_and_no_qualifying_candidate_keeps_original():
    for precision, recall in [([.3] * 3, [.5] * 3), ([.4] * 3, [.1] * 3)]:
        validation = _validation(dict(original=([.3] * 3, [.2] * 3), alternative=(precision, recall)))
        assert _select(validation)['selected_model'] == 'original'


def test_selection_competing_improvements_tie_in_declared_order():
    validation = _validation(dict(original=([.3] * 3, [.2] * 3),
        first=([.4] * 3, [.2] * 3), second=([.4] * 3, [.3] * 3)))
    assert _select(validation)['selected_model'] == 'first'


def test_undefined_validation_recall_blocks_selection():
    validation = _validation(dict(original=([.3] * 3, [.2] * 3)))
    validation['2022']['original']['summary']['macro_recall'] = None
    with pytest.raises(ValueError, match='Positive days'):
        _select(validation)


def test_cross_boundary_outcomes_rejected_including_first_evaluation_issue_day():
    evaluation = _partition('2024-01-01', '2024-01-08')
    for last_target in ('2024-01-01', '2024-01-02'):
        with pytest.raises(ValueError, match='cutoff'):
            _check_partition(_partition('2023-12-25', last_target), evaluation)
    _check_partition(_partition('2023-12-24', '2023-12-31'), evaluation)


def test_day_weights_keep_zero_positive_days_and_all_controls():
    data = dict(issue_date=np.repeat(np.array(['2022-01-01', '2022-01-02', '2022-01-03'],
                                             dtype='datetime64[D]'), 3),
                label=np.array([0, 0, 0, 1, 0, 0, 1, 1, 0]))
    weights = _daily_weights(data)
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1)
    np.testing.assert_allclose(weights[:6], 1.2)
    np.testing.assert_allclose(weights[6:], .6)
