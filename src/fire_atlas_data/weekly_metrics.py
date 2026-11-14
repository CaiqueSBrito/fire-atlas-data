"""Fixed-budget weekly ranking metrics over every supplied cell and issue day.

Average precision is pooled across rows, as in v1, and treats score ties as a
single threshold. Recall excludes zero-positive days only from its macro mean;
precision always includes their full highlight budget. Oracle fields describe
label-informed ceilings, never predictive performance. No independent-window
uncertainty assumption is made for the overlapping seven-day targets.
"""

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score


def _vector(values, name):
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(f'{name} contains null values')
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return result


def _prepare(cells, dates, scores, k):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    cells = _vector(cells, 'cell_id')
    dates = _vector(dates, 'issue_date')
    scores = _vector(scores, 'scores')
    if not len(cells) or len(cells) != len(dates) or len(cells) != len(scores):
        raise ValueError('inputs must be nonempty and have equal lengths')
    if any(not isinstance(cell, (str, np.str_)) or not cell for cell in cells):
        raise ValueError('cell_id must contain nonempty strings')
    if dates.dtype.kind not in ('M', 'O', 'U', 'S'):
        raise ValueError('issue_date must contain dates')
    try:
        dates = dates.astype('datetime64[D]')
        scores = scores.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid issue_date or scores') from exc
    if np.isnat(dates).any():
        raise ValueError('issue_date contains null dates')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    key_order = np.lexsort((cells, dates))
    ordered_dates = dates[key_order]
    same_day = ordered_dates[1:] == ordered_dates[:-1]
    if np.any(same_day & (cells[key_order[1:]] == cells[key_order[:-1]])):
        raise ValueError('duplicate (cell_id, issue_date) key')
    starts = np.r_[0, np.flatnonzero(~same_day) + 1]
    counts = np.diff(np.r_[starts, len(cells)])
    if np.any(counts < k):
        raise ValueError('every issue_date must have at least k cells')
    if np.any(counts > np.iinfo(np.int32).max):
        raise ValueError('daily rank exceeds int32 capacity')
    order = np.lexsort((cells, -scores, dates))
    rank = np.empty(len(cells), dtype=np.int32)
    rank[order] = np.arange(len(cells)) - np.repeat(starts, counts) + 1
    return dates, scores, order, starts, counts, rank



def _aggregate(labels, scores, daily, k):
    positives = int(labels.sum())
    tp = sum(day['tp'] for day in daily)
    highlights = sum(day['highlights'] for day in daily)
    oracle_tp = sum(day['oracle_tp'] for day in daily)
    recalls = [day['recall'] for day in daily if day['positives']]
    oracle_recalls = [day['oracle_recall'] for day in daily if day['positives']]
    precision = tp / highlights
    prevalence = positives / len(labels)
    return dict(
        rows=len(labels), days=len(daily), positives=positives,
        tp=tp, fp=highlights - tp, fn=positives - tp, highlights=highlights,
        zero_positive_days=sum(day['positives'] == 0 for day in daily),
        macro_recall=float(np.mean(recalls)) if recalls else None,
        micro_recall=tp / positives if positives else None,
        precision=precision,
        average_precision=float(average_precision_score(labels, scores)) if positives else None,
        prevalence=prevalence, precision_lift=precision / prevalence if prevalence else None,
        oracle_tp=oracle_tp, oracle_precision=oracle_tp / highlights,
        oracle_macro_recall=float(np.mean(oracle_recalls)) if oracle_recalls else None,
        precision_gap_to_80=0.8 - precision, top_k=int(k),
    )


def ranks_and_selection(cells, dates, scores, k=10):
    """Return one-based int32 ranks and bool top-k flags in input row order.

    Rank each calendar issue day by descending score then ascending cell_id.
    Reject duplicate keys and insufficient daily populations rather than
    silently changing the fixed daily highlight budget.
    """
    *_, rank = _prepare(cells, dates, scores, k)
    return rank, rank <= k



def summarize(data: Mapping, scores, k=10):
    """Return JSON-safe overall, daily, and YYYY-MM metrics without dropping rows.

    Required columns are cell_id, issue_date, and binary nonnull label; other
    columns are ignored. precision_gap_to_80 is signed (0.8 minus precision),
    so negative values mean the descriptive precision exceeds the target.
    """
    dates, scores, order, starts, counts, rank = _prepare(
        data['cell_id'], data['issue_date'], scores, k,
    )
    labels = _vector(data['label'], 'label')
    if len(labels) != len(dates):
        raise ValueError('label must have the same length as issue_date')
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError('label must contain only nonnull binary 0/1 values')
    labels = labels.astype(np.int8)
    selected = rank <= k
    positive_counts = np.add.reduceat(labels[order], starts, dtype=np.int64)
    hit_counts = np.add.reduceat((labels * selected)[order], starts, dtype=np.int64)
    daily = []
    for start, count, positives, tp in zip(starts, counts, positive_counts, hit_counts):
        positives, tp, count = int(positives), int(tp), int(count)
        oracle_tp = min(int(k), positives)
        daily.append(dict(
            date=str(dates[order[start]]), rows=count, days=1,
            positives=positives, tp=tp, fp=int(k) - tp, fn=positives - tp,
            highlights=int(k), zero_positive_days=int(positives == 0),
            precision=tp / int(k), recall=tp / positives if positives else None,
            oracle_tp=oracle_tp, oracle_fp=int(k) - oracle_tp,
            oracle_fn=positives - oracle_tp, oracle_precision=oracle_tp / int(k),
            oracle_recall=oracle_tp / positives if positives else None, top_k=int(k),
        ))
    monthly = {}
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        month_name = str(month)
        mask = months == month
        month_daily = [day for day in daily if day['date'][:7] == month_name]
        monthly[month_name] = _aggregate(labels[mask], scores[mask], month_daily, k)
    return dict(summary=_aggregate(labels, scores, daily, k), daily=daily, monthly=monthly)
