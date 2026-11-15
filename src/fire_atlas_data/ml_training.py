"""XGBoost model on gold/ml_dataset: P(any detection in [D cutoff, D+7 cutoff)) per H3 res6 cell (RF10/CA08).

Chronological protocol: train < embargo < valid < embargo < test (the dataset `eval` split). Hyperparameters
are chosen by weighted PR-AUC on valid with early stopping; test is scored once, for the chosen model and
two baselines (persistence of recent activity; regional x month-of-year climatology from the train period).
Only labeled rows are used: `cell` and `quiet` rows with a known target and positive weight. `quiet` rows
stand for many all-zero-input cells through their weight, and train negatives carry the 1/sample_rate weight,
so every metric below is weighted.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys
import time

import duckdb
from deltalake import DeltaTable
import numpy as np
import xgboost as xgb

from .ml_dataset import FEATURES, METHOD_VERSION as DATASET_METHOD_VERSION, TARGET_DAYS

METHOD_VERSION = "1.0.0"
SCORE_BINS = 1 << 20
YEAR_SCORE_BINS = 1 << 16
CALIBRATION_BINS = 10
TOP_K = (100, 1_000, 10_000)
MAX_BIN = 256
SAMPLE_SEED = "rf10-ca08-train"
LIMITATIONS = [
    "Valid rows come from the sampled train split (active negatives at sample_rate, weighted 1/rate); "
    "only test is unsampled.",
    "Average precision is computed over score bins; ties inside a bin count as one threshold.",
    "Persistence is a ranking baseline, not a probability: no Brier, log loss or calibration for it.",
    "Climatology is regional (H3 res1) x month of year, because quiet negatives are aggregated per res1.",
    "No acceptance thresholds were approved: metrics are reported, not judged.",
    "Train and valid keep a deterministic md5 share (row_sample_rate) of `cell` rows, weighted 1/rate, to fit "
    "GPU memory; test is complete.",
    "Dataset limitations apply (pass is not a clear view; never-burned cells excluded; sensors not harmonized).",
]


class _MonthBatches(xgb.DataIter):
    """Feeds labeled rows month by month. XGBoost iterates several times while building the matrix, so
    each month's sampled arrays are kept after the first query (a few GB at the default sample rate)."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, months: list[tuple[int, int]],
                 first: date, last: date, sample_rate: float = 1.0) -> None:
        self._connection, self._months, self._first, self._last = connection, months, first, last
        self._sample_rate = sample_rate
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._index = 0
        super().__init__()

    def next(self, input_data) -> bool:
        while self._index < len(self._months):
            if self._index not in self._cache:
                year, month = self._months[self._index]
                batch = _load(self._connection, year, month, self._first, self._last, sample_rate=self._sample_rate)
                self._cache[self._index] = {k: batch[k] for k in ("x", "target", "weight")}
            batch = self._cache[self._index]
            self._index += 1
            if len(batch["target"]):
                input_data(data=batch["x"], label=batch["target"], weight=batch["weight"])
                return True
        return False

    def reset(self) -> None:
        self._index = 0


def _connect(dataset: str, memory_limit: str, threads: int) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute(f"SET threads={threads}")
    connection.execute("SET enable_progress_bar=false")
    connection.execute("SET TimeZone='UTC'")
    connection.register("gold", DeltaTable(dataset).to_pyarrow_dataset())
    return connection


def _table_months(dataset: str, first: date, last: date) -> list[tuple[int, int]]:
    present = {(int(p["year"]), int(p["month"])) for p in DeltaTable(dataset).partitions()}
    return sorted(m for m in present if (first.year, first.month) <= m <= (last.year, last.month))


def _load(connection: duckdb.DuckDBPyConnection, year: int, month: int, first: date, last: date,
          extra: str = "", sample_rate: float = 1.0) -> dict[str, np.ndarray]:
    """Labeled rows of one month within [first, last]; NULL feature means become NaN (missing).

    sample_rate < 1 keeps a deterministic md5 share of `cell` rows with weight / rate (unbiased weighted
    totals); aggregated `quiet` rows are always kept whole.
    """
    columns = ", ".join(f"coalesce({f}::DOUBLE, 'NaN'::DOUBLE) AS {f}" for f in FEATURES)
    if sample_rate < 1:
        drawn = (f"(md5_number(concat(h3_r6::VARCHAR, '|', forecast_date::VARCHAR, '|', '{SAMPLE_SEED}'))"
                 f" % 1000000) < {round(sample_rate * 1_000_000)}")
        sampled = f"AND (row_kind = 'quiet' OR {drawn})"
        weight = f"CASE WHEN row_kind = 'quiet' THEN weight ELSE weight / {sample_rate} END"
    else:
        sampled, weight = "", "weight"
    arrays = connection.execute(f"""
        SELECT {columns}, target::FLOAT AS target, ({weight})::FLOAT AS weight{extra}
        FROM gold
        WHERE year = {year} AND month = {month}
          AND forecast_date BETWEEN DATE '{first.isoformat()}' AND DATE '{last.isoformat()}'
          AND row_kind IN ('cell', 'quiet') AND target IS NOT NULL AND weight > 0 {sampled}
    """).fetchnumpy()
    arrays["x"] = np.column_stack([np.asarray(arrays.pop(f), dtype=np.float32) for f in FEATURES]) \
        if len(arrays["target"]) else np.empty((0, len(FEATURES)), dtype=np.float32)
    arrays["target"] = np.asarray(arrays["target"], dtype=np.float32)
    arrays["weight"] = np.asarray(arrays["weight"], dtype=np.float32)
    return arrays


def _search_configs(count: int, seed: int) -> list[dict]:
    """Seeded random search space; the list itself is recorded in the report."""
    rng = np.random.default_rng(seed)
    return [{"max_depth": int(rng.choice([4, 6, 8, 10])),
             "learning_rate": round(float(10 ** rng.uniform(np.log10(0.03), np.log10(0.3))), 4),
             "min_child_weight": round(float(10 ** rng.uniform(0, 3)), 2),
             "subsample": round(float(rng.uniform(0.5, 1.0)), 3),
             "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 3),
             "reg_lambda": round(float(10 ** rng.uniform(-1, 2)), 3)} for _ in range(count)]


def _climatology_sql(first: date, last: date) -> str:
    """Weighted target rate per (res1 region, month of year) and per month of year, from train rows only."""
    where = (f"forecast_date BETWEEN DATE '{first.isoformat()}' AND DATE '{last.isoformat()}'"
             " AND row_kind IN ('cell', 'quiet') AND target IS NOT NULL AND weight > 0")
    return f"""
        CREATE OR REPLACE TEMP TABLE clim_region AS
        SELECT h3_r1, month, sum(weight * target) / sum(weight) AS rate FROM gold WHERE {where} GROUP BY ALL;
        CREATE OR REPLACE TEMP TABLE clim_month AS
        SELECT month, sum(weight * target) / sum(weight) AS rate FROM gold WHERE {where} GROUP BY ALL;
    """


def _new_accumulator(probabilistic: bool) -> dict:
    return {"pos": np.zeros(SCORE_BINS), "neg": np.zeros(SCORE_BINS), "years": {},
            "probabilistic": probabilistic, "brier": 0.0, "logloss": 0.0, "weight": 0.0,
            "cal_p": np.zeros(CALIBRATION_BINS), "cal_y": np.zeros(CALIBRATION_BINS),
            "cal_w": np.zeros(CALIBRATION_BINS), "topk": {k: [] for k in TOP_K}}


def _accumulate(acc: dict, score: np.ndarray, target: np.ndarray, weight: np.ndarray,
                day: np.ndarray, year: int) -> None:
    score = np.clip(score.astype(np.float64), 0.0, 1.0)
    positive_w, negative_w = weight * target, weight * (1 - target)
    bins = np.minimum((score * SCORE_BINS).astype(np.int64), SCORE_BINS - 1)
    acc["pos"] += np.bincount(bins, positive_w, SCORE_BINS)
    acc["neg"] += np.bincount(bins, negative_w, SCORE_BINS)
    year_pos, year_neg = acc["years"].setdefault(year, (np.zeros(YEAR_SCORE_BINS), np.zeros(YEAR_SCORE_BINS)))
    year_bins = bins >> (20 - 16)
    year_pos += np.bincount(year_bins, positive_w, YEAR_SCORE_BINS)
    year_neg += np.bincount(year_bins, negative_w, YEAR_SCORE_BINS)
    if acc["probabilistic"]:
        p = np.clip(score, 1e-7, 1 - 1e-7)
        acc["brier"] += float(np.sum(weight * (score - target) ** 2))
        acc["logloss"] -= float(np.sum(weight * (target * np.log(p) + (1 - target) * np.log1p(-p))))
        acc["weight"] += float(weight.sum())
        cal = np.minimum((score * CALIBRATION_BINS).astype(np.int64), CALIBRATION_BINS - 1)
        acc["cal_p"] += np.bincount(cal, weight * score, CALIBRATION_BINS)
        acc["cal_y"] += np.bincount(cal, positive_w, CALIBRATION_BINS)
        acc["cal_w"] += np.bincount(cal, weight, CALIBRATION_BINS)
    order = np.lexsort((-score, day))
    sorted_day, sorted_w, sorted_pos = day[order], weight[order], positive_w[order]
    starts = np.flatnonzero(np.r_[True, sorted_day[1:] != sorted_day[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(order)]):
        cumulative_w, cumulative_pos = np.cumsum(sorted_w[start:end]), np.cumsum(sorted_pos[start:end])
        for k in TOP_K:
            # rows are weighted cells: take whole rows until k cells, then a fraction of the crossing row
            i = int(np.searchsorted(cumulative_w, k))
            if i >= len(cumulative_w):
                acc["topk"][k].append(float(cumulative_pos[-1] / cumulative_w[-1]))
                continue
            before_w = cumulative_w[i - 1] if i else 0.0
            before_pos = cumulative_pos[i - 1] if i else 0.0
            fraction = (k - before_w) / sorted_w[start + i]
            acc["topk"][k].append(float((before_pos + fraction * sorted_pos[start + i]) / k))


def _average_precision(pos: np.ndarray, neg: np.ndarray) -> float | None:
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    total = tp[-1]
    if total <= 0:
        return None
    keep = (tp + fp) > 0
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=keep)
    recall = tp / total
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _summarize(acc: dict) -> dict:
    total_pos, total_neg = float(acc["pos"].sum()), float(acc["neg"].sum())
    summary = {"average_precision": _average_precision(acc["pos"], acc["neg"]),
               "base_rate": total_pos / (total_pos + total_neg) if total_pos + total_neg else None,
               "weighted_positives": total_pos, "weighted_negatives": total_neg,
               "by_year": {str(y): {"average_precision": _average_precision(p, n),
                                    "base_rate": float(p.sum() / (p.sum() + n.sum()))}
                           for y, (p, n) in sorted(acc["years"].items())},
               "precision_at_k_daily_mean": {str(k): float(np.mean(v)) if v else None
                                             for k, v in acc["topk"].items()}}
    if acc["probabilistic"]:
        summary["brier"] = acc["brier"] / acc["weight"]
        summary["logloss"] = acc["logloss"] / acc["weight"]
        summary["calibration"] = [
            {"bin": f"[{i / CALIBRATION_BINS:.1f}, {(i + 1) / CALIBRATION_BINS:.1f})", "weight": float(w),
             "mean_predicted": float(p / w) if w else None, "observed_rate": float(y / w) if w else None}
            for i, (p, y, w) in enumerate(zip(acc["cal_p"], acc["cal_y"], acc["cal_w"]))]
    return summary


def _evaluate(connection: duckdb.DuckDBPyConnection, booster: xgb.Booster, months: list[tuple[int, int]],
              first: date, last: date) -> dict:
    """Score test once for the model and both baselines, streaming month by month."""
    accumulators = {"model": _new_accumulator(True), "climatology": _new_accumulator(True),
                    "persistence": _new_accumulator(False)}
    extra = (", (forecast_date - DATE '1970-01-01')::INTEGER AS day,"
             " coalesce((SELECT rate FROM clim_region c WHERE c.h3_r1 = gold.h3_r1 AND c.month = gold.month),"
             " (SELECT rate FROM clim_month c WHERE c.month = gold.month), 0.0) AS climatology")
    rows = 0
    for year, month in months:
        batch = _load(connection, year, month, first, last, extra)
        if not len(batch["target"]):
            continue
        rows += len(batch["target"])
        x, target, weight, day = batch["x"], batch["target"], batch["weight"], batch["day"]
        recent = (np.nan_to_num(x[:, FEATURES.index("det_7d")]) + 1e-3 * np.nan_to_num(x[:, FEATURES.index("ring1_det_7d")])
                  + 1e-6 * np.nan_to_num(x[:, FEATURES.index("ring2_det_7d")]))
        scores = {"model": booster.inplace_predict(x), "climatology": np.asarray(batch["climatology"]),
                  "persistence": recent / (1.0 + recent)}
        for name, score in scores.items():
            _accumulate(accumulators[name], np.asarray(score), target, weight, day, year)
        print(f"ml-train test {year}-{month:02d}: {len(target)} rows", file=sys.stderr, flush=True)
    return {"rows": rows, "metrics": {name: _summarize(acc) for name, acc in accumulators.items()}}


def _search_key(dataset: str, splits: dict, seed: int, max_rounds: int, early_stopping: int,
                 row_sample_rate: float, device: str) -> dict:
    """Settings that must match for configs of an earlier run in the same output dir to be reused."""
    return {"method_version": METHOD_VERSION, "dataset": dataset, "dataset_version": DeltaTable(dataset).version(),
            "splits": splits, "seed": seed, "max_rounds": max_rounds, "early_stopping": early_stopping,
            "row_sample_rate": row_sample_rate, "sample_seed": SAMPLE_SEED, "device": device}


def train(dataset: str, output_dir: str | Path, *, train_start: date = date(2012, 2, 1),
          valid_start: date = date(2021, 1, 1), test_start: date = date(2023, 1, 1),
          test_end: date | None = None, embargo_days: int = 14, configs: int = 20, seed: int = 20261114,
          max_rounds: int = 3000, early_stopping: int = 100, row_sample_rate: float = 0.2,
          device: str = "cuda", memory_limit: str = "8GB", threads: int = 8) -> dict:
    """Random search on train/valid, keep the best valid PR-AUC model, then score test once.

    Resumable: configs already in `<output_dir>/search.json` with their model file are reused when
    `search-key.json` matches the current settings; a mismatch refuses to mix runs."""
    if embargo_days < TARGET_DAYS:
        raise ValueError(f"embargo_days must be >= {TARGET_DAYS} (target window length)")
    train_end = valid_start - timedelta(days=embargo_days + 1)
    valid_end = test_start - timedelta(days=embargo_days + 1)
    if not train_start <= train_end < valid_start <= valid_end < test_start:
        raise ValueError("dates must satisfy train_start <= train_end < valid_start <= valid_end < test_start")
    if not 0 < row_sample_rate <= 1:
        raise ValueError("row_sample_rate must be in (0, 1]")
    output = Path(output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    connection = _connect(dataset, memory_limit, threads)
    if test_end is None:
        test_end = connection.execute("SELECT max(forecast_date) FROM gold").fetchone()[0]
    split_months = {"train": _table_months(dataset, train_start, train_end),
                    "valid": _table_months(dataset, valid_start, valid_end),
                    "test": _table_months(dataset, test_start, test_end)}
    splits = {"train": [train_start.isoformat(), train_end.isoformat()],
              "valid": [valid_start.isoformat(), valid_end.isoformat()],
              "test": [test_start.isoformat(), test_end.isoformat()], "embargo_days": embargo_days}
    key = _search_key(dataset, splits, seed, max_rounds, early_stopping, row_sample_rate, device)
    key_path, search_path = output / "search-key.json", output / "search.json"
    previous = {}
    if search_path.exists():
        if not key_path.exists() or json.loads(key_path.read_text()) != key:
            raise ValueError(f"{search_path} belongs to other settings (see {key_path}); use a new output dir")
        previous = {r["name"]: r for r in json.loads(search_path.read_text())}
    key_path.write_text(json.dumps(key, indent=1))
    started = time.monotonic()
    dtrain = xgb.QuantileDMatrix(_MonthBatches(connection, split_months["train"], train_start, train_end,
                                               row_sample_rate), max_bin=MAX_BIN)
    dvalid = xgb.QuantileDMatrix(_MonthBatches(connection, split_months["valid"], valid_start, valid_end,
                                               row_sample_rate), ref=dtrain)
    print(f"ml-train matrices: train {dtrain.num_row()} rows, valid {dvalid.num_row()} rows "
          f"({time.monotonic() - started:.0f}s)", file=sys.stderr, flush=True)
    base = {"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_bin": MAX_BIN,
            "eval_metric": ["logloss", "aucpr"], "seed": seed}  # early stopping uses the last metric
    search, best = [], None
    for index, config in enumerate(_search_configs(configs, seed)):
        name = f"config-{index:02d}"
        model_path = output / "models" / f"{name}.ubj"
        prior = previous.get(name)
        if prior is not None and prior["params"] == config and model_path.exists():
            booster, result = xgb.Booster(model_file=str(model_path)), prior
            print(f"ml-train {name}: reused from the earlier run", file=sys.stderr, flush=True)
        else:
            history: dict = {}
            tic = time.monotonic()
            booster = xgb.train({**base, **config}, dtrain, num_boost_round=max_rounds,
                                evals=[(dvalid, "valid")], early_stopping_rounds=early_stopping,
                                evals_result=history, verbose_eval=False)
            booster = booster[: booster.best_iteration + 1]
            booster.save_model(model_path)
            result = {"name": name, "params": config, "best_iteration": booster.num_boosted_rounds() - 1,
                      "valid_aucpr": history["valid"]["aucpr"][booster.num_boosted_rounds() - 1],
                      "valid_logloss": history["valid"]["logloss"][booster.num_boosted_rounds() - 1],
                      "seconds": round(time.monotonic() - tic, 1)}
            print(f"ml-train {name}: {json.dumps(result)}", file=sys.stderr, flush=True)
        search.append(result)
        search_path.write_text(json.dumps(search, indent=1))
        if best is None or result["valid_aucpr"] > best[0]["valid_aucpr"]:
            best = (result, booster)
    del dtrain, dvalid
    chosen, booster = best
    booster.save_model(output / "model.ubj")
    connection.execute(_climatology_sql(train_start, train_end))
    test = _evaluate(connection, booster, split_months["test"], test_start, test_end)
    report = {"method_version": METHOD_VERSION, "dataset": dataset,
              "dataset_version": DeltaTable(dataset).version(), "dataset_method_version": DATASET_METHOD_VERSION,
              "features": FEATURES, "splits": splits,
              "base_params": base, "max_rounds": max_rounds, "early_stopping_rounds": early_stopping,
              "seed": seed, "row_sample_rate": row_sample_rate, "sample_seed": SAMPLE_SEED,
              "search": search, "reused_configs": sorted(set(previous) & {r["name"] for r in search}),
              "chosen": chosen, "test": test,
              "seconds": round(time.monotonic() - started, 1), "limitations": LIMITATIONS,
              "counts": {"configs": len(search), "test_rows": test["rows"],
                         "chosen": chosen["name"], "valid_aucpr": chosen["valid_aucpr"],
                         "test_average_precision": test["metrics"]["model"]["average_precision"]}}
    (output / "report.json").write_text(json.dumps(report, indent=1))
    return report
