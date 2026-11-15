"""Cold-start experiment for RF10/CA08 (retrospective diagnostic, not a product table).

Question: for H3 res6 land cells with no detection before D (absent from the ml_dataset universe), do the
existing neighbourhood inputs (ring1/ring2 counts of windows D-7..D-1) rank the first-ever detection in
[D cutoff, D+7 cutoff)?

Population of D: land cells (centroid inside a Natural Earth admin-0 polygon) with no detection in any
window before D and at least one detection in their ring 1..2 during D-7..D-1. Cold cells without recent
neighbour activity are outside the population; `cold_positive_all` vs `cold_positive_in_population` per
month measures what that leaves out.

Rows use the ml_dataset schema, windows, feature SQL and target/coverage rule, so `ml-train` and the
existing model run on them unchanged; own-cell inputs are zero by construction. Negatives are drawn by
md5 at a split-specific rate with weight 1/rate (test included, to bound size); positives are kept whole.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile
import time

from deltalake import CommitProperties, DeltaTable, write_deltalake
import xgboost as xgb

from . import ml_dataset, ml_training
from .places import _countries_sql, _local_copy
from .silver import _literal, _sha256

METHOD_VERSION = "0.1.0"
RING = 2
SAMPLE_SEED = "rf10-coldstart"
LIMITATIONS = [
    "Retrospective diagnostic: says nothing about cells without neighbour activity in D-7..D-1 "
    "(see cold_positive_all vs cold_positive_in_population).",
    "'No history' means no detection of any hotspot type in silver since 2000-11 before D; MODIS-only years "
    "make early cells look colder than they would under VIIRS.",
    "Land = cell centroid inside a Natural Earth 1:10m admin-0 polygon; coastal cells with a sea centroid are out.",
    "Negatives need a pass of the res4 parent in all 7 windows (pass, not clear view); the rest is unknown and dropped.",
    "Negatives are md5-sampled in every split (weight 1/rate); metrics are weighted, not exhaustive.",
    "First-ever detections include industrial/static heat sources not flagged in NRT data.",
    "Same inputs and latency assumptions as ml_dataset (fixed cutoff margin, sensors not harmonized).",
]


def _candidates_sql(cutoff_hour: int) -> str:
    """Table `cand`: land cells within RING of any universe cell, with their first detection window of any
    hotspot type (or NULL). The universe ignores type 3 (offshore), but a cell with such a detection before D
    has own inputs and is not cold."""
    return f"""
        CREATE OR REPLACE TEMP TABLE cand AS
        WITH cells AS (SELECT DISTINCT unnest(h3_grid_disk(c6::UBIGINT, {RING}))::BIGINT AS c6 FROM universe),
             land AS (
                 SELECT DISTINCT c.c6 FROM cells c
                 JOIN countries k ON ST_Intersects(k.geom, ST_Point(h3_cell_to_lng(c.c6::UBIGINT),
                                                                     h3_cell_to_lat(c.c6::UBIGINT)))),
             offshore AS (
                 SELECT h3_r6 AS c6, min((acquired_at - INTERVAL {cutoff_hour} HOUR)::DATE) AS first_window
                 FROM silver WHERE hotspot_type = 3 GROUP BY 1)
        SELECT l.c6, h3_cell_to_parent(l.c6::UBIGINT, 4)::BIGINT AS c4,
               h3_cell_to_parent(l.c6::UBIGINT, 1)::BIGINT AS c1, least(u.first_window, o.first_window) AS first_window
        FROM land l LEFT JOIN universe u USING (c6) LEFT JOIN offshore o USING (c6)"""


def _month_sql(first: date, last: date, cutoff_hour: int) -> str:
    """Temp table `classified`: one row per population cell and D in [first, last] (target may be NULL)."""
    window_lo = first - timedelta(days=ml_dataset.INPUT_DAYS)
    window_hi = last + timedelta(days=ml_dataset.TARGET_DAYS - 1)
    return ml_dataset._det_sql(window_lo, window_hi, cutoff_hour) + f"""
        CREATE OR REPLACE TEMP TABLE classified AS
        WITH {ml_dataset._feature_ctes(first, last)}, {ml_dataset._target_ctes(first, last)}
        SELECT ring.c6, cand.c1, ring.d, {ml_dataset.FEATURE_COLUMNS_SQL}, (own.c6 IS NOT NULL) AS own_input,
               CASE WHEN positive.c6 IS NOT NULL THEN 1 WHEN covered.c4 IS NOT NULL THEN 0 END::TINYINT AS target
        FROM ring
        JOIN cand ON cand.c6 = ring.c6 AND (cand.first_window IS NULL OR cand.first_window >= ring.d)
        LEFT JOIN own ON own.c6 = ring.c6 AND own.d = ring.d
        LEFT JOIN positive ON positive.c6 = ring.c6 AND positive.d = ring.d
        LEFT JOIN covered ON covered.c4 = cand.c4 AND covered.d = ring.d;
    """


def _rows_sql(year: int, month: int, eval_start: date, sample_rate: float, test_sample_rate: float) -> str:
    """Labeled rows in the ml_dataset schema; negatives md5-sampled per split with weight 1/rate."""
    is_eval = f"d >= DATE '{eval_start.isoformat()}'"
    rate = f"CASE WHEN {is_eval} THEN {test_sample_rate} ELSE {sample_rate} END"
    drawn = (f"(md5_number(concat(c6::VARCHAR, '|', d::VARCHAR, '|', {_literal(SAMPLE_SEED)})) % 1000000)"
             f" < round({rate} * 1000000)")
    return f"""
        SELECT d AS forecast_date, CASE WHEN {is_eval} THEN 'eval' ELSE 'train' END AS split,
               'cell' AS row_kind, c6 AS h3_r6, c1 AS h3_r1, target,
               CASE WHEN target = 1 THEN 1.0 ELSE 1.0 / {rate} END AS weight, 1::BIGINT AS cells,
               {", ".join(ml_dataset.FEATURES)}, {year}::INTEGER AS year, {month}::INTEGER AS month
        FROM classified WHERE target = 1 OR (target = 0 AND {drawn})"""


def _month_counts(connection, first: date, last: date) -> dict:
    """Population sizes and the share of all cold first detections the population contains."""
    rows, positive, negative, unknown, own_input = connection.execute("""
        SELECT count(*), count(*) FILTER (WHERE target = 1), count(*) FILTER (WHERE target = 0),
               count(*) FILTER (WHERE target IS NULL), count(*) FILTER (WHERE own_input) FROM classified""").fetchall()[0]
    if own_input:
        raise RuntimeError(f"{own_input} population rows have own-cell inputs: cold-cell filter is wrong")
    cold_all = connection.execute(f"""
        SELECT count(*) FROM cand, range(0, {ml_dataset.TARGET_DAYS}) t(k)
        WHERE first_window - k::INTEGER BETWEEN DATE '{first.isoformat()}' AND DATE '{last.isoformat()}'
    """).fetchall()[0][0]
    return {"population_cell_days": rows, "positive": positive, "negative": negative, "unknown": unknown,
            "cold_positive_all": cold_all, "cold_positive_in_population": positive}


def build(silver: str, coverage: str, countries: str | Path, target: str, *, first_month: tuple[int, int] = (2012, 2),
          months: list[tuple[int, int]] | None = None, eval_start: date = date(2023, 1, 1),
          sample_rate: float = 0.05, test_sample_rate: float = 0.2, cutoff_hour: int = 9,
          storage_options: dict[str, str] | None = None, memory_limit: str = "48GB", threads: int = 16,
          temp_dir: str | Path | None = None) -> dict:
    """Write one Delta commit per forecast month; months already committed are skipped (resumable)."""
    if not (0 < sample_rate <= 1 and 0 < test_sample_rate <= 1):
        raise ValueError("sample rates must be in (0, 1]")
    silver_version = DeltaTable(silver, storage_options=storage_options).version()
    coverage_version = DeltaTable(coverage, storage_options=storage_options).version()
    counts = {}
    with tempfile.TemporaryDirectory(dir=temp_dir) as scratch, \
            ml_dataset._connect(silver, coverage, memory_limit, threads, temp_dir, storage_options) as connection:
        countries_path = _local_copy(countries, Path(scratch))
        countries_sha = _sha256(countries_path)
        connection.execute("INSTALL spatial")
        connection.execute("LOAD spatial")
        connection.execute(_countries_sql(countries_path))
        tic = time.monotonic()
        connection.execute(ml_dataset._universe_sql(cutoff_hour))
        bounds = ml_dataset._forecast_bounds(connection, cutoff_hour)
        connection.execute(_candidates_sql(cutoff_hour))
        n_cand, n_cold = connection.execute(
            "SELECT count(*), count(*) FILTER (WHERE first_window IS NULL) FROM cand").fetchall()[0]
        print(f"coldstart candidates: {n_cand} land cells ({n_cold} never detected), "
              f"{time.monotonic() - tic:.0f}s", file=sys.stderr, flush=True)
        if months is None:
            written = ml_dataset._written_months(target, storage_options)
            months, current = [], date(*first_month, 1)
            while current <= bounds[1]:
                if (current.year, current.month) not in written:
                    months.append((current.year, current.month))
                current = (current + timedelta(days=32)).replace(day=1)
        for year, month in months:
            clamped = ml_dataset._clamp(year, month, bounds)
            if clamped is None:
                continue
            tic = time.monotonic()
            connection.execute(_month_sql(*clamped, cutoff_hour))
            month_counts = _month_counts(connection, *clamped)
            reader = connection.sql(_rows_sql(year, month, eval_start, sample_rate, test_sample_rate)) \
                .to_arrow_reader(1_000_000)
            exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
            key = f"{year}-{month:02d}"
            metadata = {"method_version": METHOD_VERSION, "dataset_method_version": ml_dataset.METHOD_VERSION,
                        "silver_version": str(silver_version), "coverage_version": str(coverage_version),
                        "countries_sha256": countries_sha, "cutoff_hour": str(cutoff_hour), "ring": str(RING),
                        "eval_start": eval_start.isoformat(), "sample_rate": str(sample_rate),
                        "test_sample_rate": str(test_sample_rate), "sample_seed": SAMPLE_SEED,
                        "forecast_month": key, "counts": json.dumps(month_counts)}
            write_deltalake(target, reader, mode="overwrite", partition_by=ml_dataset.PARTITIONS,
                            predicate=f"year = {year} AND month = {month}" if exists else None,
                            storage_options=storage_options, commit_properties=CommitProperties(custom_metadata=metadata))
            counts[key] = month_counts
            print(f"coldstart {key}: {json.dumps(month_counts)} ({time.monotonic() - tic:.0f}s)",
                  file=sys.stderr, flush=True)
    return {"method_version": METHOD_VERSION, "target": target, "silver_version": silver_version,
            "coverage_version": coverage_version, "candidates": n_cand, "never_detected_candidates": n_cold,
            "counts": counts, "limitations": LIMITATIONS}


def summarize(target: str, storage_options: dict[str, str] | None = None) -> dict:
    """Totals per split of the month counts recorded in the commits (latest commit per month)."""
    eval_start, per_month = None, {}
    for commit in reversed(DeltaTable(target, storage_options=storage_options).history()):
        if "forecast_month" in commit:
            per_month[commit["forecast_month"]] = json.loads(commit["counts"])
            eval_start = commit["eval_start"]
    totals: dict[str, dict] = {}
    for key, values in per_month.items():
        split = "eval" if f"{key}-01" >= eval_start else "train"  # eval_start is a month start (default)
        bucket = totals.setdefault(split, {})
        for name, value in values.items():
            bucket[name] = bucket.get(name, 0) + value
    for bucket in totals.values():
        bucket["population_recall_ceiling"] = (bucket["cold_positive_in_population"] / bucket["cold_positive_all"]
                                               if bucket["cold_positive_all"] else None)
    return {"months": len(per_month), "eval_start": eval_start, "totals": totals}


def evaluate_model(dataset: str, model: str, output: str | Path, *, train_start: date = date(2012, 2, 1),
                   train_end: date = date(2020, 12, 17), test_start: date = date(2023, 1, 1),
                   test_end: date | None = None, memory_limit: str = "48GB", threads: int = 16) -> dict:
    """Score the test period with an already trained model (e.g. the warm-cell xgb-rf10), plus the
    ml-train baselines (persistence of neighbour activity; res1 x month climatology from train rows)."""
    connection = ml_training._connect(dataset, memory_limit, threads)
    if test_end is None:
        test_end = connection.execute("SELECT max(forecast_date) FROM gold").fetchall()[0][0]
    booster = xgb.Booster()
    booster.load_model(model)
    connection.execute(ml_training._climatology_sql(train_start, train_end))
    test = ml_training._evaluate(connection, booster, ml_training._table_months(dataset, test_start, test_end),
                                 test_start, test_end)
    report = {"method_version": METHOD_VERSION, "dataset": dataset,
              "dataset_version": DeltaTable(dataset).version(), "model": model, "model_sha256": _sha256(Path(model)),
              "test": [test_start.isoformat(), test_end.isoformat()],
              "climatology_train": [train_start.isoformat(), train_end.isoformat()],
              "result": test, "population": summarize(dataset), "limitations": LIMITATIONS,
              "counts": {"test_rows": test["rows"],
                         "average_precision": test["metrics"]["model"]["average_precision"]}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=1))
    return report
