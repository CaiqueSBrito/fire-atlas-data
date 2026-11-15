"""Gold ML dataset: H3 res6 cell x forecast day D, inputs up to the cutoff, 7-day target (RF10/CA08).

Time is cut in 24 h windows starting at `cutoff_hour` UTC; window w spans [w HH:00, w+1 HH:00).
For forecast day D:
- inputs use windows D-7..D-1, i.e. only detections acquired before D HH:00 UTC;
- target uses windows D..D+6, i.e. [D HH:00, D+7 HH:00): 1 if any detection (no confidence
  filter), 0 only when the cell's H3 res4 parent had a recorded satellite pass in each of the
  7 windows, otherwise unknown (NULL).
Universe on D: res6 cells with at least one non-offshore detection (FIRMS type != 3) before
D HH:00 UTC, so membership itself never uses future data.

Rows (`row_kind`):
- `cell`: one cell. All positives; every known negative in the eval split; in the train split,
  "active" negatives (any detection in the cell or its 2-ring during the input windows) are
  kept with a deterministic md5 sample and weight 1/rate.
- `quiet`: per (D, H3 res1 region), the known negatives with no activity in the input windows.
  Their inputs are all zero, so one row with weight = cells represents them exactly.
- `unknown`: per (D, res1), cells whose target is unknown; weight 0, never a label.
- `sampled_out`: per (D, res1), train active negatives not drawn by the sample; weight 0, audit only.
`cells` always counts represented cells, so sum(cells) per D equals the universe size.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
import json
import multiprocessing
from pathlib import Path
import sys

import duckdb
import pyarrow.dataset as pads
from deltalake import CommitProperties, DeltaTable, write_deltalake

from .silver import _literal, load_h3

METHOD_VERSION = "1.0.0"
PARTITIONS = ["year", "month"]
INPUT_DAYS = 7
TARGET_DAYS = 7
LIMITATIONS = [
    "Negatives need a satellite pass, not a clear view: cloud, glint and bad scans are not excluded.",
    "Coverage is evaluated at the H3 res4 parent center; res6 cells inherit it (approximate at swath edges).",
    "Universe excludes cells never burned before D: the model says nothing about first-ever fires.",
    "Inputs ignore FIRMS publication latency beyond the fixed cutoff margin (available_at unknown).",
    "Train rows near the eval start have target windows inside the eval period; apply an embargo when training.",
    "Brightness means are per sensor (MODIS ch21/22, VIIRS I4) and not comparable across sensors.",
    "Sensors are not harmonized; detection counts mix platforms whose number changes over time.",
]
FEATURES = ["det_24h", "det_3d", "det_7d", "frp_mean_7d", "modis_brightness_mean_7d",
            "viirs_brightness_mean_7d", "ring1_det_24h", "ring1_det_7d", "ring2_det_7d"]
# Feature select list over the `own`/`ring` CTEs of `_feature_ctes`; shared by training rows and inference.
FEATURE_COLUMNS_SQL = """
               coalesce(own.det_24h, 0)::INTEGER AS det_24h, coalesce(own.det_3d, 0)::INTEGER AS det_3d,
               coalesce(own.det_7d, 0)::INTEGER AS det_7d, own.frp_mean_7d,
               own.modis_brightness_mean_7d, own.viirs_brightness_mean_7d,
               coalesce(ring.ring1_det_24h, 0)::INTEGER AS ring1_det_24h,
               coalesce(ring.ring1_det_7d, 0)::INTEGER AS ring1_det_7d,
               coalesce(ring.ring2_det_7d, 0)::INTEGER AS ring2_det_7d,
               (own.c6 IS NOT NULL OR ring.c6 IS NOT NULL) AS active"""


def _months_touching(first: date, last: date) -> str:
    """Silver partition filter covering acquisition dates first..last."""
    months, current = [], first.replace(day=1)
    while current <= last:
        months.append(f"(acquisition_year = {current.year} AND acquisition_month = {current.month})")
        current = (current + timedelta(days=32)).replace(day=1)
    return "(" + " OR ".join(months) + ")"


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    return first, (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def _universe_sql(cutoff_hour: int) -> str:
    return f"""
        CREATE OR REPLACE TEMP TABLE universe AS
        SELECT h3_r6 AS c6, any_value(h3_r4) AS c4, any_value(h3_r1) AS c1,
               min((acquired_at - INTERVAL {cutoff_hour} HOUR)::DATE) AS first_window
        FROM silver WHERE hotspot_type IS DISTINCT FROM 3
        GROUP BY 1
    """


def _det_sql(window_lo: date, window_hi: date, cutoff_hour: int) -> str:
    """Temp table `det`: per res6 cell and 24 h window in [window_lo, window_hi], counts and sensor sums."""
    start = f"TIMESTAMPTZ '{window_lo.isoformat()} {cutoff_hour:02d}:00:00+00'"
    end = f"TIMESTAMPTZ '{(window_hi + timedelta(days=1)).isoformat()} {cutoff_hour:02d}:00:00+00'"
    return f"""
        CREATE OR REPLACE TEMP TABLE det AS
        SELECT h3_r6 AS c6, (acquired_at - INTERVAL {cutoff_hour} HOUR)::DATE AS w, count(*) AS n,
               sum(frp_mw) AS frp_sum, count(frp_mw) AS frp_n,
               sum(brightness_k) FILTER (WHERE sensor = 'MODIS') AS mb_sum,
               count(brightness_k) FILTER (WHERE sensor = 'MODIS') AS mb_n,
               sum(brightness_i4_k) FILTER (WHERE sensor = 'VIIRS') AS vb_sum,
               count(brightness_i4_k) FILTER (WHERE sensor = 'VIIRS') AS vb_n
        FROM silver
        WHERE {_months_touching(window_lo, window_hi + timedelta(days=1))}
          AND acquired_at >= {start} AND acquired_at < {end}
        GROUP BY ALL;
    """


def _feature_ctes(first: date, last: date) -> str:
    """CTEs `grid` (universe cell x D), `own` and `ring`: inputs from windows D-7..D-1 of `det` only."""
    d0, d1 = f"DATE '{first.isoformat()}'", f"DATE '{last.isoformat()}'"
    return f"""
        grid AS (  -- per-cell day series: avoids a non-spilling inequality join
            SELECT c6, c4, c1, unnest(generate_series(greatest(first_window + 1, {d0}), {d1}, INTERVAL 1 DAY))::DATE AS d
            FROM universe WHERE first_window + 1 <= {d1}),
        own AS (
            SELECT c6, (w + k::INTEGER) AS d,
                   sum(n) FILTER (WHERE k = 1) AS det_24h, sum(n) FILTER (WHERE k <= 3) AS det_3d,
                   sum(n) AS det_7d, sum(frp_sum) / nullif(sum(frp_n), 0) AS frp_mean_7d,
                   sum(mb_sum) / nullif(sum(mb_n), 0) AS modis_brightness_mean_7d,
                   sum(vb_sum) / nullif(sum(vb_n), 0) AS viirs_brightness_mean_7d
            FROM det, range(1, {INPUT_DAYS + 1}) t(k)
            WHERE w + k::INTEGER BETWEEN {d0} AND {d1}
            GROUP BY 1, 2),
        ring AS (
            SELECT neighbor AS c6, d,
                   sum(n) FILTER (WHERE r = 1 AND k = 1) AS ring1_det_24h,
                   sum(n) FILTER (WHERE r = 1) AS ring1_det_7d,
                   sum(n) FILTER (WHERE r = 2) AS ring2_det_7d
            FROM (SELECT unnest(h3_grid_ring(c6::UBIGINT, r::INTEGER))::BIGINT AS neighbor, r, k,
                         (w + k::INTEGER) AS d, n
                  FROM det, range(1, 3) rr(r), range(1, {INPUT_DAYS + 1}) t(k)
                  WHERE w + k::INTEGER BETWEEN {d0} AND {d1})
            GROUP BY 1, 2)"""


def _target_ctes(first: date, last: date) -> str:
    """CTEs `positive` (any detection in windows D..D+6) and `covered` (res4 pass in all 7 windows) for D in
    [first, last]; need `det` over windows first..last+6 and the `coverage` relation."""
    window_hi = last + timedelta(days=TARGET_DAYS - 1)
    d0, d1 = f"DATE '{first.isoformat()}'", f"DATE '{last.isoformat()}'"
    return f"""
        positive AS (
            SELECT DISTINCT c6, (w - k::INTEGER) AS d FROM det, range(0, {TARGET_DAYS}) t(k)
            WHERE w - k::INTEGER BETWEEN {d0} AND {d1}),
        covered AS (
            SELECT h3_r4 AS c4, (window_date - k::INTEGER) AS d
            FROM coverage, range(0, {TARGET_DAYS}) t(k)
            WHERE year BETWEEN {first.year} AND {window_hi.year}
              AND window_date BETWEEN {d0} AND {d1} + {TARGET_DAYS - 1}
              AND window_date - k::INTEGER BETWEEN {d0} AND {d1}
            GROUP BY 1, 2 HAVING count(*) = {TARGET_DAYS})"""


def _classified_sql(first: date, last: date, cutoff_hour: int) -> str:
    """One row per universe cell and D in [first, last] with inputs, target and coverage."""
    window_lo, window_hi = first - timedelta(days=INPUT_DAYS), last + timedelta(days=TARGET_DAYS - 1)
    return _det_sql(window_lo, window_hi, cutoff_hour) + f"""
        CREATE OR REPLACE TEMP TABLE classified AS
        WITH {_feature_ctes(first, last)}, {_target_ctes(first, last)}
        SELECT g.c6, g.c1, g.d, {FEATURE_COLUMNS_SQL},
               CASE WHEN positive.c6 IS NOT NULL THEN 1 WHEN covered.c4 IS NOT NULL THEN 0 END::TINYINT AS target
        FROM grid g
        LEFT JOIN own ON own.c6 = g.c6 AND own.d = g.d
        LEFT JOIN ring ON ring.c6 = g.c6 AND ring.d = g.d
        LEFT JOIN positive ON positive.c6 = g.c6 AND positive.d = g.d
        LEFT JOIN covered ON covered.c4 = g.c4 AND covered.d = g.d;
    """


def features_sql(issue_date: date, cutoff_hour: int) -> str:
    """Temp table `features`: one row per universe cell for forecast day `issue_date`, inputs only.

    Same windows and expressions as the training rows; no target or coverage, so it needs data only up to
    the `issue_date` cutoff. Requires the temp table `universe` (see `_universe_sql`)."""
    return _det_sql(issue_date - timedelta(days=INPUT_DAYS), issue_date - timedelta(days=1), cutoff_hour) + f"""
        CREATE OR REPLACE TEMP TABLE features AS
        WITH {_feature_ctes(issue_date, issue_date)}
        SELECT g.c6, g.c1, g.d, {FEATURE_COLUMNS_SQL}
        FROM grid g
        LEFT JOIN own ON own.c6 = g.c6 AND own.d = g.d
        LEFT JOIN ring ON ring.c6 = g.c6 AND ring.d = g.d;
    """


def _rows_sql(year: int, month: int, eval_start: date, sample_rate: float, seed: str) -> str:
    split = f"CASE WHEN d >= DATE '{eval_start.isoformat()}' THEN 'eval' ELSE 'train' END"
    drawn = (f"(md5_number(concat(c6::VARCHAR, '|', d::VARCHAR, '|', {_literal(seed)})) % 1000000)"
             f" < {round(sample_rate * 1_000_000)}")
    kind = f"""CASE WHEN target IS NULL THEN 'unknown'
                    WHEN target = 1 THEN 'cell'
                    WHEN NOT active THEN 'quiet'
                    WHEN {split} = 'eval' OR {drawn} THEN 'cell'
                    ELSE 'sampled_out' END"""
    negative_weight = f"CASE WHEN {split} = 'eval' OR target = 1 THEN 1.0 ELSE {1 / sample_rate} END"
    features = ", ".join(FEATURES)
    zero_features = ("0 AS det_24h, 0 AS det_3d, 0 AS det_7d, NULL::DOUBLE AS frp_mean_7d, "
                     "NULL::DOUBLE AS modis_brightness_mean_7d, NULL::DOUBLE AS viirs_brightness_mean_7d, "
                     "0 AS ring1_det_24h, 0 AS ring1_det_7d, 0 AS ring2_det_7d")
    null_features = ", ".join(f"NULL::{'DOUBLE' if 'mean' in f else 'INTEGER'} AS {f}" for f in FEATURES)
    return f"""
        WITH labeled AS (SELECT *, {kind} AS row_kind, {split} AS split FROM classified)
        SELECT d AS forecast_date, split, row_kind, c6 AS h3_r6, c1 AS h3_r1, target,
               {negative_weight} AS weight, 1::BIGINT AS cells, {features},
               {year}::INTEGER AS year, {month}::INTEGER AS month
        FROM labeled WHERE row_kind = 'cell'
        UNION ALL
        SELECT d, split, row_kind, NULL::BIGINT, c1, 0::TINYINT,
               CASE WHEN row_kind = 'quiet' THEN count(*) ELSE 0 END::DOUBLE, count(*)::BIGINT,
               {zero_features}, {year}::INTEGER, {month}::INTEGER
        FROM labeled WHERE row_kind = 'quiet' GROUP BY d, split, row_kind, c1
        UNION ALL
        SELECT d, split, row_kind, NULL::BIGINT, c1, NULL::TINYINT, 0.0, count(*)::BIGINT,
               {null_features}, {year}::INTEGER, {month}::INTEGER
        FROM labeled WHERE row_kind IN ('unknown', 'sampled_out') GROUP BY d, split, row_kind, c1
    """


def _connect(silver: str, coverage: str | None, memory_limit: str, threads: int, temp_dir: str | Path | None,
             storage_options: dict[str, str] | None) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={_literal(memory_limit)}")
    connection.execute(f"SET threads={threads}")
    connection.execute("SET TimeZone='UTC'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET enable_progress_bar=false")
    if temp_dir is not None:
        connection.execute(f"SET temp_directory={_literal(str(temp_dir))}")
    load_h3(connection)
    connection.register("silver", DeltaTable(silver, storage_options=storage_options).to_pyarrow_dataset())
    if coverage is not None:
        connection.register("coverage", DeltaTable(coverage, storage_options=storage_options).to_pyarrow_dataset())
    return connection


def _coverage_hour(coverage: str, storage_options: dict[str, str] | None) -> int:
    hours = {commit.get("window_start_hour") for commit in DeltaTable(coverage, storage_options=storage_options)
             .history() if commit.get("operation") == "WRITE"}
    if len(hours) != 1:
        raise ValueError(f"Coverage table mixes window start hours: {sorted(map(str, hours))}")
    return int(hours.pop())


def _forecast_bounds(connection: duckdb.DuckDBPyConnection, cutoff_hour: int) -> tuple[date, date]:
    """First D with all input windows inside the data; last D whose target windows are complete
    in both silver and coverage (the window holding the newest record may be partial)."""
    first_window, silver_last = connection.execute(
        f"SELECT min(first_window), (SELECT max(acquired_at) FROM silver) FROM universe").fetchone()
    coverage_last = connection.execute("SELECT max(window_date) FROM coverage").fetchone()[0]
    last_window = min((silver_last - timedelta(hours=cutoff_hour)).date(), coverage_last) - timedelta(days=1)
    return first_window + timedelta(days=INPUT_DAYS), last_window - timedelta(days=TARGET_DAYS - 1)


def _clamp(year: int, month: int, bounds: tuple[date, date]) -> tuple[date, date] | None:
    first, last = _month_bounds(year, month)
    first, last = max(first, bounds[0]), min(last, bounds[1])
    return (first, last) if first <= last else None


def _written_months(target: str, storage_options: dict[str, str] | None) -> set[tuple[int, int]]:
    if not DeltaTable.is_deltatable(target, storage_options=storage_options):
        return set()
    return {tuple(int(v) for v in commit["forecast_month"].split("-"))
            for commit in DeltaTable(target, storage_options=storage_options).history() if "forecast_month" in commit}


def _verify_month(connection: duckdb.DuckDBPyConnection, relation: str, year: int, month: int,
                  bounds: tuple[date, date], sample_size: int, cutoff_hour: int) -> dict:
    """Universe check plus a from-silver recomputation of sampled `cell` rows of one month in `relation`."""
    first, last = _clamp(year, month, bounds)
    bad_universe = connection.execute(f"""
        WITH g AS (SELECT forecast_date AS d, sum(cells) AS cells FROM {relation}
                   WHERE year = {year} AND month = {month} GROUP BY 1),
             u AS (SELECT range::DATE AS d FROM range(DATE '{first}', DATE '{last}' + 1, INTERVAL 1 DAY))
        SELECT count(*) FROM u LEFT JOIN g USING (d)
        WHERE coalesce(g.cells, 0) <> (SELECT count(*) FROM universe WHERE first_window <= u.d - 1)
    """).fetchone()[0]
    window_lo, window_hi = first - timedelta(days=INPUT_DAYS), last + timedelta(days=TARGET_DAYS)
    checked, mismatched = connection.execute(f"""
        WITH sample AS MATERIALIZED (
            SELECT * FROM (
                SELECT forecast_date AS d, h3_r6, target, det_24h, det_7d, ring1_det_7d, ring2_det_7d
                FROM {relation} WHERE year = {year} AND month = {month} AND row_kind = 'cell')
            USING SAMPLE reservoir({sample_size} ROWS) REPEATABLE (7)),
        cutoff AS (SELECT *, d::TIMESTAMPTZ + INTERVAL {cutoff_hour} HOUR AS t FROM sample),
        s AS (SELECT h3_r6, acquired_at FROM silver WHERE {_months_touching(window_lo, window_hi)}),
        recomputed AS (
            SELECT c.d, c.h3_r6,
              (SELECT count(*) FROM s WHERE s.h3_r6 = c.h3_r6
                 AND s.acquired_at >= c.t - INTERVAL 1 DAY AND s.acquired_at < c.t) AS det_24h,
              (SELECT count(*) FROM s WHERE s.h3_r6 = c.h3_r6
                 AND s.acquired_at >= c.t - INTERVAL {INPUT_DAYS} DAY AND s.acquired_at < c.t) AS det_7d,
              (SELECT count(*) FROM s WHERE list_contains(h3_grid_ring(c.h3_r6::UBIGINT, 1)::BIGINT[], s.h3_r6)
                 AND s.acquired_at >= c.t - INTERVAL {INPUT_DAYS} DAY AND s.acquired_at < c.t) AS ring1_det_7d,
              (SELECT count(*) FROM s WHERE list_contains(h3_grid_ring(c.h3_r6::UBIGINT, 2)::BIGINT[], s.h3_r6)
                 AND s.acquired_at >= c.t - INTERVAL {INPUT_DAYS} DAY AND s.acquired_at < c.t) AS ring2_det_7d,
              (SELECT count(*) > 0 FROM s WHERE s.h3_r6 = c.h3_r6
                 AND s.acquired_at >= c.t AND s.acquired_at < c.t + INTERVAL {TARGET_DAYS} DAY) AS positive
            FROM cutoff c)
        SELECT count(*), count(*) FILTER (WHERE r.det_24h <> g.det_24h OR r.det_7d <> g.det_7d
            OR r.ring1_det_7d <> g.ring1_det_7d OR r.ring2_det_7d <> g.ring2_det_7d
            OR r.positive <> (g.target = 1))
        FROM sample g JOIN recomputed r USING (d, h3_r6)
    """).fetchone()
    return {"days_with_wrong_universe": bad_universe, "rows_checked": checked, "rows_mismatched": mismatched}


_WORKER: dict = {}  # per-process state of build workers, set by _init_worker


def _init_worker(silver: str, coverage: str, cutoff_hour: int, memory_limit: str, threads: int,
                 temp_dir: str | None, storage_options: dict[str, str] | None) -> None:
    connection = _connect(silver, coverage, memory_limit, threads, temp_dir, storage_options)
    connection.execute(_universe_sql(cutoff_hour))
    _WORKER.update(connection=connection, bounds=_forecast_bounds(connection, cutoff_hour), cutoff_hour=cutoff_hour)


def _build_month(task: tuple) -> tuple[int, int, tuple, dict]:
    """Build one month into a staging Parquet file and verify it there, before any commit."""
    year, month, staging, eval_start, sample_rate, seed, sample_size = task
    connection, bounds, cutoff_hour = _WORKER["connection"], _WORKER["bounds"], _WORKER["cutoff_hour"]
    clamped = _clamp(year, month, bounds)
    if clamped is None:
        raise ValueError(f"{year}-{month:02d} is outside the forecast bounds {bounds}")
    connection.execute(_classified_sql(*clamped, cutoff_hour))
    path = Path(staging) / f"{year}-{month:02d}.parquet"
    connection.execute(f"COPY ({_rows_sql(year, month, eval_start, sample_rate, seed)}) "
                       f"TO {_literal(str(path))} (FORMAT parquet, ROW_GROUP_SIZE 1000000)")
    counts = connection.execute(
        "SELECT count(*) FILTER (WHERE target = 1), count(*) FILTER (WHERE target = 0), "
        "count(*) FILTER (WHERE target IS NULL), count(*) FROM classified").fetchone()
    report = _verify_month(connection, f"read_parquet({_literal(str(path))})", year, month, bounds,
                           sample_size, cutoff_hour)
    connection.execute("DROP TABLE classified; DROP TABLE det")
    return year, month, counts, report


def build(silver: str, coverage: str, target: str, *, months: list[tuple[int, int]] | None = None,
          eval_start: date = date(2023, 1, 1), sample_rate: float = 0.1, seed: str = "rf10-ca08",
          cutoff_hour: int = 9, workers: int = 1, verify_sample: int = 2000, staging_dir: str | Path | None = None,
          storage_options: dict[str, str] | None = None, memory_limit: str = "3GB", threads: int = 4,
          temp_dir: str | Path | None = None) -> dict:
    """Write (or replace) forecast months in the Delta table at `target`, one commit per month.

    `workers` processes build and verify months into local staging Parquet files; this process is the
    only Delta writer and commits each month only after its verification passed. Without `months`,
    every month inside the data bounds not yet written is built (resumable).
    """
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if _coverage_hour(coverage, storage_options) != cutoff_hour:
        raise ValueError("Coverage windows must start at the cutoff hour")
    silver_version = DeltaTable(silver, storage_options=storage_options).version()
    coverage_version = DeltaTable(coverage, storage_options=storage_options).version()
    with _connect(silver, coverage, memory_limit, threads, temp_dir, storage_options) as connection:
        connection.execute(_universe_sql(cutoff_hour))
        bounds = _forecast_bounds(connection, cutoff_hour)
    if months is None:
        written = _written_months(target, storage_options)
        months = [(y, m) for y in range(bounds[0].year, bounds[1].year + 1) for m in range(1, 13)
                  if _clamp(y, m, bounds) and (y, m) not in written]
    staging = Path(staging_dir if staging_dir is not None else f"{target.rstrip('/')}.staging")
    if "://" in str(staging):
        raise ValueError("staging_dir must be a local directory")
    staging.mkdir(parents=True, exist_ok=True)
    tasks = [(y, m, str(staging), eval_start, sample_rate, seed, verify_sample) for y, m in sorted(set(months))]
    counts, reports = {}, {}
    # spawn: deltalake's runtime does not survive fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker,
                             initargs=(silver, coverage, cutoff_hour, memory_limit, threads,
                                       None if temp_dir is None else str(temp_dir), storage_options)) as pool:
        futures = [pool.submit(_build_month, task) for task in tasks]
        try:
            for future in as_completed(futures):
                year, month, month_counts, report = future.result()
                key = f"{year}-{month:02d}"
                path = staging / f"{key}.parquet"
                if report["days_with_wrong_universe"] or report["rows_mismatched"]:
                    raise RuntimeError(f"ml_dataset {key} failed verification, not committed: {json.dumps(report)}")
                exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
                metadata = {"method_version": METHOD_VERSION, "silver_version": str(silver_version),
                            "coverage_version": str(coverage_version), "cutoff_hour": str(cutoff_hour),
                            "eval_start": eval_start.isoformat(), "sample_rate": str(sample_rate), "seed": seed,
                            "forecast_month": key, "verification": json.dumps(report)}
                write_deltalake(target, pads.dataset(path).scanner().to_reader(), mode="overwrite",
                                partition_by=PARTITIONS,
                                predicate=f"year = {year} AND month = {month}" if exists else None,
                                storage_options=storage_options,
                                commit_properties=CommitProperties(custom_metadata=metadata))
                path.unlink()
                counts[key], reports[key] = month_counts, report
                print(f"ml_dataset {key}: {month_counts} {report}", file=sys.stderr, flush=True)
        except BaseException:
            # without this, leaving the `with` block waits for every queued month before re-raising
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return {"method_version": METHOD_VERSION, "target": target, "cutoff_hour": cutoff_hour,
            "eval_start": eval_start.isoformat(), "sample_rate": sample_rate, "seed": seed,
            "forecast_bounds": [bounds[0].isoformat(), bounds[1].isoformat()],
            "counts": {m: dict(zip(["positive", "negative", "unknown", "universe_cell_days"], c))
                       for m, c in sorted(counts.items())},
            "verification": dict(sorted(reports.items())), "limitations": LIMITATIONS}


def verify(target: str, silver: str, coverage: str, *, months: list[tuple[int, int]], sample_size: int = 2000,
           cutoff_hour: int = 9, storage_options: dict[str, str] | None = None, memory_limit: str = "3GB",
           threads: int = 4, temp_dir: str | Path | None = None) -> dict:
    """Re-run the month verification on committed data (e.g. after a copy to S3). Raises on any mismatch."""
    with _connect(silver, coverage, memory_limit, threads, temp_dir, storage_options) as connection:
        connection.register("gold", DeltaTable(target, storage_options=storage_options).to_pyarrow_dataset())
        connection.execute(_universe_sql(cutoff_hour))
        bounds = _forecast_bounds(connection, cutoff_hour)
        report = {f"{y}-{m:02d}": _verify_month(connection, "gold", y, m, bounds, sample_size, cutoff_hour)
                  for y, m in sorted(set(months))}
    failed = {m: r for m, r in report.items() if r["days_with_wrong_universe"] or r["rows_mismatched"]}
    if failed:
        raise RuntimeError(f"ml_dataset verification failed: {json.dumps(failed)}")
    return report
