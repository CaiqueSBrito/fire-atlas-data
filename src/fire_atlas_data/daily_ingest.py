"""Daily job: FIRMS Area API (world, NRT) -> normalized Parquet -> silver append -> 7-day forecast (RF10/CA08, RF11)
-> optional place lookup rebuild -> optional harmonized activity refresh (RF07).

One run for issue date D, after the D cutoff:
1. Per product, the fetch starts `OVERLAP_DAYS` before the date of its newest silver `acquired_at` within the
   lookback (or at the lookback start when it has none), so records that arrive late for earlier passes are seen.
2. Area API windows (at most 5 days each) cover that start through the current UTC date.
3. Every non-empty download is normalized; `silver.append` adds only observations silver lacks, oldest month
   first. A repeated or resumed run therefore appends nothing twice.
4. `ml_forecast.forecast` replaces the rows of D in the forecast table.
5. With `places_target`, `places.build` rebuilds continent/country/nearest city per H3 cell from the new
   silver version and `places.verify` checks it.
6. With `harmonize_target`, `harmonization.apply` rewrites the local-day years that the appended data can
   change (outage detection looks `OUTAGE_WINDOW_DAYS` around each day) and `harmonization.verify` checks them.
   The tables must already exist: a daily run never creates a table holding only recent years.
Raw CSVs, receipts, normalized datasets and `report.json` stay in `<work-dir>/<D>/run-<UTC time>/`.
Silver, model, report and forecast may be local paths or s3:// URIs (credentials from the standard AWS chain).
A file lock in the work dir keeps a single writer; the lock is local, not a distributed one.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time as clock

import duckdb
from deltalake import DeltaTable
import pyarrow.fs as pafs

from . import ml_dataset, silver as silver_layer
from .acquisition import fetch_area
from .ml_forecast import forecast
from . import harmonization, places as places_layer
from .normalization import normalize

METHOD_VERSION = "1.0.0"
# VIIRS S-NPP stopped producing new data on 2026-11-02; pass --product to add it back.
PRODUCTS = ("MODIS_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
AREA_MAX_DAYS = 5
# NRT passes are published out of order: a later pass can be ingested before an earlier one.
OVERLAP_DAYS = 1
LIMITATIONS = [
    f"Records published more than {OVERLAP_DAYS} day(s) after the newest ingested acquisition date are missed.",
    "An empty download is unknown coverage, not absence of fire; a product without rows adds nothing.",
    "NRT has no FIRMS type: static industrial sources are not removed before the forecast.",
    "The lock is a local file lock; two hosts writing the same table are not prevented.",
]


def _windows(first: date, last: date) -> list[tuple[date, int]]:
    """Consecutive Area API windows (start date, days <= 5) covering first..last."""
    windows = []
    while first <= last:
        days = min(AREA_MAX_DAYS, (last - first).days + 1)
        windows.append((first, days))
        first += timedelta(days=days)
    return windows


def _fetch_starts(silver: str, products: list[str], floor: datetime, now: datetime) -> dict[str, date]:
    """Per product: OVERLAP_DAYS before the newest silver acquisition date within [floor, now], never before floor."""
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.register("silver", DeltaTable(silver).to_pyarrow_dataset())
    listed = ", ".join(silver_layer._literal(p) for p in products)
    newest = dict(connection.execute(
        f"SELECT source_product, max(acquired_at) FROM silver "
        f"WHERE {ml_dataset._months_touching(floor.date(), now.date())} AND source_product IN ({listed}) "
        f"AND acquired_at >= TIMESTAMPTZ {silver_layer._literal(floor.isoformat())} GROUP BY 1").fetchall())
    return {p: max(newest[p].date() - timedelta(days=OVERLAP_DAYS), floor.date()) if p in newest else floor.date()
            for p in products}


@contextmanager
def _lock(path: Path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another daily-ingest run holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _log(event: str, **fields) -> None:
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields},
                     default=str), flush=True)


def _archive(run_dir: Path, destination: str) -> None:
    """Copy every file of `run_dir` under `destination` object by object; no directory markers are created."""
    filesystem, base = pafs.FileSystem.from_uri(destination)
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        with path.open("rb") as source, filesystem.open_output_stream(f"{base}/{path.relative_to(run_dir).as_posix()}") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)


def run(issue_date: date, *, silver: str, model: str, report: str | None, forecast_target: str,
        work_dir: str | Path, products: list[str] | tuple[str, ...] = PRODUCTS, lookback_days: int = 10,
        cutoff_hour: int = 9, key_env: str = "FIRMS_MAP_KEY", optimize: bool = False,
        memory_limit: str = "3GB", threads: int = 4, temp_dir: str | Path | None = None,
        archive_uri: str | None = None, places_target: str | None = None, places_countries: str | None = None,
        places_cities: str | None = None, harmonize_model: str | None = None, harmonize_target: str | None = None,
        harmonize_availability: str | None = None, now: datetime | None = None) -> dict:
    """Fetch, normalize, append and forecast for one issue date; returns the run report.

    With `archive_uri` (e.g. s3://bucket/bronze/daily-ingest) the run directory is copied to
    `<archive_uri>/<D>/<run>/` after success, for hosts whose local disk is ephemeral."""
    now = now or datetime.now(timezone.utc)
    cutoff = datetime.combine(issue_date, time(cutoff_hour), timezone.utc)
    if now < cutoff:
        raise ValueError(f"Issue date {issue_date} needs inputs up to {cutoff}; run after the cutoff")
    if not 1 <= lookback_days <= 60:
        raise ValueError("lookback_days must be within 1..60")
    if not DeltaTable.is_deltatable(silver):
        raise ValueError("daily-ingest needs the historical silver table (it defines the forecast universe)")
    if (places_target is None) != (places_countries is None) or (places_target is None) != (places_cities is None):
        raise ValueError("places_target, places_countries and places_cities go together")
    harmonize = (harmonize_model, harmonize_target, harmonize_availability)
    if any(v is None for v in harmonize) and any(v is not None for v in harmonize):
        raise ValueError("harmonize_model, harmonize_target and harmonize_availability go together")
    if harmonize_target is not None and not all(DeltaTable.is_deltatable(t) for t in harmonize[1:]):
        raise ValueError("harmonize_target and harmonize_availability must be existing Delta tables "
                         "(build them once with `harmonize`)")
    work = Path(work_dir).absolute()
    work.mkdir(parents=True, exist_ok=True)
    # DuckDB spills to ".tmp" under the current directory by default, which a container may not let us create.
    temp_dir = temp_dir if temp_dir is not None else work / "duckdb-spill"
    with _lock(work / ".daily-ingest.lock"):
        run_dir = work / issue_date.isoformat() / now.strftime("run-%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True)
        starts = _fetch_starts(silver, list(products), cutoff - timedelta(days=lookback_days), now)
        downloads, datasets = [], []
        for product in products:
            for first, days in _windows(starts[product], now.date()):
                raw = run_dir / "raw" / product / f"{first.isoformat()}_{days}d"
                receipt = fetch_area(raw, product=product, area="world", days=days, start_date=first.isoformat(),
                                     key_env=key_env)
                rows = receipt["raw"]["rows"]
                downloads.append({"product": product, "date": first.isoformat(), "days": days, "rows": rows,
                                  "sha256": receipt["raw"]["sha256"]})
                _log("downloaded", product=product, date=first, days=days, rows=rows)
                if rows:
                    normalized = run_dir / "normalized" / product / f"{first.isoformat()}_{days}d"
                    normalize(raw / "raw.csv", normalized, source_product=product, source_url=receipt["source"]["url"],
                              memory_limit=memory_limit, threads=threads, temp_dir=temp_dir)
                    datasets.append(normalized)
        since = datetime.combine(min(starts.values()), time(0), timezone.utc)
        appended = silver_layer.append(datasets, silver, since=since, memory_limit=memory_limit, threads=threads,
                                       temp_dir=temp_dir)["counts"] if datasets else {}
        _log("silver_appended", months=appended)
        optimized = silver_layer.optimize(silver) if optimize and appended else {}
        predicted = forecast(silver, model, forecast_target, issue_date=issue_date, report=report,
                             cutoff_hour=cutoff_hour, memory_limit=memory_limit, threads=threads, temp_dir=temp_dir)
        placed = None
        if places_target is not None:
            placed = places_layer.build(silver, places_countries, places_cities, places_target,
                                        memory_limit=memory_limit, threads=threads, temp_dir=temp_dir)
            placed["verification"] = places_layer.verify(places_target, silver, memory_limit=memory_limit)
            _log("places_rebuilt", **placed["counts"])
        harmonized = None
        if harmonize_target is not None:
            first = min(min(starts.values()), issue_date) - timedelta(days=harmonization.OUTAGE_WINDOW_DAYS)
            years = list(range(first.year, now.date().year + 1))
            harmonized = harmonization.apply(silver, harmonize_model, harmonize_target, harmonize_availability,
                                             years=years, memory_limit=memory_limit, threads=threads,
                                             temp_dir=temp_dir)
            harmonized["verification"] = harmonization.verify(harmonize_target, silver, years=years,
                                                              memory_limit=memory_limit, threads=threads,
                                                              temp_dir=temp_dir)
            _log("harmonized", years=years)
        result = {"method_version": METHOD_VERSION, "issue_date": issue_date.isoformat(),
                  "started_at": now.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                  "fetch_starts": {p: d.isoformat() for p, d in starts.items()}, "downloads": downloads,
                  "silver_appended": appended, "silver_optimized": optimized,
                  "silver_version": DeltaTable(silver).version(), "forecast": predicted, "places": placed,
                  "harmonized": harmonized,
                  "counts": predicted["counts"], "limitations": LIMITATIONS}
        with (run_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if archive_uri is not None:
            destination = f"{archive_uri.rstrip('/')}/{issue_date.isoformat()}/{run_dir.name}"
            _archive(run_dir, destination)
            _log("archived", destination=destination)
    return result


def run_daily(at: time, *, retry_minutes: int = 60, poll_seconds: int = 300, **options) -> None:
    """Run once per UTC day at or after `at`; a failed day is retried every `retry_minutes` until it succeeds."""
    done, next_try = None, datetime.min.replace(tzinfo=timezone.utc)
    _log("scheduler_started", daily_at=at.isoformat(), retry_minutes=retry_minutes)
    while True:
        now = datetime.now(timezone.utc)
        if now.time() >= at and done != now.date() and now >= next_try:
            try:
                result = run(now.date(), **options)
                done = now.date()
                _log("run_succeeded", **result["counts"])
            except Exception as error:  # keep the scheduler alive; the message never holds the MAP_KEY
                next_try = now + timedelta(minutes=retry_minutes)
                _log("run_failed", issue_date=now.date(), error=f"{type(error).__name__}: {error}",
                     retry_at=next_try)
                print(f"daily-ingest failed for {now.date()}: {error}", file=sys.stderr, flush=True)
        clock.sleep(poll_seconds)
