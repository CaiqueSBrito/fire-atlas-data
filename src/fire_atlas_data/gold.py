"""Gold layer, map product: daily detection aggregates per H3 cell (res 0..6) and sensor, in Delta Lake.

Built from the silver table one year at a time; each (resolution, year)
partition is replaced in its own Delta commit, bounding memory. No confidence filter. Rows exist
only where at least one detection happened: a missing row is unknown, not a
confirmed absence of fire. Sensors are kept apart (never summed), because
MODIS and VIIRS differ in pixel size and sensitivity and are not harmonized.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
from deltalake import CommitProperties, DeltaTable, write_deltalake

from .silver import RESOLUTIONS, _literal

METHOD_VERSION = "1.0.0"
PARTITIONS = ["resolution", "year"]
ZORDER = ["day", "h3_cell"]
LIMITATIONS = [
    "Cells only where detections exist; absence of a row is unknown, not confirmed absence of fire.",
    "No confidence filter. VIIRS confidence is counted per category (l/n/h); MODIS keeps its 0-100 percent mean.",
    "brightness_max_k is MODIS channel 21/22 or VIIRS I4: comparable only within the same sensor.",
    "Sensors are not harmonized nor deduplicated; platforms of one sensor are summed (e.g. S-NPP + NOAA-20 + NOAA-21).",
    "day is the UTC acquisition date, not local date.",
]


def _cells_sql(year: int, res: int) -> str:
    return f"""
        SELECT {res}::INTEGER AS resolution, h3_r{res} AS h3_cell, acquisition_date AS day, sensor,
               count(*)::BIGINT AS detections,
               count(DISTINCT platform)::INTEGER AS platforms,
               sum(frp_mw) AS frp_sum_mw, avg(frp_mw) AS frp_mean_mw, max(frp_mw) AS frp_max_mw,
               max(coalesce(brightness_k, brightness_i4_k)) AS brightness_max_k,
               count(*) FILTER (WHERE confidence_category = 'l')::BIGINT AS viirs_confidence_low,
               count(*) FILTER (WHERE confidence_category = 'n')::BIGINT AS viirs_confidence_nominal,
               count(*) FILTER (WHERE confidence_category = 'h')::BIGINT AS viirs_confidence_high,
               avg(confidence_percent) AS modis_confidence_mean,
               {year}::INTEGER AS year
        FROM silver WHERE acquisition_year = {year}
        GROUP BY ALL
    """


def _pending_zorder(table: DeltaTable) -> set[tuple[int, int]]:
    """(resolution, year) partitions whose last WRITE is newer than their last Z-order."""
    done, pending = set(), set()
    for commit in table.history():
        if commit["operation"] == "OPTIMIZE":
            res, year = (int(v.split("'")[1]) for v in json.loads(commit["operationParameters"]["predicate"]))
            done.add((res, year))
        elif commit["operation"] == "WRITE" and "years" in commit:
            written = [int(commit["resolution"])] if "resolution" in commit else RESOLUTIONS
            pending |= {(res, year) for year in json.loads(commit["years"]) for res in written} - done
    return pending


def build_map_cells(silver: str, target: str, *, years: list[int] | None = None,
                    storage_options: dict[str, str] | None = None, memory_limit: str = "2GB",
                    threads: int = 2, temp_dir: str | Path | None = None) -> dict:
    """Write (or replace) map-cell years, one (resolution, year) Delta commit each; return per-year detections."""
    if threads < 1:
        raise ValueError("threads must be positive")
    source = DeltaTable(silver, storage_options=storage_options)
    counts = {}
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={threads}")
        connection.execute("SET preserve_insertion_order=false")
        if temp_dir is not None:
            connection.execute(f"SET temp_directory={_literal(str(temp_dir))}")
        dataset = source.to_pyarrow_dataset()
        connection.register("silver", dataset)
        available = [row[0] for row in connection.execute(
            "SELECT DISTINCT acquisition_year FROM silver ORDER BY 1").fetchall()]
        selected = available if years is None else sorted(set(years))
        missing = set(selected) - set(available)
        if missing:
            raise ValueError(f"Years absent from silver: {sorted(missing)}")
        for year in selected:
            expected = connection.execute(
                f"SELECT count(*) FROM silver WHERE acquisition_year = {year}").fetchone()[0]
            for res in RESOLUTIONS:
                cursor = connection.cursor()
                cursor.register("silver", dataset)
                reader = cursor.sql(_cells_sql(year, res)).to_arrow_reader(1_000_000)
                exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
                metadata = {"method_version": METHOD_VERSION, "silver": silver,
                            "silver_version": str(source.version()), "years": json.dumps([year]),
                            "resolution": str(res), "expected_detections": str(expected)}
                write_deltalake(target, reader, mode="overwrite", partition_by=PARTITIONS,
                                predicate=f"resolution = {res} AND year = {year}" if exists else None,
                                storage_options=storage_options,
                                commit_properties=CommitProperties(custom_metadata=metadata))
                cursor.close()
            counts[str(year)] = expected
    return {"method_version": METHOD_VERSION, "target": target, "silver": silver,
            "silver_version": source.version(), "resolutions": list(RESOLUTIONS),
            "counts": counts, "limitations": LIMITATIONS}


def verify_map_cells(target: str, silver: str, *, storage_options: dict[str, str] | None = None,
                     memory_limit: str = "2GB") -> dict:
    """Check each built year has resolutions 0..6, each summing to that year's silver detections.

    Years present in silver but not yet built are reported in `unbuilt_years`, not treated as errors.
    """
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.register("gold", DeltaTable(target, storage_options=storage_options).to_pyarrow_dataset())
        connection.register("silver", DeltaTable(silver, storage_options=storage_options).to_pyarrow_dataset())
        expected = dict(connection.execute(
            "SELECT acquisition_year, count(*) FROM silver GROUP BY 1").fetchall())
        totals = connection.execute(
            "SELECT year, resolution, sum(detections), count(*) FROM gold GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    built = {y for y, *_ in totals}
    mismatches = [(y, r, n, expected.get(y)) for y, r, n, _ in totals if n != expected.get(y)]
    incomplete = sorted(y for y in built if {r for yy, r, *_ in totals if yy == y} != set(RESOLUTIONS))
    if mismatches or incomplete:
        raise RuntimeError(f"map_cells disagrees with silver: mismatches={mismatches[:10]} "
                           f"incomplete_years={incomplete}")
    return {"version": DeltaTable(target, storage_options=storage_options).version(),
            "rows": {f"{y}/r{r}": cells for y, r, _, cells in totals},
            "unbuilt_years": sorted(set(expected) - built)}


def optimize_map_cells(target: str, *, storage_options: dict[str, str] | None = None,
                       temp_dir: str | Path | None = None) -> dict:
    """Z-order on (day, h3_cell) every (resolution, year) written since its last Z-order.

    `temp_dir` redirects the Z-order spill (TMPDIR); the default /tmp may be RAM-backed.
    """
    if temp_dir is not None:
        os.environ["TMPDIR"] = str(Path(temp_dir).resolve())
    table = DeltaTable(target, storage_options=storage_options)
    results = {}
    for res, year in sorted(_pending_zorder(table)):
        metrics = table.optimize.z_order(
            ZORDER, partition_filters=[("resolution", "=", str(res)), ("year", "=", str(year))],
            max_concurrent_tasks=1, max_spill_size=3 * 1024 ** 3)
        results[f"r{res}/{year}"] = {"files_added": metrics["numFilesAdded"],
                                     "files_removed": metrics["numFilesRemoved"]}
    return results
