"""Gold layer, map product: daily detection aggregates per H3 cell (res 0..6) and sensor, in Delta Lake.

Built from the silver table one year at a time; each (resolution, year)
is replaced in its own Delta commit, partitioned by resolution/year/month. No confidence filter. Rows exist
only where at least one detection happened: a missing row is unknown, not a
confirmed absence of fire. Sensors are kept apart (never summed), because
MODIS and VIIRS differ in pixel size and sensitivity and are not harmonized.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import duckdb
import pyarrow as pa
from deltalake import CommitProperties, DeltaTable, write_deltalake

from .silver import RESOLUTIONS, _literal

METHOD_VERSION = "1.1.0"
PARTITIONS = ["resolution", "year", "month"]
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
               {year}::INTEGER AS year, month(acquisition_date)::INTEGER AS month
        FROM silver WHERE acquisition_year = {year}
        GROUP BY ALL
    """


def _pending_zorder(table: DeltaTable) -> set[tuple[int, int, int]]:
    """Active monthly partitions whose newest relevant operation is a WRITE."""
    actions = pa.table(table.get_add_actions(flatten=True))
    active = set(zip(*(actions[f"partition.{key}"].to_pylist() for key in PARTITIONS)))
    seen, pending = set(), set()
    for commit in table.history():
        if commit["operation"] == "OPTIMIZE":
            predicate = json.loads(commit["operationParameters"]["predicate"])
            filters = {key: int(value) for key, value in
                       re.findall(r"(\w+)\s*=\s*'(\d+)'", " ".join(predicate))}
            affected = {p for p in active if all(p[PARTITIONS.index(k)] == v for k, v in filters.items())}
        elif commit["operation"] == "WRITE" and ("years" in commit or "repartition" in commit):
            years = json.loads(commit["years"]) if "years" in commit else None
            affected = {p for p in active if (years is None or p[1] in years)
                        and ("resolution" not in commit or p[0] == int(commit["resolution"]))}
            pending |= affected - seen
        else:
            continue
        seen |= affected
    return pending


def _fingerprint(connection: duckdb.DuckDBPyConnection, relation: str) -> list:
    columns = [row[0] for row in connection.execute(f"DESCRIBE {relation}").fetchall()]
    values = ", ".join(f'"{column}"' for column in columns if column != "month")
    return connection.execute(
        f"SELECT resolution, year, month(day), count(*), sum(detections), "
        f"sum(hash({values})::HUGEINT) FROM {relation} GROUP BY 1, 2, 3 ORDER BY 1, 2, 3").fetchall()


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
        if DeltaTable.is_deltatable(target, storage_options=storage_options):
            if DeltaTable(target, storage_options=storage_options).metadata().partition_columns != PARTITIONS:
                raise ValueError("Run gold-map-repartition before updating the legacy map table")
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
    """Check each built month has resolutions 0..6 summing to that month's silver detections.

    Years present in silver but not yet built are reported in `unbuilt_years`, not treated as errors.
    """
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        table = DeltaTable(target, storage_options=storage_options)
        if table.metadata().partition_columns != PARTITIONS:
            raise ValueError(f"Expected partition columns {PARTITIONS}")
        connection.register("gold", table.to_pyarrow_dataset())
        connection.register("silver", DeltaTable(silver, storage_options=storage_options).to_pyarrow_dataset())
        expected = {(y, m): n for y, m, n in connection.execute(
            "SELECT acquisition_year, acquisition_month, count(*) FROM silver GROUP BY 1, 2").fetchall()}
        totals = connection.execute(
            "SELECT year, month, resolution, sum(detections), count(*) FROM gold "
            "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3").fetchall()
        invalid = connection.execute(
            "SELECT count(*) FROM gold WHERE day IS NULL OR year IS DISTINCT FROM year(day) "
            "OR month IS DISTINCT FROM month(day)").fetchone()[0]
    built = {(y, m) for y, m, *_ in totals}
    built_years = {y for y, _ in built}
    mismatches = [(y, m, r, n, expected.get((y, m))) for y, m, r, n, _ in totals
                  if n != expected.get((y, m))]
    incomplete = sorted((y, m) for y, m in built
                        if {r for yy, mm, r, *_ in totals if (yy, mm) == (y, m)} != set(RESOLUTIONS))
    missing = sorted(p for p in set(expected) - built if p[0] in built_years)
    if mismatches or incomplete or missing or invalid:
        raise RuntimeError(f"map_cells disagrees with silver: mismatches={mismatches[:10]} "
                           f"incomplete_months={incomplete} missing_months={missing} invalid_dates={invalid}")
    return {"version": DeltaTable(target, storage_options=storage_options).version(),
            "rows": {f"{y}/{m:02d}/r{r}": cells for y, m, r, _, cells in totals},
            "unbuilt_years": sorted({y for y, _ in expected} - built_years),
            "unbuilt_months": [f"{y}-{m:02d}" for y, m in sorted(set(expected) - built)]}


def repartition_map_cells(target: str, *, storage_options: dict[str, str] | None = None,
                          memory_limit: str = "2GB", threads: int = 2,
                          temp_dir: str | Path | None = None) -> dict:
    """Atomically add month partitions without recalculating aggregates or vacuuming old files."""
    source = DeltaTable(target, storage_options=storage_options)
    source_version = source.version()
    partitions = source.metadata().partition_columns
    if partitions == PARTITIONS:
        return {"target": target, "version": source.version(), "already_partitioned": True}
    if partitions != ["resolution", "year"]:
        raise ValueError(f"Unsupported source partitions: {partitions}")
    if threads < 1:
        raise ValueError("threads must be positive")
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={threads}")
        connection.execute("SET preserve_insertion_order=false")
        if temp_dir is not None:
            connection.execute(f"SET temp_directory={_literal(str(temp_dir))}")
        connection.register("original", source.to_pyarrow_dataset())
        before = _fingerprint(connection, "original")
        invalid = connection.execute(
            "SELECT count(*) FROM original WHERE day IS NULL OR year IS DISTINCT FROM year(day)").fetchall()[0][0]
        if invalid:
            raise ValueError(f"{invalid} rows have invalid partition dates")
        reader = connection.sql(
            "SELECT *, month(day)::INTEGER AS month FROM original").to_arrow_reader(100_000)
        write_deltalake(source, reader, mode="overwrite", schema_mode="overwrite", partition_by=PARTITIONS,
                        storage_options=storage_options, commit_properties=CommitProperties(custom_metadata={
                            "method_version": METHOD_VERSION, "repartition": "month",
                            "source_version": str(source_version)}))
        result = DeltaTable(target, storage_options=storage_options)
        connection.register("rewritten", result.to_pyarrow_dataset())
        after = _fingerprint(connection, "rewritten")
        if before != after:
            raise RuntimeError("Repartition verification failed; original Delta version remains available")
    return {"target": target, "source_version": source_version, "version": result.version(),
            "partition_columns": result.metadata().partition_columns, "partitions": len(after),
            "rows": sum(row[3] for row in after), "fingerprint_mismatches": 0,
            "verification": "Per resolution/year/month: row count, detection sum and sum of 64-bit row hashes",
            "limitations": ["Hash conservation is not a collision-free proof of row equality."]}


def optimize_map_cells(target: str, *, storage_options: dict[str, str] | None = None,
                       temp_dir: str | Path | None = None) -> dict:
    """Z-order on (day, h3_cell) every monthly partition written since its last Z-order.

    `temp_dir` redirects the Z-order spill (TMPDIR); the default /tmp may be RAM-backed.
    """
    if temp_dir is not None:
        os.environ["TMPDIR"] = str(Path(temp_dir).resolve())
    table = DeltaTable(target, storage_options=storage_options)
    results = {}
    for res, year, month in sorted(_pending_zorder(table)):
        metrics = table.optimize.z_order(
            ZORDER, partition_filters=[("resolution", "=", str(res)), ("year", "=", str(year)),
                                      ("month", "=", str(month))],
            max_concurrent_tasks=1, max_spill_size=3 * 1024 ** 3)
        results[f"r{res}/{year}/{month:02d}"] = {"files_added": metrics["numFilesAdded"],
                                               "files_removed": metrics["numFilesRemoved"]}
    return results
