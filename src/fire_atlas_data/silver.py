"""Silver layer: normalized FIRMS observations plus H3 cells at resolutions 0..6, in Delta Lake (RF11/CA09).

Every accepted observation is kept (no confidence filter). Each row carries one
H3 cell id per resolution 0..6; coarser cells are H3 parents of the res6 cell,
so all levels nest exactly. Months are written one at a time with a partition
predicate, so reprocessing a month replaces only that month in one Delta commit.
Rows inside a month are sorted by `h3_r6`; `optimize` then Z-orders on `h3_r6`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
from deltalake import CommitProperties, DeltaTable, write_deltalake

from .map_grid import load_h3

METHOD_VERSION = "1.0.0"
RESOLUTIONS = tuple(range(7))
FINEST = max(RESOLUTIONS)
H3_COLUMNS = tuple(f"h3_r{res}" for res in RESOLUTIONS)
PARTITIONS = ["acquisition_year", "acquisition_month"]
LIMITATIONS = [
    "No confidence filter: low-confidence detections are kept; filtering is a gold-layer decision.",
    "H3 cell ids are signed 64-bit (Delta has no unsigned integers); values are identical to the H3 index.",
    "Cells come from the detection pixel center; pixel footprint (scan/track) is not distributed across cells.",
    "Sensors are not harmonized or deduplicated; overlapping platforms may report the same fire.",
    "Absence of a row is unknown, not confirmed absence of fire.",
]


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(datasets: list[str | Path]) -> tuple[list[dict], str]:
    roots = [Path(d).resolve(strict=True) for d in datasets]
    if not roots:
        raise ValueError("At least one normalized dataset is required")
    sources = []
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append({"path": str(root), "manifest_sha256": _sha256(manifest_path),
                        "schema_version": manifest["schema_version"],
                        "source_product": manifest["source"]["product"]})
    globs = "[" + ",".join(_literal(str(r / "accepted" / "**" / "*.parquet")) for r in roots) + "]"
    return sources, f"read_parquet({globs}, hive_partitioning=true, union_by_name=true)"


def _month_sql(relation: str, year: int, month: int) -> str:
    finest = f"h3_latlng_to_cell(latitude, longitude, {FINEST})"
    parents = ", ".join(f"h3_cell_to_parent({finest}, {res})::BIGINT AS h3_r{res}" for res in RESOLUTIONS[:-1])
    return f"""
        SELECT * EXCLUDE (acquisition_year, acquisition_month),
               {parents}, {finest}::BIGINT AS h3_r{FINEST},
               acquisition_year::INTEGER AS acquisition_year, acquisition_month::INTEGER AS acquisition_month
        FROM {relation}
        WHERE acquisition_year = {year} AND acquisition_month = {month}
        ORDER BY h3_r{FINEST}, acquired_at
    """


def build(datasets: list[str | Path], target: str, *, months: list[tuple[int, int]] | None = None,
          storage_options: dict[str, str] | None = None, memory_limit: str = "2GB", threads: int = 2,
          temp_dir: str | Path | None = None) -> dict:
    """Write (or replace) silver months in the Delta table at `target`; return per-month counts."""
    if threads < 1:
        raise ValueError("threads must be positive")
    sources, relation = _sources(datasets)
    counts = {}
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={threads}")
        connection.execute("SET TimeZone='UTC'")
        if temp_dir is not None:
            connection.execute(f"SET temp_directory={_literal(str(temp_dir))}")
        load_h3(connection)
        available = [tuple(row) for row in connection.execute(
            f"SELECT DISTINCT acquisition_year, acquisition_month FROM {relation} ORDER BY 1, 2").fetchall()]
        selected = available if months is None else sorted(set(months))
        missing = set(selected) - set(available)
        if missing:
            raise ValueError(f"Months absent from sources: {sorted(missing)}")
        for year, month in selected:
            expected = connection.execute(
                f"SELECT count(*) FROM {relation} WHERE acquisition_year = {year} "
                f"AND acquisition_month = {month}").fetchone()[0]
            cursor = connection.cursor()
            reader = cursor.sql(_month_sql(relation, year, month)).to_arrow_reader(1_000_000)
            exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
            metadata = {"method_version": METHOD_VERSION, "h3_resolutions": json.dumps(list(RESOLUTIONS)),
                        "sources": json.dumps(sources), "partition": f"{year}-{month:02d}",
                        "expected_rows": str(expected)}
            write_deltalake(target, reader, mode="overwrite", partition_by=PARTITIONS,
                            predicate=(f"acquisition_year = {year} AND acquisition_month = {month}" if exists else None),
                            schema_mode="merge" if exists else None, storage_options=storage_options,
                            commit_properties=CommitProperties(custom_metadata=metadata))
            cursor.close()
            counts[f"{year}-{month:02d}"] = expected
    return {"method_version": METHOD_VERSION, "target": target, "h3_resolutions": list(RESOLUTIONS),
            "sources": sources, "counts": counts, "limitations": LIMITATIONS}


def verify(target: str, *, storage_options: dict[str, str] | None = None, memory_limit: str = "2GB") -> dict:
    """Check every silver row has nested H3 cells 0..6 and return per-month row counts."""
    table = DeltaTable(target, storage_options=storage_options)
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        load_h3(connection)
        connection.register("silver", table.to_pyarrow_dataset())
        checks = " OR ".join(
            [f"h3_r{r} IS NULL OR h3_get_resolution(h3_r{r}::UBIGINT) <> {r}" for r in RESOLUTIONS]
            + [f"h3_cell_to_parent(h3_r{FINEST}::UBIGINT, {r})::BIGINT <> h3_r{r}" for r in RESOLUTIONS[:-1]]
            + [f"h3_latlng_to_cell(latitude, longitude, {FINEST})::BIGINT <> h3_r{FINEST}"])
        invalid = connection.execute(f"SELECT count(*) FROM silver WHERE {checks}").fetchone()[0]
        counts = {f"{y}-{m:02d}": n for y, m, n in connection.execute(
            "SELECT acquisition_year, acquisition_month, count(*) FROM silver GROUP BY 1, 2 ORDER BY 1, 2").fetchall()}
    if invalid:
        raise RuntimeError(f"{invalid} silver rows have missing or inconsistent H3 cells")
    return {"version": table.version(), "counts": counts, "invalid_h3_rows": invalid}


def _pending_zorder(table: DeltaTable) -> set[str]:
    """Partitions whose last WRITE is newer than their last Z-order (history is newest first)."""
    done, pending = set(), set()
    for commit in table.history():
        if commit["operation"] == "OPTIMIZE":
            year, month = (int(v.split("'")[1]) for v in json.loads(commit["operationParameters"]["predicate"]))
            done.add(f"{year}-{month:02d}")
        elif commit["operation"] == "WRITE" and "partition" in commit and commit["partition"] not in done:
            pending.add(commit["partition"])
    return pending


def optimize(target: str, *, storage_options: dict[str, str] | None = None) -> dict:
    """Z-order on h3_r6 every partition written since its last Z-order, one partition at a time."""
    table = DeltaTable(target, storage_options=storage_options)
    results = {}
    for partition in sorted(_pending_zorder(table)):
        year, month = (int(v) for v in partition.split("-"))
        metrics = table.optimize.z_order(
            [f"h3_r{FINEST}"], partition_filters=[("acquisition_year", "=", str(year)),
                                                  ("acquisition_month", "=", str(month))],
            max_concurrent_tasks=1, max_spill_size=3 * 1024 ** 3)
        results[partition] = {"files_added": metrics["numFilesAdded"],
                              "files_removed": metrics["numFilesRemoved"]}
    return results


def vacuum(target: str, *, storage_options: dict[str, str] | None = None) -> int:
    """Delete files no longer referenced by the latest version; earlier versions become unreadable."""
    table = DeltaTable(target, storage_options=storage_options)
    return len(table.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False))
