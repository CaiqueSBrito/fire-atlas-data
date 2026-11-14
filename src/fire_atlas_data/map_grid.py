"""H3 multi-resolution daily map aggregation over normalized FIRMS Parquet (RF11/CA09).

Counts detections per H3 cell, UTC acquisition day and sensor (or platform) at
several resolutions, one per map zoom band, so a map can render any number of
observations without one feature per pixel. Daily rows let any date interval be
summed at query time. Map resolutions are a display decision; ML uses only H3
res5 and builds its own grid from the normalized observations.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb

METHOD_VERSION = "2.0.0"
DEFAULT_RESOLUTIONS = (2, 3, 4, 5, 6, 7, 8)
GROUPS = ("sensor", "platform")
LIMITATIONS = [
    "Counts are detections (pixels), not fire events, burned area or ignition points.",
    "Sensors are never merged: MODIS and VIIRS counts are not harmonized or comparable.",
    "Overlapping platforms can observe the same fire; detections are not deduplicated.",
    "Cells without detections are absent from the output: unknown, not confirmed absence of fire.",
    "Days use the UTC acquisition date, not local calendar days or availability time.",
    "Map resolutions are for display only; they do not define ML features or labels (ML uses H3 res5).",
]


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filters(start: date | None, end: date | None, bbox: tuple[float, float, float, float] | None) -> list[str]:
    clauses = []
    if start is not None:
        # Year predicate lets DuckDB prune Hive partitions before reading files.
        clauses += [f"acquisition_year >= {start.year}", f"acquisition_date >= DATE '{start.isoformat()}'"]
    if end is not None:
        clauses += [f"acquisition_year <= {end.year}", f"acquisition_date <= DATE '{end.isoformat()}'"]
    if bbox is not None:
        west, south, east, north = bbox
        clauses += [f"longitude BETWEEN {west!r} AND {east!r}", f"latitude BETWEEN {south!r} AND {north!r}"]
    return clauses


def _validate(resolutions: tuple[int, ...], group_by: str, start: date | None, end: date | None,
              bbox: tuple[float, float, float, float] | None) -> None:
    if not resolutions or any(not isinstance(r, int) or not 0 <= r <= 15 for r in resolutions):
        raise ValueError("resolutions must be H3 resolutions between 0 and 15")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not repeat")
    if group_by not in GROUPS:
        raise ValueError(f"group_by must be one of {', '.join(GROUPS)}")
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")
    if bbox is not None:
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be west,south,east,north within EPSG:4326 limits; antimeridian crossing is unsupported")


def _source(globs: list[str]) -> str:
    if not globs:
        raise ValueError("At least one normalized Parquet source is required")
    return "[" + ",".join(_literal(g) for g in globs) + "]"


def load_h3(connection: duckdb.DuckDBPyConnection) -> None:
    """Load the DuckDB community H3 extension, installing it on first use."""
    connection.execute("INSTALL h3 FROM community")
    connection.execute("LOAD h3")


def grid_sql(parquet_glob: str | list[str], *, resolution: int, group_by: str = "sensor",
             start: date | None = None, end: date | None = None,
             bbox: tuple[float, float, float, float] | None = None) -> str:
    """Return daily aggregation SQL for one resolution; run on a connection with H3 loaded."""
    _validate((resolution,), group_by, start, end, bbox)
    globs = [parquet_glob] if isinstance(parquet_glob, str) else list(parquet_glob)
    keys = "sensor" if group_by == "sensor" else "sensor, platform"
    where = _filters(start, end, bbox)
    return f"""
        SELECT h3_latlng_to_cell(latitude, longitude, {resolution}) AS cell,
               acquisition_date AS period_start, {keys},
               count(*)::INTEGER AS detections, count(frp_mw)::INTEGER AS detections_with_frp,
               sum(frp_mw) AS frp_sum_mw, max(frp_mw) AS frp_max_mw
        FROM read_parquet({_source(globs)}, hive_partitioning=true, union_by_name=true)
        {'WHERE ' + ' AND '.join(where) if where else ''}
        GROUP BY ALL
    """


def _write_year(connection: duckdb.DuckDBPyConnection, cells_path: Path, year: int,
                resolutions: tuple[int, ...], keys: str) -> dict[int, tuple[int, int]]:
    finest = max(resolutions)
    counts = {}
    for res in resolutions:
        cell = "cell" if res == finest else f"h3_cell_to_parent(cell, {res})"
        folder = cells_path / f"resolution={res}" / f"period_year={year}"
        folder.mkdir(parents=True)
        connection.execute(f"""
            CREATE OR REPLACE TABLE level AS
            SELECT {cell} AS cell, period_start, {keys},
                   sum(detections)::INTEGER AS detections,
                   sum(detections_with_frp)::INTEGER AS detections_with_frp,
                   sum(frp_sum_mw) AS frp_sum_mw, max(frp_max_mw) AS frp_max_mw
            FROM base GROUP BY ALL""")
        counts[res] = connection.execute("SELECT count(*), sum(detections) FROM level").fetchone()
        connection.execute(f"COPY (SELECT * FROM level ORDER BY period_start, cell) TO "
                           f"{_literal(str(folder / 'data.parquet'))} "
                           f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
    return counts


def aggregate(datasets: list[str | Path], output_dir: str | Path, *,
              resolutions: tuple[int, ...] = DEFAULT_RESOLUTIONS, group_by: str = "sensor",
              start: date | None = None, end: date | None = None,
              bbox: tuple[float, float, float, float] | None = None,
              memory_limit: str = "1GB", threads: int = 2) -> dict:
    """Materialize the daily H3 pyramid from normalized datasets; publish atomically.

    Each year is aggregated once at the finest resolution and coarser levels are
    derived by H3 parent, so memory scales with one year, not the whole archive.
    """
    resolutions = tuple(sorted(resolutions))
    _validate(resolutions, group_by, start, end, bbox)
    roots = [Path(d).resolve(strict=True) for d in datasets]
    if not roots:
        raise ValueError("At least one normalized dataset is required")
    if threads < 1:
        raise ValueError("threads must be positive")
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(f"Output must not already exist: {target}")
    sources = []
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append({"path": str(root), "manifest_sha256": _sha256(manifest_path),
                        "schema_version": manifest["schema_version"],
                        "source_product": manifest["source"]["product"],
                        "accepted": manifest["counts"]["accepted"]})
    globs = [str(r / "accepted" / "**" / "*.parquet") for r in roots]
    keys = "sensor" if group_by == "sensor" else "sensor, platform"
    finest = max(resolutions)
    where = _filters(start, end, bbox)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        cells_path = stage / "cells"
        cells_path.mkdir()
        totals = {res: [0, 0] for res in resolutions}
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            connection.execute(f"SET memory_limit={_literal(memory_limit)}")
            connection.execute(f"SET threads={threads}")
            connection.execute(f"SET temp_directory={_literal(str(stage / 'spill'))}")
            connection.execute("SET TimeZone='UTC'")
            connection.execute("SET preserve_insertion_order=false")
            load_h3(connection)
            years = [row[0] for row in connection.execute(
                f"SELECT DISTINCT acquisition_year FROM read_parquet({_source(globs)}, hive_partitioning=true, "
                f"union_by_name=true) {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY 1").fetchall()]
            for year in years:
                year_start = max(start, date(year, 1, 1)) if start else date(year, 1, 1)
                year_end = min(end, date(year, 12, 31)) if end else date(year, 12, 31)
                year_sql = grid_sql(globs, resolution=finest, group_by=group_by,
                                    start=year_start, end=year_end, bbox=bbox)
                connection.execute(f"CREATE OR REPLACE TABLE base AS {year_sql}")
                for res, (cells, detections) in _write_year(connection, cells_path, year, resolutions, keys).items():
                    totals[res][0] += cells
                    totals[res][1] += detections
            if not years:
                empty_sql = grid_sql(globs, resolution=finest, group_by=group_by, start=start, end=end, bbox=bbox)
                connection.execute(f"COPY (SELECT * FROM ({empty_sql}) LIMIT 0) TO "
                                   f"{_literal(str(cells_path / 'empty.parquet'))} (FORMAT PARQUET)")
        (stage / "work.duckdb").unlink()
        shutil.rmtree(stage / "spill", ignore_errors=True)
        detections = totals[resolutions[0]][1]
        if any(total[1] != detections for total in totals.values()):
            raise RuntimeError(f"Detections differ across resolutions: {totals}")
        outputs = [{"path": str(p.relative_to(stage)), "size_bytes": p.stat().st_size, "sha256": _sha256(p)}
                   for p in sorted(stage.rglob("*.parquet"))]
        manifest = {
            "method_version": METHOD_VERSION, "engine": {"name": "duckdb", "version": duckdb.__version__},
            "parameters": {"grid": "h3", "resolutions": list(resolutions), "period": "day", "group_by": group_by,
                           "start": start.isoformat() if start else None, "end": end.isoformat() if end else None,
                           "bbox": list(bbox) if bbox else None, "crs": "EPSG:4326"},
            "resources": {"memory_limit": memory_limit, "threads": threads},
            "sources": sources,
            "counts": {"detections": detections,
                       "cells_by_resolution": {str(res): total[0] for res, total in totals.items()}},
            "outputs": outputs, "limitations": LIMITATIONS,
        }
        with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        if target.exists():
            raise FileExistsError(f"Output appeared during processing: {target}")
        os.rename(stage, target)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
