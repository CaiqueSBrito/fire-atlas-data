"""Build a frozen, explicitly parameterized fine-resolution static-source mask."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3

from .normalization import identifier, literal, sha256
from .ranking import _json

SCHEMA = "static-source-mask/1.0.0"
LIMITATIONS = [
    "This is an exclusion proxy, not a validated inventory of static sources.",
    "Recurrence uses distinct acquisition years of FIRMS type 2 through reference_end only.",
    "External coordinates have caller-declared provenance and temporal validity; neither is independently verified.",
    "External points declare knowledge through reference_end; file-year labels do not establish historical availability.",
    "ORNL flare positions labeled 2012 in the 2012-2016 catalog used all 2012-2016 observations and are not causal 2012 inputs.",
    "Resolution, recurrence threshold and buffer are explicit operator choices, not scientific defaults.",
    "A missing mask cell does not establish vegetation fire or absence of a static source.",
    "Freeze reference_end on or before every issue date using this mask to avoid future leakage.",
]


def _configure(connection, stage: Path, memory_limit: str, threads: int) -> None:
    connection.execute(f"SET memory_limit={literal(memory_limit)}")
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET temp_directory={literal(str(stage / 'spill'))}")
    connection.execute("SET TimeZone='UTC'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET partitioned_write_max_open_files=2")
    connection.execute("SET partitioned_write_flush_threshold=16384")
    connection.create_function("h3_cell", h3.latlng_to_cell,
                               ["DOUBLE", "DOUBLE", "INTEGER"], "VARCHAR")
    connection.create_function("h3_disk", _disk, ["VARCHAR", "INTEGER"], "VARCHAR[]")


def _disk(cell: str, rings: int) -> list[str]:
    return sorted(h3.grid_disk(cell, rings))


def _target(output_dir, threads: int) -> Path:
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    target = Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output must not already exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _normalized_sources(datasets) -> tuple[list[dict], list[str]]:
    sources, queries = [], []
    seen_hashes = set()
    for index, dataset in enumerate(datasets):
        root = Path(dataset).resolve(strict=True)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.3.0":
            raise ValueError(f"Normalized schema 1.3.0 required: {root}")
        source_hash = manifest["source"]["sha256"]
        if source_hash in seen_hashes:
            raise ValueError(f"Duplicate normalized source hash: {source_hash}")
        seen_hashes.add(source_hash)
        outputs = []
        for entry in manifest["outputs"]:
            relative = Path(entry["path"])
            if not relative.parts or relative.parts[0] != "accepted":
                continue
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or path.suffix != ".parquet":
                raise ValueError(f"Invalid accepted output path: {relative}")
            digest = sha256(path)
            if digest != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Normalized output hash/size mismatch: {path}")
            outputs.append({"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
        if not outputs:
            raise ValueError(f"No accepted Parquet outputs in manifest: {root}")
        sources.append({"path": str(root), "manifest_sha256": sha256(manifest_path),
                        "source_sha256": source_hash, "schema_version": "1.3.0",
                        "source_product": manifest["source"]["product"], "outputs": outputs})
        paths = "[" + ",".join(literal(p["path"]) for p in outputs) + "]"
        queries.append(f"SELECT {index}::INTEGER AS dataset_id, schema_version, source_sha256, "
                       "latitude, longitude, acquisition_date, acquired_at, sensor, platform, "
                       f"hotspot_type, frp_mw FROM read_parquet({paths}, hive_partitioning=true)")
    return sources, queries


def _observations(connection, queries: list[str], sources: list[dict]) -> None:
    if queries:
        connection.execute("CREATE VIEW observations AS " + " UNION ALL ".join(queries))
    else:
        connection.execute("""CREATE VIEW observations AS SELECT NULL::INTEGER AS dataset_id,
            NULL::VARCHAR AS schema_version, NULL::VARCHAR AS source_sha256,
            NULL::DOUBLE AS latitude, NULL::DOUBLE AS longitude,
            NULL::DATE AS acquisition_date, NULL::TIMESTAMPTZ AS acquired_at,
            NULL::VARCHAR AS sensor, NULL::VARCHAR AS platform,
            NULL::SMALLINT AS hotspot_type, NULL::DOUBLE AS frp_mw WHERE false""")
    # Validate source identity inside Parquet too, without loading observations into Python.
    for index, source in enumerate(sources):
        invalid = connection.execute(f"""SELECT count(*) FROM observations
            WHERE dataset_id={index} AND (schema_version IS DISTINCT FROM '1.3.0'
            OR source_sha256 IS DISTINCT FROM {literal(source['source_sha256'])})""").fetchone()[0]
        if invalid:
            raise ValueError("Normalized row identity does not match its manifest")
    connection.execute("""CREATE TABLE streams AS SELECT dataset_id, sensor, platform,
        min(acquired_at) AS start_time, max(acquired_at) AS end_time
        FROM observations GROUP BY dataset_id, sensor, platform""")
    overlap = connection.execute("""SELECT count(*) FROM streams a JOIN streams b
        ON a.dataset_id < b.dataset_id AND a.sensor=b.sensor AND a.platform=b.platform
        AND a.start_time <= b.end_time AND b.start_time <= a.end_time""").fetchone()[0]
    if overlap:
        raise ValueError("Overlapping same-platform stream acquisition intervals")


def _copy(connection, query: str, path: Path) -> None:
    connection.execute(f"COPY ({query}) TO {literal(str(path))} "
                       "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)")


def _publish(stage: Path, target: Path, manifest: dict) -> dict:
    (stage / "work.duckdb").unlink()
    shutil.rmtree(stage / "spill", ignore_errors=True)
    manifest["outputs"] = [{"path": str(p.relative_to(stage)), "size_bytes": p.stat().st_size,
                            "sha256": sha256(p)} for p in sorted(stage.rglob("*.parquet"))]
    _json(stage / "manifest.json", manifest)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output appeared during processing: {target}")
    os.rename(stage, target)
    return manifest


def _external(connection, external_points, resolution: int) -> dict | None:
    if external_points is None:
        return None
    path = Path(external_points).resolve(strict=True)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(set(header)) != len(header) or len({h.lower() for h in header}) != len(header):
        raise ValueError("External CSV requires unique column names")
    latitude = "latitude" if "latitude" in header else "lat"
    longitude = "longitude" if "longitude" in header else "lon"
    if latitude not in header or longitude not in header:
        raise ValueError("External CSV requires latitude/longitude (lat/lon aliases allowed)")
    digest = sha256(path)
    columns = ",".join(f"{literal(name)}:'VARCHAR'" for name in header)
    connection.execute(f"""CREATE TABLE external_points AS SELECT
        try_cast({identifier(latitude)} AS DOUBLE) AS latitude,
        try_cast({identifier(longitude)} AS DOUBLE) AS longitude
        FROM read_csv({literal(str(path))}, header=true, auto_detect=false,
            columns={{{columns}}}, strict_mode=true, ignore_errors=false)""")
    invalid = connection.execute("""SELECT count(*) FROM external_points WHERE
        latitude IS NULL OR longitude IS NULL OR NOT isfinite(latitude) OR NOT isfinite(longitude)
        OR latitude NOT BETWEEN -90 AND 90 OR longitude NOT BETWEEN -180 AND 180""").fetchone()[0]
    if invalid:
        raise ValueError("External CSV contains invalid coordinates")
    connection.execute(f"INSERT INTO seeds SELECT DISTINCT h3_cell(latitude, longitude, {resolution}), "
                       "0::INTEGER FROM external_points")
    if sha256(path) != digest:
        raise RuntimeError("External CSV changed while reading")
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size,
            "kind": "external_points", "temporal_validity": "caller_declared_not_verified"}


def build(datasets, external_points, output_dir, *, reference_end: date,
          min_recurrent_years: int, mask_resolution: int, buffer_rings: int,
          memory_limit="2GB", threads=2) -> dict:
    """Publish a fine H3 mask; no scientific parameter is implicit."""
    if not isinstance(reference_end, date) or type(reference_end) is not date:
        raise ValueError("reference_end must be a date")
    for name, value, minimum, maximum in (("min_recurrent_years", min_recurrent_years, 1, None),
                                          ("mask_resolution", mask_resolution, 6, 15),
                                          ("buffer_rings", buffer_rings, 0, None)):
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"Invalid {name}")
    target = _target(output_dir, threads)
    sources, queries = _normalized_sources(datasets)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            _configure(connection, stage, memory_limit, threads)
            _observations(connection, queries, sources)
            connection.execute("CREATE TABLE seeds(cell_id VARCHAR, priority INTEGER)")
            connection.execute(f"""INSERT INTO seeds SELECT
                h3_cell(latitude, longitude, {mask_resolution}) AS cell_id, 1::INTEGER
                FROM observations WHERE hotspot_type=2
                AND acquisition_date <= DATE {literal(reference_end.isoformat())}
                GROUP BY cell_id HAVING count(DISTINCT year(acquisition_date)) >= {min_recurrent_years}""")
            external = _external(connection, external_points, mask_resolution)
            if external is not None:
                external["declared_knowledge_cutoff"] = reference_end.isoformat()
                sources.append(external)
            connection.execute(f"""CREATE TABLE mask AS WITH candidates AS (
                SELECT cell_id, priority FROM seeds UNION ALL
                SELECT unnest(h3_disk(cell_id, {buffer_rings})), 2::INTEGER FROM (SELECT DISTINCT cell_id FROM seeds)
                ) SELECT cell_id, CASE min(priority) WHEN 0 THEN 'external'
                WHEN 1 THEN 'recurrent_type' ELSE 'buffer' END AS origin
                FROM candidates GROUP BY cell_id""")
            counts = dict(connection.execute("SELECT origin, count(*) FROM mask GROUP BY origin").fetchall())
            _copy(connection, "SELECT * FROM mask ORDER BY cell_id", stage / "mask.parquet")
        return _publish(stage, target, {
            "schema": SCHEMA, "parameters": {"reference_end": reference_end.isoformat(),
                "min_recurrent_years": min_recurrent_years, "mask_resolution": mask_resolution,
                "buffer_rings": buffer_rings}, "sources": sources,
            "resources": {"memory_limit": memory_limit, "threads": threads},
            "engine": {"duckdb": duckdb.__version__, "h3": h3.__version__},
            "counts": {"cells": sum(counts.values()), "origins": counts}, "limitations": LIMITATIONS})
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--external-points")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-end", type=date.fromisoformat, required=True)
    parser.add_argument("--min-recurrent-years", type=int, required=True)
    parser.add_argument("--mask-resolution", type=int, required=True)
    parser.add_argument("--buffer-rings", type=int, required=True)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    print(json.dumps(build(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
