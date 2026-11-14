"""Aggregate accepted multisensor observations into global H3 resolution-5 activity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import duckdb
import h3

from .normalization import literal, sha256
from .static_mask import (
    SCHEMA as MASK_SCHEMA, _configure, _copy, _normalized_sources, _observations,
    _publish, _target,
)

SCHEMA = "global-cell-activity/1.0.0"
H3_RESOLUTION = 5
LIMITATIONS = [
    "Activity is an observation proxy, not an ignition, event, burned area or physical fire-negative label.",
    "A non-static proxy requires hotspot_type NULL or 0 and coordinates outside the fine mask; NULL type is unknown, not verified vegetation fire.",
    "Static detections still establish cells, first detection dates and platform reporting days.",
    "A static fine cell never removes other fine cells in the same resolution-5 cell.",
    "Platform days indicate observed reporting, not complete spatial coverage or guaranteed satellite operation.",
    "Absent cell-days are unknown, not confirmed absence of fire; missing FRP is identified by zero FRP rows.",
    "Sensor-family counts are not harmonized; different platforms may observe the same phenomenon.",
    "Duplicate raw-source hashes and overlapping acquisition intervals within the same sensor/platform are rejected, including SP versus NRT overlap.",
    "Acquisition dates are UTC; unknown availability time prevents a claim of operational availability.",
    "The mask is frozen: downstream issue dates must not precede its reference_end.",
]


def _center(cell: str) -> list[float]:
    return list(h3.cell_to_latlng(cell))


def _mask_resolution(cell: str) -> int:
    return h3.get_resolution(cell) if h3.is_valid_cell(cell) else -1


def _mask_source(mask_dir) -> tuple[Path, dict, str]:
    root = Path(mask_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MASK_SCHEMA:
        raise ValueError("Expected static-source-mask/1.0.0 manifest")
    resolution = manifest["parameters"]["mask_resolution"]
    if type(resolution) is not int or not 6 <= resolution <= 15:
        raise ValueError("Mask resolution must be an integer from 6 through 15")
    outputs = [entry for entry in manifest["outputs"] if entry["path"] == "mask.parquet"]
    if len(outputs) != 1:
        raise ValueError("Mask manifest must identify mask.parquet exactly once")
    path = root / "mask.parquet"
    if sha256(path) != outputs[0]["sha256"] or path.stat().st_size != outputs[0]["size_bytes"]:
        raise ValueError("Mask output hash/size mismatch")
    return path, manifest, sha256(manifest_path)


def build(datasets, mask_dir, output_dir, *, memory_limit="2GB", threads=2) -> dict:
    """Publish all accepted cells and non-static daily family counts atomically."""
    target = _target(output_dir, threads)
    sources, queries = _normalized_sources(datasets)
    if not queries:
        raise ValueError("At least one normalized dataset is required")
    mask_path, mask_manifest, mask_hash = _mask_source(mask_dir)
    resolution = mask_manifest["parameters"]["mask_resolution"]
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            _configure(connection, stage, memory_limit, threads)
            connection.create_function("h3_center", _center, ["VARCHAR"], "DOUBLE[]")
            connection.create_function("h3_parent", h3.cell_to_parent, ["VARCHAR", "INTEGER"], "VARCHAR")
            connection.create_function("mask_resolution", _mask_resolution, ["VARCHAR"], "INTEGER")
            _observations(connection, queries, sources)
            connection.execute(f"CREATE TABLE mask AS SELECT * FROM read_parquet({literal(str(mask_path))})")
            invalid_mask = connection.execute(f"""SELECT count(*) FROM mask WHERE cell_id IS NULL
                OR mask_resolution(cell_id) <> {resolution} OR origin IS NULL
                OR origin NOT IN ('external','recurrent_type','buffer')""").fetchone()[0]
            duplicate_mask = connection.execute("SELECT count(*)-count(DISTINCT cell_id) FROM mask").fetchone()[0]
            if invalid_mask or duplicate_mask:
                raise ValueError("Mask contains invalid or duplicate cells/origins")
            connection.execute(f"""CREATE TABLE mapped AS SELECT
                h3_cell(latitude, longitude, 5) AS cell_id,
                h3_cell(latitude, longitude, {resolution}) AS mask_cell_id,
                acquisition_date, sensor, platform, hotspot_type, frp_mw FROM observations""")
            accepted, start, end = connection.execute("""SELECT count(*), min(acquisition_date),
                max(acquisition_date) FROM mapped""").fetchone()
            if not accepted:
                raise ValueError("Activity requires at least one accepted observation for acquisition support")
            connection.execute("""CREATE TABLE cells AS WITH first_dates AS (
                SELECT cell_id, min(acquisition_date) AS first_detection_date FROM mapped GROUP BY cell_id
                ), centers AS (SELECT *, h3_center(cell_id) AS center FROM first_dates)
                SELECT cell_id, center[1]::DOUBLE AS latitude, center[2]::DOUBLE AS longitude,
                first_detection_date, h3_parent(cell_id, 4) AS parent4_id,
                h3_parent(cell_id, 3) AS parent3_id FROM centers""")
            connection.execute("""CREATE TABLE neighbors AS SELECT cell_id, neighbor_id FROM (
                SELECT cell_id, unnest(h3_disk(cell_id, 1)) AS neighbor_id FROM cells
                ) WHERE cell_id <> neighbor_id""")
            connection.execute("""CREATE TABLE platform_days AS SELECT sensor, platform,
                acquisition_date, count(*)::BIGINT AS detections FROM mapped GROUP BY ALL""")
            connection.execute("""CREATE TABLE nonstatic AS SELECT m.cell_id, m.acquisition_date,
                m.sensor, m.frp_mw FROM mapped m ANTI JOIN mask s ON m.mask_cell_id=s.cell_id
                WHERE m.hotspot_type IS NULL OR m.hotspot_type=0""")
            connection.execute("""CREATE TABLE daily AS SELECT cell_id, acquisition_date,
                count(*) FILTER (WHERE sensor='VIIRS')::BIGINT AS viirs_detections,
                count(*) FILTER (WHERE sensor='MODIS')::BIGINT AS modis_detections,
                coalesce(sum(frp_mw) FILTER (WHERE sensor='VIIRS'), 0)::DOUBLE AS viirs_frp_sum_mw,
                count(frp_mw) FILTER (WHERE sensor='VIIRS')::BIGINT AS viirs_frp_rows,
                coalesce(sum(frp_mw) FILTER (WHERE sensor='MODIS'), 0)::DOUBLE AS modis_frp_sum_mw,
                count(frp_mw) FILTER (WHERE sensor='MODIS')::BIGINT AS modis_frp_rows,
                year(acquisition_date)::INTEGER AS year FROM nonstatic GROUP BY ALL""")
            counts = {"accepted_detections": accepted}
            for table in ("cells", "neighbors", "platform_days", "daily", "nonstatic"):
                counts[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            counts["excluded_static_detections"] = accepted - counts["nonstatic"]
            for table, order in (("cells", "cell_id"), ("neighbors", "cell_id, neighbor_id"),
                                 ("platform_days", "acquisition_date, sensor, platform")):
                _copy(connection, f"SELECT * FROM {table} ORDER BY {order}", stage / f"{table}.parquet")
            daily_path = stage / "daily"
            if counts["daily"]:
                connection.execute(f"COPY (SELECT * FROM daily ORDER BY acquisition_date, cell_id) "
                                   f"TO {literal(str(daily_path))} (FORMAT PARQUET, COMPRESSION ZSTD, "
                                   "PARTITION_BY(year), ROW_GROUP_SIZE 16384, FILE_SIZE_BYTES '128MB')")
            else:
                daily_path.mkdir()
                _copy(connection, "SELECT * FROM daily", daily_path / "empty.parquet")
        return _publish(stage, target, {
            "schema": SCHEMA, "h3_resolution": H3_RESOLUTION,
            "acquisition_range": {"min": start.isoformat(), "max": end.isoformat()},
            "mask": {"manifest_sha256": mask_hash, "parameters": mask_manifest["parameters"],
                     "path": str(mask_path.parent)},
            "sources": sources, "counts": counts,
            "resources": {"memory_limit": memory_limit, "threads": threads},
            "engine": {"duckdb": duckdb.__version__, "h3": h3.__version__}, "limitations": LIMITATIONS})
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    print(json.dumps(build(**vars(parser.parse_args(argv))), indent=2))


if __name__ == "__main__":
    main()
