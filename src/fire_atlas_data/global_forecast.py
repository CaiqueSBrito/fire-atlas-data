"""Experimental calibrated global weekly forecasts on the historical cell domain."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from fire_atlas_data import global_features
from fire_atlas_data.normalization import literal, sha256
from fire_atlas_data.ranking import _json

SCHEMA = "global-weekly-forecast/1.0.0"
LIMITATIONS = [
    "Experimental probabilities; no operational capability is asserted.",
    "The target is any non-static multisensor detection during the next seven UTC dates, not fire ignition.",
    "Cells outside the historical eligible center-in-bbox domain are missing, not zero probability.",
    "Detection coverage and the static-source mask limit interpretation.",
]


def _manifest(root):
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _verify_outputs(root, manifest, required):
    entries = manifest.get("outputs", [])
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()) or str(relative) in seen:
            raise ValueError("Invalid or duplicate artifact path")
        seen.add(str(relative))
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {relative}")
    if not set(required).issubset(seen):
        raise ValueError("Required artifacts are absent from manifest")


def _bounds(bbox):
    if len(bbox) != 4:
        raise ValueError("bbox must contain west, south, east, north")
    west, south, east, north = map(float, bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("bbox coordinates must be finite")
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90) or west == east:
        raise ValueError("Invalid bbox bounds")
    return west, south, east, north


def _load_model(root, activity, issue_date):
    manifest = _manifest(root)
    if (manifest.get("schema") != "global-weekly-model/1.0.0"
            or manifest.get("feature_names") != list(global_features.FEATURE_NAMES)
            or manifest.get("horizon_days") != global_features.HORIZON_DAYS):
        raise ValueError("Incompatible model schema, features or horizon")
    if (manifest.get("feature_method_version") is not None
            and manifest["feature_method_version"] != global_features.METHOD_VERSION):
        raise ValueError("Model feature transformation version mismatch")
    mask = activity.get("mask", {}).get("manifest_sha256")
    if not mask or manifest.get("activity", {}).get("mask_manifest_sha256") != mask:
        raise ValueError("Model and activity mask identities differ")
    for field in ("training_cutoff_date", "data_cutoff_date", "model_cutoff_date"):
        if manifest.get(field) and date.fromisoformat(manifest[field]) > issue_date:
            raise ValueError("Model cutoff is later than the issue date")
    _verify_outputs(root, manifest, ("model.json", "calibrator.json"))
    calibration = json.loads((root / "calibrator.json").read_text(encoding="utf-8"))
    x = np.asarray(calibration.get("x", []), dtype=float)
    y = np.asarray(calibration.get("y", []), dtype=float)
    if (calibration.get("method") != "isotonic" or x.ndim != 1 or y.ndim != 1
            or not len(x) or len(x) != len(y) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or (np.diff(x) <= 0).any()
            or (np.diff(y) < 0).any() or (x < 0).any() or (x > 1).any()
            or (y < 0).any() or (y > 1).any()):
        raise ValueError("Invalid isotonic calibration knots")
    model = xgb.Booster()
    model.load_model(root / "model.json")
    config = json.loads(model.save_config())
    if (model.feature_names != list(global_features.FEATURE_NAMES)
            or model.num_features() != len(global_features.FEATURE_NAMES)
            or config["learner"]["objective"]["name"] != "binary:logistic"):
        raise ValueError("Native model feature schema or objective mismatch")
    return manifest, model, x, y


def _predict_batches(features_path, output_path, model, x, y):
    rows = 0
    writer = None
    try:
        for batch in pq.ParquetFile(features_path).iter_batches(batch_size=65536):
            table = pa.Table.from_batches([batch])
            values = np.column_stack([table[name].to_numpy(zero_copy_only=False)
                                      for name in global_features.FEATURE_NAMES])
            raw = model.predict(xgb.DMatrix(values, feature_names=list(global_features.FEATURE_NAMES)))
            if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
                raise ValueError("Model produced invalid probabilities")
            result = table.select(["cell_id", "issue_date", "target_start_date", "target_end_date", "horizon_days"])
            result = result.append_column("raw_probability", pa.array(raw, type=pa.float64()))
            result = result.append_column("probability", pa.array(np.interp(raw, x, y), type=pa.float64()))
            if writer is None:
                writer = pq.ParquetWriter(output_path, result.schema, compression="zstd")
            writer.write_table(result)
            rows += len(result)
    finally:
        if writer is not None:
            writer.close()
    return rows


def predict(activity_dir, model_dir, output_dir, *, bbox, issue_date, memory_limit="2GB", threads=2):
    """Publish an experimental forecast without refitting the model or calibrator."""
    bounds = _bounds(bbox)
    issue_date = date.fromisoformat(issue_date) if isinstance(issue_date, str) else issue_date
    if type(issue_date) is not date or type(threads) is not int or threads < 1:
        raise ValueError("A UTC issue date and positive integer threads are required")
    activity_root, model_root = Path(activity_dir).resolve(strict=True), Path(model_dir).resolve(strict=True)
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    activity = _manifest(activity_root)
    if activity.get("schema") != "global-cell-activity/1.0.0" or activity.get("h3_resolution") != 5:
        raise ValueError("Incompatible activity schema or H3 resolution")
    _verify_outputs(activity_root, activity, ("cells.parquet",))
    model_manifest, model, x, y = _load_model(model_root, activity, issue_date)
    model.set_param({"nthread": threads})
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        west, south, east, north = bounds
        longitude = f"longitude BETWEEN {west} AND {east}" if west < east else f"(longitude >= {west} OR longitude <= {east})"
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            connection.execute(f"SET memory_limit={literal(memory_limit)}")
            connection.execute(f"SET threads={threads}")
            connection.execute(f"SET temp_directory={literal(str(stage / 'spill'))}")
            connection.execute(f"CREATE TABLE keys AS SELECT cell_id, DATE {literal(issue_date.isoformat())} AS issue_date "
                               f"FROM read_parquet({literal(str(activity_root / 'cells.parquet'))}) "
                               f"WHERE first_detection_date <= DATE {literal(issue_date.isoformat())} "
                               f"AND latitude BETWEEN {south} AND {north} AND {longitude}")
            count, unique = connection.execute("SELECT count(*), count(DISTINCT cell_id) FROM keys").fetchone()
            if not count or count != unique:
                raise ValueError("Empty or duplicate eligible forecast domain")
            connection.execute(f"COPY (SELECT * FROM keys ORDER BY cell_id) TO {literal(str(stage / 'keys.parquet'))} (FORMAT PARQUET)")
        feature_manifest = global_features.build(activity_root, stage / "keys.parquet", stage / "features",
                                                 labels=False, memory_limit=memory_limit, threads=threads)
        if (feature_manifest.get("schema") != "global-weekly-features/1.0.0"
                or feature_manifest.get("feature_names") != list(global_features.FEATURE_NAMES)
                or feature_manifest.get("feature_method_version") != global_features.METHOD_VERSION
                or feature_manifest.get("labels") is not False
                or feature_manifest.get("rows") != count
                or feature_manifest.get("activity", {}).get("manifest_sha256") != sha256(activity_root / "manifest.json")
                or feature_manifest.get("activity", {}).get("mask_manifest_sha256") != activity["mask"]["manifest_sha256"]):
            raise ValueError("Feature contract mismatch")
        _verify_outputs(stage / "features", feature_manifest, ("features.parquet",))
        features_path = stage / "features" / "features.parquet"
        expected = ["cell_id", "issue_date", "target_start_date", "target_end_date", "horizon_days", *global_features.FEATURE_NAMES]
        schema = pq.read_schema(features_path)
        expected_types = [pa.string(), pa.date32(), pa.date32(), pa.date32(), pa.int32()]
        expected_types.extend([pa.float64()] * len(global_features.FEATURE_NAMES))
        if schema.names != expected or schema.types != expected_types:
            raise ValueError("Feature parquet schema mismatch")
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            bad = connection.execute(f"SELECT count(*) FROM read_parquet({literal(str(features_path))}) f "
                                     "FULL JOIN keys k USING (cell_id, issue_date) WHERE f.cell_id IS NULL OR k.cell_id IS NULL "
                                     f"OR horizon_days IS DISTINCT FROM 7 OR target_start_date IS DISTINCT FROM DATE {literal((issue_date + timedelta(days=1)).isoformat())} "
                                     f"OR target_end_date IS DISTINCT FROM DATE {literal((issue_date + timedelta(days=7)).isoformat())}").fetchone()[0]
            if bad:
                raise ValueError("Feature keys or target horizon mismatch")
        rows = _predict_batches(features_path, stage / "predictions.parquet", model, x, y)
        if rows != count:
            raise ValueError("Prediction row count mismatch")
        feature_hash = sha256(stage / "features" / "manifest.json")
        shutil.copyfile(stage / "features" / "manifest.json", stage / "features-manifest.json")
        shutil.rmtree(stage / "features")
        (stage / "work.duckdb").unlink()
        shutil.rmtree(stage / "spill", ignore_errors=True)
        outputs = [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in (stage / "keys.parquet", stage / "predictions.parquet", stage / "features-manifest.json")]
        manifest = {"schema": SCHEMA, "status": "experimental", "h3_resolution": 5,
                    "feature_names": list(global_features.FEATURE_NAMES), "horizon_days": 7,
                    "feature_method_version": global_features.METHOD_VERSION,
                    "parameters": {"bbox": list(bounds), "issue_date": issue_date.isoformat()},
                    "domain": {"selection": "historically observed cell centers inside bbox, first_detection_date <= issue_date",
                               "outside_domain": "missing, not zero", "antimeridian": west > east},
                    "activity": {"path": str(activity_root), "manifest_sha256": sha256(activity_root / "manifest.json"),
                                 "mask_manifest_sha256": activity["mask"]["manifest_sha256"]},
                    "model": {"path": str(model_root), "manifest_sha256": sha256(model_root / "manifest.json"),
                              "schema": model_manifest["schema"]},
                    "features_manifest_sha256": feature_hash,
                    "resources": {"memory_limit": memory_limit, "threads": threads},
                    "counts": {"eligible_cells": count, "predictions": rows},
                    "outputs": outputs, "limitations": LIMITATIONS}
        _json(stage / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage, target)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--issue-date", required=True, type=date.fromisoformat)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args(argv)
    predict(args.activity, args.model, args.output, bbox=args.bbox, issue_date=args.issue_date,
            memory_limit=args.memory_limit, threads=args.threads)


if __name__ == "__main__":
    main()
