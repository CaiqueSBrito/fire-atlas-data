"""Daily inference of the RF10/CA08 XGBoost model: P(any detection in [D cutoff, D+7 cutoff)) per H3 res6 cell.

Inputs come from `ml_dataset.features_sql`, the same SQL as the training rows, over silver detections acquired
before the D cutoff only. The universe is the training one: res6 cells with a non-offshore detection before the
cutoff. Each run replaces the rows of one `issue_date` in a Delta table partitioned by year/month; the commit
records the model hash, silver version and newest silver record, so every published probability is traceable.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from deltalake import CommitProperties, DeltaTable, write_deltalake
import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import xgboost as xgb

from . import ml_dataset
from .ml_dataset import FEATURES

METHOD_VERSION = "1.0.0"
PARTITIONS = ["year", "month"]
LIMITATIONS = [
    "Probability of at least one FIRMS detection in the res6 cell within 7 days, not of a fire event, its spread "
    "or an ignition location; clouds can hide fire.",
    "Cells never detected before the cutoff are outside the universe: no forecast, which is not a zero risk.",
    "The last input window may be partial because of NRT latency; `input_lag_hours` records the gap.",
    "No acceptance thresholds were approved (CA08): probabilities are experimental.",
]


def _read_bytes(path: str | Path) -> bytes:
    """Local path or URI with a scheme (e.g. s3://bucket/key, credentials from the standard AWS chain)."""
    if "://" not in str(path):
        return Path(path).read_bytes()
    filesystem, key = pafs.FileSystem.from_uri(str(path))
    with filesystem.open_input_stream(key) as stream:
        return stream.read()


def _load_model(model: str | Path, report: str | Path | None, threads: int) -> tuple[xgb.Booster, dict]:
    """Booster on CPU plus the checks that tie it to the current feature SQL."""
    raw = _read_bytes(model)
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    booster.set_param({"device": "cpu", "nthread": threads})
    if booster.num_features() != len(FEATURES) or booster.feature_names not in (None, FEATURES):
        raise ValueError(f"Model expects {booster.num_features()} features {booster.feature_names}, not {FEATURES}")
    info = {"model_sha256": hashlib.sha256(raw).hexdigest(), "model_method_version": None}
    if report is not None:
        trained = json.loads(_read_bytes(report).decode("utf-8"))
        if trained.get("features") != FEATURES:
            raise ValueError(f"Model report features {trained.get('features')} differ from {FEATURES}")
        if trained.get("dataset_method_version") != ml_dataset.METHOD_VERSION:
            raise ValueError(f"Model trained on dataset method {trained.get('dataset_method_version')}, "
                             f"features are method {ml_dataset.METHOD_VERSION}")
        info["model_method_version"] = trained.get("method_version")
    return booster, info


def forecast(silver: str, model: str | Path, target: str, *, issue_date: date, report: str | Path | None = None,
             cutoff_hour: int = 9, max_input_lag_hours: float = 6.0,
             storage_options: dict[str, str] | None = None, memory_limit: str = "3GB", threads: int = 4,
             temp_dir: str | Path | None = None) -> dict:
    """Score every universe cell for `issue_date` and replace that day's rows in the Delta table at `target`.

    Refuses when the newest silver record is older than `max_input_lag_hours` before the cutoff (stale inputs).
    """
    booster, info = _load_model(model, report, threads)
    cutoff = datetime(issue_date.year, issue_date.month, issue_date.day, cutoff_hour, tzinfo=timezone.utc)
    silver_version = DeltaTable(silver, storage_options=storage_options).version()
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        silver_last = connection.execute("SELECT max(acquired_at) FROM silver").fetchone()[0]
        if silver_last is None or silver_last < cutoff - timedelta(hours=max_input_lag_hours):
            raise ValueError(f"Silver ends at {silver_last}; inputs for {issue_date} need data up to {cutoff} "
                             f"(max lag {max_input_lag_hours} h)")
        connection.execute(ml_dataset._universe_sql(cutoff_hour))
        connection.execute(ml_dataset.features_sql(issue_date, cutoff_hour))
        columns = ", ".join(f"coalesce({f}::DOUBLE, 'NaN'::DOUBLE) AS {f}" for f in FEATURES)
        arrays = connection.execute(f"SELECT c6, c1, active, {columns} FROM features ORDER BY c6").fetchnumpy()
    if not len(arrays["c6"]):
        raise ValueError(f"No universe cell has a detection before {cutoff}")
    # Same matrix as training: float32 with NULL means as NaN (missing).
    x = np.column_stack([np.asarray(arrays[f], dtype=np.float32) for f in FEATURES])
    probability = np.asarray(booster.inplace_predict(x), dtype=np.float32)
    rows = len(probability)
    table = pa.table({
        "issue_date": pa.array([issue_date] * rows, pa.date32()),
        "h3_r6": pa.array(arrays["c6"], pa.int64()),
        "h3_r1": pa.array(arrays["c1"], pa.int64()),
        **{f: pa.array(arrays[f], pa.float64(), from_pandas=True) for f in FEATURES},
        "active": pa.array(arrays["active"], pa.bool_()),
        "prob_fire_7d": pa.array(probability, pa.float32()),
        "year": pa.array([issue_date.year] * rows, pa.int32()),
        "month": pa.array([issue_date.month] * rows, pa.int32()),
    })
    lag_hours = round((cutoff - silver_last).total_seconds() / 3600, 2)
    metadata = {"method_version": METHOD_VERSION, "dataset_method_version": ml_dataset.METHOD_VERSION,
                "model_sha256": info["model_sha256"], "model_method_version": str(info["model_method_version"]),
                "silver_version": str(silver_version), "silver_last": silver_last.isoformat(),
                "cutoff_hour": str(cutoff_hour), "issue_date": issue_date.isoformat(),
                "input_lag_hours": str(lag_hours)}
    exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
    write_deltalake(target, table, mode="overwrite", partition_by=PARTITIONS,
                    predicate=f"issue_date = '{issue_date.isoformat()}'" if exists else None,
                    storage_options=storage_options, commit_properties=CommitProperties(custom_metadata=metadata))
    active = arrays["active"].astype(bool)
    counts = {"issue_date": issue_date.isoformat(), "cells": rows, "active_cells": int(active.sum()),
              "mean_prob_active": float(probability[active].mean()) if active.any() else None,
              "input_lag_hours": lag_hours}
    return {**metadata, "target": target, "counts": counts, "limitations": LIMITATIONS}
