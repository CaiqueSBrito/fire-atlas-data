"""Synthetic technical checks only; these fixtures do not validate scientific skill."""
from datetime import date
import json

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from fire_atlas_data import global_features
from fire_atlas_data.global_forecast import predict
from fire_atlas_data.normalization import sha256


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _outputs(root):
    return [{"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(root.rglob("*.parquet"))]


def _fixture(tmp_path):
    activity = tmp_path / "activity"
    activity.mkdir()
    cells = [h3.latlng_to_cell(0, lon, 5) for lon in (0, 4, 179, -179, 20)]
    centers = [h3.cell_to_latlng(c) for c in cells]
    pq.write_table(pa.table({"cell_id": cells, "latitude": [c[0] for c in centers],
                            "longitude": [c[1] for c in centers],
                            "first_detection_date": [date(2012, 1, 1)] * 4 + [date(2021, 1, 1)],
                            "parent4_id": [h3.cell_to_parent(c, 4) for c in cells],
                            "parent3_id": [h3.cell_to_parent(c, 3) for c in cells]}), activity / "cells.parquet")
    pairs = [(c, n) for c in cells for n in sorted(h3.grid_disk(c, 1)) if n != c]
    pq.write_table(pa.table({"cell_id": [p[0] for p in pairs], "neighbor_id": [p[1] for p in pairs]}), activity / "neighbors.parquet")
    daily = activity / "daily" / "year=2020"
    daily.mkdir(parents=True)
    pq.write_table(pa.table({"cell_id": cells[:4], "acquisition_date": [date(2020, 1, 1)] * 4,
                            "viirs_detections": [1] * 4, "modis_detections": [1] * 4,
                            "viirs_frp_sum_mw": [2.] * 4, "viirs_frp_rows": [1] * 4,
                            "modis_frp_sum_mw": [2.] * 4, "modis_frp_rows": [1] * 4}), daily / "part.parquet")
    pq.write_table(pa.table({"sensor": ["VIIRS", "MODIS"], "platform": ["SNPP", "Terra"],
                            "acquisition_date": [date(2020, 1, 1)] * 2, "detections": [4, 4]}), activity / "platform_days.parquet")
    _json(activity / "manifest.json", {"schema": "global-cell-activity/1.0.0", "h3_resolution": 5,
          "acquisition_range": {"min": "2012-01-01", "max": "2021-01-01"},
          "mask": {"manifest_sha256": "a" * 64, "parameters": {"reference_end": "2011-12-31"}},
          "outputs": _outputs(activity), "sources": [], "counts": {}, "limitations": []})
    model = tmp_path / "model"
    model.mkdir()
    values = np.zeros((4, len(global_features.FEATURE_NAMES)))
    values[2:] = 1
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 1, "min_child_weight": 0, "nthread": 1},
                        xgb.DMatrix(values, label=[0, 0, 1, 1], feature_names=list(global_features.FEATURE_NAMES)), num_boost_round=2)
    booster.save_model(model / "model.json")
    _json(model / "calibrator.json", {"method": "isotonic", "x": [0., .5, 1.], "y": [.1, .2, .9]})
    artifacts = [{"path": name, "size_bytes": (model / name).stat().st_size, "sha256": sha256(model / name)}
                 for name in ("model.json", "calibrator.json")]
    _json(model / "manifest.json", {"schema": "global-weekly-model/1.0.0", "feature_names": list(global_features.FEATURE_NAMES),
          "horizon_days": 7, "activity": {"mask_manifest_sha256": "a" * 64}, "outputs": artifacts,
          "data_cutoff_date": "2019-12-31", "sustained_improvement": False})
    return activity, model, cells


def _run(activity, model, output, bbox):
    return predict(activity, model, output, bbox=bbox, issue_date=date(2020, 1, 1))


def test_bbox_invariance_domain_and_calibration(tmp_path):
    activity, model, cells = _fixture(tmp_path)
    wide = tmp_path / "wide"
    small = tmp_path / "small"
    manifest = _run(activity, model, wide, (-30, -5, 30, 5))
    _run(activity, model, small, (-1, -1, 1, 1))
    broad = pq.read_table(wide / "predictions.parquet").to_pylist()
    narrow = pq.read_table(small / "predictions.parquet").to_pylist()
    assert len(narrow) == 1
    assert narrow[0] == next(row for row in broad if row["cell_id"] == cells[0])
    assert {row["cell_id"] for row in broad} == set(cells[:2])
    assert cells[4] not in {row["cell_id"] for row in broad}
    for row in broad:
        assert row["probability"] == pytest.approx(np.interp(row["raw_probability"], [0, .5, 1], [.1, .2, .9]))
        assert row["target_start_date"] == date(2020, 1, 2)
        assert row["target_end_date"] == date(2020, 1, 8)
    assert manifest["counts"]["predictions"] == 2
    assert manifest["status"] == "experimental"
    assert sha256(wide / "features-manifest.json") == manifest["features_manifest_sha256"]
    assert manifest["feature_method_version"] == global_features.METHOD_VERSION


def test_antimeridian(tmp_path):
    activity, model, cells = _fixture(tmp_path)
    output = tmp_path / "forecast"
    _run(activity, model, output, (170, -5, -170, 5))
    assert set(pq.read_table(output / "predictions.parquet")["cell_id"].to_pylist()) == set(cells[2:4])


@pytest.mark.parametrize("bbox", [(10, -1, 11, 1), (0, 1, 1, -1), (0, 0, float("nan"), 1), (0, 0, 0, 1)])
def test_empty_or_invalid_domain_not_published(tmp_path, bbox):
    activity, model, _ = _fixture(tmp_path)
    output = tmp_path / "forecast"
    with pytest.raises(ValueError):
        _run(activity, model, output, bbox)
    assert not output.exists()
    assert not list(tmp_path.glob(".forecast-*"))


@pytest.mark.parametrize("field,value", [("horizon_days", 8), ("feature_names", []),
                                         ("schema", "wrong"), ("data_cutoff_date", "2020-01-02"),
                                         ("feature_method_version", "incompatible"),
                                         ("activity", {"mask_manifest_sha256": "different"})])
def test_model_contract_corruption(tmp_path, field, value):
    activity, model, _ = _fixture(tmp_path)
    path = model / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    _json(path, manifest)
    with pytest.raises(ValueError):
        _run(activity, model, tmp_path / "forecast", (-1, -1, 1, 1))
    assert not (tmp_path / "forecast").exists()


def test_artifact_hash_corruption(tmp_path):
    activity, model, _ = _fixture(tmp_path)
    with (model / "calibrator.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="integrity"):
        _run(activity, model, tmp_path / "forecast", (-1, -1, 1, 1))
    assert not (tmp_path / "forecast").exists()


@pytest.mark.parametrize("x,y", [([0, 0], [0, 1]), ([0, 1], [1, 0]), ([0, 1], [0, 2]),
                                 ([0, float("nan")], [0, 1]), ([], [])])
def test_invalid_calibration_is_rejected(tmp_path, x, y):
    activity, model, _ = _fixture(tmp_path)
    path = model / "calibrator.json"
    _json(path, {"method": "isotonic", "x": x, "y": y})
    manifest = json.loads((model / "manifest.json").read_text())
    manifest["outputs"][1].update(size_bytes=path.stat().st_size, sha256=sha256(path))
    _json(model / "manifest.json", manifest)
    with pytest.raises(ValueError, match="calibration"):
        _run(activity, model, tmp_path / "forecast", (-1, -1, 1, 1))
    assert not (tmp_path / "forecast").exists()
