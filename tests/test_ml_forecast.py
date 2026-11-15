from datetime import date, datetime, timedelta, timezone

from deltalake import DeltaTable, write_deltalake
import duckdb
import h3
import numpy as np
import pyarrow as pa
import pytest
import xgboost as xgb

from fire_atlas_data import ml_dataset
from fire_atlas_data.silver import load_h3
from fire_atlas_data.ml_dataset import FEATURES
from fire_atlas_data.ml_forecast import forecast

CENTER = h3.latlng_to_cell(-16.5, -56.5, 6)
RING1 = sorted(h3.grid_ring(CENTER, 1))
RING2 = sorted(h3.grid_ring(CENTER, 2))
ISSUE = date(2025, 9, 10)
CUTOFF = datetime(2025, 9, 10, 9, tzinfo=timezone.utc)


def _detections(rows):
    """Silver-shaped table from (cell, acquired_at, sensor, frp, hotspot_type)."""
    return pa.table({
        "h3_r6": pa.array([h3.str_to_int(c) for c, *_ in rows], pa.int64()),
        "h3_r4": pa.array([h3.str_to_int(h3.cell_to_parent(c, 4)) for c, *_ in rows], pa.int64()),
        "h3_r1": pa.array([h3.str_to_int(h3.cell_to_parent(c, 1)) for c, *_ in rows], pa.int64()),
        "acquired_at": pa.array([t for _, t, *_ in rows], pa.timestamp("us", tz="UTC")),
        "sensor": pa.array([s for _, _, s, *_ in rows]),
        "frp_mw": pa.array([f for *_, f, _ in rows], pa.float64()),
        "brightness_k": pa.array([330.0 if s == "MODIS" else None for _, _, s, *_ in rows], pa.float64()),
        "brightness_i4_k": pa.array([340.0 if s == "VIIRS" else None for _, _, s, *_ in rows], pa.float64()),
        "hotspot_type": pa.array([k for *_, k in rows], pa.int32()),
        "acquisition_year": pa.array([t.year for _, t, *_ in rows], pa.int32()),
        "acquisition_month": pa.array([t.month for _, t, *_ in rows], pa.int32()),
    })


def _history():
    hour = timedelta(hours=1)
    return [
        (CENTER, datetime(2025, 8, 1, 12, tzinfo=timezone.utc), "VIIRS", 5.0, 0),  # puts the cells in the universe
        (RING1[0], datetime(2025, 8, 2, 12, tzinfo=timezone.utc), "MODIS", 4.0, 0),
        (RING2[0], datetime(2025, 8, 2, 12, tzinfo=timezone.utc), "VIIRS", None, 0),
        (CENTER, CUTOFF - hour, "VIIRS", 10.0, 0),                    # last 24 h window
        (CENTER, CUTOFF - 2 * 24 * hour, "MODIS", 20.0, 0),           # within 3 days
        (CENTER, CUTOFF - 5 * 24 * hour, "VIIRS", None, 0),           # within 7 days, FRP missing
        (CENTER, CUTOFF - 8 * 24 * hour, "VIIRS", 99.0, 0),           # before the input windows
        (RING1[0], CUTOFF - 3 * hour, "VIIRS", 1.0, 0),
        (RING1[1], CUTOFF - 4 * 24 * hour, "MODIS", 1.0, 0),
        (RING2[0], CUTOFF - 6 * 24 * hour, "VIIRS", 1.0, 0),
        (RING2[1], CUTOFF - 2 * hour, "VIIRS", 1.0, 3),               # offshore: counted, never a universe cell
    ]


def _future():
    hour = timedelta(hours=1)
    return [
        (CENTER, CUTOFF, "VIIRS", 500.0, 0),                          # exactly at the cutoff: target, not input
        (CENTER, CUTOFF + 30 * hour, "MODIS", 500.0, 0),
        (RING1[2], CUTOFF + 2 * hour, "VIIRS", 1.0, 0),               # first detection after the cutoff
    ]


def _connection(rows):
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    load_h3(connection)
    connection.register("silver", _detections(rows))
    connection.register("coverage", pa.table({"h3_r4": pa.array([], pa.int64()),
                                              "window_date": pa.array([], pa.date32()),
                                              "year": pa.array([], pa.int32())}))
    connection.execute(ml_dataset._universe_sql(9))
    return connection


def _features(connection):
    connection.execute(ml_dataset.features_sql(ISSUE, 9))
    return connection.execute(f"SELECT c6, {', '.join(FEATURES)}, active FROM features ORDER BY c6").fetchall()


def test_inference_features_equal_training_rows_and_ignore_the_future():
    training = _connection(_history() + _future())
    training.execute(ml_dataset._classified_sql(ISSUE, ISSUE, 9))
    expected = training.execute(
        f"SELECT c6, {', '.join(FEATURES)}, active FROM classified ORDER BY c6").fetchall()

    assert _features(_connection(_history())) == expected
    assert _features(_connection(_history() + _future())) == expected

    by_cell = {row[0]: row for row in expected}
    assert h3.str_to_int(RING1[2]) not in by_cell  # first seen after the cutoff: no forecast row
    center = dict(zip(["c6", *FEATURES, "active"], by_cell[h3.str_to_int(CENTER)]))
    assert (center["det_24h"], center["det_3d"], center["det_7d"]) == (1, 2, 3)
    assert center["frp_mean_7d"] == pytest.approx(15.0)
    assert (center["ring1_det_24h"], center["ring1_det_7d"], center["ring2_det_7d"]) == (1, 2, 2)


def _model(path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(200, len(FEATURES))).astype(np.float32)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2}, xgb.DMatrix(x, label=x[:, 2] > 1),
                        num_boost_round=3)
    booster.save_model(path)
    return booster


def test_forecast_scores_universe_and_replaces_its_issue_date(tmp_path):
    silver, target, model = str(tmp_path / "silver"), str(tmp_path / "forecast"), tmp_path / "model.ubj"
    write_deltalake(silver, _detections(_history()), partition_by=["acquisition_year", "acquisition_month"])
    booster = _model(model)

    first = forecast(silver, model, target, issue_date=ISSUE, threads=1)
    forecast(silver, model, target, issue_date=ISSUE, threads=1)
    forecast(silver, model, target, issue_date=ISSUE - timedelta(days=1), threads=1)

    table = DeltaTable(target).to_pyarrow_table().to_pandas()
    today = table[table.issue_date == ISSUE].sort_values("h3_r6")
    # center, ring1[0], ring1[1], ring2[0]; the offshore cell never enters the universe
    assert len(today) == first["counts"]["cells"] == 4
    assert (table.issue_date == ISSUE - timedelta(days=1)).sum() == 4
    x = today[FEATURES].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(today.prob_fire_7d, booster.inplace_predict(x), rtol=1e-6)
    assert first["counts"]["input_lag_hours"] == 1.0


def test_forecast_refuses_stale_inputs(tmp_path):
    silver = str(tmp_path / "silver")
    write_deltalake(silver, _detections(_history()), partition_by=["acquisition_year", "acquisition_month"])
    _model(tmp_path / "model.ubj")
    with pytest.raises(ValueError, match="Silver ends at"):
        forecast(silver, tmp_path / "model.ubj", str(tmp_path / "forecast"), issue_date=ISSUE + timedelta(days=2),
                 threads=1)
