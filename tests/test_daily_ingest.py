"""Daily job contracts with a local fake of the Area API; not proof of NASA access."""
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib

from deltalake import DeltaTable
import numpy as np
import pytest
import xgboost as xgb

from fire_atlas_data import daily_ingest
from fire_atlas_data.ml_dataset import FEATURES
from fire_atlas_data.normalization import normalize
from fire_atlas_data.silver import build

HEADER = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time", "satellite",
          "instrument", "confidence", "version", "bright_t31", "frp", "daynight"]
ISSUE = date(2025, 10, 2)
CUTOFF = datetime(2025, 10, 2, 9, tzinfo=timezone.utc)


def _row(at: datetime, lat: float = -16.5) -> list:
    return [lat, -56.5, 330.0, 1.0, 1.0, at.date().isoformat(), at.strftime("%H%M"), "Aqua", "MODIS", 80,
            "6.1NRT", 300.0, 12.0, "D"]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)


def _fake_area(archive: list[list]):
    """Area API stand-in: rows whose acq_date falls in [date, date + days)."""
    def fetch(output_dir, *, product, area, days, start_date, key_env):
        first = date.fromisoformat(start_date)
        rows = [r for r in archive if first <= date.fromisoformat(r[5]) < first + timedelta(days=days)]
        _write(output_dir / "raw.csv", rows)
        digest = hashlib.sha256((output_dir / "raw.csv").read_bytes()).hexdigest()
        return {"raw": {"rows": len(rows), "sha256": digest}, "source": {"url": "https://example.invalid/area"}}
    return fetch


def _model(path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=(200, len(FEATURES))).astype(np.float32)
    xgb.train({"objective": "binary:logistic", "max_depth": 2}, xgb.DMatrix(x, label=x[:, 2] > 1),
              num_boost_round=3).save_model(path)


def test_reruns_add_late_and_new_rows_once_across_months(tmp_path, monkeypatch):
    hour = timedelta(hours=1)
    history = [_row(datetime(2025, 9, 1, 12, tzinfo=timezone.utc)), _row(datetime(2025, 9, 29, 3, tzinfo=timezone.utc))]
    # A record published late for a time before the newest ingested one, then new rows across the month boundary.
    new = [_row(datetime(2025, 9, 29, 1, tzinfo=timezone.utc), lat=-16.7),
           _row(datetime(2025, 9, 30, 5, tzinfo=timezone.utc)), _row(datetime(2025, 10, 1, 14, tzinfo=timezone.utc)),
           _row(CUTOFF - hour, lat=-16.6)]
    _write(tmp_path / "history.csv", history)
    normalize(tmp_path / "history.csv", tmp_path / "history", source_product="MODIS_NRT", threads=1)
    silver = str(tmp_path / "silver")
    build([tmp_path / "history"], silver, threads=1)
    _model(tmp_path / "model.ubj")
    monkeypatch.setattr(daily_ingest, "fetch_area", _fake_area(history + new))
    options = dict(silver=silver, model=str(tmp_path / "model.ubj"), report=None,
                   forecast_target=str(tmp_path / "forecast"), work_dir=tmp_path / "work",
                   products=["MODIS_NRT"], threads=1)

    first = daily_ingest.run(ISSUE, now=CUTOFF + hour, **options)
    second = daily_ingest.run(ISSUE, now=CUTOFF + 2 * hour, **options)

    assert first["fetch_starts"] == {"MODIS_NRT": "2025-09-28"}
    assert first["silver_appended"] == {"2025-09": 2, "2025-10": 2}
    assert second["silver_appended"] == {}
    rows = DeltaTable(silver).to_pyarrow_table(columns=["acquired_at"]).column("acquired_at").to_pylist()
    assert len(rows) == len(history) + len(new) == len(set(rows))
    assert second["counts"]["cells"] == 3  # cells detected before the cutoff, scored once per issue date
    assert DeltaTable(str(tmp_path / "forecast")).to_pyarrow_table().num_rows == 3


def test_refuses_an_issue_date_before_its_cutoff(tmp_path):
    with pytest.raises(ValueError, match="run after the cutoff"):
        daily_ingest.run(ISSUE, silver=str(tmp_path), model="m", report=None, forecast_target=str(tmp_path / "f"),
                         work_dir=tmp_path, now=CUTOFF - timedelta(minutes=1))
