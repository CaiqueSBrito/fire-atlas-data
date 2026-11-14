"""Distinct historical-observation behavior, without asserting SP/NRT equivalence."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from fire_atlas_data import history
from fire_atlas_data.normalization import sha256


BASE_ROW = {"latitude": "-15.2", "longitude": "-47.1", "bright_ti4": "310.2",
            "scan": "0.4", "track": "0.4", "acq_date": "2020-09-01", "acq_time": "7",
            "satellite": "N20", "instrument": "VIIRS", "confidence": "n", "version": "2",
            "bright_ti5": "290.1", "frp": "2.1", "daynight": "N", "type": "0"}


def source_archive(tmp_path: Path, rows: list[dict], *, absent=()):
    root = tmp_path / "archive"
    raw = root / "raw" / "viirs-jpss1_2020_Brazil.csv"
    raw.parent.mkdir(parents=True)
    header = [name for name in BASE_ROW if name not in absent]
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows({name: row[name] for name in header} for row in rows)
    receipt = {"size_bytes": raw.stat().st_size, "sha256": sha256(raw),
               "received_at_utc": "2026-11-12T10:00:00+00:00"}
    manifest = {"product_declared": history.PRODUCT, "country": "Brazil", "files": [
        {"year": 2020, "raw_path": str(raw.relative_to(root)), "receipt": receipt,
         "url": history.BASE + "/viirs-jpss1/2020/viirs-jpss1_2020_Brazil.csv"}]}
    (root / "acquisition.json").write_text(json.dumps(manifest))
    return root, raw


def test_distinct_schema_preserves_source_and_reconciles_region(tmp_path):
    rows = [dict(BASE_ROW), dict(BASE_ROW, latitude="-19", longitude="-50", acq_time="2359"),
            dict(BASE_ROW, latitude="-13", longitude="-44"),
            dict(BASE_ROW, latitude="-20"), dict(BASE_ROW, latitude="nan")]
    root, raw = source_archive(tmp_path, rows)
    raw_hash = sha256(raw)
    output = tmp_path / "prepared"
    manifest = history.prepare(root, output, memory_limit="128MB", threads=1)
    assert manifest["schema"] == "historical-observations/1.0.0"
    assert manifest["product_declared"] == "VIIRS_NOAA20_SP"
    assert manifest["counts"] == {"brazil_raw": 5, "brazil_valid": 4, "rejected": 1,
                                  "region": 3, "outside_region_valid": 1}
    year = manifest["years"][0]
    assert year["count_reconciliation"] is True
    assert year["rejection_counts"] == {"invalid_latitude": 1}
    assert year["metadata_fields_available"] == dict.fromkeys(["instrument", "satellite", "version", "type", "daynight"], True)
    assert sha256(raw) == raw_hash
    with duckdb.connect() as con:
        result = con.execute("""SELECT historical_schema,source_product_declared,platform,
            instrument_source,source_version,acquisition_date::VARCHAR,acquisition_time_utc,
            acquired_at_utc,source_record_number,source_fields.acq_time,
            confidence_category,frp_mw,detection_type_source FROM read_parquet(?) ORDER BY source_record_number""",
            [str(output / "year=2020/observations.parquet")]).fetchall()
    assert result[0] == (history.SCHEMA, history.PRODUCT, "NOAA-20", "VIIRS", "2", "2020-09-01", "0007",
                         datetime(2020, 9, 1, 0, 7, tzinfo=timezone.utc), 1, "7", "n", 2.1, "0")
    assert len(result) == 3
    assert all(file["sha256"] == sha256(output / file["path"]) for file in manifest["outputs"])
    assert "available_at" not in {column["name"] for column in year["output_columns"]}


@pytest.mark.parametrize(("changes", "reason"), [
    ({"satellite": "N21"}, "platform_product_mismatch"),
    ({"satellite": ""}, "platform_product_mismatch"),
    ({"instrument": "MODIS"}, "instrument_product_mismatch"),
    ({"version": "2.0NRT"}, "nrt_version_in_sp_history"),
    ({"acq_date": "2020-02-30"}, "invalid_acquisition_datetime"),
    ({"acq_date": "2019-12-31"}, "acquisition_year_mismatch"),
    ({"acq_date": "2020-9-01"}, "invalid_acquisition_datetime"),
    ({"acq_time": "2400"}, "invalid_acquisition_datetime"),
    ({"acq_time": "1260"}, "invalid_acquisition_datetime"),
    ({"latitude": "91"}, "invalid_latitude"),
    ({"longitude": "-181"}, "invalid_longitude"),
    ({"longitude": "inf"}, "invalid_longitude"),
    ({"confidence": "80"}, "invalid_confidence"),
    ({"frp": "-0.01"}, "invalid_frp"),
    ({"frp": "nan"}, "invalid_frp"),
    ({"scan": "0"}, "invalid_scan"),
])
def test_invalid_observations_are_retained_as_rejections(tmp_path, changes, reason):
    root, _ = source_archive(tmp_path, [dict(BASE_ROW, **changes)])
    output = tmp_path / "prepared"
    manifest = history.prepare(root, output, memory_limit="128MB", threads=1)
    assert manifest["counts"]["region"] == 0
    assert manifest["years"][0]["rejection_counts"] == {reason: 1}
    with duckdb.connect() as con:
        row = con.execute("SELECT reason,source_fields FROM read_parquet(?)",
                          [str(output / "year=2020/rejected.parquet")]).fetchone()
    assert row[0] == reason
    assert row[1] == dict(BASE_ROW, **changes)


def test_optional_metadata_is_not_invented(tmp_path):
    root, _ = source_archive(tmp_path, [dict(BASE_ROW)], absent=("instrument", "version", "type", "daynight"))
    output = tmp_path / "prepared"
    manifest = history.prepare(root, output, memory_limit="128MB", threads=1)
    assert manifest["years"][0]["metadata_fields_available"] == {
        "instrument": False, "satellite": True, "version": False, "type": False, "daynight": False}
    with duckdb.connect() as con:
        row = con.execute("SELECT platform,instrument_source,source_version,detection_type_source FROM read_parquet(?)",
                          [str(output / "year=2020/observations.parquet")]).fetchone()
    assert row == ("NOAA-20", None, None, None)


def test_changed_raw_bytes_refuse_preparation(tmp_path):
    root, raw = source_archive(tmp_path, [dict(BASE_ROW)])
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="immutable download receipt"):
        history.prepare(root, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_inventory_uses_exact_brazil_filename_and_size(tmp_path):
    inventory = tmp_path / "inventory.txt"
    inventory.write_text("filename,size,updated\nviirs-jpss1_2020_Brazil.csv,123,2025-06-20 12:00:00\n"
                         "viirs-jpss1_2020_all_countries.zip,999,2025-06-20 12:00:00\n")
    selected = history.inventory_selection(inventory, (2020,))
    assert selected == [{"year": 2020, "filename": "viirs-jpss1_2020_Brazil.csv", "size_bytes": 123,
                         "inventory_updated_text": "2025-06-20 12:00:00",
                         "url": history.BASE + "/viirs-jpss1/2020/viirs-jpss1_2020_Brazil.csv"}]
    with pytest.raises(ValueError, match="absent"):
        history.inventory_selection(inventory, (2021,))
    with pytest.raises(ValueError, match="2018 through 2025"):
        history.inventory_selection(inventory, (2026,))
