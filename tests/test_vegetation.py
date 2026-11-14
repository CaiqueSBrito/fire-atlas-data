"""Synthetic offline engineering fixtures; not evidence of NASA scientific validity."""
from copy import deepcopy
from datetime import date, datetime, timezone
import json
from urllib.parse import parse_qs, urlsplit

import duckdb
import h3
import pytest

from fire_atlas_data import vegetation as veg


def band_metadata():
    rows = []
    for name, low, high, fill in [(veg.NDVI, -2000, 10000, -3000), (veg.EVI, -2000, 10000, -3000),
                                  (veg.RELIABILITY, 0, 3, -1), (veg.QUALITY, 0, 65534, -1),
                                  (veg.DAY, 1, 366, -1)]:
        row = {"band": name, "valid_range": f"{low} to {high}", "fill_value": str(fill)}
        if name in (veg.NDVI, veg.EVI):
            row.update(scale_factor="0.0001", add_offset="0")
        rows.append(row)
    return {"bands": rows}


def patch(start="A2026241", proc="2026258093231", latitude=-16.0, longitude=-47.0):
    arrays = {veg.NDVI: [5000, 8000, -3000, 9000], veg.EVI: [2000, 4000, 4000, 5000],
              veg.RELIABILITY: [0, 1, 0, 3], veg.QUALITY: [0, 0, 0, 0],
              veg.DAY: [245, 246, 247, 248]}
    return {"latitude": latitude, "longitude": longitude, "cellsize": "231.656358264",
            "xllcorner": "-5000000", "yllcorner": "-1700000", "nrows": 2, "ncols": 2,
            "band": "all", "subset": [{"band": b, "modis_date": start, "tile": "h13v10",
                                         "proc_date": proc, "data": values} for b, values in arrays.items()]}


def parse(payload=None):
    return veg._patch(payload or patch(), ["A2026241"], -16, -47, veg._metadata(band_metadata()))


def test_strict_common_quality_mask_and_units():
    composites, bounds = parse()
    result = composites[0]
    assert result["ndvi"] == pytest.approx(0.5)
    assert result["evi"] == pytest.approx(0.2)
    assert result["valid_fraction"] == 0.25
    assert result["valid_pixels"] == 1
    assert result["version"] is None
    assert bounds["collection_version_missing"] is True
    values, selected = veg._select(composites, date(2026, 9, 15))
    assert selected["start"] == date(2026, 8, 29)
    assert values == pytest.approx([0.5, 0.2, 0.25, 2.0, 13.0])


def test_no_valid_pixels_means_null_not_zero():
    payload = patch()
    next(b for b in payload["subset"] if b["band"] == veg.RELIABILITY)["data"] = [3] * 4
    composites, _ = parse(payload)
    values, _ = veg._select(composites, date(2026, 9, 15))
    assert values[:3] == [None, None, 0.0]
    assert values[3:] == [2.0, None]


def test_out_of_range_values_and_wrong_observation_days_are_missing():
    for band, value in [(veg.NDVI, 10001), (veg.EVI, -2001), (veg.DAY, 300), (veg.DAY, -1), (veg.QUALITY, -1)]:
        payload = patch()
        next(b for b in payload["subset"] if b["band"] == band)["data"][0] = value
        composites, _ = parse(payload)
        assert composites[0]["ndvi"] is None
        assert composites[0]["evi"] is None
        assert composites[0]["valid_fraction"] == 0


def test_latest_production_guard_and_no_quality_backfill():
    composites, _ = parse()
    original = composites[0]
    newer = {**original, "start": date(2026, 9, 14), "end": date(2026, 9, 29),
             "production": datetime(2026, 10, 2, tzinfo=timezone.utc), "ndvi": None,
             "evi": None, "valid_fraction": 0.0, "mean_observation_ordinal": None}
    values, selected = veg._select([original, newer], date(2026, 10, 1))
    assert selected == original
    values, selected = veg._select([original, newer], date(2026, 10, 2))
    assert selected == newer
    assert values[:3] == [None, None, 0.0]
    assert veg._select([original], date(2026, 9, 12)) == ([None] * len(veg.VEGETATION_FEATURES), None)
    # End-of-day, not midnight, is the stated issue cutoff.
    boundary = {**original, "production": datetime(2026, 9, 15, 23, 59, 59, tzinfo=timezone.utc)}
    assert veg._select([boundary], date(2026, 9, 15))[1] == boundary


def test_unknown_production_is_preserved_not_invented():
    composites, _ = parse(patch(proc=None))
    assert composites[0]["production"] is None
    assert veg._select(composites, date(2026, 9, 15))[1] is not None


@pytest.mark.parametrize("value", ["2026241", "A2026366", "A2026000", "A2026241x", None])
def test_invalid_composite_dates(value):
    with pytest.raises((ValueError, TypeError)):
        veg._modis_date(value)


@pytest.mark.parametrize("value", ["2026366093231", "2026241093261", "2026241", 2026241093231])
def test_invalid_production_timestamps(value):
    with pytest.raises((ValueError, TypeError)):
        veg._production(value)


def test_inventory_window_and_missing_or_duplicate_dates():
    inventory = {"dates": [{"modis_date": d, "calendar_date": veg._modis_date(d).isoformat()}
                            for d in ["A2026209", "A2026225", "A2026241", "A2026257", "A2026273", "A2026289"]]}
    assert veg._dates(inventory, date(2026, 8, 1), date(2026, 10, 6)) == ["A2026225", "A2026241", "A2026257", "A2026273"]
    with pytest.raises(ValueError, match="No MOD13Q1"):
        veg._dates(inventory, date(2025, 1, 1), date(2025, 2, 1))
    duplicate = deepcopy(inventory)
    duplicate["dates"].append(duplicate["dates"][0])
    with pytest.raises(ValueError, match="duplicate"):
        veg._dates(duplicate, date(2026, 8, 1), date(2026, 10, 6))


@pytest.mark.parametrize("mutation", ["missing_band", "missing_date", "duplicate", "array_length", "different_proc", "different_tile", "different_version", "different_center", "different_resolution", "bad_rank", "bad_quality"])
def test_malformed_or_different_metadata_fails_visibly(mutation):
    payload = patch()
    if mutation == "missing_band":
        payload["subset"].pop()
    elif mutation == "missing_date":
        payload["subset"] = []
    elif mutation == "duplicate":
        payload["subset"].append(payload["subset"][0])
    elif mutation == "array_length":
        payload["subset"][0]["data"] = [5000]
    elif mutation == "different_proc":
        payload["subset"][0]["proc_date"] = "2026259093231"
    elif mutation == "different_tile":
        payload["subset"][0]["tile"] = "h14v10"
    elif mutation == "different_version":
        payload["subset"][0]["version"] = "061"
    elif mutation == "different_center":
        payload["latitude"] = -15
    elif mutation == "different_resolution":
        payload["cellsize"] = 500
    elif mutation == "bad_rank":
        next(b for b in payload["subset"] if b["band"] == veg.RELIABILITY)["data"][0] = 4
    elif mutation == "bad_quality":
        next(b for b in payload["subset"] if b["band"] == veg.QUALITY)["data"][0] = 65535
    with pytest.raises(ValueError):
        parse(payload)


@pytest.mark.parametrize("field,value", [("scale_factor", "0.001"), ("fill_value", "-9999"), ("valid_range", "0 to 10000")])
def test_different_band_metadata_rejected(field, value):
    metadata = band_metadata()
    metadata["bands"][0][field] = value
    with pytest.raises(ValueError, match="metadata"):
        veg._metadata(metadata)


def feature_file(tmp_path, issue=date(2026, 9, 15)):
    cell = h3.latlng_to_cell(-16, -47, 5)
    source = tmp_path / "features.parquet"
    connection = duckdb.connect()
    connection.execute("CREATE TABLE f(cell_id VARCHAR, issue_date DATE, label INTEGER)")
    connection.executemany("INSERT INTO f VALUES (?,?,?)", [(cell, issue, 0), (cell, issue, 1)])
    connection.execute(f"COPY f TO '{source}' (FORMAT PARQUET)")
    connection.close()
    return source, cell


def fake_fetch(url, stage, name):
    if name == "bands":
        payload = band_metadata()
    elif name == "dates":
        payload = {"dates": [{"modis_date": "A2026241", "calendar_date": "2026-08-29"}]}
    else:
        params = parse_qs(urlsplit(url).query)
        assert params["kmAboveBelow"] == ["1"]
        assert params["kmLeftRight"] == ["1"]
        assert "band" not in params
        payload = patch(latitude=float(params["latitude"][0]), longitude=float(params["longitude"][0]))
    raw = stage / "raw" / (name + ".json")
    raw.write_text(json.dumps(payload))
    receipt = {"url": url, "path": str(raw.relative_to(stage)), "sha256": veg.sha256(raw), "available_at": None}
    veg._json(stage / "receipts" / (name + ".json"), receipt)
    return payload, receipt


def test_acquisition_unique_keys_schema_hashes_and_no_overwrite(monkeypatch, tmp_path):
    source, cell = feature_file(tmp_path)
    monkeypatch.setattr(veg, "_fetch", fake_fetch)
    target = tmp_path / "vegetation"
    manifest = veg.acquire(source, target, concurrency=1)
    assert manifest["columns"] == veg.VEGETATION_FEATURES
    assert manifest["counts"]["rows"] == 1
    assert manifest["counts"]["cells"] == 1
    assert manifest["source"]["available_at"] is None
    assert manifest["source"]["processing"].startswith("scientific")
    for output in manifest["outputs"]:
        assert veg.sha256(target / output["path"]) == output["sha256"]
    connection = duckdb.connect()
    result = connection.execute(f"SELECT * FROM read_parquet('{target / 'covariates.parquet'}')").fetchone()
    schema = connection.description
    connection.close()
    assert result[:2] == (cell, date(2026, 9, 15))
    assert [str(c[1]) for c in schema] == ["VARCHAR", "DATE"] + ["DOUBLE"] * 5
    with pytest.raises(FileExistsError):
        veg.acquire(source, target)


def test_future_reserved_is_never_requested(monkeypatch, tmp_path):
    source, _ = feature_file(tmp_path, issue=date(2026, 11, 13))
    def forbidden(*args):
        raise AssertionError("Network must not be accessed")
    monkeypatch.setattr(veg, "_fetch", forbidden)
    with pytest.raises(ValueError, match="Reserved"):
        veg.acquire(source, tmp_path / "forbidden")


def test_failed_remote_validation_retains_evidence_not_success_manifest(monkeypatch, tmp_path):
    source, _ = feature_file(tmp_path)
    def broken(url, stage, name):
        if name.startswith("subset"):
            payload, receipt = fake_fetch(url, stage, name)
            payload["subset"] = []
            return payload, receipt
        return fake_fetch(url, stage, name)
    monkeypatch.setattr(veg, "_fetch", broken)
    target = tmp_path / "vegetation"
    with pytest.raises(RuntimeError, match="evidence retained"):
        veg.acquire(source, target, concurrency=1)
    assert not target.exists()
    failures = list(tmp_path.glob(".vegetation-*/failure.json"))
    assert len(failures) == 1
    assert (failures[0].parent / "raw" / "bands.json").exists()
    assert not (failures[0].parent / "manifest.json").exists()


@pytest.mark.parametrize("body", [b"<html>not JSON</html>", b"[]"])
def test_non_json_body_retains_raw_receipt(monkeypatch, tmp_path, body):
    import io
    from email.message import Message

    class Response(io.BytesIO):
        status = 200
        url = "https://modis.ornl.gov/test"
        headers = Message()

    monkeypatch.setattr(veg, "urlopen", lambda *args, **kwargs: Response(body))
    (tmp_path / "raw").mkdir()
    (tmp_path / "receipts").mkdir()
    with pytest.raises(ValueError):
        veg._fetch("https://modis.ornl.gov/test", tmp_path, "bad")
    assert (tmp_path / "raw/bad.json").read_bytes() == body
    receipt = json.loads((tmp_path / "receipts/bad.json").read_text())
    assert receipt["sha256"] == veg.sha256(tmp_path / "raw/bad.json")
    assert receipt["available_at"] is None


def test_oversized_response_is_bounded_and_preserved(monkeypatch, tmp_path):
    import io
    from email.message import Message

    class Response(io.BytesIO):
        status = 200
        url = "https://modis.ornl.gov/test"
        headers = Message()

    monkeypatch.setattr(veg, "MAX_BYTES", 8)
    monkeypatch.setattr(veg, "urlopen", lambda *args, **kwargs: Response(b"0123456789012345"))
    (tmp_path / "raw").mkdir()
    (tmp_path / "receipts").mkdir()
    with pytest.raises(ValueError, match="oversized"):
        veg._fetch("https://modis.ornl.gov/test", tmp_path, "large")
    assert (tmp_path / "raw/large.json").stat().st_size == 9


def test_publication_failure_removes_own_links_but_retains_evidence(monkeypatch, tmp_path):
    source, _ = feature_file(tmp_path)
    target = tmp_path / "vegetation"
    monkeypatch.setattr(veg, "_fetch", fake_fetch)
    real_link = veg.os.link

    def link(source_path, destination):
        if destination.name == "manifest.json":
            raise OSError("synthetic publication failure")
        return real_link(source_path, destination)

    monkeypatch.setattr(veg.os, "link", link)
    with pytest.raises(RuntimeError, match="publication failure"):
        veg.acquire(source, target, concurrency=1)
    assert not target.exists()
    failures = list(tmp_path.glob(".vegetation-*/failure.json"))
    assert len(failures) == 1
    assert (failures[0].parent / "covariates.parquet").exists()


def test_destination_race_keeps_unrelated_files(monkeypatch, tmp_path):
    source, _ = feature_file(tmp_path)
    target = tmp_path / "vegetation"

    def racing_fetch(url, stage, name):
        result = fake_fetch(url, stage, name)
        if name == "bands":
            target.mkdir()
            (target / "unrelated").write_text("keep")
        return result

    monkeypatch.setattr(veg, "_fetch", racing_fetch)
    with pytest.raises(RuntimeError, match="evidence retained"):
        veg.acquire(source, target, concurrency=1)
    assert (target / "unrelated").read_text() == "keep"
    assert not (target / "manifest.json").exists()
