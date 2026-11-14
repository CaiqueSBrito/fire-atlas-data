"""Consumer-visible static-mask and global activity invariants."""
import csv
from datetime import date
import json

import duckdb
import h3
import pytest

from fire_atlas_data import global_activity, static_mask
from fire_atlas_data.normalization import normalize, sha256

HEADER = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
          "satellite", "instrument", "confidence", "version", "bright_t31", "frp", "daynight", "type"]
PARAMETERS = {"reference_end": date(2012, 12, 31), "min_recurrent_years": 2,
              "mask_resolution": 8, "buffer_rings": 0, "memory_limit": "256MB", "threads": 1}


def observation(point, day="2024-01-01", hotspot="", frp="", sensor="VIIRS", platform=None):
    return [*point, "330", "1", "1", day, "0100", platform or ("N20" if sensor == "VIIRS" else "Aqua"),
            sensor, "n" if sensor == "VIIRS" else "80", "2", "280", frp, "D", hotspot]


def dataset(tmp_path, name, rows, product="VIIRS_NOAA20_SP"):
    raw = tmp_path / f"{name}.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    output = tmp_path / name
    normalize(raw, output, source_product=product, memory_limit="256MB", threads=1)
    return output


def parquet(path):
    with duckdb.connect() as connection:
        result = connection.execute("SELECT * FROM read_parquet(?, hive_partitioning=true)", [str(path)])
        names = [column[0] for column in result.description]
        return [dict(zip(names, row)) for row in result.fetchall()]


def mask(tmp_path, points=(), datasets=(), **parameters):
    external = None
    if points:
        external = tmp_path / "external.csv"
        with external.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["latitude", "longitude"])
            writer.writerows(points)
    output = tmp_path / "mask"
    static_mask.build(datasets, external, output, **(PARAMETERS | parameters))
    return output


def fail_write(*args):
    raise OSError("disk failure")


def test_fine_mask_keeps_coarse_cell_and_unknown_type_and_missing_frp(tmp_path):
    coarse = h3.latlng_to_cell(-15, -47, 5)
    children = sorted(cell for cell in h3.cell_to_children(coarse, 8)
                      if h3.latlng_to_cell(*h3.cell_to_latlng(cell), 5) == coarse)
    blocked, free = h3.cell_to_latlng(children[0]), h3.cell_to_latlng(children[-1])
    mask_dir = mask(tmp_path, [blocked])
    viirs = dataset(tmp_path, "viirs", [
        observation(blocked, "2023-12-30", "0", "100"),
        observation(free, hotspot="0"), observation(free, hotspot="", frp="4"),
        observation(free, hotspot="1", frp="10"), observation(free, hotspot="2", frp="10"),
        observation(free, hotspot="3", frp="10")])
    modis = dataset(tmp_path, "modis", [observation(free, sensor="MODIS")], product="MODIS_SP")
    output = tmp_path / "activity"
    manifest = global_activity.build([viirs, modis], mask_dir, output, memory_limit="256MB", threads=1)
    daily = parquet(output / "daily" / "**" / "*.parquet")
    assert daily == [{"cell_id": coarse, "acquisition_date": date(2024, 1, 1),
                      "viirs_detections": 2, "modis_detections": 1,
                      "viirs_frp_sum_mw": 4.0, "viirs_frp_rows": 1,
                      "modis_frp_sum_mw": 0.0, "modis_frp_rows": 0, "year": 2024}]
    cells = parquet(output / "cells.parquet")
    assert len(cells) == 1
    assert cells[0]["first_detection_date"] == date(2023, 12, 30)
    assert cells[0]["parent4_id"] == h3.cell_to_parent(coarse, 4)
    assert cells[0]["parent3_id"] == h3.cell_to_parent(coarse, 3)
    assert (cells[0]["latitude"], cells[0]["longitude"]) == h3.cell_to_latlng(coarse)
    neighbors = parquet(output / "neighbors.parquet")
    assert {row["neighbor_id"] for row in neighbors} == set(h3.grid_disk(coarse, 1)) - {coarse}
    assert sum(row["detections"] for row in parquet(output / "platform_days.parquet")) == 7
    assert manifest["counts"]["excluded_static_detections"] == 4
    assert manifest["mask"]["manifest_sha256"] == sha256(mask_dir / "manifest.json")
    assert manifest["sources"][0]["manifest_sha256"] == sha256(viirs / "manifest.json")
    for entry in manifest["outputs"]:
        assert sha256(output / entry["path"]) == entry["sha256"]


def test_mask_recurrence_counts_distinct_years_at_inclusive_cutoff_and_origin_priority(tmp_path):
    seed = h3.latlng_to_cell(-15, -47, 8)
    neighbor = next(cell for cell in sorted(h3.grid_disk(seed, 1)) if cell != seed)
    far = h3.latlng_to_cell(10, 20, 8)
    same_year = h3.latlng_to_cell(30, 40, 8)
    point, other, repeated = map(h3.cell_to_latlng, (seed, far, same_year))
    data = dataset(tmp_path, "history", [
        observation(point, "2011-01-01", "2"), observation(point, "2012-12-31", "2"),
        observation(other, "2011-01-01", "2"), observation(other, "2013-01-01", "2"),
        observation(repeated, "2012-01-01", "2"), observation(repeated, "2012-12-31", "2")])
    directory = mask(tmp_path, [point, h3.cell_to_latlng(neighbor)], [data], buffer_rings=1)
    rows = {row["cell_id"]: row["origin"] for row in parquet(directory / "mask.parquet")}
    assert rows[seed] == "external" and rows[neighbor] == "external"
    manifest = json.loads((directory / "manifest.json").read_text())
    external = next(source for source in manifest["sources"] if source.get("kind") == "external_points")
    assert external["declared_knowledge_cutoff"] == "2012-12-31"
    assert external["temporal_validity"] == "caller_declared_not_verified"
    assert external["sha256"] == sha256(tmp_path / "external.csv")
    assert far not in rows and same_year not in rows
    expected = set(h3.grid_disk(seed, 1)) | set(h3.grid_disk(neighbor, 1))
    assert set(rows) == expected
    assert all(rows[cell] == "buffer" for cell in expected - {seed, neighbor})
    # Without the external override the recurrent origin remains visible.
    second = tmp_path / "recurrent"
    static_mask.build([data], None, second, **PARAMETERS)
    assert parquet(second / "mask.parquet") == [{"cell_id": seed, "origin": "recurrent_type"}]


def test_static_only_cells_and_platform_days_survive_with_empty_typed_daily(tmp_path):
    point = (0, 0)
    directory = mask(tmp_path)
    data = dataset(tmp_path, "static", [observation(point, hotspot="1")])
    output = tmp_path / "activity"
    global_activity.build([data], directory, output, memory_limit="256MB", threads=1)
    assert parquet(output / "daily" / "**" / "*.parquet") == []
    assert len(parquet(output / "cells.parquet")) == 1
    assert parquet(output / "platform_days.parquet")[0]["detections"] == 1
    with duckdb.connect() as connection:
        columns = connection.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                     [str(output / "daily" / "empty.parquet")]).fetchall()
    assert dict((row[0], row[1]) for row in columns)["year"] == "INTEGER"


def test_duplicate_hash_and_overlapping_same_platform_streams_are_rejected(tmp_path):
    directory = mask(tmp_path)
    first = dataset(tmp_path, "first", [observation((0, 0), "2024-01-01"),
                                         observation((0, 0), "2024-01-03")])
    overlap = dataset(tmp_path, "overlap", [observation((1, 1), "2024-01-02")],
                      product="VIIRS_NOAA20_NRT")
    for sources, message in (([first, first], "Duplicate"), ([first, overlap], "Overlapping")):
        with pytest.raises(ValueError, match=message):
            global_activity.build(sources, directory, tmp_path / "bad", memory_limit="256MB", threads=1)
        assert not (tmp_path / "bad").exists()
        assert not list(tmp_path.glob(".bad-*"))


def test_different_platform_overlap_and_disjoint_same_platform_intervals_are_allowed(tmp_path):
    directory = mask(tmp_path)
    first = dataset(tmp_path, "first", [observation((0, 0))])
    second = dataset(tmp_path, "second", [observation((0, 0), platform="N21")], "VIIRS_NOAA21_SP")
    later = dataset(tmp_path, "later", [observation((0, 0), "2024-01-02")])
    output = tmp_path / "activity"
    result = global_activity.build([first, second, later], directory, output, memory_limit="256MB", threads=1)
    assert result["counts"]["accepted_detections"] == 3
    assert sorted(row["viirs_detections"] for row in parquet(output / "daily" / "**" / "*.parquet")) == [1, 2]


@pytest.mark.parametrize("parameters", [{"mask_resolution": 4}, {"mask_resolution": 5}, {"mask_resolution": 16},
                                         {"min_recurrent_years": 0}, {"buffer_rings": -1}, {"threads": 0}])
def test_invalid_mask_parameters_never_publish(tmp_path, parameters):
    with pytest.raises(ValueError):
        static_mask.build([], None, tmp_path / "bad", **(PARAMETERS | parameters))
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_invalid_external_coordinate_aborts_without_partial_output(tmp_path):
    external = tmp_path / "invalid.csv"
    external.write_text("lat,longitude\n91,0\n")
    with pytest.raises(ValueError, match="invalid coordinates"):
        static_mask.build([], external, tmp_path / "bad", **PARAMETERS)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_existing_outputs_and_failed_publication_preserve_inputs(tmp_path, monkeypatch):
    directory = mask(tmp_path)
    data = dataset(tmp_path, "input", [observation((0, 0))])
    mask_before = (directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        static_mask.build([], None, directory, **PARAMETERS)
    with pytest.raises(FileExistsError):
        global_activity.build([data], directory, directory)
    assert (directory / "manifest.json").read_bytes() == mask_before
    monkeypatch.setattr(global_activity, "_copy", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        global_activity.build([data], directory, tmp_path / "bad", memory_limit="256MB", threads=1)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_input_schema_and_artifact_hash_are_checked(tmp_path):
    directory = mask(tmp_path)
    data = dataset(tmp_path, "input", [observation((0, 0))])
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "1.2.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="1.3.0"):
        global_activity.build([data], directory, tmp_path / "bad")
    manifest["schema_version"] = "1.3.0"
    manifest_path.write_text(json.dumps(manifest))
    with (directory / "mask.parquet").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        global_activity.build([data], directory, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_empty_accepted_inputs_are_not_an_observed_negative(tmp_path):
    directory = mask(tmp_path)
    data = dataset(tmp_path, "empty", [])
    with pytest.raises(ValueError, match="at least one accepted"):
        global_activity.build([data], directory, tmp_path / "bad", memory_limit="256MB", threads=1)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))


def test_mask_write_failure_cleans_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(static_mask, "_copy", fail_write)
    with pytest.raises(OSError, match="disk failure"):
        static_mask.build([], None, tmp_path / "bad", **PARAMETERS)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))
