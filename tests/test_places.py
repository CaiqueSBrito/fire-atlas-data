import zipfile

from deltalake import DeltaTable, write_deltalake
import duckdb
import h3
import pyarrow as pa

from fire_atlas_data.places import build, verify

# Two square "countries" side by side, sea everywhere else.
AAA = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"
BBB = "POLYGON((10 0, 20 0, 20 10, 10 10, 10 0))"


def _countries(tmp_path):
    folder = tmp_path / "ne"
    folder.mkdir()
    with duckdb.connect() as connection:
        connection.execute("INSTALL spatial")
        connection.execute("LOAD spatial")
        connection.execute(f"""COPY (
            SELECT * FROM (VALUES ('AAA', 'AA', 'Aland', 'Alândia', 'South America', ST_GeomFromText('{AAA}')),
                                  ('BBB', 'BB', 'Beland', 'Belândia', 'South America', ST_GeomFromText('{BBB}')))
                t(ADM0_A3, ISO_A2_EH, NAME, NAME_PT, CONTINENT, geom))
            TO '{folder}/ne_10m_admin_0_countries.shp' (FORMAT GDAL, DRIVER 'ESRI Shapefile')""")
    path = tmp_path / "countries.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in folder.iterdir():
            archive.write(file, file.name)
    return path


def _cities(tmp_path):
    rows = [(1, "Alfa", 5.0, 9.4, "AA"), (2, "Beta", 5.0, 10.6, "BB")]
    text = "".join(f"{i}\t{name}\t{name}\t\t{lat}\t{lon}\tP\tPPL\t{cc}\t\t\t\t\t\t1000\t\t0\tUTC\t2026-01-01\n"
                   for i, name, lat, lon, cc in rows)
    path = tmp_path / "cities.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("cities1000.txt", text)
    return path


def _straddling_point():
    """A point inside AAA whose res6 cell centroid lies north of AAA (lat > 10)."""
    for step in range(1, 200):
        lat = 10 - step * 0.0005
        lat_c, _ = h3.cell_to_latlng(h3.latlng_to_cell(lat, 5.0, 6))
        if lat_c > 10:
            return lat, 5.0
    raise AssertionError("no straddling cell found")


def _silver(points, types=None):
    cells = [h3.latlng_to_cell(lat, lon, 6) for lat, lon in points]
    columns = {f"h3_r{r}": pa.array([h3.str_to_int(h3.cell_to_parent(c, r)) for c in cells], pa.int64())
               for r in range(7)}
    return pa.table({"latitude": [p[0] for p in points], "longitude": [p[1] for p in points], **columns,
                     "hotspot_type": pa.array(types or [None] * len(points), pa.int16()),
                     "acquisition_year": [2025] * len(points), "acquisition_month": [8] * len(points)})


def _row(table, resolution, lat, lon):
    cell = h3.str_to_int(h3.latlng_to_cell(lat, lon, resolution))
    return next(r for r in table if r["resolution"] == resolution and r["h3_cell"] == cell)


def test_places_assign_country_by_fallbacks_and_predominance(tmp_path):
    inside_a, inside_b = (5.0, 9.5), (5.0, 10.5)
    straddling = _straddling_point()
    coastal = (10.04, 5.0)        # sea, within 0.05 degree of AAA
    far = (40.0, 40.0)            # open sea
    lat_c, _ = h3.cell_to_latlng(h3.latlng_to_cell(*coastal, 6))
    assert lat_c > 10
    assert h3.latlng_to_cell(*inside_a, 1) == h3.latlng_to_cell(*inside_b, 1)
    silver, target = str(tmp_path / "silver"), str(tmp_path / "places")
    write_deltalake(silver, _silver([inside_a] * 3 + [inside_b] + [straddling, coastal, far]),
                    partition_by=["acquisition_year", "acquisition_month"])

    build(silver, _countries(tmp_path), _cities(tmp_path), target, threads=1)
    report = verify(target, silver)
    rows = DeltaTable(target).to_pyarrow_table().to_pylist()

    a = _row(rows, 6, *inside_a)
    assert (a["country_code"], a["country_method"], a["country_share"], a["continent"]) == \
        ("AAA", "centroid", 1.0, "South America")
    assert (a["country_name_pt"], a["city_name"], a["city_country_code"]) == ("Alândia", "Alfa", "AA")
    assert 0 < a["city_distance_km"] < 15
    assert _row(rows, 6, *straddling)["country_method"] == "detection_mean"
    assert (_row(rows, 6, *coastal)["country_code"], _row(rows, 6, *coastal)["country_method"]) == ("AAA", "nearest")
    sea = _row(rows, 6, *far)
    assert sea["country_code"] is None and sea["continent"] is None and sea["country_share"] is None

    shared = _row(rows, 1, *inside_a)
    members = sum(h3.latlng_to_cell(*p, 1) == h3.latlng_to_cell(*inside_a, 1)
                  for p in [inside_a] * 3 + [inside_b, straddling, coastal])
    assert (shared["country_code"], shared["country_method"], shared["city_name"]) == ("AAA", "predominant", None)
    assert shared["detections"] == members
    assert shared["country_share"] == (members - 1) / members
    assert report["silver_detections"] == 7 and report["detections_without_country"]["r6"] == 1
    assert a["static_source"] is None and a["static_share"] is None    # untyped (NRT-like) detections
    assert shared["static_source"] is None                            # parents carry no flag


def test_places_flag_cells_dominated_by_firms_static_types(tmp_path):
    flare, mixed, fire = (5.0, 9.5), (5.0, 10.5), (2.0, 2.0)
    points = [flare] * 10 + [mixed] * 10 + [fire] * 12
    types = [2] * 9 + [0] + [2] * 4 + [0] * 6 + [0] * 11 + [None]
    silver, target = str(tmp_path / "silver"), str(tmp_path / "places")
    write_deltalake(silver, _silver(points, types), partition_by=["acquisition_year", "acquisition_month"])
    manifest = build(silver, _countries(tmp_path), _cities(tmp_path), target, threads=1)
    rows = DeltaTable(target).to_pyarrow_table().to_pylist()

    assert (_row(rows, 6, *flare)["static_source"], _row(rows, 6, *flare)["static_share"]) == (True, 0.9)
    assert (_row(rows, 6, *mixed)["static_source"], _row(rows, 6, *mixed)["static_share"]) == (False, 0.4)
    assert (_row(rows, 6, *fire)["static_source"], _row(rows, 6, *fire)["static_share"]) == (False, 0.0)
    assert manifest["static_sources"] == {"cells": 1, "detections": 10, "cells_without_enough_typed": 0}
