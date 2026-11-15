"""Gold layer, place lookup: continent, country and nearest city for every H3 cell (res 0..6) with detections.

One row per (resolution, h3_cell) present in silver, so the map joins it on `h3_cell` instead of
denormalizing the daily aggregates. Countries come from Natural Earth 1:10m admin-0, cities from
GeoNames cities1000 (populated places with population >= 1000).

- res6: country of the polygon containing the cell centroid; otherwise the polygon containing the
  mean position of the cell's detections; otherwise the nearest polygon within COAST_TOLERANCE_DEG.
- res0..5: predominant country by detections among the res6 children (`country_share` = detections
  in that country / all detections in the cell, including those without a country).
- city: nearest GeoNames place to the cell centroid, only for res >= CITY_MIN_RESOLUTION.
- static source (res6 only): share of the cell's detections with a FIRMS archive `type` that FIRMS classed
  as volcano, other static land source or offshore (types 1, 2, 3). `static_source` when that share is at
  least STATIC_SHARE over at least STATIC_MIN_TYPED typed detections; NULL when too few are typed (NRT has no type).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
import shutil
import tempfile
import zipfile

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
from deltalake import CommitProperties, DeltaTable, write_deltalake
from sklearn.neighbors import BallTree

from .silver import RESOLUTIONS, _literal, _sha256, load_h3

METHOD_VERSION = "1.1.0"
PARTITIONS = ["resolution"]
CITY_MIN_RESOLUTION = 4
COAST_TOLERANCE_DEG = 0.05
STATIC_SHARE = 0.5        # measured 2026-11-14: per-cell shares are bimodal (<10% or >=90%), few in between
STATIC_MIN_TYPED = 10
EARTH_RADIUS_KM = 6371.0088
COUNTRIES_LAYER = "ne_10m_admin_0_countries.shp"
CITIES_MEMBER = "cities1000.txt"
LIMITATIONS = [
    "Rows only for cells with detections in the pinned silver version; new cells need a rebuild.",
    "res6 country is decided per cell (~36 km2), so detections near a border may belong to the neighbour.",
    "res0..5 country is the predominant one by detections; country_share gives how predominant.",
    "Country NULL: no Natural Earth polygon at the centroid, the detection mean or within 0.05 degree (mostly sea).",
    "Borders follow the Natural Earth default point of view; disputed areas are not adjudicated.",
    "city is the nearest GeoNames place (>= 1000 inhabitants) to the cell centroid, not the municipality "
    "containing the fire; it may lie in another country (city_country_code).",
    "static_source relies on the FIRMS archive classification (types 1-3); cells seen only in NRT data have no "
    "type and get NULL, and a fire burning next to a flare inside one res6 cell is hidden with it.",
]


def _countries_sql(path: Path) -> str:
    source = _literal(f"/vsizip/{path.resolve()}/{COUNTRIES_LAYER}")
    return f"""CREATE TABLE countries AS
        SELECT ADM0_A3 AS country_code, nullif(ISO_A2_EH, '-99') AS country_iso_a2, NAME AS country_name,
               NAME_PT AS country_name_pt, CONTINENT AS continent, geom
        FROM ST_Read({source})"""


def _assign_r6_sql() -> str:
    return f"""CREATE TABLE r6_country AS
        WITH centroid AS (
            SELECT r.h3_cell, k.country_code FROM r6 r
            JOIN countries k ON ST_Intersects(k.geom, ST_Point(r.cell_lon, r.cell_lat))),
        mean AS (
            SELECT r.h3_cell, k.country_code FROM (SELECT * FROM r6 ANTI JOIN centroid USING (h3_cell)) r
            JOIN countries k ON ST_Intersects(k.geom, ST_Point(r.mean_lon, r.mean_lat))),
        rest AS (SELECT * FROM r6 ANTI JOIN centroid USING (h3_cell) ANTI JOIN mean USING (h3_cell)),
        nearest AS (
            SELECT r.h3_cell, arg_min(k.country_code, ST_Distance(k.geom, ST_Point(r.mean_lon, r.mean_lat))) AS country_code
            FROM rest r JOIN countries k ON ST_DWithin(k.geom, ST_Point(r.mean_lon, r.mean_lat), {COAST_TOLERANCE_DEG})
            GROUP BY r.h3_cell)
        SELECT r.h3_cell, r.detections, coalesce(c.country_code, m.country_code, n.country_code) AS country_code,
               CASE WHEN c.country_code IS NOT NULL THEN 'centroid'
                    WHEN m.country_code IS NOT NULL THEN 'detection_mean'
                    WHEN n.country_code IS NOT NULL THEN 'nearest' END AS country_method
        FROM r6 r LEFT JOIN centroid c USING (h3_cell) LEFT JOIN mean m USING (h3_cell)
        LEFT JOIN nearest n USING (h3_cell)"""


def _places_sql() -> str:
    parents = " UNION ALL ".join(
        f"SELECT {res} AS resolution, h3_cell_to_parent(h3_cell::UBIGINT, {res})::BIGINT AS h3_cell, "
        f"country_code, NULL AS country_method, detections FROM r6_country" for res in RESOLUTIONS[:-1])
    return f"""
        WITH members AS (
            SELECT {max(RESOLUTIONS)} AS resolution, h3_cell, country_code, country_method, detections
            FROM r6_country UNION ALL {parents}),
        per_country AS (
            SELECT resolution, h3_cell, country_code, any_value(country_method) AS country_method,
                   sum(detections) AS detections
            FROM members GROUP BY ALL),
        ranked AS (
            SELECT *, sum(detections) OVER cell AS cell_detections,
                   row_number() OVER (cell ORDER BY (country_code IS NULL), detections DESC, country_code) AS rank
            FROM per_country WINDOW cell AS (PARTITION BY resolution, h3_cell))
        SELECT r.resolution::INTEGER AS resolution, r.h3_cell,
               h3_cell_to_lat(r.h3_cell::UBIGINT) AS latitude, h3_cell_to_lng(r.h3_cell::UBIGINT) AS longitude,
               k.continent, r.country_code, k.country_iso_a2, k.country_name, k.country_name_pt,
               CASE WHEN r.country_code IS NOT NULL THEN r.detections / r.cell_detections END AS country_share,
               CASE WHEN r.country_code IS NULL THEN NULL
                    WHEN r.resolution = {max(RESOLUTIONS)} THEN r.country_method ELSE 'predominant' END
                   AS country_method,
               r.cell_detections::BIGINT AS detections,
               CASE WHEN r.resolution = {max(RESOLUTIONS)} AND s.typed >= {STATIC_MIN_TYPED}
                    THEN s.static / s.typed END AS static_share,
               CASE WHEN r.resolution = {max(RESOLUTIONS)} AND s.typed >= {STATIC_MIN_TYPED}
                    THEN s.static >= {STATIC_SHARE} * s.typed END AS static_source
        FROM ranked r LEFT JOIN countries k USING (country_code)
        LEFT JOIN r6 s ON r.resolution = {max(RESOLUTIONS)} AND s.h3_cell = r.h3_cell
        WHERE r.rank = 1
        ORDER BY r.resolution, r.h3_cell"""


def _load_cities(path: Path) -> tuple[np.ndarray, pa.Table]:
    """GeoNames cities: (lat/lon radians for BallTree, attribute table aligned by row)."""
    ids, names, countries, lats, lons = [], [], [], [], []
    with zipfile.ZipFile(path) as archive, archive.open(CITIES_MEMBER) as raw:
        for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(int(row[0]))
            names.append(row[1])
            lats.append(float(row[4]))
            lons.append(float(row[5]))
            countries.append(row[8] or None)
    coordinates = np.radians(np.column_stack([lats, lons]))
    return coordinates, pa.table({"city_geonameid": pa.array(ids, pa.int64()), "city_name": names,
                                  "city_country_code": countries})


def _with_cities(places: pa.Table, cities_path: Path) -> pa.Table:
    coordinates, cities = _load_cities(cities_path)
    tree = BallTree(coordinates, metric="haversine")
    eligible = np.flatnonzero(places["resolution"].to_numpy() >= CITY_MIN_RESOLUTION)
    points = np.radians(np.column_stack([places["latitude"].to_numpy()[eligible],
                                         places["longitude"].to_numpy()[eligible]]))
    distance, index = tree.query(points, k=1)
    mask = np.ones(places.num_rows, dtype=bool)
    mask[eligible] = False
    full_index = np.zeros(places.num_rows, dtype=np.int64)
    full_index[eligible] = index[:, 0]
    full_distance = np.zeros(places.num_rows)
    full_distance[eligible] = distance[:, 0] * EARTH_RADIUS_KM
    for name in cities.column_names:
        column = cities[name].take(pa.array(full_index))
        places = places.append_column(name, pc.if_else(pa.array(mask), pa.nulls(len(column), column.type), column))
    return places.append_column("city_distance_km", pa.array(full_distance, mask=mask))


def _local_copy(source: str | Path, folder: Path) -> Path:
    """Local path of a reference file; s3:// (or other pyarrow) URIs are copied into `folder` first."""
    if "://" not in str(source):
        return Path(source)
    filesystem, path = pafs.FileSystem.from_uri(str(source))
    local = folder / Path(path).name
    with filesystem.open_input_stream(path) as stream, local.open("wb") as copy:
        shutil.copyfileobj(stream, copy, 8 * 1024 * 1024)
    return local


def build(silver: str, countries: str | Path, cities: str | Path, target: str, *,
          storage_options: dict[str, str] | None = None, memory_limit: str = "4GB", threads: int = 4,
          temp_dir: str | Path | None = None) -> dict:
    """Rebuild the whole place lookup from a pinned silver version in one Delta overwrite.

    `countries` and `cities` may be local paths or s3:// URIs."""
    if threads < 1:
        raise ValueError("threads must be positive")
    source = DeltaTable(silver, storage_options=storage_options)
    with tempfile.TemporaryDirectory(dir=temp_dir) as scratch:
        countries, cities = _local_copy(countries, Path(scratch)), _local_copy(cities, Path(scratch))
        with duckdb.connect() as connection:
            connection.execute(f"SET memory_limit={_literal(memory_limit)}")
            connection.execute(f"SET threads={threads}")
            connection.execute("SET preserve_insertion_order=false")
            if temp_dir is not None:
                connection.execute(f"SET temp_directory={_literal(str(temp_dir))}")
            connection.execute("INSTALL spatial")
            connection.execute("LOAD spatial")
            load_h3(connection)
            connection.register("silver", source.to_pyarrow_dataset())
            connection.execute(_countries_sql(countries))
            connection.execute(f"""CREATE TABLE r6 AS
                SELECT h3_r{max(RESOLUTIONS)} AS h3_cell, count(*)::BIGINT AS detections,
                       avg(latitude) AS mean_lat, avg(longitude) AS mean_lon,
                       h3_cell_to_lat(h3_r{max(RESOLUTIONS)}::UBIGINT) AS cell_lat,
                       h3_cell_to_lng(h3_r{max(RESOLUTIONS)}::UBIGINT) AS cell_lon,
                       count(hotspot_type) AS typed, count(*) FILTER (WHERE hotspot_type IN (1, 2, 3)) AS static
                FROM silver GROUP BY h3_r{max(RESOLUTIONS)}""")
            connection.execute(_assign_r6_sql())
            places = connection.sql(_places_sql()).to_arrow_table()
            methods = dict(connection.execute(
                "SELECT coalesce(country_method, 'none'), sum(detections) FROM r6_country GROUP BY 1").fetchall())
            static = connection.execute(f"""
                SELECT count(*) FILTER (WHERE typed >= {STATIC_MIN_TYPED} AND static >= {STATIC_SHARE} * typed),
                       coalesce(sum(detections) FILTER (WHERE typed >= {STATIC_MIN_TYPED}
                                                        AND static >= {STATIC_SHARE} * typed), 0),
                       count(*) FILTER (WHERE typed < {STATIC_MIN_TYPED}) FROM r6""").fetchall()[0]
        places = _with_cities(places, cities)
        metadata = {"method_version": METHOD_VERSION, "silver": silver, "silver_version": str(source.version()),
                    "countries_sha256": _sha256(countries), "cities_sha256": _sha256(cities),
                    "static_share": str(STATIC_SHARE), "static_min_typed": str(STATIC_MIN_TYPED)}
    write_deltalake(target, places, mode="overwrite", partition_by=PARTITIONS, schema_mode="overwrite",
                    storage_options=storage_options, commit_properties=CommitProperties(custom_metadata=metadata))
    counts = {f"r{res}": n for res, n in zip(*np.unique(places["resolution"].to_numpy(), return_counts=True))}
    return {**metadata, "target": target, "counts": {k: int(v) for k, v in counts.items()},
            "r6_detections_by_method": {k: int(v) for k, v in methods.items()},
            "static_sources": {"cells": int(static[0]), "detections": int(static[1]),
                               "cells_without_enough_typed": int(static[2])},
            "limitations": LIMITATIONS}


def verify(target: str, silver: str, *, storage_options: dict[str, str] | None = None,
           memory_limit: str = "4GB") -> dict:
    """Check the lookup against the silver version it was built from: one row per silver cell per
    resolution, detections summing to silver, consistent country/city fields."""
    table = DeltaTable(target, storage_options=storage_options)
    built_from = int(table.history(1)[0]["silver_version"])
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.register("places", table.to_pyarrow_dataset())
        connection.register("silver", DeltaTable(silver, version=built_from,
                                                 storage_options=storage_options).to_pyarrow_dataset())
        total = connection.execute("SELECT count(*) FROM silver").fetchone()[0]
        expected = dict(zip(RESOLUTIONS, connection.execute(
            "SELECT " + ", ".join(f"count(DISTINCT h3_r{r})" for r in RESOLUTIONS) + " FROM silver").fetchone()))
        rows = connection.execute(f"""
            SELECT resolution, count(*), count(DISTINCT h3_cell), sum(detections),
                   count(*) FILTER (WHERE (country_code IS NULL) <> (continent IS NULL)
                                       OR (country_code IS NULL) <> (country_share IS NULL)
                                       OR country_share <= 0 OR country_share > 1),
                   count(*) FILTER (WHERE (city_name IS NULL) <> (resolution < {CITY_MIN_RESOLUTION})),
                   sum(detections) FILTER (WHERE country_code IS NULL)
            FROM places GROUP BY 1 ORDER BY 1""").fetchall()
    errors = [(r, n, distinct, detections, bad_country, bad_city) for r, n, distinct, detections, bad_country, bad_city, _
              in rows if n != distinct or n != expected[r] or detections != total or bad_country or bad_city]
    if errors or {r for r, *_ in rows} != set(RESOLUTIONS):
        raise RuntimeError(f"h3_places disagrees with silver v{built_from}: {errors}")
    return {"version": table.version(), "silver_version": built_from, "silver_detections": total,
            "rows": {f"r{r}": n for r, n, *_ in rows},
            "detections_without_country": {f"r{r}": int(u or 0) for r, *_, u in rows}}
