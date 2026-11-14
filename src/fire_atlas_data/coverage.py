"""Satellite pass coverage from NASA CMR granule footprints (RF10/CA08, confirmation of target=0).

`fetch_granules` stores, untouched, the footprint polygon and time range of every
active-fire L2 swath granule (5-6 min of orbit) in a Delta table partitioned by
product and year. Only CMR metadata is read; no imagery is downloaded.

A footprint proves the satellite passed over an area, not that it saw the ground:
cloud, sun glint and bad scans are not excluded. Coverage without a detection is
therefore stronger than "no detection", but is not proof of absence of fire.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import h3
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
from deltalake import CommitProperties, DeltaTable, write_deltalake
from sklearn.neighbors import KDTree

METHOD_VERSION = "1.0.0"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
PAGE_SIZE = 2000
# product -> (CMR collection concept id, sensor, platform as named in silver, first year with data)
PRODUCTS = {
    "MOD14": ("C2271754179-LPCLOUD", "MODIS", "Terra", 2000),
    "MYD14": ("C2278858993-LPCLOUD", "MODIS", "Aqua", 2002),
    "VNP14IMG": ("C2734202914-LPCLOUD", "VIIRS", "Suomi-NPP", 2012),
    "VJ114IMG": ("C2734197957-LPCLOUD", "VIIRS", "NOAA-20", 2018),
    "VJ214IMG": ("C2831626262-LPCLOUD", "VIIRS", "NOAA-21", 2023),
}
GRANULE_PARTITIONS = ["product", "year"]
GRANULE_SCHEMA = pa.schema([
    ("granule_id", pa.string()), ("producer_granule_id", pa.string()), ("collection_concept_id", pa.string()),
    ("sensor", pa.string()), ("platform", pa.string()),
    ("time_start", pa.timestamp("us", tz="UTC")), ("time_end", pa.timestamp("us", tz="UTC")),
    ("day_night_flag", pa.string()), ("footprint", pa.string()), ("cmr_updated", pa.string()),
    ("product", pa.string()), ("year", pa.int32()),
])
LIMITATIONS = [
    "Footprint means the satellite passed, not that the ground was visible (cloud, glint, bad scans not excluded).",
    "Granules absent from CMR (outages, reprocessing gaps) leave the area unknown, never confirmed negative.",
    "footprint is the raw CMR polygon string ('lat lon lat lon ...'); NULL when CMR has no polygon.",
]
CELL_RESOLUTION = 4
PASS_COLUMNS = {"MOD14": "passes_terra", "MYD14": "passes_aqua", "VNP14IMG": "passes_snpp",
                "VJ114IMG": "passes_noaa20", "VJ214IMG": "passes_noaa21"}
CELL_SCHEMA = pa.schema([("h3_r4", pa.int64()), ("window_date", pa.date32())]
                        + [(column, pa.int16()) for column in PASS_COLUMNS.values()] + [("year", pa.int32())])
CELL_LIMITATIONS = LIMITATIONS[:2] + [
    f"Coverage is tested at the center of each H3 res{CELL_RESOLUTION} cell (edge ~26 km), not per res6 cell or pixel; "
    "cells cut by a swath edge may be marked uncovered (unknown) or covered.",
    "Footprint edges are treated as great-circle arcs between the four CMR corners.",
    "Only cells present in the gold map table (at least one detection in history) are evaluated.",
    "A (cell, window) without a row had no recorded pass: unknown, never a confirmed negative.",
]
_CENTERS: np.ndarray | None = None  # per-process state of the cell workers, set by _init_cells
_TREE: KDTree | None = None


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _get_page(params: dict[str, str], search_after: str | None) -> tuple[list[dict], int, str | None]:
    """One CMR page: (entries, total hits, next search-after token). Retries transient network errors."""
    headers = {"CMR-Search-After": search_after} if search_after else {}
    request = Request(f"{CMR_GRANULES}?{urlencode(params)}", headers=headers)
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120) as response:
                entries = json.load(response)["feed"]["entry"]
                return entries, int(response.headers["CMR-Hits"]), response.headers.get("CMR-Search-After")
        except OSError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _year_granules(product: str, year: int) -> tuple[pa.Table, int]:
    """All granules of `product` whose time_start falls in `year`, plus the CMR hit count of the query."""
    concept_id, sensor, platform, _ = PRODUCTS[product]
    params = {"collection_concept_id": concept_id, "page_size": str(PAGE_SIZE), "sort_key": "start_date",
              "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z"}
    rows, fetched, search_after = [], 0, None
    while True:
        entries, hits, search_after = _get_page(params, search_after)
        fetched += len(entries)
        for entry in entries:
            start = _timestamp(entry["time_start"])
            if start.year != year:  # started in the previous year; belongs to that partition
                continue
            polygons = entry.get("polygons") or [[None]]
            rows.append({
                "granule_id": entry["id"], "producer_granule_id": entry.get("producer_granule_id"),
                "collection_concept_id": concept_id, "sensor": sensor, "platform": platform,
                "time_start": start, "time_end": _timestamp(entry["time_end"]),
                "day_night_flag": entry.get("day_night_flag"), "footprint": polygons[0][0],
                "cmr_updated": entry.get("updated"), "product": product, "year": year})
        if len(entries) < PAGE_SIZE or search_after is None:
            break
    if fetched != hits:
        raise RuntimeError(f"{product} {year}: fetched {fetched} granules but CMR reported {hits}")
    return pa.Table.from_pylist(rows, schema=GRANULE_SCHEMA), hits


def _fetched(target: str, storage_options: dict[str, str] | None) -> set[tuple[str, int]]:
    if not DeltaTable.is_deltatable(target, storage_options=storage_options):
        return set()
    table = DeltaTable(target, storage_options=storage_options)
    return {(p["product"], int(p["year"])) for p in table.partitions()}


def _unit(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude, longitude = np.radians(latitude), np.radians(longitude)
    return np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude),
                     np.sin(latitude)], axis=-1)


def _init_cells(cells: np.ndarray) -> None:
    global _CENTERS, _TREE
    centers = np.array([h3.cell_to_latlng(h3.int_to_str(int(cell))) for cell in cells])
    _CENTERS = _unit(centers[:, 0], centers[:, 1])
    _TREE = KDTree(_CENTERS)


def _pass_counts(footprints: list[str | None]) -> tuple[np.ndarray, int]:
    """Passes over each cell center by the given footprints, and how many footprints were unusable.

    A footprint is usable when it is a spherical quadrilateral whose centroid lies on the same side
    of its four great-circle edges; a center is covered when it lies on that side of all four.
    """
    counts = np.zeros(len(_CENTERS), dtype=np.int16)
    quads = [f.split() for f in footprints if f is not None]
    quads = [q[:8] for q in quads if len(q) == 10]
    skipped = len(footprints) - len(quads)
    if not quads:
        return counts, skipped
    corners = np.array(quads, dtype=np.float64).reshape(-1, 4, 2)
    vertices = _unit(corners[..., 0], corners[..., 1])
    centroid = vertices.sum(axis=1)
    centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
    normals = np.cross(vertices, np.roll(vertices, -1, axis=1))
    signs = np.sign(np.einsum("gkc,gc->gk", normals, centroid))
    usable = np.abs(signs.sum(axis=1)) == 4
    skipped += int((~usable).sum())
    vertices, centroid, normals, signs = vertices[usable], centroid[usable], normals[usable], signs[usable]
    if not len(centroid):
        return counts, skipped
    radius = np.linalg.norm(vertices - centroid[:, None, :], axis=2).max(axis=1)
    near = _TREE.query_radius(centroid, radius)
    granule = np.repeat(np.arange(len(near)), [len(n) for n in near])
    cell = np.concatenate(near)
    inside = np.ones(len(cell), dtype=bool)
    for edge in range(4):
        inside &= (normals[granule, edge] * _CENTERS[cell]).sum(axis=1) * signs[granule, edge] >= 0
    return np.bincount(cell[inside], minlength=len(counts)).astype(np.int16), skipped


def _month_passes(task: tuple[str, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(window day ordinals, cell positions, passes[n, products], unusable footprints) of one month of windows."""
    granules, year, month, window_start_hour = task
    start = datetime(year, month, 1, window_start_hour, tzinfo=timezone.utc)
    end = datetime(year + month // 12, month % 12 + 1, 1, window_start_hour, tzinfo=timezone.utc)
    table = DeltaTable(granules).to_pyarrow_dataset().to_table(
        columns=["product", "time_start", "footprint"],
        filter=(pads.field("year").isin([year, year + 1]) & (pads.field("time_start") >= start)
                & (pads.field("time_start") < end)))
    product = np.array(table["product"].to_pylist(), dtype=object)
    footprint = np.array(table["footprint"].to_pylist(), dtype=object)
    shifted = table["time_start"].to_numpy() - np.timedelta64(window_start_hour, "h")
    ordinal = shifted.astype("datetime64[D]").astype(np.int64)
    days, cells, passes, skipped = [], [], [], 0
    for day in np.unique(ordinal):
        in_day = ordinal == day
        counts = np.zeros((len(_CENTERS), len(PASS_COLUMNS)), dtype=np.int16)
        for position, name in enumerate(PASS_COLUMNS):
            counts[:, position], unusable = _pass_counts(list(footprint[in_day & (product == name)]))
            skipped += unusable
        covered = np.flatnonzero(counts.any(axis=1))
        days.append(np.full(len(covered), day, dtype=np.int32))
        cells.append(covered.astype(np.int32))
        passes.append(counts[covered])
    if not days:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty((0, len(PASS_COLUMNS)), np.int16), 0)
    return np.concatenate(days), np.concatenate(cells), np.concatenate(passes), skipped


def _map_cells(map_cells: str, storage_options: dict[str, str] | None) -> tuple[np.ndarray, int]:
    table = DeltaTable(map_cells, storage_options=storage_options)
    with duckdb.connect() as connection:
        connection.register("gold", table.to_pyarrow_dataset())
        cells = connection.execute(
            f"SELECT DISTINCT h3_cell FROM gold WHERE resolution = {CELL_RESOLUTION} ORDER BY 1").fetchnumpy()
    return cells["h3_cell"], table.version()


def _year_batches(pool: ProcessPoolExecutor, granules: str, year: int, window_start_hour: int,
                  cells: np.ndarray, skipped: list[int]):
    tasks = [(granules, year, month, window_start_hour) for month in range(1, 13)]
    for days, positions, passes, unusable in pool.map(_month_passes, tasks):
        skipped[0] += unusable
        if len(days):
            columns = [pa.array(cells[positions]), pa.array(days, type=pa.date32())]
            columns += [pa.array(passes[:, position]) for position in range(len(PASS_COLUMNS))]
            columns.append(pa.array(np.full(len(days), year, dtype=np.int32)))
            yield pa.record_batch(columns, schema=CELL_SCHEMA)


def fetch_granules(target: str, *, products: list[str] | None = None, years: list[int] | None = None,
                   last_year: int | None = None, refresh: bool = False, workers: int = 4,
                   storage_options: dict[str, str] | None = None) -> dict:
    """Fetch CMR footprints into the Delta table at `target`, one (product, year) commit each.

    Partitions already present are skipped unless `refresh`; the current year is always incomplete,
    so pass it in `years` with `refresh` to update it. `workers` (product, year) downloads run
    concurrently; commits stay sequential.
    """
    last_year = datetime.now(timezone.utc).year if last_year is None else last_year
    done = set() if refresh else _fetched(target, storage_options)
    pending = [(product, year) for product in products or list(PRODUCTS)
               for year in years or range(PRODUCTS[product][3], last_year + 1)
               if year >= PRODUCTS[product][3] and (product, year) not in done]
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (product, year), (granules, hits) in zip(pending, pool.map(lambda p: _year_granules(*p), pending)):
            exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
            metadata = {"method_version": METHOD_VERSION, "source": CMR_GRANULES,
                        "collection_concept_id": PRODUCTS[product][0], "cmr_hits": str(hits),
                        "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            write_deltalake(target, granules, mode="overwrite", partition_by=GRANULE_PARTITIONS,
                            predicate=f"product = '{product}' AND year = {year}" if exists else None,
                            storage_options=storage_options,
                            commit_properties=CommitProperties(custom_metadata=metadata))
            counts[f"{product}/{year}"] = granules.num_rows
            print(f"{product} {year}: {granules.num_rows} granules", file=sys.stderr, flush=True)
    return {"method_version": METHOD_VERSION, "target": target, "counts": counts, "limitations": LIMITATIONS}


def build_cells(granules: str, map_cells: str, target: str, *, years: list[int] | None = None,
                window_start_hour: int = 9, workers: int = 4,
                storage_options: dict[str, str] | None = None) -> dict:
    """Write (or replace) yearly pass counts per H3 res4 cell and 24 h window in the Delta table at `target`.

    Window `window_date` spans [window_date HH:00, next day HH:00) UTC with HH = `window_start_hour`
    (the effective input cutoff of the forecast); a pass belongs to the window of its granule start.
    `granules` must be a local table: workers read it directly.
    """
    source = DeltaTable(granules)
    available = sorted({int(p["year"]) for p in source.partitions()})
    selected = available if years is None else sorted(set(years))
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(f"Years absent from granules: {sorted(missing)}")
    cells, map_version = _map_cells(map_cells, storage_options)
    counts = {}
    # spawn: deltalake's runtime does not survive fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_cells, initargs=(cells,)) as pool:
        for year in selected:
            skipped = [0]
            reader = pa.RecordBatchReader.from_batches(
                CELL_SCHEMA, _year_batches(pool, granules, year, window_start_hour, cells, skipped))
            exists = DeltaTable.is_deltatable(target, storage_options=storage_options)
            write_deltalake(target, reader, mode="overwrite", partition_by=["year"],
                            predicate=f"year = {year}" if exists else None, storage_options=storage_options,
                            commit_properties=CommitProperties(custom_metadata={
                                "method_version": METHOD_VERSION, "granules_version": str(source.version()),
                                "map_cells_version": str(map_version), "cell_resolution": str(CELL_RESOLUTION),
                                "window_start_hour": str(window_start_hour), "cells": str(len(cells)),
                                "coverage_year": str(year)}))
            counts[str(year)] = {"unusable_footprints": skipped[0]}
            print(f"coverage cells {year}: unusable footprints {skipped[0]}", file=sys.stderr, flush=True)
    return {"method_version": METHOD_VERSION, "target": target, "cell_resolution": CELL_RESOLUTION,
            "window_start_hour": window_start_hour, "cells": len(cells), "counts": counts,
            "limitations": CELL_LIMITATIONS}


def verify_cells(target: str, silver: str, *, window_start_hour: int = 9, years: list[int] | None = None,
                 storage_options: dict[str, str] | None = None, memory_limit: str = "2GB",
                 temp_dir: str | None = None) -> dict:
    """Share of silver detections whose (res4 cell, window, platform) has a recorded pass, per platform.

    Every detection was physically observed, so misses measure footprint gaps plus the
    cell-center approximation at swath edges; nothing is raised, the rates are the evidence.
    """
    platforms = {PRODUCTS[product][2]: column for product, column in PASS_COLUMNS.items()}
    passes = "CASE s.platform " + " ".join(f"WHEN '{p}' THEN c.{col}" for p, col in platforms.items()) + " END"
    year_filter = "" if years is None else f"WHERE acquisition_year IN ({', '.join(str(y) for y in years)})"
    with duckdb.connect() as connection:
        connection.execute(f"SET memory_limit='{memory_limit}'")
        connection.execute("SET TimeZone='UTC'")
        if temp_dir is not None:
            connection.execute(f"SET temp_directory='{temp_dir}'")
        connection.register("silver", DeltaTable(silver, storage_options=storage_options).to_pyarrow_dataset())
        connection.register("cells", DeltaTable(target, storage_options=storage_options).to_pyarrow_dataset())
        rows = connection.execute(f"""
            WITH s AS (
                SELECT h3_r{CELL_RESOLUTION} AS cell, platform,
                       (acquired_at - INTERVAL {window_start_hour} HOUR)::DATE AS window_date, count(*) AS detections
                FROM silver {year_filter} GROUP BY ALL)
            SELECT s.platform, sum(s.detections), sum(s.detections) FILTER (WHERE coalesce({passes}, 0) > 0),
                   count(*), count(*) FILTER (WHERE coalesce({passes}, 0) > 0)
            FROM s LEFT JOIN cells c ON c.h3_r4 = s.cell AND c.window_date = s.window_date
            GROUP BY 1 ORDER BY 1""").fetchall()
    return {"version": DeltaTable(target, storage_options=storage_options).version(),
            "platforms": {platform: {"detections": int(total), "detections_with_pass": int(matched or 0),
                                     "cell_windows": int(groups), "cell_windows_with_pass": int(hit or 0)}
                          for platform, total, matched, groups, hit in rows}}
