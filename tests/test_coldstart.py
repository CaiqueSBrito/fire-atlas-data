from datetime import date, datetime, timedelta, timezone
import zipfile

from deltalake import DeltaTable, write_deltalake
import duckdb
import h3
import pyarrow as pa

from fire_atlas_data.coldstart import build

CENTER = h3.latlng_to_cell(-16.5, -56.5, 6)
LAT, LNG = h3.cell_to_latlng(CENTER)
_WEST = sorted(h3.grid_ring(CENTER, 1), key=lambda c: h3.cell_to_latlng(c)[1])  # land side first
COLD, WARM, OFFSHORE = _WEST[0], _WEST[1], _WEST[2]
ISSUE = date(2025, 9, 10)
CUTOFF = datetime(2025, 9, 10, 9, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
# Land ends just east of CENTER: eastern neighbours are sea.
LAND = f"POLYGON((-60 -20, {LNG + 0.02} -20, {LNG + 0.02} -10, -60 -10, -60 -20))"


def _silver(path):
    rows = [(CENTER, datetime(2025, 8, 1, 12, tzinfo=timezone.utc), 0),   # warm since August
            (WARM, datetime(2025, 8, 2, 12, tzinfo=timezone.utc), 0),
            (CENTER, CUTOFF - HOUR, 0),                                     # neighbour activity of COLD at ISSUE
            (OFFSHORE, CUTOFF - 30 * HOUR, 3),                              # type 3: outside the universe, not cold
            (COLD, CUTOFF + 2 * HOUR, 0),                                   # COLD's first-ever detection, window ISSUE
            (CENTER, datetime(2025, 10, 20, 12, tzinfo=timezone.utc), 0)]   # keeps silver beyond September targets
    write_deltalake(path, pa.table({
        "h3_r6": pa.array([h3.str_to_int(c) for c, *_ in rows], pa.int64()),
        "h3_r4": pa.array([h3.str_to_int(h3.cell_to_parent(c, 4)) for c, *_ in rows], pa.int64()),
        "h3_r1": pa.array([h3.str_to_int(h3.cell_to_parent(c, 1)) for c, *_ in rows], pa.int64()),
        "acquired_at": pa.array([t for _, t, _ in rows], pa.timestamp("us", tz="UTC")),
        "sensor": ["VIIRS"] * len(rows), "frp_mw": pa.array([1.0] * len(rows), pa.float64()),
        "brightness_k": pa.array([None] * len(rows), pa.float64()),
        "brightness_i4_k": pa.array([340.0] * len(rows), pa.float64()),
        "hotspot_type": pa.array([k for *_, k in rows], pa.int32()),
        "acquisition_year": pa.array([t.year for _, t, _ in rows], pa.int32()),
        "acquisition_month": pa.array([t.month for _, t, _ in rows], pa.int32())}))


def _coverage(path):
    """A pass every window for every res4 parent around CENTER, except COLD's parent on ISSUE + 2."""
    parents = {h3.str_to_int(h3.cell_to_parent(c, 4)) for c in h3.grid_disk(CENTER, 3)}
    days = [date(2025, 8, 1) + timedelta(days=i) for i in range(80)]
    gap = (h3.str_to_int(h3.cell_to_parent(COLD, 4)), ISSUE + timedelta(days=2))
    rows = [(p, d) for p in parents for d in days if (p, d) != gap]
    write_deltalake(path, pa.table({"h3_r4": pa.array([p for p, _ in rows], pa.int64()),
                                    "window_date": pa.array([d for _, d in rows], pa.date32()),
                                    "year": pa.array([d.year for _, d in rows], pa.int32())}),
                    commit_properties=None)
    return gap


def _countries(tmp_path):
    folder = tmp_path / "ne"
    folder.mkdir()
    with duckdb.connect() as connection:
        connection.execute("INSTALL spatial")
        connection.execute("LOAD spatial")
        connection.execute(f"""COPY (
            SELECT * FROM (VALUES ('LND', 'LD', 'Land', 'Terra', 'South America', ST_GeomFromText('{LAND}')))
                t(ADM0_A3, ISO_A2_EH, NAME, NAME_PT, CONTINENT, geom))
            TO '{folder}/ne_10m_admin_0_countries.shp' (FORMAT GDAL, DRIVER 'ESRI Shapefile')""")
    path = tmp_path / "countries.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in folder.iterdir():
            archive.write(file, file.name)
    return path


def test_population_is_cold_land_cells_with_neighbour_activity_and_labels_first_detection(tmp_path):
    silver, coverage, target = str(tmp_path / "silver"), str(tmp_path / "coverage"), str(tmp_path / "cold")
    _silver(silver)
    gap_parent, _ = _coverage(coverage)
    manifest = build(silver, coverage, _countries(tmp_path), target, months=[(2025, 9)], sample_rate=1.0,
                     test_sample_rate=1.0, memory_limit="1GB", threads=1)
    rows = DeltaTable(target).to_pyarrow_table().to_pylist()
    on_issue = {r["h3_r6"]: r for r in rows if r["forecast_date"] == ISSUE}
    cold, center, warm, offshore = (h3.str_to_int(c) for c in (COLD, CENTER, WARM, OFFSHORE))

    # Warm cells never enter, nor a cell whose only earlier detection is type 3 (it has own inputs);
    # COLD is cold up to its first window and gone afterwards.
    assert center not in on_issue and warm not in on_issue and offshore not in on_issue
    assert on_issue[cold]["target"] == 1 and on_issue[cold]["weight"] == 1.0
    assert on_issue[cold]["ring1_det_24h"] == 1 and on_issue[cold]["det_7d"] == 0
    assert not [r for r in rows if r["h3_r6"] == cold and r["forecast_date"] > ISSUE]
    # Sea cells (centroid east of the land edge) are out of the population.
    assert all(h3.cell_to_latlng(h3.int_to_str(r["h3_r6"]))[1] < LNG + 0.02 for r in rows)
    # Missing pass in the target window -> unknown, not a negative (dropped from rows, counted).
    missing = [r for r in rows if r["forecast_date"] in (ISSUE - timedelta(days=4), ISSUE + timedelta(days=2))
               and h3.str_to_int(h3.cell_to_parent(h3.int_to_str(r["h3_r6"]), 4)) == gap_parent
               and r["target"] == 0]
    assert not missing
    counts = manifest["counts"]["2025-09"]
    assert counts["unknown"] > 0
    # First windows: COLD (ISSUE) and OFFSHORE (ISSUE-2), each reachable from 7 issue days. In population:
    # COLD at ISSUE (CENTER active) and at ISSUE-1 (OFFSHORE active in window ISSUE-2); OFFSHORE never.
    assert counts["cold_positive_all"] == 14 and counts["cold_positive_in_population"] == 2
