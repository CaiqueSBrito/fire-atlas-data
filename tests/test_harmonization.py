from datetime import date, datetime, timedelta, timezone

from deltalake import DeltaTable, write_deltalake
import duckdb
import h3
import pyarrow as pa
import pytest

from fire_atlas_data import harmonization
from fire_atlas_data.silver import load_h3


def _region(lat, lon):
    """700 res6 cells of one res2 region, all inside the 10-20 degree latitude band."""
    cells = sorted(c for c in h3.cell_to_children(h3.latlng_to_cell(lat, lon, 2), 6)
                   if -19.5 < h3.cell_to_latlng(c)[0] < -10.5)
    assert len(cells) >= 700
    return cells[:700]


X, Y = _region(-15.0, -55.0), _region(-15.0, -45.0)


def _rows(platform, cells, days):
    """One daytime detection per cell and day, at 13:30 local solar time."""
    rows = []
    for day in days:
        for cell in cells:
            lat, lon = h3.cell_to_latlng(cell)
            when = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc) - timedelta(hours=lon / 15)
            rows.append((platform, cell, lat, lon, when, "D"))
    return rows


def _silver(rows):
    return pa.table({
        "platform": [r[0] for r in rows],
        "h3_r6": pa.array([h3.str_to_int(r[1]) for r in rows], pa.int64()),
        "latitude": [r[2] for r in rows], "longitude": [r[3] for r in rows],
        "acquired_at": pa.array([r[4] for r in rows], pa.timestamp("us", tz="UTC")),
        "daynight": [r[5] for r in rows], "frp_mw": [10.0] * len(rows),
        "scan_km": [1.0] * len(rows), "track_km": [1.0] * len(rows),
        "confidence_category": pa.array([None] * len(rows), pa.string()),
        "confidence_percent": pa.array([None] * len(rows), pa.int16()),
        "brightness_i4_k": [340.0] * len(rows), "brightness_k": pa.array([None] * len(rows), pa.float64()),
        "brightness_i5_k": [300.0] * len(rows), "bright_t31_k": pa.array([None] * len(rows), pa.float64()),
        "acquisition_year": pa.array([r[4].year for r in rows], pa.int32()),
        "acquisition_month": pa.array([r[4].month for r in rows], pa.int32())})


def _august(year):
    return [date(year, 8, d) for d in range(10, 15)]


def _record():
    """Aqua reports a quarter of the Suomi-NPP units in X and half in Y; NOAA-20 misses one day of 2020.
    In the reserved year 2018 Aqua reports every Suomi-NPP unit, a relation the fit must not learn.
    In 2001 only Terra flies."""
    rows = []
    for cells, seen in ((X, 175), (Y, 350)):
        rows += _rows("Suomi-NPP", cells, _august(2016)) + _rows("Aqua", cells[:seen], _august(2016))
        rows += _rows("Terra", cells[:140], _august(2016))
    rows += _rows("Suomi-NPP", X, _august(2020)) + _rows("Aqua", X[:500], _august(2020))
    rows += _rows("NOAA-20", X[:350], [d for d in _august(2020) if d.day != 12])
    rows += _rows("NOAA-20", X[:350], _august(2023)) + _rows("NOAA-21", X, _august(2023))
    rows += _rows("Suomi-NPP", X, _august(2023))
    rows += _rows("Suomi-NPP", X, _august(2018)) + _rows("Aqua", X, _august(2018))
    rows += _rows("NOAA-20", X[:350], _august(2025)) + _rows("NOAA-21", X, _august(2025))
    rows += _rows("Terra", X[:140], _august(2001))
    return rows


def _sum(table, column, day, cells):
    wanted = {h3.str_to_int(c) for c in cells}
    return sum(r[column] or 0 for r in table if r["resolution"] == 6 and r["day"] == day and r["h3_cell"] in wanted)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("harmonization")
    silver, model = str(folder / "silver"), folder / "model"
    write_deltalake(silver, _silver(_record()))
    harmonization.fit(silver, model, memory_limit="1GB", threads=1)
    return silver, model


def test_shrinkage_pulls_sparse_regional_factors_toward_their_parent():
    rich, sparse = (h3.str_to_int(c) for c in sorted(h3.cell_to_children(h3.latlng_to_cell(-15.0, -55.0, 1), 2))[:2])
    parent = h3.str_to_int(h3.latlng_to_cell(-15.0, -55.0, 1))
    connection = duckdb.connect()
    connection.execute(f"""CREATE TABLE pair AS SELECT * FROM (VALUES
        ({rich}, {parent}, 8, 1000.0, 4000, 250.0), ({sparse}, {parent}, 8, 10.0, 40, 10.0))
        t(r2, r1, moy, den, units, num)""")
    connection.execute("CREATE TABLE factor (platform VARCHAR, level INTEGER, region BIGINT, moy INTEGER, "
                       "factor DOUBLE, units BIGINT)")
    connection.execute(harmonization._factor_sql("Suomi-NPP", 100.0))
    factors = {(level, region): f for level, region, f in
               connection.execute("SELECT level, region, factor FROM factor").fetchall()}
    overall = 260 / 1010
    regional = (260 + 100 * overall) / (1010 + 100)
    assert factors[(0, 0)] == pytest.approx(overall)
    assert factors[(1, parent)] == pytest.approx(regional)
    assert factors[(2, rich)] == pytest.approx((250 + 100 * regional) / 1100)       # data dominates: ~0.25
    assert factors[(2, sparse)] == pytest.approx((10 + 100 * regional) / 110)       # 10 units: near the parent
    assert abs(factors[(2, sparse)] - regional) < abs(1.0 - regional) / 5


def test_detection_model_replaces_the_viirs_curve_and_keeps_the_aqua_scale(tmp_path):
    silver, model = str(tmp_path / "silver"), tmp_path / "model"
    write_deltalake(silver, _silver(_record()))
    fitted = harmonization.fit(silver, model, kappa=30.0, detection=True, memory_limit="1GB", threads=1)
    assert fitted["detection"]["platforms"] == ["Suomi-NPP", "NOAA-20", "NOAA-21"]
    assert (model / harmonization.DETECTION_FILE).exists()
    target, availability = str(tmp_path / "cells"), str(tmp_path / "availability")
    harmonization.apply(silver, f"file://{model}", target, availability, years=[2016, 2020], memory_limit="1GB",
                        threads=1)
    harmonization.verify(target, silver, years=[2016, 2020], memory_limit="1GB", threads=1)
    cells = DeltaTable(target).to_pyarrow_table().to_pylist()
    # Factors refitted on the model probabilities still put Suomi-NPP on Aqua's regional totals (175 in X).
    assert _sum(cells, "aqua_eq_snpp", date(2016, 8, 12), X) == pytest.approx(175, rel=0.15)
    assert _sum(cells, "aqua_eq_snpp", date(2016, 8, 12), Y) == pytest.approx(350, rel=0.15)
    assert _sum(cells, "harmonized", date(2020, 8, 11), X) > 0


def test_every_platform_lands_on_the_aqua_scale_and_outages_do_not_lower_the_series(fitted, tmp_path):
    silver, model = fitted
    target, availability = str(tmp_path / "cells"), str(tmp_path / "availability")
    harmonization.apply(silver, model, target, availability, years=[2001, 2016, 2018, 2020, 2023],
                        memory_limit="1GB", threads=1)
    harmonization.verify(target, silver, years=[2001, 2016, 2018, 2020, 2023], memory_limit="1GB", threads=1)
    cells = DeltaTable(target).to_pyarrow_table().to_pylist()
    for day in (date(2018, 8, 12), date(2020, 8, 11)):
        assert _sum(cells, "harmonized_t340", day, X) == pytest.approx(_sum(cells, "harmonized", day, X))
        assert _sum(cells, "harmonized_t330", day, X) == 0
        assert _sum(cells, "harmonized_t_none", day, X) == 0

    # Regional factors: the same 700 Suomi-NPP units are worth 175 Aqua units in X and 350 in Y.
    day = date(2016, 8, 12)
    assert _sum(cells, "aqua_eq_snpp", day, X) == pytest.approx(175)
    assert _sum(cells, "aqua_eq_snpp", day, Y) == pytest.approx(350)
    assert _sum(cells, "aqua_eq_terra", day, X) == pytest.approx(175)
    assert _sum(cells, "harmonized", day, X) == pytest.approx(175)
    assert _sum(cells, "units_snpp", day, X) == 700          # the original per-platform count is kept

    # While Aqua works (up to 2019) the series is Aqua itself, not a mean with the others: in 2018 Aqua sees
    # 700 where the Suomi-NPP estimate says 175.
    reserved = date(2018, 8, 12)
    assert _sum(cells, "aqua_eq_snpp", reserved, X) == pytest.approx(175)
    assert _sum(cells, "harmonized", reserved, X) == pytest.approx(700)
    assert {(r["basis"], r["low_confidence"], r["contributing"]) for r in cells if r["day"] == reserved} == \
        {("aqua", False, 1)}
    # Terra alone (before Aqua): Terra estimate, flagged as low confidence.
    early = date(2001, 8, 12)
    assert _sum(cells, "harmonized", early, X) == pytest.approx(175)
    assert {(r["basis"], r["low_confidence"]) for r in cells if r["day"] == early} == {("terra", True)}

    # After the reference period only VIIRS contributes: 500 Aqua units do not move the series.
    assert _sum(cells, "harmonized", date(2020, 8, 11), X) == pytest.approx(175)
    assert _sum(cells, "aqua_eq_aqua", date(2020, 8, 11), X) == pytest.approx(500)
    # NOAA-20 outage on the 12th: the day rests on Suomi-NPP alone instead of dropping by half.
    outage = date(2020, 8, 12)
    assert _sum(cells, "harmonized", outage, X) == pytest.approx(175)
    assert {r["contributing"] for r in cells if r["day"] == outage} == {1}
    # A parent cell holds the sum of its res6 children: region X is one res2 cell.
    parent = h3.str_to_int(h3.cell_to_parent(X[0], 2))
    assert sum(r["harmonized"] for r in cells if r["resolution"] == 2 and r["h3_cell"] == parent
               and r["day"] == outage) == pytest.approx(175)
    flags = {(r["platform"], r["day"]): (r["available"], r["contributes"])
             for r in DeltaTable(availability).to_pyarrow_table().to_pylist()}
    assert flags[("NOAA-20", outage)] == (False, False)
    assert flags[("Aqua", outage)] == (True, False)
    assert flags[("Aqua", reserved)] == (True, True) and flags[("Suomi-NPP", reserved)] == (True, False)

    # Chain: NOAA-21, tied to the NOAA-20 estimate, reports the same activity with twice the units.
    assert _sum(cells, "aqua_eq_noaa20", date(2023, 8, 12), X) == pytest.approx(175)
    assert _sum(cells, "aqua_eq_noaa21", date(2023, 8, 12), X) == pytest.approx(175)


def test_model_loads_from_object_storage_uri(fitted, tmp_path):
    silver, model = fitted
    target, availability = str(tmp_path / "cells"), str(tmp_path / "availability")
    harmonization.apply(silver, f"file://{model}", target, availability, years=[2016], memory_limit="1GB", threads=1)
    cells = DeltaTable(target).to_pyarrow_table().to_pylist()
    assert _sum(cells, "harmonized", date(2016, 8, 12), X) == pytest.approx(175)


def test_temperature_bins_split_harmonized_by_the_brightness_of_each_unit(fitted, tmp_path):
    _, model = fitted
    brightness = [315.0, 320.0, 345.0, 360.0, 365.0, None]   # MODIS channel 21/22; the edges open the upper bin
    rows = _rows("Aqua", X[:len(brightness)], [date(2018, 8, 12)])
    table = _silver(rows)
    table = table.set_column(table.schema.get_field_index("brightness_i4_k"), "brightness_i4_k",
                             pa.array([None] * len(rows), pa.float64()))
    table = table.set_column(table.schema.get_field_index("brightness_k"), "brightness_k",
                             pa.array(brightness, pa.float64()))
    silver, target, availability = str(tmp_path / "silver"), str(tmp_path / "cells"), str(tmp_path / "availability")
    write_deltalake(silver, table)
    harmonization.apply(silver, model, target, availability, years=[2018], memory_limit="1GB", threads=1)
    harmonization.verify(target, silver, years=[2018], memory_limit="1GB", threads=1)
    cells = DeltaTable(target).to_pyarrow_table().to_pylist()
    day = date(2018, 8, 12)
    split = {name: _sum(cells, f"harmonized_{name}", day, X) for name, _ in harmonization.TEMPERATURE_BINS}
    assert split == {"t_lt320": 1, "t320": 1, "t330": 0, "t340": 1, "t350": 0, "t_ge360": 2, "t_none": 1}
    # Parents add the bins of their children like they add `harmonized`.
    parent = h3.str_to_int(h3.cell_to_parent(X[0], 0))
    assert [r["harmonized_t_ge360"] for r in cells if r["resolution"] == 0 and r["h3_cell"] == parent] == [2]


def test_final_test_scores_reserved_years_the_fit_never_saw_and_runs_once(fitted, tmp_path):
    silver, model = fitted
    report = harmonization.evaluate(silver, model, tmp_path / "report.json", memory_limit="1GB", threads=1)

    # The fit learned 175 of 700 from 2016; 2018 (Aqua sees all 700) did not leak into it.
    scored = report["test"]["Suomi-NPP"]["res2_month"]
    assert scored["aqua_units"] == 3500
    assert scored["bias"] == pytest.approx(0.25) and scored["wape"] == pytest.approx(0.75)
    assert scored["raw_bias"] == pytest.approx(1.0)
    assert report["test"]["combined_without_aqua"]["platforms"] == ["Suomi-NPP"]
    # Platforms tied by the chain agree on the same burning in years outside their fit.
    checks = {(c["platform"], c["year"]): c for c in report["consistency_checks"]}
    assert checks[("NOAA-20", 2023)]["ratio"] == pytest.approx(1.0)
    assert checks[("NOAA-21", 2025)]["ratio"] == pytest.approx(1.0)
    with pytest.raises(FileExistsError):
        harmonization.evaluate(silver, model, tmp_path / "report.json", memory_limit="1GB", threads=1)


def test_era_report_scores_each_era_composition_against_aqua(fitted, tmp_path):
    silver, model = fitted
    report = harmonization.evaluate_eras(silver, model, tmp_path / "eras.json", drift_years=(2020, 2021),
                                         memory_limit="1GB", threads=1)
    eras = report["eras"]
    # 2018: Aqua 700, Suomi-NPP estimate 175, no Terra or NOAA-20 that year.
    assert eras["E3_2018..2019_published"]["res2_month"]["bias"] == pytest.approx((700 + 175) / 2 / 700)
    assert eras["E4_2020+_viirs"]["res2_month"]["bias"] == pytest.approx(175 / 700)
    # Agreement with Aqua after 2019 (Aqua 500 vs VIIRS 175 in 2020); years without data are skipped.
    assert report["viirs_vs_aqua_agreement_not_error"]["2020"]["res2_month"]["bias"] == pytest.approx(175 / 500)
    assert "2021" not in report["viirs_vs_aqua_agreement_not_error"]
    assert report["label"].startswith("2018-2019 reused")


def test_local_fire_day_runs_from_six_to_six_local_solar_time():
    utc = lambda *args: datetime(*args, tzinfo=timezone.utc)
    cases = {"east": (0.0, 150.0, utc(2016, 8, 10, 23, 30)),    # 09:30 local on the 11th
             "west": (0.0, -150.0, utc(2016, 8, 11, 2, 0)),     # 16:00 local on the 10th
             "night": (0.0, 0.0, utc(2016, 8, 11, 1, 30)),      # night pass belongs to the day before
             "dawn": (5.0, 0.0, utc(2016, 8, 11, 6, 0))}        # 06:00 opens the 11th
    cells = {name: h3.latlng_to_cell(lat, lon, 6) for name, (lat, lon, _) in cases.items()}
    rows = [("Aqua", cells[name], lat, lon, when, "D") for name, (lat, lon, when) in cases.items()]
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    load_h3(connection)
    connection.register("silver", _silver(rows))
    connection.execute(harmonization._units_sql(date(2016, 8, 9), date(2016, 8, 12)))
    found = dict(connection.execute("SELECT c6, day FROM units").fetchall())
    days = {name: date(1970, 1, 1) + timedelta(days=found[h3.str_to_int(cell)]) for name, cell in cells.items()}
    assert days == {"east": date(2016, 8, 11), "west": date(2016, 8, 10),
                    "night": date(2016, 8, 10), "dawn": date(2016, 8, 11)}
