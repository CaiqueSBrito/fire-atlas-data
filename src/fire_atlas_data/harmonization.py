"""Gold layer, harmonized burning activity: every MODIS/VIIRS platform on the Aqua scale (RF07/CA07).

Unit: an active (H3 res6 cell, local fire day, day/night) of one platform. The local fire day runs
06:00-06:00 local solar time, so the ~10:30/13:30 and ~22:30/01:30 passes sit hours from its boundary.
Platforms are never merged before conversion: each one is converted to "Aqua-equivalent" units, the
number of units Aqua is expected to report for the same burning.

Conversion of a unit of platform P = detection curve x regional factor:
- curve: P(Aqua also active | FRP sum, detections, day/night, latitude band of the unit), fitted on the
  overlap with Aqua. VIIRS platforms share the Suomi-NPP curve (same instrument design); Terra has its own.
- factor per (H3 res2 region, month of year), falling back to res1 x month, then global: reference total
  over curve total in the fit years. For Terra and Suomi-NPP the reference is Aqua itself; NOAA-20 is tied
  to the Suomi-NPP estimate and NOAA-21 to the NOAA-20 estimate (chain), so a platform entering or leaving
  does not move the series.

`harmonized` of a cell and day estimates what Aqua (nominal 2003-2019 orbit) would have recorded: Aqua itself
when it worked that day (up to REFERENCE_END); otherwise the mean of the VIIRS platforms working that day;
otherwise Terra (up to REFERENCE_END). `basis` says which; `low_confidence` marks Terra-based days.
The cells table holds H3 res6 rows and their sums for every parent resolution 0..5 (`resolution` column),
so a map or calendar at any zoom reads one resolution instead of aggregating res6 cells.
Years in TEST_YEARS are never read by `fit`; `evaluate` scores them once.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys
import tempfile
import time

import duckdb
from deltalake import CommitProperties, DeltaTable, write_deltalake
import numpy as np
import pyarrow as pa
import xgboost as xgb

from . import ml_dataset
from .places import _local_copy
from .silver import _literal, _sha256

METHOD_VERSION = "0.1.0"
PARTITIONS = ["resolution", "year"]
LEVELS = tuple(range(7))      # H3 resolutions written; the method works on res6 units
AVAILABILITY_PARTITIONS = ["year"]
REFERENCE = "Aqua"
REFERENCE_END = date(2019, 12, 31)   # last local day on which MODIS (Aqua, then Terra) can feed `harmonized`
TEST_YEARS = (2018, 2019)
# platform -> curve owner, reference series of its regional factor, fit years (inclusive)
LINKS = {
    "Terra": {"curve": "Terra", "target": "Aqua", "fit": (2003, 2017)},
    "Suomi-NPP": {"curve": "Suomi-NPP", "target": "Aqua", "fit": (2013, 2017)},
    "NOAA-20": {"curve": "Suomi-NPP", "target": "Suomi-NPP", "fit": (2020, 2021)},
    "NOAA-21": {"curve": "Suomi-NPP", "target": "NOAA-20", "fit": (2023, 2024)},
}
SLUGS = {"Terra": "terra", "Aqua": "aqua", "Suomi-NPP": "snpp", "NOAA-20": "noaa20", "NOAA-21": "noaa21"}
VIIRS = ("Suomi-NPP", "NOAA-20", "NOAA-21")
SERIES_RULE = "aqua-first-0.2"       # composition of `harmonized`; independent of the fitted model version
# (platform, compared against, year outside both fit windows): estimates of the same burning should agree
CONSISTENCY_CHECKS = (("NOAA-20", "Suomi-NPP", 2023), ("NOAA-21", "NOAA-20", 2025))
MIN_UNITS = 500          # source units needed in the fit years before a regional factor is used
MIN_BIN = 200            # source units needed before a full curve bin is used
OUTAGE_WINDOW_DAYS = 30  # each side of the day, for the platform's typical daily units
OUTAGE_FRACTION = 0.5    # a day below this share of the typical units is an outage
COVERAGE_WINDOW_HOUR = 9  # coverage/cells windows start at 09:00 UTC (coverage.py window_start_hour)
EPOCH = date(1970, 1, 1)
DAY_SQL = "floor((epoch(acquired_at) + longitude * 240 - 21600) / 86400)::INTEGER"
DATE_SQL = "(DATE '1970-01-01' + {})"
MODEL_FILE = "model.json"
MODEL_FILES = (MODEL_FILE, "curve.parquet", "factor.parquet")
# Detection model (v0.3): XGBoost P(Aqua also active) for VIIRS units, trained on Suomi-NPP, replacing the curve.
# Inputs validated on 2026-11-15 (`harmonize-detect --exclude aqua_passes`); `din` also computes the extra
# candidates (confidence, brightness, neighbour FRP, ring 2), which did not improve the validation.
DETECTION_FILE = "detection.ubj"
DETECTION_FEATURES = ["log_frp", "log_frp_max", "n", "scan", "track", "hour", "dn_day", "ring1_units", "prev_day",
                      "next_day", "lat", "lon", "moy_sin", "moy_cos"]
DETECTION_CANDIDATES = ["conf", "bright", "bright_bg", "ring1_log_frp", "ring2_units", "prev_log_frp"]
DETECTION_PARAMS = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 8, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 50, "eval_metric": "logloss",
                    "seed": 1}
DETECTION_ROUNDS, DETECTION_EARLY_STOP = 600, 30
UNIT_LABEL = {"pt": "Atividade de fogo harmonizada (células ativas equivalentes ao MODIS Aqua)",
              "en": "Harmonized fire activity (MODIS Aqua-equivalent active cells)"}
LIMITATIONS = [
    "Aqua-equivalent active cell x half-days, not fires, burned area or detections of one sensor.",
    "The estimate is statistical: reliable for sums over regions and months, noisy for one cell and day.",
    "Burning Aqua never detects (small fires) stays outside the scale; the old record cannot recover it.",
    "After REFERENCE_END the series rests on VIIRS only; no reference exists to re-check the factors.",
    "Outages are detected per platform and day worldwide; a regional or partial-day gap still lowers that day.",
    "Terra passes in the morning: its estimate carries the regional diurnal cycle of the fit years.",
    "All hotspot types are kept (NRT has no type), so static heat sources count as activity.",
    "Recent months are NRT data and may change when the standard product replaces them.",
    "Not an accepted harmonization: thresholds for CA07 were not approved.",
]


def _day(value: date) -> int:
    return (value - EPOCH).days


def _fit_windows() -> list[tuple[int, int]]:
    """Fit year ranges of all links, merged."""
    merged: list[list[int]] = []
    for first, last in sorted(link["fit"] for link in LINKS.values()):
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return [(first, last) for first, last in merged]


def _check_protocol() -> None:
    for platform, link in LINKS.items():
        if link["fit"][0] <= TEST_YEARS[1] and TEST_YEARS[0] <= link["fit"][1]:
            raise ValueError(f"Fit years of {platform} overlap the reserved test years {TEST_YEARS}")
        if LINKS[link["curve"]]["target"] != REFERENCE:
            raise ValueError(f"Curve owner {link['curve']} must be fitted against {REFERENCE}")


def _units_sql(first: date, last: date) -> str:
    """Temp table `units`: one row per platform unit with local day in [first, last], with curve bins and the
    per-unit attributes the detection model uses (FRP, pixel size, local hour, UTC coverage window)."""
    months = ml_dataset._months_touching(first - timedelta(days=1), last + timedelta(days=1))
    platforms = ", ".join(_literal(p) for p in SLUGS)
    return f"""
        CREATE OR REPLACE TEMP TABLE units AS
        WITH raw AS (
            SELECT platform, h3_r6 AS c6, {DAY_SQL} AS day, daynight AS dn, count(*)::INTEGER AS n,
                   sum(frp_mw) AS frp, max(frp_mw) AS frp_max, avg(scan_km) AS scan, avg(track_km) AS track,
                   avg(((epoch(acquired_at) + longitude * 240) % 86400 + 86400) % 86400) / 3600 AS hour,
                   floor((avg(epoch(acquired_at)) - {COVERAGE_WINDOW_HOUR * 3600}) / 86400)::INTEGER AS win,
                   avg(CASE WHEN confidence_category = 'h' OR confidence_percent >= 80 THEN 1.0
                            WHEN confidence_category = 'l' OR confidence_percent < 30 THEN 0.0 ELSE 0.5 END) AS conf,
                   max(coalesce(brightness_i4_k, brightness_k)) AS bright,
                   avg(coalesce(brightness_i5_k, bright_t31_k)) AS bright_bg
            FROM silver
            WHERE {months} AND platform IN ({platforms}) AND {DAY_SQL} BETWEEN {_day(first)} AND {_day(last)}
            GROUP BY ALL)
        SELECT platform, c6, day, dn, n,
               CASE WHEN frp IS NULL OR frp <= 0 THEN -99
                    ELSE greatest(-6, least(30, floor(log2(frp) * 2)))::INTEGER END AS fb,
               least(floor(log2(n) * 2), 10)::INTEGER AS nb,
               floor(abs(h3_cell_to_lat(c6::UBIGINT)) / 10)::INTEGER AS lb,
               h3_cell_to_parent(c6::UBIGINT, 2)::BIGINT AS r2, h3_cell_to_parent(c6::UBIGINT, 1)::BIGINT AS r1,
               month({DATE_SQL.format('day')})::INTEGER AS moy, year({DATE_SQL.format('day')})::INTEGER AS yr,
               frp, frp_max, scan, track, hour, win, conf, bright, bright_bg
        FROM raw"""


def _availability_sql() -> str:
    """Temp table `avail`: per platform and day of `units`, units worldwide and whether it is not an outage."""
    return f"""
        CREATE OR REPLACE TEMP TABLE avail AS
        WITH daily AS (SELECT platform, day, count(*) AS units FROM units GROUP BY ALL),
             span AS (SELECT platform, min(day) AS lo, max(day) AS hi FROM daily GROUP BY 1),
             grid AS (SELECT platform, unnest(generate_series(lo, hi))::INTEGER AS day FROM span),
             filled AS (SELECT platform, day, coalesce(units, 0) AS units
                        FROM grid LEFT JOIN daily USING (platform, day)),
             typical AS (SELECT *, median(units) OVER (
                             PARTITION BY platform ORDER BY day
                             ROWS BETWEEN {OUTAGE_WINDOW_DAYS} PRECEDING AND {OUTAGE_WINDOW_DAYS} FOLLOWING) AS typical
                         FROM filled)
        SELECT platform, day, units, typical, units > 0 AND units >= {OUTAGE_FRACTION} * typical AS ok
        FROM typical"""


def _curve_views_sql() -> str:
    """Marginal and global fallbacks of the temp table `curve`."""
    return """
        CREATE OR REPLACE TEMP TABLE curve_m AS
        SELECT curve, fb, dn, sum(units * p) / sum(units) AS p FROM curve GROUP BY ALL;
        CREATE OR REPLACE TEMP TABLE curve_g AS
        SELECT curve, sum(units * p) / sum(units) AS p FROM curve GROUP BY ALL;"""


def _with_p(table: str) -> str:
    """Rows of `table` (unit rows with bins) plus `p`, the chance that Aqua reports the same unit: the detection
    model probability from `dprob` when present (blended with the curve by `dprob.blend`), else the curve."""
    owner = "CASE u.platform " + " ".join(
        f"WHEN {_literal(p)} THEN {_literal(link['curve'])}" for p, link in LINKS.items()) + " END"
    curve = f"coalesce(CASE WHEN c.units >= {MIN_BIN} THEN c.p END, m.p, g.p)"
    return f"""
        SELECT u.*, CASE WHEN u.platform = {_literal(REFERENCE)} THEN 1.0
                         WHEN d.p IS NOT NULL THEN (1 - d.blend) * d.p + d.blend * {curve}
                         ELSE {curve} END AS p
        FROM {table} u
        LEFT JOIN curve c ON c.curve = {owner} AND c.fb = u.fb AND c.nb = u.nb AND c.dn = u.dn AND c.lb = u.lb
        LEFT JOIN curve_m m ON m.curve = {owner} AND m.fb = u.fb AND m.dn = u.dn
        LEFT JOIN curve_g g ON g.curve = {owner}
        LEFT JOIN dprob d ON d.platform = u.platform AND d.c6 = u.c6 AND d.day = u.day AND d.dn = u.dn"""


def _detection_inputs_sql(source: str, platforms: tuple[str, ...]) -> str:
    """Temp table `din`: detection-model inputs (DETECTION_FEATURES) for the units of `platforms` in `source`."""
    names = ", ".join(_literal(p) for p in platforms)
    return f"""
        CREATE OR REPLACE TEMP TABLE din AS
        WITH src AS (SELECT * FROM {source} WHERE platform IN ({names})),
             ring AS (SELECT platform, day, dn, unnest(h3_grid_ring(c6::UBIGINT, 1))::BIGINT AS c6,
                             coalesce(frp, 0) AS frp FROM src),
             ring_n AS (SELECT platform, day, dn, c6, count(*) AS k, sum(frp) AS frp FROM ring GROUP BY ALL),
             ring2 AS (SELECT platform, day, dn, unnest(h3_grid_ring(c6::UBIGINT, 2))::BIGINT AS c6 FROM src),
             ring2_n AS (SELECT platform, day, dn, c6, count(*) AS k FROM ring2 GROUP BY ALL),
             seen AS (SELECT platform, c6, day, sum(coalesce(frp, 0)) AS frp FROM src GROUP BY ALL)
        SELECT s.platform, s.c6, s.day, s.dn, s.yr, s.win,
               ln(1 + coalesce(s.frp, 0)) AS log_frp, ln(1 + coalesce(s.frp_max, 0)) AS log_frp_max,
               s.n::DOUBLE AS n, s.scan, s.track, s.hour, (s.dn = 'D')::DOUBLE AS dn_day,
               s.conf, s.bright, s.bright_bg,
               coalesce(r.k, 0)::DOUBLE AS ring1_units, ln(1 + coalesce(r.frp, 0)) AS ring1_log_frp,
               coalesce(r2.k, 0)::DOUBLE AS ring2_units,
               (p.c6 IS NOT NULL)::DOUBLE AS prev_day, ln(1 + coalesce(p.frp, 0)) AS prev_log_frp,
               (q.c6 IS NOT NULL)::DOUBLE AS next_day,
               h3_cell_to_lat(s.c6::UBIGINT) AS lat, h3_cell_to_lng(s.c6::UBIGINT) AS lon,
               sin(2 * pi() * s.moy / 12) AS moy_sin, cos(2 * pi() * s.moy / 12) AS moy_cos
        FROM src s
        LEFT JOIN ring_n r ON r.platform = s.platform AND r.day = s.day AND r.dn = s.dn AND r.c6 = s.c6
        LEFT JOIN ring2_n r2 ON r2.platform = s.platform AND r2.day = s.day AND r2.dn = s.dn AND r2.c6 = s.c6
        LEFT JOIN seen p ON p.platform = s.platform AND p.c6 = s.c6 AND p.day = s.day - 1
        LEFT JOIN seen q ON q.platform = s.platform AND q.c6 = s.c6 AND q.day = s.day + 1"""


def _empty_dprob(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("CREATE OR REPLACE TEMP TABLE dprob (platform VARCHAR, c6 BIGINT, day INTEGER, dn VARCHAR, "
                       "p DOUBLE, blend DOUBLE)")


def _detection_matrix(connection: duckdb.DuckDBPyConnection, where: str, features: list[str],
                      extra: str = "") -> dict[str, np.ndarray]:
    columns = ", ".join(f"{f}::FLOAT AS {f}" for f in features)
    arrays = connection.execute(f"SELECT platform, c6, day, dn{extra}, {columns} FROM din WHERE {where}").fetchnumpy()
    arrays["x"] = np.column_stack([np.asarray(arrays.pop(f), dtype=np.float32) for f in features])
    return arrays


def _detection_predict(connection: duckdb.DuckDBPyConnection, source: str, detection: dict | None) -> None:
    """Fill `dprob` for the VIIRS units of `source` from the model's detection booster (empty without one)."""
    _empty_dprob(connection)
    if not detection:
        return
    booster = detection["booster"]
    connection.execute(_detection_inputs_sql(source, tuple(detection["platforms"])))
    for platform in detection["platforms"]:
        data = _detection_matrix(connection, f"platform = {_literal(platform)}", booster.feature_names)
        if not len(data["day"]):
            continue
        p = booster.inplace_predict(data["x"]).astype(np.float64)
        connection.register("pred", pa.table({"platform": data["platform"], "c6": data["c6"], "day": data["day"],
                                              "dn": data["dn"], "p": p,
                                              "blend": np.full(len(p), detection["blend"])}))
        connection.execute("INSERT INTO dprob SELECT platform, c6, day, dn, p, blend FROM pred")
        connection.unregister("pred")
    connection.execute("DROP TABLE din")


def _with_est(relation: str) -> str:
    """Rows of `relation` (with platform, r2, r1, moy, p) plus `est`, the Aqua-equivalent units."""
    return f"""
        SELECT x.*, CASE WHEN x.platform = {_literal(REFERENCE)} THEN 1.0
                         ELSE x.p * coalesce(f2.factor, f1.factor, f0.factor) END AS est
        FROM {relation} x
        LEFT JOIN factor f2 ON f2.platform = x.platform AND f2.level = 2 AND f2.region = x.r2 AND f2.moy = x.moy
        LEFT JOIN factor f1 ON f1.platform = x.platform AND f1.level = 1 AND f1.region = x.r1 AND f1.moy = x.moy
        LEFT JOIN factor f0 ON f0.platform = x.platform AND f0.level = 0"""


def _years_sql(years: list[int]) -> str:
    return f"yr IN ({', '.join(str(y) for y in sorted(years))})" if years else "FALSE"


def _fit_years(platform: str) -> list[int]:
    first, last = LINKS[platform]["fit"]
    return list(range(first, last + 1))


def _curve_sql(owner: str, years: list[int]) -> str:
    """Curve rows of one owner platform: share of its units that Aqua also reports, per bin, on days both ran."""
    return f"""
        INSERT INTO curve
        SELECT {_literal(owner)}, s.fb, s.nb, s.dn, s.lb, count(*), avg((a.c6 IS NOT NULL)::INTEGER)
        FROM fu s
        JOIN fa so ON so.platform = s.platform AND so.day = s.day AND so.ok
        JOIN fa ro ON ro.platform = {_literal(REFERENCE)} AND ro.day = s.day AND ro.ok
        LEFT JOIN fu a ON a.platform = {_literal(REFERENCE)} AND a.c6 = s.c6 AND a.day = s.day AND a.dn = s.dn
        WHERE s.platform = {_literal(owner)} AND s.{_years_sql(years)}
        GROUP BY ALL"""


def _pair_sql(platform: str, years: list[int]) -> str:
    """Temp table `pair`: per region and month of year, curve total of `platform` and total of its reference,
    over the days of `years` on which both ran."""
    target = LINKS[platform]["target"]
    return f"""
        CREATE OR REPLACE TEMP TABLE pair AS
        WITH days AS (SELECT a.day FROM fa a JOIN fa b ON b.day = a.day
                      WHERE a.platform = {_literal(platform)} AND b.platform = {_literal(target)}
                        AND a.ok AND b.ok),
             src AS (SELECT r2, r1, moy, sum(p) AS den, count(*) AS units FROM fp
                     WHERE platform = {_literal(platform)} AND {_years_sql(years)} AND day IN (SELECT day FROM days)
                     GROUP BY ALL),
             tgt AS (SELECT r2, r1, moy, sum(est) AS num FROM fe
                     WHERE platform = {_literal(target)} AND {_years_sql(years)} AND day IN (SELECT day FROM days)
                     GROUP BY ALL)
        SELECT r2, r1, moy, coalesce(den, 0) AS den, coalesce(units, 0) AS units, coalesce(num, 0) AS num
        FROM src FULL JOIN tgt USING (r2, r1, moy)"""


def _factor_sql(platform: str, kappa: float | None) -> str:
    """Regional factors from `pair`. kappa None: hard cutoff (MIN_UNITS) with fallback res2 -> res1 -> global.
    kappa set: empirical-Bayes style shrinkage, f = (reference + kappa * f_parent) / (curve total + kappa),
    res1 x month toward the global factor and res2 x month toward its res1 x month factor."""
    name = _literal(platform)
    if kappa is None:
        return f"""
            INSERT INTO factor
            SELECT {name}, 2, r2, moy, sum(num) / sum(den), sum(units) FROM pair
            GROUP BY r2, moy HAVING sum(units) >= {MIN_UNITS}
            UNION ALL
            SELECT {name}, 1, r1, moy, sum(num) / sum(den), sum(units) FROM pair
            GROUP BY r1, moy HAVING sum(units) >= {MIN_UNITS}
            UNION ALL
            SELECT {name}, 0, 0, 0, sum(num) / sum(den), sum(units) FROM pair"""
    return f"""
        INSERT INTO factor
        WITH g AS (SELECT sum(num) / sum(den) AS f, sum(units) AS u FROM pair),
             f1 AS (SELECT r1, moy, (sum(num) + {kappa} * any_value(g.f)) / (sum(den) + {kappa}) AS f,
                           sum(units) AS u
                    FROM pair, g GROUP BY r1, moy),
             f2 AS (SELECT p.r2, p.moy, (sum(p.num) + {kappa} * any_value(f1.f)) / (sum(p.den) + {kappa}) AS f,
                           sum(p.units) AS u
                    FROM pair p JOIN f1 ON f1.r1 = p.r1 AND f1.moy = p.moy GROUP BY p.r2, p.moy)
        SELECT {name}, 2, r2, moy, f, u FROM f2
        UNION ALL SELECT {name}, 1, r1, moy, f, u FROM f1
        UNION ALL SELECT {name}, 0, 0, 0, f, u FROM g"""


def _contributes(days_alias: str, platform_alias: str) -> str:
    """SQL condition: the platform of `platform_alias` feeds `harmonized` on a day with basis `days_alias.basis`."""
    viirs = ", ".join(_literal(p) for p in VIIRS)
    return (f"(({days_alias}.basis = 'aqua' AND {platform_alias}.platform = {_literal(REFERENCE)})"
            f" OR ({days_alias}.basis = 'viirs' AND {platform_alias}.platform IN ({viirs}))"
            f" OR ({days_alias}.basis = 'terra' AND {platform_alias}.platform = 'Terra'))")


def _estimates_sql(year: int) -> str:
    """Temp tables `days` (series basis and contributing platforms per day) and `est` (unit estimates of
    local-day year `year`, with the contribution flag); need `units`, `avail` and the model tables."""
    viirs = ", ".join(_literal(p) for p in VIIRS)
    end = _day(REFERENCE_END)
    return f"""
        CREATE OR REPLACE TEMP TABLE days AS
        WITH f AS (
            SELECT day, bool_or(platform = {_literal(REFERENCE)} AND ok AND day <= {end}) AS aqua,
                   count(*) FILTER (WHERE platform IN ({viirs}) AND ok) AS viirs,
                   bool_or(platform = 'Terra' AND ok AND day <= {end}) AS terra
            FROM avail WHERE year({DATE_SQL.format('day')}) = {year} GROUP BY day)
        SELECT day, CASE WHEN aqua THEN 'aqua' WHEN viirs > 0 THEN 'viirs' WHEN terra THEN 'terra' END AS basis,
               CASE WHEN aqua THEN 1 WHEN viirs > 0 THEN viirs WHEN terra THEN 1 ELSE 0 END AS n
        FROM f;
        CREATE OR REPLACE TEMP TABLE est AS
        SELECT e.platform, e.c6, e.day, e.dn, e.n, e.est, coalesce(a.ok, FALSE) AND {_contributes('d', 'e')} AS contributes
        FROM ({_with_est(f"({_with_p('units')})")}) e
        LEFT JOIN avail a ON a.platform = e.platform AND a.day = e.day
        LEFT JOIN days d ON d.day = e.day
        WHERE e.yr = {year};"""


def _cell_columns(alias: str) -> str:
    return ", ".join(
        f"(count(*) FILTER (WHERE {alias}.platform = {_literal(p)}))::INTEGER AS units_{slug}, "
        f"coalesce(sum({alias}.n) FILTER (WHERE {alias}.platform = {_literal(p)}), 0)::INTEGER AS detections_{slug}, "
        f"sum({alias}.est) FILTER (WHERE {alias}.platform = {_literal(p)}) AS aqua_eq_{slug}"
        for p, slug in SLUGS.items())


def _r6_sql(year: int) -> str:
    """Temp table `r6`: one row per (res6 cell, local day) with any unit: per-platform columns and `harmonized`."""
    return f"""
        CREATE OR REPLACE TEMP TABLE r6 AS
        SELECT {DATE_SQL.format('e.day')} AS day, e.c6 AS h3_cell, {_cell_columns('e')},
               coalesce(sum(e.est) FILTER (WHERE e.contributes), 0) / nullif(any_value(d.n), 0) AS harmonized,
               any_value(d.n)::TINYINT AS contributing, any_value(d.basis) AS basis
        FROM est e JOIN days d ON d.day = e.day
        GROUP BY e.day, e.c6"""


def _cells_sql(year: int) -> str:
    """Rows of every resolution: the res6 rows of `r6` and their sums per parent cell and day."""
    values = [f"{kind}_{slug}" for slug in SLUGS.values() for kind in ("units", "detections", "aqua_eq")]
    columns = f"h3_cell, day, {', '.join(values)}, harmonized, contributing, basis"
    parts = [f"SELECT 6 AS resolution, {columns} FROM r6"]
    sums = ", ".join(f"sum({v})::{'DOUBLE' if v.startswith('aqua_eq') else 'INTEGER'} AS {v}" for v in values)
    parts += [f"""SELECT {res} AS resolution, h3_cell_to_parent(h3_cell::UBIGINT, {res})::BIGINT AS h3_cell, day,
                         {sums}, sum(harmonized) AS harmonized, any_value(contributing) AS contributing,
                         any_value(basis) AS basis
                  FROM r6 GROUP BY 2, 3""" for res in LEVELS if res < 6]
    return f"""
        SELECT resolution::INTEGER AS resolution, h3_cell, day, {', '.join(values)}, harmonized, contributing,
               basis, basis = 'terra' AS low_confidence, {year}::INTEGER AS year
        FROM ({' UNION ALL '.join(parts)})"""


def _availability_rows_sql(year: int) -> str:
    return f"""
        SELECT {DATE_SQL.format('a.day')} AS day, a.platform, a.units::INTEGER AS units,
               a.typical AS typical_units, a.ok AS available, a.ok AND {_contributes('d', 'a')} AS contributes,
               d.basis, {year}::INTEGER AS year
        FROM avail a JOIN days d ON d.day = a.day"""


def _compare_sql(series: str, platforms: list[str]) -> str:
    """Rows for temp table `cmp`: per (res4, month), estimate of `platforms` (mean over the ones running that
    day) and their raw units (summed, as a naive merge would) against Aqua units, on days Aqua and at least
    one of them ran."""
    names = ", ".join(_literal(p) for p in platforms)
    return f"""
        INSERT INTO cmp
        WITH k AS (SELECT day, count(*) AS n FROM avail WHERE ok AND platform IN ({names}) GROUP BY day),
             days AS (SELECT k.day, k.n FROM k
                      JOIN avail a ON a.day = k.day AND a.platform = {_literal(REFERENCE)} AND a.ok),
             s AS (SELECT e.r4, e.ym, sum(e.est / d.n) AS pred, count(*)::DOUBLE AS raw
                   FROM e JOIN days d ON d.day = e.day WHERE e.platform IN ({names}) GROUP BY ALL),
             ref AS (SELECT e.r4, e.ym, count(*)::DOUBLE AS actual
                     FROM e JOIN days d ON d.day = e.day WHERE e.platform = {_literal(REFERENCE)} GROUP BY ALL)
        SELECT {_literal(series)}, r4, ym, coalesce(pred, 0), coalesce(raw, 0), coalesce(actual, 0)
        FROM s FULL JOIN ref USING (r4, ym)"""


def _load_window(connection: duckdb.DuckDBPyConnection, first: date, last: date, detection: dict | None = None) -> None:
    """Temp tables `units`, `avail` and `dprob` (detection probabilities of the window's VIIRS units)."""
    connection.execute(_units_sql(first, last))
    connection.execute(_availability_sql())
    _detection_predict(connection, "units", detection)


def _load_model(connection: duckdb.DuckDBPyConnection, model_dir: str | Path) -> dict:
    """Create the temp tables `curve`, `curve_m`, `curve_g`, `factor` and an empty `dprob` from a fitted model
    directory (local path or s3:// URI). With a detection model, `model["detection"]["booster"]` holds it."""
    with tempfile.TemporaryDirectory() as scratch:
        if "://" in str(model_dir):
            folder = Path(scratch)
            _local_copy(f"{str(model_dir).rstrip('/')}/{MODEL_FILE}", folder)
            listed = json.loads((folder / MODEL_FILE).read_text())
            extra = (DETECTION_FILE,) if listed.get("detection") else ()
            for name in (*MODEL_FILES[1:], *extra):
                _local_copy(f"{str(model_dir).rstrip('/')}/{name}", folder)
        else:
            folder = Path(model_dir)
        model = json.loads((folder / MODEL_FILE).read_text())
        if model["method_version"] != METHOD_VERSION:
            raise ValueError(f"Model method {model['method_version']} does not match {METHOD_VERSION}")
        for name in ("curve", "factor"):
            path = folder / f"{name}.parquet"
            if _sha256(path) != model["sha256"][name]:
                raise ValueError(f"{path} does not match the hash recorded in {MODEL_FILE}")
            connection.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS "
                               f"SELECT * FROM read_parquet({_literal(str(path))})")
        if model.get("detection"):
            path = folder / DETECTION_FILE
            if _sha256(path) != model["detection"]["sha256"]:
                raise ValueError(f"{path} does not match the hash recorded in {MODEL_FILE}")
            model["detection"]["booster"] = xgb.Booster(model_file=str(path))
    connection.execute(_curve_views_sql())
    _empty_dprob(connection)
    return model


def _metrics(connection: duckdb.DuckDBPyConnection, series: str, level: int) -> dict:
    """Bias and WAPE of the estimate and of the raw units against Aqua, over (res `level` region, month)."""
    groups, actual, bias, wape, raw_bias, raw_wape = connection.execute(f"""
        WITH x AS (SELECT h3_cell_to_parent(r4::UBIGINT, {level}) AS k, ym, sum(pred) AS pred, sum(raw) AS raw,
                          sum(actual) AS actual FROM cmp WHERE series = {_literal(series)} GROUP BY ALL)
        SELECT count(*), sum(actual), sum(pred) / sum(actual), sum(abs(pred - actual)) / sum(actual),
               sum(raw) / sum(actual), sum(abs(raw - actual)) / sum(actual) FROM x""").fetchall()[0]
    return {"groups": groups, "aqua_units": actual, "bias": bias, "wape": wape,
            "raw_bias": raw_bias, "raw_wape": raw_wape}


def _month_ranking(connection: duckdb.DuckDBPyConnection, series: str) -> dict:
    """Rank correlation between estimated and Aqua monthly totals inside each res2 region (ties: lowest rank)."""
    regions, median, p10 = connection.execute(f"""
        WITH x AS (SELECT h3_cell_to_parent(r4::UBIGINT, 2) AS k, ym, sum(pred) AS pred, sum(actual) AS actual
                   FROM cmp WHERE series = {_literal(series)} GROUP BY ALL),
             kept AS (SELECT k FROM x GROUP BY k
                      HAVING sum(actual) >= 1000 AND count(*) FILTER (WHERE actual > 0) >= 12),
             r AS (SELECT k, rank() OVER (PARTITION BY k ORDER BY pred) AS rp,
                          rank() OVER (PARTITION BY k ORDER BY actual) AS ra
                   FROM x WHERE k IN (SELECT k FROM kept)),
             c AS (SELECT k, corr(rp, ra) AS rho FROM r GROUP BY k)
        SELECT count(*), median(rho), quantile_cont(rho, 0.1) FROM c""").fetchall()[0]
    return {"regions": regions, "median": median, "p10": p10}


def _agreement(connection: duckdb.DuckDBPyConnection, platform: str, other: str, year: int,
               detection: dict | None = None) -> dict:
    """Estimates of two platforms for the same year, per (res2, month), on days both ran."""
    _load_window(connection, date(year, 1, 1) - timedelta(days=OUTAGE_WINDOW_DAYS),
                 date(year, 12, 31) + timedelta(days=OUTAGE_WINDOW_DAYS), detection)
    groups, total, ratio, wape = connection.execute(f"""
        WITH days AS (SELECT a.day FROM avail a JOIN avail b ON b.day = a.day
                      WHERE a.platform = {_literal(platform)} AND b.platform = {_literal(other)} AND a.ok AND b.ok),
             e AS (SELECT platform, r2, moy, est FROM ({_with_est(f"({_with_p('units')})")})
                   WHERE yr = {year} AND day IN (SELECT day FROM days)),
             x AS (SELECT r2, moy, coalesce(sum(est) FILTER (WHERE platform = {_literal(platform)}), 0) AS a,
                          coalesce(sum(est) FILTER (WHERE platform = {_literal(other)}), 0) AS b
                   FROM e WHERE platform IN ({_literal(platform)}, {_literal(other)}) GROUP BY ALL)
        SELECT count(*), sum(b), sum(a) / sum(b), sum(abs(a - b)) / sum(b) FROM x""").fetchall()[0]
    return {"platform": platform, "against": other, "year": year, "groups": groups, "against_total": total,
            "ratio": ratio, "wape_res2_month": wape}


def _load_fit_units(connection: duckdb.DuckDBPyConnection, windows: list[tuple[int, int]], tic: float) -> None:
    """Temp tables `fu` (units) and `fa` (availability) over the year windows."""
    for index, (first, last) in enumerate(windows):
        _load_window(connection, date(first, 1, 1), date(last, 12, 31))
        for kept, loaded in (("fu", "units"), ("fa", "avail")):
            connection.execute(f"INSERT INTO {kept} SELECT * FROM {loaded}" if index
                               else f"CREATE OR REPLACE TEMP TABLE {kept} AS SELECT * FROM {loaded}")
        print(f"harmonize units {first}-{last} loaded ({time.monotonic() - tic:.0f}s)", file=sys.stderr, flush=True)


def _fit_curves(connection: duckdb.DuckDBPyConnection, years: dict[str, list[int]]) -> dict:
    """Temp tables `curve` (+ views) and `fp` (unit probabilities) from `fu`/`fa`; `years` maps each curve
    owner to its fit years."""
    connection.execute("CREATE OR REPLACE TEMP TABLE curve (curve VARCHAR, fb INTEGER, nb INTEGER, dn VARCHAR, "
                       "lb INTEGER, units BIGINT, p DOUBLE)")
    for owner, owner_years in sorted(years.items()):
        connection.execute(_curve_sql(owner, owner_years))
    connection.execute(_curve_views_sql())
    curves = {owner: {"units": units, "p_global": p} for owner, units, p in connection.execute(
        "SELECT g.curve, (SELECT sum(units) FROM curve c WHERE c.curve = g.curve), g.p FROM curve_g g").fetchall()}
    if set(curves) != set(years):
        raise ValueError(f"No fit data for curve(s): {sorted(set(years) - set(curves))}")
    connection.execute(f"CREATE OR REPLACE TEMP TABLE fp AS SELECT platform, day, r2, r1, moy, yr, p "
                       f"FROM ({_with_p('fu')}) WHERE platform <> {_literal(REFERENCE)}")
    return curves


def _fit_factors(connection: duckdb.DuckDBPyConnection, years: dict[str, list[int]], kappa: float | None) -> dict:
    """Temp table `factor` for the platforms of `years` (in LINKS order, so chained targets come first)."""
    connection.execute(f"CREATE OR REPLACE TEMP TABLE fe AS SELECT platform, day, r2, r1, moy, yr, 1.0::DOUBLE AS est "
                       f"FROM fu WHERE platform = {_literal(REFERENCE)}")
    connection.execute("CREATE OR REPLACE TEMP TABLE factor (platform VARCHAR, level INTEGER, region BIGINT, "
                       "moy INTEGER, factor DOUBLE, units BIGINT)")
    factors = {}
    for platform in (p for p in LINKS if p in years):
        connection.execute(_pair_sql(platform, years[platform]))
        connection.execute(_factor_sql(platform, kappa))
        connection.execute(f"INSERT INTO fe SELECT platform, day, r2, r1, moy, yr, est FROM "
                           f"({_with_est('fp')}) WHERE platform = {_literal(platform)}")
        levels = dict(connection.execute(f"SELECT level, count(*) FROM factor WHERE platform = "
                                         f"{_literal(platform)} AND factor IS NOT NULL GROUP BY 1").fetchall())
        overall, units = connection.execute(f"SELECT factor, units FROM factor WHERE platform = "
                                            f"{_literal(platform)} AND level = 0").fetchall()[0]
        if overall is None:
            raise ValueError(f"No fit data for {platform}: its fit years have no day shared with "
                             f"{LINKS[platform]['target']}")
        factors[platform] = {"global_factor": overall, "fit_units": units,
                             "res2_month_factors": levels.get(2, 0), "res1_month_factors": levels.get(1, 0)}
    return factors


def _train_detection(connection: duckdb.DuckDBPyConnection, years: list[int], threads: int) -> xgb.Booster:
    """Detection classifier on usable Suomi-NPP units of `years` in `fu` (target: Aqua active on the same
    cell, day and day/night); a random 10% is the early-stopping set."""
    connection.execute(_detection_inputs_sql("fu", ("Suomi-NPP",)))
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE din AS
        SELECT d.*, (a.c6 IS NOT NULL)::FLOAT AS target FROM din d
        JOIN fa so ON so.platform = d.platform AND so.day = d.day AND so.ok
        JOIN fa ao ON ao.platform = {_literal(REFERENCE)} AND ao.day = d.day AND ao.ok
        LEFT JOIN fu a ON a.platform = {_literal(REFERENCE)} AND a.c6 = d.c6 AND a.day = d.day AND a.dn = d.dn
        WHERE d.{_years_sql(years)}""")
    data = _detection_matrix(connection, "TRUE", DETECTION_FEATURES, ", target")
    valid = np.random.default_rng(7).random(len(data["target"])) < 0.1
    target = np.asarray(data["target"], dtype=np.float32)
    dtrain = xgb.DMatrix(data["x"][~valid], label=target[~valid], feature_names=DETECTION_FEATURES, nthread=threads)
    dvalid = xgb.DMatrix(data["x"][valid], label=target[valid], feature_names=DETECTION_FEATURES, nthread=threads)
    connection.execute("DROP TABLE din")
    return xgb.train({**DETECTION_PARAMS, "nthread": threads}, dtrain, DETECTION_ROUNDS, evals=[(dvalid, "valid")],
                     early_stopping_rounds=DETECTION_EARLY_STOP, verbose_eval=False)


def fit(silver: str, model_dir: str | Path, *, kappa: float | None = None, detection: bool = False,
        blend: float = 0.0, storage_options: dict[str, str] | None = None, memory_limit: str = "16GB",
        threads: int = 8, temp_dir: str | Path | None = None) -> dict:
    """Fit the curves and regional factors from the fit years of LINKS and write them to `model_dir`.
    `kappa` None keeps the v0.1 hard cutoff for regional factors; a value shrinks them toward their parent.
    `detection` trains the VIIRS detection model on the Suomi-NPP fit years; its probabilities (mixed with the
    curve by `blend`) replace the curve for every VIIRS platform before the factors are fitted."""
    _check_protocol()
    folder = Path(model_dir)
    if (folder / MODEL_FILE).exists():
        raise FileExistsError(f"{folder / MODEL_FILE} already exists; fit into a new directory")
    folder.mkdir(parents=True, exist_ok=True)
    silver_version = DeltaTable(silver, storage_options=storage_options).version()
    tic = time.monotonic()
    detector = None
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        _load_fit_units(connection, _fit_windows(), tic)
        if detection:
            booster = _train_detection(connection, _fit_years("Suomi-NPP"), threads)
            booster.save_model(folder / DETECTION_FILE)
            detector = {"booster": booster, "platforms": list(VIIRS), "blend": blend}
            _detection_predict(connection, "fu", detector)
            print(f"harmonize-fit detection model trained ({time.monotonic() - tic:.0f}s)", file=sys.stderr, flush=True)
        curves = _fit_curves(connection, {link["curve"]: _fit_years(link["curve"]) for link in LINKS.values()})
        factors = _fit_factors(connection, {p: _fit_years(p) for p in LINKS}, kappa)
        connection.execute(f"COPY (SELECT * FROM curve ORDER BY ALL) TO {_literal(str(folder / 'curve.parquet'))} "
                           "(FORMAT PARQUET)")
        connection.execute(f"COPY (SELECT * FROM factor WHERE factor IS NOT NULL ORDER BY ALL) TO "
                           f"{_literal(str(folder / 'factor.parquet'))} (FORMAT PARQUET)")
    model = {"method_version": METHOD_VERSION, "silver": silver, "silver_version": silver_version,
             "reference": REFERENCE, "reference_end": REFERENCE_END.isoformat(), "test_years": list(TEST_YEARS),
             "links": LINKS, "factor_rule": "cutoff" if kappa is None else "shrinkage", "kappa": kappa,
             "min_units": MIN_UNITS, "min_bin": MIN_BIN,
             "outage_window_days": OUTAGE_WINDOW_DAYS, "outage_fraction": OUTAGE_FRACTION,
             "sha256": {name: _sha256(folder / f"{name}.parquet") for name in ("curve", "factor")},
             "detection": None if detector is None else {
                 "sha256": _sha256(folder / DETECTION_FILE), "features": DETECTION_FEATURES,
                 "platforms": list(VIIRS), "blend": blend, "trained_on": {"Suomi-NPP": _fit_years("Suomi-NPP")},
                 "params": DETECTION_PARAMS, "best_iteration": detector["booster"].best_iteration},
             "curves": curves, "factors": factors, "seconds": round(time.monotonic() - tic, 1),
             "limitations": LIMITATIONS, "counts": {p: f["res2_month_factors"] for p, f in factors.items()}}
    (folder / MODEL_FILE).write_text(json.dumps(model, indent=1))
    return model


# Year-block validation inside the fit years (DS protocol, 2026-11-15): each fold holds out a Terra block and
# a Suomi-NPP year; years within CV_GAP of a held year stay out of the fit. NOAA-20/21 are chained to Suomi-NPP
# and are not refitted here.
CV_FOLDS = ({"Terra": (2003, 2004, 2005), "Suomi-NPP": (2013,)},
            {"Terra": (2009, 2010, 2011), "Suomi-NPP": (2015,)},
            {"Terra": (2015, 2016, 2017), "Suomi-NPP": (2017,)})
CV_GAP = 1
KAPPAS = (30.0, 100.0, 300.0, 1000.0, 3000.0)
EXTRAPOLATION = {"platform": "Terra", "fit": (2006, 2017), "test": (2003, 2004)}   # scored once
# Non-inferiority rule proposed by the DS and adopted by the user's delegation (2026-11-15).
ACCEPTANCE = {"wape_margin": {"res1": 0.005, "res2": 0.005, "res4": 0.010}, "bias": (0.97, 1.03),
              "spearman_drop": 0.01, "sparse_gain_min_folds": 2}


def _score(connection: duckdb.DuckDBPyConnection, name: str, platform: str, years: list[int]) -> dict:
    """Estimate of `platform` against Aqua units in `years`, from the current curve and factor tables."""
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE e AS
        SELECT x.platform, x.day, h3_cell_to_parent(x.c6::UBIGINT, 4)::BIGINT AS r4, x.yr * 100 + x.moy AS ym, x.est
        FROM ({_with_est(f"({_with_p('fu')})")}) x
        JOIN fa a ON a.platform = x.platform AND a.day = x.day AND a.ok
        WHERE x.platform IN ({_literal(platform)}, {_literal(REFERENCE)}) AND x.{_years_sql(years)}""")
    connection.execute(_compare_sql(name, [platform]))
    return {**{f"res{level}": _metrics(connection, name, level) for level in (1, 2, 4)},
            "spearman_res2": _month_ranking(connection, name),
            "res2_by_volume_quartile": _volume_quartiles(connection, name)}


def _key(kappa: float | None) -> str:
    return "cutoff" if kappa is None else f"kappa_{kappa:g}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _acceptance(folds: list[dict], platform: str, new: str) -> dict:
    """DS rule (non-inferiority + gain in sparse regions) of variant `new` against `cutoff` over the folds."""
    old_scores = [f["scores"]["cutoff"][platform] for f in folds]
    new_scores = [f["scores"][new][platform] for f in folds]
    checks = {}
    for level, margin in ACCEPTANCE["wape_margin"].items():
        old, new_value = _mean([s[level]["wape"] for s in old_scores]), _mean([s[level]["wape"] for s in new_scores])
        checks[f"wape_{level}"] = {"cutoff": old, "new": new_value, "pass": new_value <= old + margin}
    bias = _mean([s["res2"]["bias"] for s in new_scores])
    checks["bias_res2"] = {"new": bias, "pass": ACCEPTANCE["bias"][0] <= bias <= ACCEPTANCE["bias"][1]}
    old, new_value = (_mean([s["spearman_res2"]["median"] for s in scores]) for scores in (old_scores, new_scores))
    checks["spearman_res2"] = {"cutoff": old, "new": new_value, "pass": new_value >= old - ACCEPTANCE["spearman_drop"]}
    sparse = [(o["res2_by_volume_quartile"]["q1"]["wape"], n["res2_by_volume_quartile"]["q1"]["wape"])
              for o, n in zip(old_scores, new_scores)]
    better = sum(n < o for o, n in sparse)
    checks["sparse_q1_res2"] = {"cutoff": _mean([o for o, _ in sparse]), "new": _mean([n for _, n in sparse]),
                                "folds_better": better,
                                "pass": _mean([n for _, n in sparse]) < _mean([o for o, _ in sparse])
                                and better >= ACCEPTANCE["sparse_gain_min_folds"]}
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def cross_validate(silver: str, output: str | Path, *, kappas: tuple[float, ...] = KAPPAS,
                   storage_options: dict[str, str] | None = None, memory_limit: str = "16GB", threads: int = 8,
                   temp_dir: str | Path | None = None) -> dict:
    """Compare the v0.1 cutoff factors with shrinkage (each kappa) by year-block validation inside the fit
    years, pick the kappa with the lowest mean sparse-region error, apply the acceptance rule, then score the
    Terra extrapolation test once for cutoff and the chosen kappa. Never reads TEST_YEARS."""
    report_path = Path(output)
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists: the extrapolation test is scored once")
    first = min(min(min(f[p]) for p in f) for f in CV_FOLDS)
    last = max(max(max(f[p]) for p in f) for f in CV_FOLDS)
    if first <= TEST_YEARS[1] and TEST_YEARS[0] <= last:
        raise ValueError("Cross-validation years overlap the reserved test years")
    tic = time.monotonic()
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        _load_fit_units(connection, [(first, last)], tic)
        connection.execute("CREATE OR REPLACE TEMP TABLE avail AS SELECT * FROM fa")
        connection.execute("CREATE OR REPLACE TEMP TABLE cmp (series VARCHAR, r4 BIGINT, ym INTEGER, pred DOUBLE, "
                           "raw DOUBLE, actual DOUBLE)")
        folds = []
        for index, held in enumerate(CV_FOLDS):
            years = {p: [y for y in _fit_years(p) if all(abs(y - h) > CV_GAP for h in held[p])] for p in held}
            _fit_curves(connection, years)
            scores = {}
            for kappa in (None, *kappas):
                _fit_factors(connection, years, kappa)
                scores[_key(kappa)] = {p: _score(connection, f"fold{index}|{_key(kappa)}|{p}", p, list(held[p]))
                                       for p in held}
            folds.append({"held": {p: list(v) for p, v in held.items()}, "fit_years": years, "scores": scores})
            print(f"harmonize-cv fold {index} done ({time.monotonic() - tic:.0f}s)", file=sys.stderr, flush=True)
        platforms = list(CV_FOLDS[0])
        sparse = {_key(k): _mean([f["scores"][_key(k)][p]["res2_by_volume_quartile"]["q1"]["wape"]
                                  for f in folds for p in platforms]) for k in kappas}
        chosen = min(kappas, key=lambda k: sparse[_key(k)])
        acceptance = {p: _acceptance(folds, p, _key(chosen)) for p in platforms}
        target = EXTRAPOLATION["platform"]
        years = {target: list(range(EXTRAPOLATION["fit"][0], EXTRAPOLATION["fit"][1] + 1))}
        _fit_curves(connection, years)
        extrapolation = {}
        for kappa in (None, chosen):
            _fit_factors(connection, years, kappa)
            extrapolation[_key(kappa)] = _score(connection, f"extrapolation|{_key(kappa)}", target,
                                                list(range(EXTRAPOLATION["test"][0], EXTRAPOLATION["test"][1] + 1)))
    old, new = extrapolation["cutoff"], extrapolation[_key(chosen)]
    extrapolation_checks = {
        **{f"wape_{level}": new[level]["wape"] <= old[level]["wape"] + margin
           for level, margin in ACCEPTANCE["wape_margin"].items()},
        "bias_res2": ACCEPTANCE["bias"][0] <= new["res2"]["bias"] <= ACCEPTANCE["bias"][1]}
    accepted = all(a["pass"] for a in acceptance.values()) and all(extrapolation_checks.values())
    report = {"method_version": METHOD_VERSION, "silver": silver,
              "silver_version": DeltaTable(silver, storage_options=storage_options).version(),
              "folds": folds, "cv_gap_years": CV_GAP, "kappas": list(kappas), "sparse_q1_res2_by_variant": sparse,
              "chosen_kappa": chosen, "acceptance_rule": ACCEPTANCE, "acceptance": acceptance,
              "extrapolation": {"setup": EXTRAPOLATION, "scores": extrapolation, "checks": extrapolation_checks},
              "accepted": accepted, "seconds": round(time.monotonic() - tic, 1),
              "label": "kappa chosen on the same folds that judge it (optimistic); the Terra extrapolation is "
                       "the only untouched check",
              "counts": {"chosen_kappa": chosen, "accepted": accepted}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    return report


def apply(silver: str, model_dir: str | Path, target: str, availability: str, *, years: list[int] | None = None,
          storage_options: dict[str, str] | None = None, memory_limit: str = "16GB", threads: int = 8,
          temp_dir: str | Path | None = None) -> dict:
    """Write the harmonized cells and the platform availability of each local-day year, one commit per year."""
    silver_version = DeltaTable(silver, storage_options=storage_options).version()
    counts = {}
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        model = _load_model(connection, model_dir)
        if years is None:
            first, last = connection.execute(
                "SELECT min(acquisition_year), max(acquisition_year) FROM silver").fetchall()[0]
            years = list(range(first, last + 1))
        metadata = {"method_version": METHOD_VERSION, "silver": silver, "silver_version": str(silver_version),
                    "curve_sha256": model["sha256"]["curve"], "factor_sha256": model["sha256"]["factor"],
                    "unit_label_pt": UNIT_LABEL["pt"], "unit_label_en": UNIT_LABEL["en"], "series_rule": SERIES_RULE}
        for year in sorted(set(years)):
            tic = time.monotonic()
            _load_window(connection, date(year, 1, 1) - timedelta(days=OUTAGE_WINDOW_DAYS),
                         date(year, 12, 31) + timedelta(days=OUTAGE_WINDOW_DAYS), model.get("detection"))
            connection.execute(_estimates_sql(year))
            missing = connection.execute("SELECT DISTINCT platform FROM est WHERE est IS NULL").fetchall()
            if missing:
                raise ValueError(f"Model has no factor for platform(s): {sorted(p for p, in missing)}")
            platforms = {p: {"units": units, "aqua_equivalent": total} for p, units, total in connection.execute(
                "SELECT platform, count(*), sum(est) FROM est GROUP BY 1 ORDER BY 1").fetchall()}
            harmonized, without_data = connection.execute("""
                SELECT (SELECT sum(s / d.n) FROM (SELECT day, sum(est) FILTER (WHERE contributes) AS s
                                                  FROM est GROUP BY 1) e JOIN days d USING (day) WHERE d.n > 0),
                       (SELECT count(*) FROM days WHERE n = 0)""").fetchall()[0]
            connection.execute(_r6_sql(year))
            for path, sql, partitions in ((target, _cells_sql(year), PARTITIONS),
                                          (availability, _availability_rows_sql(year), AVAILABILITY_PARTITIONS)):
                exists = DeltaTable.is_deltatable(path, storage_options=storage_options)
                write_deltalake(path, connection.sql(sql).to_arrow_reader(1_000_000), mode="overwrite",
                                partition_by=partitions, predicate=f"year = {year}" if exists else None,
                                storage_options=storage_options,
                                commit_properties=CommitProperties(custom_metadata={**metadata, "year": str(year)}))
            counts[str(year)] = {"harmonized": harmonized, "days_without_contributing_platform": without_data,
                                 "platforms": platforms}
            print(f"harmonize {year}: harmonized {harmonized} ({time.monotonic() - tic:.0f}s)",
                  file=sys.stderr, flush=True)
    return {**metadata, "target": target, "availability": availability, "counts": counts,
            "limitations": LIMITATIONS}


def verify(target: str, silver: str, *, years: list[int], storage_options: dict[str, str] | None = None,
           memory_limit: str = "16GB", threads: int = 8, temp_dir: str | Path | None = None) -> dict:
    """Per local-day year and resolution: detections per platform equal silver, one row per cell and day,
    and the same `harmonized` total at every resolution."""
    sums = ", ".join(f"sum(detections_{slug})" for slug in SLUGS.values())
    report = {}
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        connection.register("cells", DeltaTable(target, storage_options=storage_options).to_pyarrow_dataset())
        for year in sorted(set(years)):
            first, last = date(year, 1, 1), date(year, 12, 31)
            months = ml_dataset._months_touching(first - timedelta(days=1), last + timedelta(days=1))
            expected = dict(connection.execute(f"""
                SELECT platform, count(*) FROM silver
                WHERE {months} AND {DAY_SQL} BETWEEN {_day(first)} AND {_day(last)} GROUP BY 1""").fetchall())
            levels = connection.execute(f"""
                SELECT resolution, count(*), count(DISTINCT (h3_cell, day)), coalesce(sum(harmonized), 0), {sums}
                FROM cells WHERE year = {year} GROUP BY 1 ORDER BY 1""").fetchall()
            if not levels and not expected:
                report[str(year)] = {}
                continue
            totals = [h for _, _, _, h, *_ in levels]
            for resolution, rows, distinct, _, *written in levels:
                found = {p: int(n or 0) for p, n in zip(SLUGS, written)}
                if rows != distinct or any(found[p] != expected.get(p, 0) for p in SLUGS):
                    raise RuntimeError(f"harmonized {year} r{resolution} does not reconcile: rows={rows} "
                                       f"distinct={distinct} silver={expected} table={found}")
            if [r for r, *_ in levels] != list(LEVELS) or max(totals) - min(totals) > 1e-6 * max(1.0, max(totals)):
                raise RuntimeError(f"harmonized {year}: resolutions {[r for r, *_ in levels]}, totals {totals}")
            report[str(year)] = {f"r{r}": rows for r, rows, *_ in levels} | {"harmonized": totals[0]}
    return report


def _comparison_tables(connection: duckdb.DuckDBPyConnection, first_year: int, last_year: int,
                       detection: dict | None = None) -> list[str]:
    """Temp tables `e` (unit estimates on days their platform worked, local-day years first..last) and an
    empty `cmp`; returns the platforms present."""
    _load_window(connection, date(first_year, 1, 1) - timedelta(days=OUTAGE_WINDOW_DAYS),
                 date(last_year, 12, 31) + timedelta(days=OUTAGE_WINDOW_DAYS), detection)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE e AS
        SELECT x.platform, x.day, h3_cell_to_parent(x.c6::UBIGINT, 4)::BIGINT AS r4,
               x.yr * 100 + x.moy AS ym, x.est
        FROM ({_with_est(f"({_with_p('units')})")}) x
        JOIN avail a ON a.platform = x.platform AND a.day = x.day AND a.ok
        WHERE x.yr BETWEEN {first_year} AND {last_year}""")
    connection.execute("CREATE OR REPLACE TEMP TABLE cmp (series VARCHAR, r4 BIGINT, ym INTEGER, pred DOUBLE, "
                       "raw DOUBLE, actual DOUBLE)")
    return [p for p, in connection.execute("SELECT DISTINCT platform FROM e ORDER BY 1").fetchall()]


def _volume_quartiles(connection: duckdb.DuckDBPyConnection, series: str) -> dict:
    """WAPE over (res2 region, month) within quartiles of regions ranked by Aqua units (q1 = fewest)."""
    rows = connection.execute(f"""
        WITH x AS (SELECT h3_cell_to_parent(r4::UBIGINT, 2) AS k, ym, sum(pred) AS pred, sum(actual) AS actual
                   FROM cmp WHERE series = {_literal(series)} GROUP BY ALL),
             q AS (SELECT k, ntile(4) OVER (ORDER BY sum(actual)) AS quartile FROM x GROUP BY k)
        SELECT quartile, count(DISTINCT k), sum(actual), sum(abs(pred - actual)) / nullif(sum(actual), 0),
               sum(pred) / nullif(sum(actual), 0)
        FROM x JOIN q USING (k) GROUP BY 1 ORDER BY 1""").fetchall()
    return {f"q{q}": {"regions": n, "aqua_units": a, "wape": w, "bias": b} for q, n, a, w, b in rows}


def _series_report(connection: duckdb.DuckDBPyConnection, name: str, platforms: list[str]) -> dict:
    connection.execute(_compare_sql(name, platforms))
    return {"platforms": platforms,
            **{f"res{level}_month": _metrics(connection, name, level) for level in (4, 2, 1)},
            "month_ranking_res2": _month_ranking(connection, name),
            "res2_month_by_volume_quartile": _volume_quartiles(connection, name),
            "yearly": {str(y): {"estimate": pred, "raw_units": raw, "aqua_units": actual}
                       for y, pred, raw, actual in connection.execute(f"""
                           SELECT ym // 100, sum(pred), sum(raw), sum(actual) FROM cmp
                           WHERE series = {_literal(name)} GROUP BY 1 ORDER BY 1""").fetchall()}}


def evaluate(silver: str, model_dir: str | Path, output: str | Path, *,
             storage_options: dict[str, str] | None = None, memory_limit: str = "16GB", threads: int = 8,
             temp_dir: str | Path | None = None) -> dict:
    """Final test on TEST_YEARS (estimates against Aqua, which the fit never read there) plus the
    cross-platform consistency checks. Refuses to run twice into the same report."""
    report_path = Path(output)
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists: the final test is scored once")
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        model = _load_model(connection, model_dir)
        running = _comparison_tables(connection, *TEST_YEARS, model.get("detection"))
        series = {p: [p] for p in LINKS if p in running}
        series["combined_without_aqua"] = sorted(series)
        test = {name: _series_report(connection, name, platforms) for name, platforms in series.items()}
        checks = [_agreement(connection, *check, model.get("detection")) for check in CONSISTENCY_CHECKS]
    combined = test["combined_without_aqua"]
    report = {"method_version": METHOD_VERSION, "silver": silver,
              "silver_version": DeltaTable(silver, storage_options=storage_options).version(),
              "model_sha256": model["sha256"], "test_years": list(TEST_YEARS), "reference": REFERENCE,
              "test": test, "consistency_checks": checks, "limitations": LIMITATIONS,
              "counts": {"combined_bias": combined["res2_month"]["bias"],
                         "combined_wape_res2_month": combined["res2_month"]["wape"],
                         "raw_wape_res2_month": combined["res2_month"]["raw_wape"]}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    return report


# Era -> platforms averaged in that era under the v0.1.0 rule (simple mean of the working platforms).
ERAS = {"E0_2000-11..2002-06_terra": ["Terra"],
        "E1_2002-07..2011_terra_aqua": ["Terra", "Aqua"],
        "E2_2012..2017_terra_aqua_snpp": ["Terra", "Aqua", "Suomi-NPP"],
        "E3_2018..2019_published": ["Terra", "Aqua", "Suomi-NPP", "NOAA-20"],
        "E4_2020+_viirs": ["Suomi-NPP", "NOAA-20"]}


def evaluate_eras(silver: str, model_dir: str | Path, output: str | Path, *, drift_years: tuple[int, int] = (2020, 2025),
                  storage_options: dict[str, str] | None = None, memory_limit: str = "16GB", threads: int = 8,
                  temp_dir: str | Path | None = None) -> dict:
    """Error of the series as composed in each era, against Aqua, with the fitted model fixed.

    Every era composition is rebuilt inside TEST_YEARS (second read of those years: not independent).
    Extras: Terra alone in 2003-2004 (inside Terra's fit years: optimistic bound for the Terra-only era) and,
    per year of `drift_years`, the VIIRS mean against Aqua (agreement with a drifting Aqua, not error)."""
    report_path = Path(output)
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists")
    with ml_dataset._connect(silver, None, memory_limit, threads, temp_dir, storage_options) as connection:
        model = _load_model(connection, model_dir)
        _comparison_tables(connection, *TEST_YEARS, model.get("detection"))
        eras = {name: _series_report(connection, name, platforms) for name, platforms in ERAS.items()}
        _comparison_tables(connection, 2003, 2004, model.get("detection"))
        terra_in_fit = _series_report(connection, "terra_2003_2004", ["Terra"])
        drift = {}
        for year in range(drift_years[0], drift_years[1] + 1):
            running = _comparison_tables(connection, year, year, model.get("detection"))
            viirs = [p for p in VIIRS if p in running]
            if not viirs or REFERENCE not in running:
                continue
            name = f"viirs_{year}"
            connection.execute(_compare_sql(name, viirs))
            drift[str(year)] = {"platforms": viirs, "res2_month": _metrics(connection, name, 2),
                                "res1_month": _metrics(connection, name, 1)}
    report = {"method_version": METHOD_VERSION, "series_rule_measured": "v0.1.0 mean of working platforms",
              "silver": silver, "silver_version": DeltaTable(silver, storage_options=storage_options).version(),
              "model_sha256": model["sha256"], "reference": REFERENCE,
              "label": f"{TEST_YEARS[0]}-{TEST_YEARS[1]} reused (second read, model {METHOD_VERSION} fixed); "
                       "not independent for a v0.2",
              "eras": eras, "terra_alone_inside_fit_2003_2004": terra_in_fit,
              "viirs_vs_aqua_agreement_not_error": drift, "limitations": LIMITATIONS,
              "counts": {name: r["res2_month"]["wape"] for name, r in eras.items()}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    return report
