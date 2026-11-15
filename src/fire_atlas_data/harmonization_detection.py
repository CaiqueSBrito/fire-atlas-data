"""Experiment (RF07/CA07): replace the binned detection curve of the harmonization with a learned detection
model, and measure what explains the disagreement between platforms.

For each Terra or Suomi-NPP unit (active res6 cell x local fire day x day/night), an XGBoost classifier
estimates P(Aqua also active) from the unit itself (FRP sum and max, detections, pixel size, local hour),
its context (neighbour units the same day, same cell the day before and after) and whether Aqua passed
over the res4 parent in the unit's UTC window (coverage/cells, from NASA CMR swath footprints). The regional
factors of v0.2 (shrinkage, KAPPA) are refitted on top of the model probabilities.

Protocol: the v0.2 year-block folds (harmonization.CV_FOLDS, gap harmonization.CV_GAP), the curve-based v0.2 as
baseline, the v0.2 acceptance rule; plus Suomi-NPP 2012, never used for fitting or choosing, scored once.
The Terra 2003-2004 extrapolation is scored again for reference only (already used to accept v0.2).
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import time

import duckdb
from deltalake import DeltaTable
import numpy as np
import pyarrow as pa
import xgboost as xgb

from . import harmonization as h, ml_dataset
from .silver import _literal

METHOD_VERSION = "0.3.0-experiment"
KAPPA = 30.0
SOURCES = ("Terra", "Suomi-NPP")
FEATURES = [*h.DETECTION_FEATURES, *h.DETECTION_CANDIDATES, "aqua_passes"]
PARAMS = h.DETECTION_PARAMS
ROUNDS, EARLY_STOP = h.DETECTION_ROUNDS, h.DETECTION_EARLY_STOP
CLEAN_TEST = {"platform": "Suomi-NPP", "years": (2012,)}


def _features_sql(first_year: int, last_year: int) -> str:
    """Temp table `feat`: production model inputs (harmonization `din`) plus Aqua passes, target and usability
    for every source-platform unit in `fu`; run after `harmonization._detection_inputs_sql('fu', SOURCES)`."""
    return f"""
        CREATE OR REPLACE TEMP TABLE feat AS
        WITH cov AS (SELECT h3_r4, window_date, passes_aqua FROM coverage
                     WHERE year BETWEEN {first_year - 1} AND {last_year + 1})
        SELECT d.*, cv.passes_aqua::DOUBLE AS aqua_passes,
               (a.c6 IS NOT NULL)::FLOAT AS target, (so.ok AND ao.ok) AS usable
        FROM din d
        LEFT JOIN cov cv ON cv.h3_r4 = h3_cell_to_parent(d.c6::UBIGINT, 4)::BIGINT
                        AND cv.window_date = DATE '1970-01-01' + d.win
        LEFT JOIN fu a ON a.platform = {_literal(h.REFERENCE)} AND a.c6 = d.c6 AND a.day = d.day AND a.dn = d.dn
        JOIN fa so ON so.platform = d.platform AND so.day = d.day
        JOIN fa ao ON ao.platform = {_literal(h.REFERENCE)} AND ao.day = d.day"""


def _matrix(connection: duckdb.DuckDBPyConnection, platform: str, where: str, features: list[str]) -> dict[str, np.ndarray]:
    columns = ", ".join(f"{f}::FLOAT AS {f}" for f in features)
    arrays = connection.execute(f"SELECT c6, day, dn, target, {columns} FROM feat "
                                f"WHERE platform = {_literal(platform)} AND {where}").fetchnumpy()
    arrays["x"] = np.column_stack([np.asarray(arrays.pop(f), dtype=np.float32) for f in features])
    return arrays


def _train(connection: duckdb.DuckDBPyConnection, platform: str, years: list[int], features: list[str],
           threads: int) -> xgb.Booster:
    """Classifier on usable units of `years`; a random 10% of them is the early-stopping set."""
    data = _matrix(connection, platform, f"usable AND {h._years_sql(years)}", features)
    rng = np.random.default_rng(7)
    valid = rng.random(len(data["target"])) < 0.1
    dtrain = xgb.DMatrix(data["x"][~valid], label=data["target"][~valid], feature_names=features, nthread=threads)
    dvalid = xgb.DMatrix(data["x"][valid], label=data["target"][valid], feature_names=features, nthread=threads)
    return xgb.train({**PARAMS, "nthread": threads}, dtrain, ROUNDS, evals=[(dvalid, "valid")],
                     early_stopping_rounds=EARLY_STOP, verbose_eval=False)


def _predict(connection: duckdb.DuckDBPyConnection, models: dict[str, xgb.Booster], threads: int) -> None:
    """Temp tables `pu` (all `fu` units with model p; Aqua p = 1) and `fp` (source units only)."""
    parts = []
    for platform, booster in models.items():
        data = _matrix(connection, platform, "TRUE", booster.feature_names)
        p = booster.inplace_predict(data["x"]) if len(data["target"]) else np.empty(0, np.float32)
        parts.append(pa.table({"platform": pa.array([platform] * len(p)), "c6": data["c6"], "day": data["day"],
                               "dn": data["dn"], "p": pa.array(p, pa.float64())}))
    connection.register("pred", pa.concat_tables(parts))
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE pu AS
        SELECT u.*, CASE WHEN u.platform = {_literal(h.REFERENCE)} THEN 1.0 ELSE pr.p END AS p
        FROM fu u LEFT JOIN pred pr ON pr.platform = u.platform AND pr.c6 = u.c6 AND pr.day = u.day AND pr.dn = u.dn
        WHERE u.platform = {_literal(h.REFERENCE)} OR pr.p IS NOT NULL""")
    connection.execute("CREATE OR REPLACE TEMP TABLE fp AS SELECT platform, day, r2, r1, moy, yr, p FROM pu "
                       f"WHERE platform <> {_literal(h.REFERENCE)}")
    connection.unregister("pred")


def _score_model(connection: duckdb.DuckDBPyConnection, name: str, platform: str, years: list[int]) -> dict:
    """Like harmonization._score, with the model probabilities of `pu` instead of the curve."""
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE e AS
        SELECT x.platform, x.day, h3_cell_to_parent(x.c6::UBIGINT, 4)::BIGINT AS r4, x.yr * 100 + x.moy AS ym, x.est
        FROM ({h._with_est('pu')}) x
        JOIN fa a ON a.platform = x.platform AND a.day = x.day AND a.ok
        WHERE x.platform IN ({_literal(platform)}, {_literal(h.REFERENCE)}) AND x.{h._years_sql(years)}""")
    connection.execute(h._compare_sql(name, [platform]))
    return {**{f"res{level}": h._metrics(connection, name, level) for level in (1, 2, 4)},
            "spearman_res2": h._month_ranking(connection, name),
            "res2_by_volume_quartile": h._volume_quartiles(connection, name)}


def _importance(connection: duckdb.DuckDBPyConnection, booster: xgb.Booster, platform: str, years: list[int]) -> dict:
    """Mean |SHAP| per feature on up to 200k usable units of `years`, plus gain importance."""
    features = booster.feature_names
    data = _matrix(connection, platform, f"usable AND {h._years_sql(years)}", features)
    take = np.random.default_rng(11).permutation(len(data["target"]))[:200_000]
    contribs = booster.predict(xgb.DMatrix(data["x"][take], feature_names=features), pred_contribs=True)
    shap = np.abs(contribs[:, :-1]).mean(axis=0)
    gain = booster.get_score(importance_type="gain")
    order = np.argsort(-shap)
    return {"mean_abs_shap": {features[i]: float(shap[i]) for i in order},
            "gain": {f: float(gain.get(f, 0.0)) for f in features}, "rows": int(len(take)),
            "base_rate": float(data["target"].mean()) if len(data["target"]) else None}


def _baseline(connection: duckdb.DuckDBPyConnection, years: dict[str, list[int]], name: str,
              held: dict[str, list[int]]) -> dict:
    h._fit_curves(connection, years)
    h._fit_factors(connection, years, KAPPA)
    return {p: h._score(connection, f"{name}|v0.2|{p}", p, held[p]) for p in held}


def _model(connection: duckdb.DuckDBPyConnection, years: dict[str, list[int]], name: str,
           held: dict[str, list[int]], features: dict[str, list[str]], threads: int,
           blend: float = 0.0) -> tuple[dict, dict[str, xgb.Booster]]:
    """Model probabilities (optionally blended with the curve: p = (1 - blend) p_model + blend p_curve),
    refitted factors, scores on `held`."""
    models = {p: _train(connection, p, years[p], features[p], threads) for p in years}
    if blend:
        h._fit_curves(connection, years)
    _predict(connection, models, threads)
    if blend:
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE pu AS
            SELECT m.* EXCLUDE (p), CASE WHEN m.platform = {_literal(h.REFERENCE)} THEN 1.0
                                         ELSE {1 - blend} * m.p + {blend} * c.p END AS p
            FROM pu m LEFT JOIN ({h._with_p('fu')}) c
              ON c.platform = m.platform AND c.c6 = m.c6 AND c.day = m.day AND c.dn = m.dn""")
        connection.execute("CREATE OR REPLACE TEMP TABLE fp AS SELECT platform, day, r2, r1, moy, yr, p FROM pu "
                           f"WHERE platform <> {_literal(h.REFERENCE)}")
    h._fit_factors(connection, years, KAPPA)
    return {p: _score_model(connection, f"{name}|model|{p}", p, held[p]) for p in held}, models


def _acceptance(folds: list[dict], platform: str) -> dict:
    """v0.2 acceptance rule with the curve-based v0.2 as the reference variant."""
    renamed = [{"scores": {"cutoff": {platform: f["scores"]["v0.2"][platform]},
                           "model": {platform: f["scores"]["model"][platform]}}} for f in folds]
    return h._acceptance(renamed, platform, "model")


def run(silver: str, coverage: str, output: str | Path, *, exclude: tuple[str, ...] = (),
        exclude_for: dict[str, tuple[str, ...]] | None = None, blend: float = 0.0,
        storage_options: dict[str, str] | None = None, memory_limit: str = "16GB", threads: int = 8,
        temp_dir: str | Path | None = None) -> dict:
    """`exclude` drops model inputs for every platform, `exclude_for` per platform; `blend` mixes the curve
    probability in. The S-NPP 2012 clean test was first read by the run with all inputs (2026-11-15); later
    runs read it again and are labelled as such."""
    report_path = Path(output)
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists: the clean test is scored once")
    exclude_for = exclude_for or {}
    unknown = (set(exclude) | {f for fs in exclude_for.values() for f in fs}) - set(FEATURES)
    if unknown or set(exclude_for) - set(SOURCES):
        raise ValueError(f"Unknown inputs or platforms to exclude: {sorted(unknown)} {sorted(exclude_for)}")
    features = {p: [f for f in FEATURES if f not in exclude and f not in exclude_for.get(p, ())] for p in SOURCES}
    first, last = 2003, 2017
    clean_platform, clean_years = CLEAN_TEST["platform"], list(CLEAN_TEST["years"])
    if any(y in h._fit_years(clean_platform) for y in clean_years):
        raise ValueError("The clean test years must lie outside the fit years of the clean-test platform")
    tic = time.monotonic()
    with ml_dataset._connect(silver, coverage, memory_limit, threads, temp_dir, storage_options) as connection:
        h._load_fit_units(connection, [(min(first, *clean_years), max(last, *clean_years))], tic)
        connection.execute("CREATE OR REPLACE TEMP TABLE avail AS SELECT * FROM fa")
        connection.execute("CREATE OR REPLACE TEMP TABLE cmp (series VARCHAR, r4 BIGINT, ym INTEGER, pred DOUBLE, "
                           "raw DOUBLE, actual DOUBLE)")
        connection.execute(h._detection_inputs_sql("fu", SOURCES))
        connection.execute(_features_sql(first, last))
        connection.execute("DROP TABLE din")
        coverage_rate = connection.execute(
            "SELECT platform, avg((aqua_passes IS NOT NULL)::INTEGER), avg((aqua_passes > 0)::INTEGER) FROM feat "
            "GROUP BY 1").fetchall()
        print(f"harmonize-detect features ready ({time.monotonic() - tic:.0f}s)", file=sys.stderr, flush=True)
        folds = []
        for index, held_years in enumerate(h.CV_FOLDS):
            held = {p: list(v) for p, v in held_years.items()}
            years = {p: [y for y in h._fit_years(p) if all(abs(y - x) > h.CV_GAP for x in held[p])] for p in held}
            base = _baseline(connection, years, f"fold{index}", held)
            model, _ = _model(connection, years, f"fold{index}", held, features, threads, blend)
            folds.append({"held": held, "fit_years": years, "scores": {"v0.2": base, "model": model}})
            print(f"harmonize-detect fold {index} done ({time.monotonic() - tic:.0f}s)", file=sys.stderr, flush=True)
        acceptance = {p: _acceptance(folds, p) for p in SOURCES}
        # Final models on all fit years: importance, then the clean Suomi-NPP test and the Terra reference.
        years = {p: h._fit_years(p) for p in SOURCES}
        clean_years_fit = {**years, clean_platform: [y for y in years[clean_platform]
                                                     if all(abs(y - c) > 0 for c in clean_years)]}
        held = {clean_platform: clean_years}
        clean = {"v0.2": _baseline(connection, clean_years_fit, "clean", held)[clean_platform]}
        clean_model, models = _model(connection, clean_years_fit, "clean", held, features, threads, blend)
        clean["model"] = clean_model[clean_platform]
        importance = {p: _importance(connection, models[p], p, years[p]) for p in SOURCES}
        terra_years = {"Terra": list(range(h.EXTRAPOLATION["fit"][0], h.EXTRAPOLATION["fit"][1] + 1))}
        terra_held = {"Terra": list(range(h.EXTRAPOLATION["test"][0], h.EXTRAPOLATION["test"][1] + 1))}
        terra = {"v0.2": _baseline(connection, terra_years, "terra_ref", terra_held)["Terra"],
                 "model": _model(connection, terra_years, "terra_ref", terra_held, features, threads, blend)[0]["Terra"]}
    accepted = all(a["pass"] for a in acceptance.values())
    report = {"method_version": METHOD_VERSION, "silver": silver, "coverage": coverage,
              "silver_version": DeltaTable(silver, storage_options=storage_options).version(),
              "features": features, "excluded": list(exclude), "excluded_for": exclude_for, "blend": blend,
              "params": PARAMS, "rounds": ROUNDS, "kappa": KAPPA,
              "coverage_rate": [[p, float(a), float(b)] for p, a, b in coverage_rate],
              "folds": folds, "acceptance_vs_v0.2": acceptance, "accepted": accepted,
              "clean_test_snpp_2012": clean,
              "clean_test_note": "S-NPP 2012 is read again by every run after the first (all inputs, 2026-11-15); "
                                 "not independent",
              "terra_2003_2004_reference_reused": terra, "importance": importance,
              "seconds": round(time.monotonic() - tic, 1), "counts": {"accepted": accepted}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, default=float))
    return report
