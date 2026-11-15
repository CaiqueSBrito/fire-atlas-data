"""Command-line interface for FIRMS normalization, the Delta lake, harmonization and the XGBoost forecast."""
import argparse
from datetime import date, datetime, time, timezone
import json
import sys

from .coldstart import build as build_coldstart, evaluate_model as evaluate_coldstart
from .coverage import PRODUCTS as COVERAGE_PRODUCTS, build_cells, fetch_granules
from .gold import build_map_cells, optimize_map_cells, verify_map_cells
from .harmonization import (apply as apply_harmonization, cross_validate as cross_validate_harmonization,
                            evaluate as evaluate_harmonization,
                            evaluate_eras as evaluate_harmonization_eras, fit as fit_harmonization,
                            verify as verify_harmonization)
from .harmonization_detection import run as run_detection_experiment
from .daily_ingest import PRODUCTS as DAILY_PRODUCTS, run as run_daily_ingest, run_daily
from .ml_dataset import build as build_ml_dataset
from .ml_forecast import forecast as forecast_ml
from .ml_training import train as train_ml
from .places import build as build_places, verify as verify_places
from .normalization import PRODUCTS, normalize
from .silver import build as build_silver, optimize as optimize_silver, vacuum as vacuum_silver, verify as verify_silver


def main() -> int:
    parser = argparse.ArgumentParser(description="FIRMS normalization, Delta lake layers, harmonization and forecast")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("normalize")
    command.add_argument("input")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--source-product", required=True, choices=sorted(PRODUCTS))
    command.add_argument("--source-url", help="Optional user-declared provenance reference; never fetched or verified")
    command.add_argument("--memory-limit", default="1GB")
    command.add_argument("--threads", type=int, default=2)
    command.add_argument("--temp-dir")
    silver = commands.add_parser("silver", help="Write normalized datasets with H3 res0..6 to a Delta table")
    silver.add_argument("datasets", nargs="+", help="Normalized dataset directories (with manifest.json)")
    silver.add_argument("--target", required=True, help="Delta table path or s3:// URI")
    silver.add_argument("--month", action="append", type=lambda v: tuple(int(p) for p in v.split("-")),
                        help="YYYY-MM to (re)write; repeatable; default all months in the sources")
    silver.add_argument("--optimize", action="store_true",
                        help="Z-order on h3_r6 every partition written since its last Z-order")
    silver.add_argument("--vacuum", action="store_true",
                        help="Delete files unreferenced by the latest version (older versions become unreadable)")
    silver.add_argument("--memory-limit", default="2GB")
    silver.add_argument("--threads", type=int, default=2)
    silver.add_argument("--temp-dir")
    gold_map = commands.add_parser("gold-map", help="Aggregate silver into daily H3 res0..6 map cells per sensor")
    gold_map.add_argument("--silver", required=True, help="Silver Delta table path or s3:// URI")
    gold_map.add_argument("--target", required=True, help="Gold map_cells Delta table path or s3:// URI")
    gold_map.add_argument("--year", action="append", type=int,
                          help="Year to (re)write; repeatable; default all years in silver")
    gold_map.add_argument("--optimize", action="store_true",
                          help="Z-order on (day, h3_cell) every partition written since its last Z-order")
    gold_map.add_argument("--vacuum", action="store_true",
                          help="Delete files unreferenced by the latest version (older versions become unreadable)")
    gold_map.add_argument("--memory-limit", default="2GB")
    gold_map.add_argument("--threads", type=int, default=2)
    gold_map.add_argument("--temp-dir", help="Spill directory for DuckDB and Z-order (default /tmp may be RAM)")
    places = commands.add_parser("gold-places",
                                 help="Continent, country and nearest city per H3 res0..6 cell with detections")
    places.add_argument("--silver", required=True, help="Silver Delta table path or s3:// URI")
    places.add_argument("--countries", required=True, help="Natural Earth ne_10m_admin_0_countries.zip")
    places.add_argument("--cities", required=True, help="GeoNames cities1000.zip")
    places.add_argument("--target", required=True, help="Gold h3_places Delta table path or s3:// URI")
    places.add_argument("--memory-limit", default="4GB")
    places.add_argument("--threads", type=int, default=4)
    places.add_argument("--temp-dir")
    footprints = commands.add_parser("coverage-granules",
                                     help="Fetch NASA CMR swath footprints (metadata only) into a Delta table")
    footprints.add_argument("--target", required=True, help="Granule footprints Delta table path")
    footprints.add_argument("--product", action="append", choices=sorted(COVERAGE_PRODUCTS),
                            help="Repeatable; default all active-fire L2 products")
    footprints.add_argument("--year", action="append", type=int, help="Repeatable; default first year..current")
    footprints.add_argument("--refresh", action="store_true", help="Refetch (product, year) already present")
    footprints.add_argument("--workers", type=int, default=4)
    passes = commands.add_parser("coverage-cells",
                                 help="Count satellite passes per H3 res4 cell and 24 h window from footprints")
    passes.add_argument("--granules", required=True, help="Local granule footprints Delta table")
    passes.add_argument("--map-cells", required=True, help="Gold map table; supplies the cells to evaluate")
    passes.add_argument("--target", required=True, help="Coverage cells Delta table path or s3:// URI")
    passes.add_argument("--year", action="append", type=int, help="Repeatable; default all years in granules")
    passes.add_argument("--window-start-hour", type=int, default=9,
                        help="UTC hour where each 24 h window starts (effective input cutoff)")
    passes.add_argument("--workers", type=int, default=4)
    ml = commands.add_parser("ml-dataset", help="Build the D -> [D cutoff, D+7 cutoff) H3 res6 ML dataset")
    ml.add_argument("--silver", required=True)
    ml.add_argument("--coverage", required=True, help="Coverage cells Delta table")
    ml.add_argument("--target", required=True)
    ml.add_argument("--month", action="append", type=lambda v: tuple(int(p) for p in v.split("-")),
                    help="YYYY-MM forecast month to (re)write; repeatable; default every month not yet written")
    ml.add_argument("--eval-start", type=date.fromisoformat, default=date(2023, 1, 1),
                    help="First forecast day of the full (unsampled) eval split")
    ml.add_argument("--sample-rate", type=float, default=0.1, help="Train share of active negatives kept")
    ml.add_argument("--cutoff-hour", type=int, default=9, help="UTC hour of the effective input cutoff")
    ml.add_argument("--verify-sample", type=int, default=2000,
                    help="Rows per month recomputed from silver before the month is committed")
    ml.add_argument("--workers", type=int, default=1, help="Months built in parallel (each ~6 GB RAM peak)")
    ml.add_argument("--staging-dir", help="Local dir for verified months awaiting commit (default <target>.staging)")
    ml.add_argument("--memory-limit", default="3GB")
    ml.add_argument("--threads", type=int, default=4)
    ml.add_argument("--temp-dir")
    fit = commands.add_parser("ml-train", help="XGBoost random search on gold/ml_dataset; test scored once")
    fit.add_argument("--dataset", required=True, help="gold/ml_dataset Delta table")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--train-start", type=date.fromisoformat, default=date(2012, 2, 1))
    fit.add_argument("--valid-start", type=date.fromisoformat, default=date(2021, 1, 1))
    fit.add_argument("--test-start", type=date.fromisoformat, default=date(2023, 1, 1))
    fit.add_argument("--test-end", type=date.fromisoformat, help="Default: last forecast day in the dataset")
    fit.add_argument("--embargo-days", type=int, default=14, help="Gap after train and after valid (>= 7)")
    fit.add_argument("--configs", type=int, default=20, help="Random search size")
    fit.add_argument("--seed", type=int, default=20261114)
    fit.add_argument("--max-rounds", type=int, default=3000)
    fit.add_argument("--early-stopping", type=int, default=100)
    fit.add_argument("--row-sample-rate", type=float, default=0.2,
                     help="Share of train/valid `cell` rows kept (weight 1/rate) to fit GPU memory; test is complete")
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--memory-limit", default="8GB")
    fit.add_argument("--threads", type=int, default=8)
    cold = commands.add_parser("ml-coldstart",
                               help="Experiment rows for never-detected land cells with neighbour activity")
    cold.add_argument("--silver", required=True)
    cold.add_argument("--coverage", required=True, help="Coverage cells Delta table")
    cold.add_argument("--countries", required=True, help="Natural Earth ne_10m_admin_0_countries.zip (land mask)")
    cold.add_argument("--target", required=True)
    cold.add_argument("--month", action="append", type=lambda v: tuple(int(p) for p in v.split("-")),
                      help="YYYY-MM forecast month to (re)write; repeatable; default every month not yet written")
    cold.add_argument("--first-month", type=lambda v: tuple(int(p) for p in v.split("-")), default=(2012, 2))
    cold.add_argument("--eval-start", type=date.fromisoformat, default=date(2023, 1, 1))
    cold.add_argument("--sample-rate", type=float, default=0.05, help="Share of train/valid negatives kept")
    cold.add_argument("--test-sample-rate", type=float, default=0.2, help="Share of eval negatives kept")
    cold.add_argument("--cutoff-hour", type=int, default=9)
    cold.add_argument("--memory-limit", default="48GB")
    cold.add_argument("--threads", type=int, default=16)
    cold.add_argument("--temp-dir")
    cold_eval = commands.add_parser("ml-coldstart-eval", help="Score cold-start test rows with a trained model")
    cold_eval.add_argument("--dataset", required=True)
    cold_eval.add_argument("--model", required=True, help="Local model.ubj")
    cold_eval.add_argument("--output", required=True, help="Report JSON path")
    cold_eval.add_argument("--train-start", type=date.fromisoformat, default=date(2012, 2, 1))
    cold_eval.add_argument("--train-end", type=date.fromisoformat, default=date(2020, 12, 17))
    cold_eval.add_argument("--test-start", type=date.fromisoformat, default=date(2023, 1, 1))
    cold_eval.add_argument("--test-end", type=date.fromisoformat)
    cold_eval.add_argument("--memory-limit", default="48GB")
    cold_eval.add_argument("--threads", type=int, default=16)
    harmonize_fit = commands.add_parser("harmonize-fit",
                                        help="Fit the MODIS/VIIRS -> Aqua-equivalent curves and regional factors")
    harmonize = commands.add_parser("harmonize",
                                    help="Write harmonized activity per H3 res6 cell and local day, per year")
    harmonize_eval = commands.add_parser("harmonize-eval",
                                         help="Final test of the fitted model on the reserved years (runs once)")
    harmonize_eras = commands.add_parser("harmonize-eras",
                                         help="Error of the series per era against Aqua, model fixed")
    for harmonize_command in (harmonize_fit, harmonize, harmonize_eval, harmonize_eras):
        harmonize_command.add_argument("--silver", required=True, help="Silver Delta table path or s3:// URI")
        harmonize_command.add_argument("--model-dir", required=True, help="Local directory of the fitted model")
        harmonize_command.add_argument("--memory-limit", default="16GB")
        harmonize_command.add_argument("--threads", type=int, default=8)
        harmonize_command.add_argument("--temp-dir")
    harmonize.add_argument("--target", required=True, help="Harmonized cells Delta table path or s3:// URI")
    harmonize.add_argument("--availability", required=True, help="Platform availability Delta table")
    harmonize.add_argument("--year", action="append", type=int,
                           help="Local-day year to (re)write; repeatable; default every year in silver")
    harmonize_eval.add_argument("--output", required=True, help="Report JSON path; must not exist")
    harmonize_eras.add_argument("--output", required=True, help="Report JSON path; must not exist")
    harmonize_fit.add_argument("--kappa", type=float,
                               help="Shrink regional factors toward their parent with this weight (default: cutoff)")
    harmonize_fit.add_argument("--detection", action="store_true",
                               help="Train the VIIRS detection model (v0.3) instead of using the curve for VIIRS")
    harmonize_fit.add_argument("--blend", type=float, default=0.0,
                               help="Weight of the curve mixed into the detection probability")
    harmonize_cv = commands.add_parser("harmonize-cv",
                                       help="Year-block validation of shrinkage vs cutoff, then the Terra "
                                            "extrapolation test (runs once)")
    harmonize_cv.add_argument("--silver", required=True)
    harmonize_cv.add_argument("--output", required=True, help="Report JSON path; must not exist")
    harmonize_cv.add_argument("--memory-limit", default="16GB")
    harmonize_cv.add_argument("--threads", type=int, default=8)
    harmonize_cv.add_argument("--temp-dir")
    detect = commands.add_parser("harmonize-detect",
                                 help="Experiment: learned detection model vs the v0.2 curve (CV + clean S-NPP 2012)")
    detect.add_argument("--silver", required=True)
    detect.add_argument("--coverage", required=True, help="coverage/cells Delta table (Aqua passes per res4 window)")
    detect.add_argument("--output", required=True, help="Report JSON path; must not exist")
    detect.add_argument("--exclude", action="append", default=[], help="Model input to leave out; repeatable")
    detect.add_argument("--exclude-for", action="append", default=[], metavar="PLATFORM:INPUT",
                        help="Model input to leave out for one platform (e.g. Terra:lat); repeatable")
    detect.add_argument("--blend", type=float, default=0.0, help="Weight of the curve probability mixed in")
    detect.add_argument("--memory-limit", default="16GB")
    detect.add_argument("--threads", type=int, default=8)
    detect.add_argument("--temp-dir")
    predict = commands.add_parser("ml-forecast",
                                  help="Score every universe H3 res6 cell for one issue date with the trained model")
    predict.add_argument("--silver", required=True)
    predict.add_argument("--model", required=True, help="model.ubj written by ml-train; local path or s3:// URI")
    predict.add_argument("--report", help="report.json written by ml-train (local or s3://); checks features")
    predict.add_argument("--target", required=True, help="Forecast Delta table path or s3:// URI")
    predict.add_argument("--date", required=True, type=date.fromisoformat, help="Issue date D (YYYY-MM-DD)")
    predict.add_argument("--cutoff-hour", type=int, default=9, help="UTC hour of the input cutoff (as in training)")
    predict.add_argument("--max-input-lag-hours", type=float, default=6.0,
                         help="Refuse when the newest silver record is older than this before the cutoff")
    predict.add_argument("--memory-limit", default="3GB")
    predict.add_argument("--threads", type=int, default=4)
    predict.add_argument("--temp-dir")
    daily = commands.add_parser("daily-ingest",
                                help="FIRMS Area API (world, NRT) -> normalize -> silver append -> ml-forecast")
    daily.add_argument("--silver", required=True)
    daily.add_argument("--model", required=True)
    daily.add_argument("--report")
    daily.add_argument("--forecast-target", required=True, help="Forecast Delta table")
    daily.add_argument("--work-dir", required=True, help="Run directories with raw CSVs, receipts and reports")
    daily.add_argument("--date", type=date.fromisoformat, help="Issue date D; default today (UTC)")
    daily.add_argument("--daily-at", type=time.fromisoformat,
                       help="HH:MM UTC: keep running and process each day at this time (container mode)")
    daily.add_argument("--retry-minutes", type=int, default=60)
    daily.add_argument("--product", action="append", choices=PRODUCTS,
                       help=f"Repeatable; default {', '.join(DAILY_PRODUCTS)}")
    daily.add_argument("--lookback-days", type=int, default=10, help="Furthest back a missing product is fetched")
    daily.add_argument("--cutoff-hour", type=int, default=9)
    daily.add_argument("--key-env", default="FIRMS_MAP_KEY")
    daily.add_argument("--optimize", action="store_true", help="Z-order the silver months appended")
    daily.add_argument("--memory-limit", default="3GB")
    daily.add_argument("--threads", type=int, default=4)
    daily.add_argument("--archive-uri", help="Copy each successful run directory here, e.g. s3://bucket/bronze/daily-ingest")
    daily.add_argument("--places-target", help="gold/h3_places Delta table rebuilt after the forecast")
    daily.add_argument("--places-countries", help="Natural Earth admin-0 zip (path or s3:// URI)")
    daily.add_argument("--places-cities", help="GeoNames cities1000.zip (path or s3:// URI)")
    daily.add_argument("--harmonize-model", help="Fitted harmonization model directory (path or s3:// URI)")
    daily.add_argument("--harmonize-target", help="Existing harmonized cells Delta table; recent years rewritten")
    daily.add_argument("--harmonize-availability", help="Existing platform availability Delta table")
    daily.add_argument("--temp-dir")
    args = parser.parse_args()
    try:
        if args.command == "silver":
            manifest = build_silver(args.datasets, args.target, months=args.month,
                                    memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir)
            if args.optimize:
                optimize_silver(args.target)
            if args.vacuum:
                vacuum_silver(args.target)
            manifest["counts"] = verify_silver(args.target, memory_limit=args.memory_limit)
        elif args.command == "gold-map":
            manifest = build_map_cells(args.silver, args.target, years=args.year, memory_limit=args.memory_limit,
                                       threads=args.threads, temp_dir=args.temp_dir)
            if args.optimize:
                optimize_map_cells(args.target, temp_dir=args.temp_dir)
            if args.vacuum:
                vacuum_silver(args.target)
            manifest["counts"] = verify_map_cells(args.target, args.silver, memory_limit=args.memory_limit)["rows"]
        elif args.command == "gold-places":
            manifest = build_places(args.silver, args.countries, args.cities, args.target,
                                    memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir)
            manifest["verification"] = verify_places(args.target, args.silver, memory_limit=args.memory_limit)
            print(json.dumps(manifest, allow_nan=False, default=str), file=sys.stderr)
        elif args.command == "coverage-granules":
            manifest = fetch_granules(args.target, products=args.product, years=args.year,
                                      refresh=args.refresh, workers=args.workers)
        elif args.command == "coverage-cells":
            manifest = build_cells(args.granules, args.map_cells, args.target, years=args.year,
                                   window_start_hour=args.window_start_hour, workers=args.workers)
        elif args.command == "ml-dataset":
            manifest = build_ml_dataset(args.silver, args.coverage, args.target, months=args.month,
                                        eval_start=args.eval_start, sample_rate=args.sample_rate,
                                        cutoff_hour=args.cutoff_hour, workers=args.workers,
                                        verify_sample=args.verify_sample, staging_dir=args.staging_dir,
                                        memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir)
        elif args.command == "ml-train":
            manifest = train_ml(args.dataset, args.output_dir, train_start=args.train_start,
                                valid_start=args.valid_start, test_start=args.test_start, test_end=args.test_end,
                                embargo_days=args.embargo_days, configs=args.configs, seed=args.seed,
                                max_rounds=args.max_rounds, early_stopping=args.early_stopping,
                                row_sample_rate=args.row_sample_rate,
                                device=args.device, memory_limit=args.memory_limit, threads=args.threads)
        elif args.command == "ml-coldstart":
            manifest = build_coldstart(args.silver, args.coverage, args.countries, args.target,
                                       first_month=args.first_month, months=args.month, eval_start=args.eval_start,
                                       sample_rate=args.sample_rate, test_sample_rate=args.test_sample_rate,
                                       cutoff_hour=args.cutoff_hour, memory_limit=args.memory_limit,
                                       threads=args.threads, temp_dir=args.temp_dir)
        elif args.command == "ml-coldstart-eval":
            manifest = evaluate_coldstart(args.dataset, args.model, args.output, train_start=args.train_start,
                                          train_end=args.train_end, test_start=args.test_start,
                                          test_end=args.test_end, memory_limit=args.memory_limit,
                                          threads=args.threads)
        elif args.command == "harmonize-fit":
            manifest = fit_harmonization(args.silver, args.model_dir, kappa=args.kappa, detection=args.detection,
                                         blend=args.blend, memory_limit=args.memory_limit, threads=args.threads,
                                         temp_dir=args.temp_dir)
        elif args.command == "harmonize":
            manifest = apply_harmonization(args.silver, args.model_dir, args.target, args.availability,
                                           years=args.year, memory_limit=args.memory_limit, threads=args.threads,
                                           temp_dir=args.temp_dir)
            verify_harmonization(args.target, args.silver, years=[int(y) for y in manifest["counts"]],
                                 memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir)
        elif args.command == "harmonize-eval":
            manifest = evaluate_harmonization(args.silver, args.model_dir, args.output,
                                              memory_limit=args.memory_limit, threads=args.threads,
                                              temp_dir=args.temp_dir)
        elif args.command == "harmonize-eras":
            manifest = evaluate_harmonization_eras(args.silver, args.model_dir, args.output,
                                                   memory_limit=args.memory_limit, threads=args.threads,
                                                   temp_dir=args.temp_dir)
        elif args.command == "harmonize-cv":
            manifest = cross_validate_harmonization(args.silver, args.output, memory_limit=args.memory_limit,
                                                    threads=args.threads, temp_dir=args.temp_dir)
        elif args.command == "harmonize-detect":
            exclude_for: dict[str, tuple[str, ...]] = {}
            for item in args.exclude_for:
                platform, name = item.split(":", 1)
                exclude_for[platform] = (*exclude_for.get(platform, ()), name)
            manifest = run_detection_experiment(args.silver, args.coverage, args.output, exclude=tuple(args.exclude),
                                                exclude_for=exclude_for, blend=args.blend,
                                                memory_limit=args.memory_limit, threads=args.threads,
                                                temp_dir=args.temp_dir)
        elif args.command == "ml-forecast":
            manifest = forecast_ml(args.silver, args.model, args.target, issue_date=args.date, report=args.report,
                                   cutoff_hour=args.cutoff_hour, max_input_lag_hours=args.max_input_lag_hours,
                                   memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir)
        elif args.command == "daily-ingest":
            options = dict(silver=args.silver, model=args.model, report=args.report,
                           forecast_target=args.forecast_target, work_dir=args.work_dir,
                           products=args.product or DAILY_PRODUCTS, lookback_days=args.lookback_days,
                           cutoff_hour=args.cutoff_hour, key_env=args.key_env, optimize=args.optimize,
                           memory_limit=args.memory_limit, threads=args.threads, temp_dir=args.temp_dir,
                           archive_uri=args.archive_uri, places_target=args.places_target,
                           places_countries=args.places_countries, places_cities=args.places_cities,
                           harmonize_model=args.harmonize_model, harmonize_target=args.harmonize_target,
                           harmonize_availability=args.harmonize_availability)
            if args.daily_at is not None:
                run_daily(args.daily_at.replace(tzinfo=None), retry_minutes=args.retry_minutes, **options)
            manifest = run_daily_ingest(args.date or datetime.now(timezone.utc).date(), **options)
        else:
            manifest = normalize(args.input, args.output_dir, source_product=args.source_product,
                                 source_url=args.source_url, memory_limit=args.memory_limit,
                                 threads=args.threads, temp_dir=args.temp_dir)
    except Exception as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
