"""Command-line interface for local FIRMS normalization and map aggregation."""
import argparse
from datetime import date
import json
import sys

from .coverage import PRODUCTS as COVERAGE_PRODUCTS, build_cells, fetch_granules
from .map_grid import DEFAULT_RESOLUTIONS, GROUPS, aggregate
from .gold import build_map_cells, optimize_map_cells, verify_map_cells
from .ml_dataset import build as build_ml_dataset
from .ml_training import train as train_ml
from .normalization import PRODUCTS, normalize
from .silver import build as build_silver, optimize as optimize_silver, vacuum as vacuum_silver, verify as verify_silver


def _bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox requires west,south,east,north")
    return tuple(parts)


def _resolutions(value: str) -> tuple[int, ...]:
    return tuple(int(p) for p in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize FIRMS CSV to bounded-memory Parquet datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("normalize")
    command.add_argument("input")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--source-product", required=True, choices=sorted(PRODUCTS))
    command.add_argument("--source-url", help="Optional user-declared provenance reference; never fetched or verified")
    command.add_argument("--memory-limit", default="1GB")
    command.add_argument("--threads", type=int, default=2)
    command.add_argument("--temp-dir")
    grid = commands.add_parser("map-grid", help="Aggregate normalized datasets into daily H3 map cells")
    grid.add_argument("datasets", nargs="+", help="Normalized dataset directories (with manifest.json)")
    grid.add_argument("--output-dir", required=True)
    grid.add_argument("--resolutions", type=_resolutions, default=DEFAULT_RESOLUTIONS,
                      help="Comma-separated H3 resolutions, one per zoom band (default 2..8)")
    grid.add_argument("--group-by", choices=GROUPS, default="sensor")
    grid.add_argument("--start", type=date.fromisoformat)
    grid.add_argument("--end", type=date.fromisoformat)
    grid.add_argument("--bbox", type=_bbox, help="west,south,east,north in EPSG:4326")
    grid.add_argument("--memory-limit", default="1GB")
    grid.add_argument("--threads", type=int, default=2)
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
        elif args.command == "map-grid":
            manifest = aggregate(args.datasets, args.output_dir, resolutions=args.resolutions,
                                 group_by=args.group_by, start=args.start,
                                 end=args.end, bbox=args.bbox, memory_limit=args.memory_limit,
                                 threads=args.threads)
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
