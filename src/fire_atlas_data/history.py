"""Public FIRMS Brazil country history, deliberately separate from NRT normalization."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

import duckdb

from .normalization import identifier, literal, sha256

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country"
INVENTORY_URL = BASE + "/yearly_summary_files.txt"
README_URL = BASE + "/AttributesReadme.txt"
PRODUCT = "VIIRS_NOAA20_SP"
SCHEMA = "historical-observations/1.0.0"
YEARS = tuple(range(2018, 2026))
BOUNDS = (-50, -19, -44, -13)
REQUIRED = {"latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp"}
LIMITATIONS = [
    "Standard-processing (SP) history is not the NOAA-20 NRT target vintage; no training or label integration is performed.",
    "SHA256 and HTTPS receipts establish local byte preservation and source retrieval, not NASA authenticity proof or scientific equivalence.",
    "Receipt time and inventory updated text are not historical observation availability; availability remains unknown.",
    "Country files contain Brazil only, not global observations; Brazil counts are never described as world counts.",
    "No observability mask, negative labels, confidence filtering or deduplication is inferred from detection points.",
    "Annual files and acquisition spans do not prove complete satellite coverage; 2018 is an initial partial year and 2025 may be partial.",
    "Seasonal coverage and SP/NRT processing differences require an explicit scientific contract before experiments can mix these observations.",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def _download(url: str, path: Path, *, cap: int, expected_size: int | None = None) -> dict:
    """Stream at most cap+1 bytes; never replace a published raw file or receipt."""
    receipt_path = path.with_name(path.name + ".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError(f"Incomplete preserved download: {path}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt["url"] != url or path.stat().st_size != receipt["size_bytes"]
                or sha256(path) != receipt["sha256"]
                or (expected_size is not None and receipt["size_bytes"] != expected_size)):
            raise ValueError(f"Preserved download does not match receipt/inventory: {path}")
        return receipt
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    started = _now()
    # A stale partial is not raw evidence and is intentionally not reused.
    with temporary.open("wb") as stream:
        try:
            request = Request(url, headers={"User-Agent": "fire-atlas-data/0.1", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=120) as response:
                if response.status != 200 or not response.geturl().startswith(BASE + "/"):
                    raise ValueError("Public FIRMS download returned unexpected status or redirect")
                size = 0
                while chunk := response.read(min(1024 * 1024, cap - size + 1)):
                    size += len(chunk)
                    if size > cap:
                        raise ValueError("Download exceeds request byte cap")
                    stream.write(chunk)
                if expected_size is not None and size != expected_size:
                    raise ValueError(f"Inventory/download byte mismatch: {size} != {expected_size}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size:
                    raise ValueError("Incomplete HTTP download")
                metadata = {"url": url, "final_url": response.geturl(), "http_status": response.status,
                            "response_headers": dict(response.headers.items())}
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    receipt = {**metadata, "started_at_utc": started, "received_at_utc": _now(),
               "size_bytes": size, "sha256": sha256(temporary), "path": path.name}
    os.link(temporary, path)
    temporary.unlink()
    path.chmod(0o444)
    _json_new(receipt_path, receipt)
    return receipt


def inventory_selection(path: Path, years: tuple[int, ...]) -> list[dict]:
    if not years or len(set(years)) != len(years) or any(y not in YEARS for y in years):
        raise ValueError("Years must be unique values from 2018 through 2025")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filename", "size", "updated"]:
            raise ValueError("Unexpected public inventory columns")
        wanted = {f"viirs-jpss1_{year}_Brazil.csv": year for year in years}
        selected = {}
        for row in reader:
            filename = row["filename"]
            if filename in wanted:
                if filename in selected or not row["size"].isdigit() or int(row["size"]) < 1:
                    raise ValueError("Duplicate or invalid inventory entry")
                year = wanted[filename]
                selected[filename] = {"year": year, "filename": filename, "size_bytes": int(row["size"]),
                                      "inventory_updated_text": row["updated"],
                                      "url": f"{BASE}/viirs-jpss1/{year}/{filename}"}
        if set(selected) != set(wanted):
            raise ValueError(f"Years absent from actual inventory: {sorted(set(wanted) - set(selected))}")
    return sorted(selected.values(), key=lambda row: row["year"])


def acquire(output_dir: str | Path, *, years: tuple[int, ...] = YEARS,
            max_bytes: int = 1_000_000_000) -> dict:
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    inventory = _download(INVENTORY_URL, root / "yearly_summary_files.txt", cap=2 * 1024 * 1024)
    selected = inventory_selection(root / "yearly_summary_files.txt", years)
    planned_bytes = sum(row["size_bytes"] for row in selected)
    if planned_bytes > max_bytes:
        raise ValueError(f"Inventory selection exceeds quota: {planned_bytes} > {max_bytes}")
    # Room for raw bytes plus on-disk DuckDB staging and Parquet; no global ZIP.
    if shutil.disk_usage(root).free < 3 * planned_bytes + 128 * 1024 * 1024:
        raise ValueError("Insufficient disk for raw files, preparation and spill")
    readme = _download(README_URL, root / "AttributesReadme.txt", cap=64 * 1024)
    receipts = []
    for entry in selected:
        receipt = _download(entry["url"], root / "raw" / entry["filename"],
                            cap=entry["size_bytes"], expected_size=entry["size_bytes"])
        receipts.append({**entry, "raw_path": "raw/" + entry["filename"], "receipt": receipt})
    manifest = {"schema": "historical-country-acquisition/1.0.0", "product_declared": PRODUCT,
                "processing_declared": "standard", "country": "Brazil", "years": list(years),
                "inventory": inventory, "attributes_readme": readme, "total_raw_bytes": planned_bytes,
                "files": receipts, "limitations": LIMITATIONS}
    destination = root / "acquisition.json"
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError("Existing acquisition manifest disagrees; choose a new directory")
    else:
        _json_new(destination, manifest)
    return manifest


def _source_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if (not header or any(not h for h in header) or len({h.lower() for h in header}) != len(header)
            or not REQUIRED.issubset(header)):
        raise ValueError("History CSV needs unique source columns and observation fields")
    if any(h.lower().startswith("history_") for h in header):
        raise ValueError("Source column collides with historical preparation fields")
    return header


def _prepare_year(connection, entry: dict, root: Path, stage: Path) -> dict:
    source = root / entry["raw_path"]
    receipt = entry["receipt"]
    before = source.stat()
    if before.st_size != receipt["size_bytes"] or sha256(source) != receipt["sha256"]:
        raise ValueError("Raw bytes disagree with immutable download receipt")
    year = entry["year"]
    header = _source_header(source)
    field = lambda name: f"nullif(trim({identifier(name)}),'')" if name in header else "NULL::VARCHAR"
    columns = ",".join(f"{literal(h)}:'VARCHAR'" for h in header)
    raw_struct = ",".join(f"{identifier(h)} := {identifier(h)}" for h in header)
    date_text = "trim(acq_date) || ' ' || lpad(trim(acq_time),4,'0')"
    parsed_time = ("CASE WHEN regexp_full_match(trim(acq_date),'[0-9]{4}-[0-9]{2}-[0-9]{2}') "
                   "AND regexp_full_match(trim(acq_time),'[0-9]{1,4}') "
                   "AND try_cast(left(lpad(trim(acq_time),4,'0'),2) AS INTEGER) BETWEEN 0 AND 23 "
                   "AND try_cast(right(lpad(trim(acq_time),4,'0'),2) AS INTEGER) BETWEEN 0 AND 59 "
                   f"THEN try_strptime({date_text}, '%Y-%m-%d %H%M') END")
    numeric = ("latitude", "longitude", "frp", "scan", "track", "bright_ti4", "bright_ti5")
    casts = ",".join(f"try_cast({field(n)} AS DOUBLE) AS history_{n}" for n in numeric)
    reasons = [
        "WHEN history_latitude IS NULL OR NOT isfinite(history_latitude) OR history_latitude NOT BETWEEN -90 AND 90 THEN 'invalid_latitude'",
        "WHEN history_longitude IS NULL OR NOT isfinite(history_longitude) OR history_longitude NOT BETWEEN -180 AND 180 THEN 'invalid_longitude'",
        "WHEN history_time IS NULL THEN 'invalid_acquisition_datetime'",
        f"WHEN year(history_time) <> {year} THEN 'acquisition_year_mismatch'",
        "WHEN upper(trim(satellite)) <> 'N20' OR nullif(trim(satellite),'') IS NULL THEN 'platform_product_mismatch'",
        f"WHEN {field('instrument')} IS NOT NULL AND upper({field('instrument')}) <> 'VIIRS' THEN 'instrument_product_mismatch'",
        f"WHEN {field('version')} IS NOT NULL AND upper({field('version')}) LIKE '%NRT%' THEN 'nrt_version_in_sp_history'",
        "WHEN lower(trim(confidence)) NOT IN ('l','n','h') OR nullif(trim(confidence),'') IS NULL THEN 'invalid_confidence'",
        "WHEN history_frp IS NULL OR NOT isfinite(history_frp) OR history_frp < 0 THEN 'invalid_frp'",
    ]
    for name in numeric[3:]:
        reasons.append(f"WHEN {field(name)} IS NOT NULL AND (history_{name} IS NULL OR NOT isfinite(history_{name}) OR history_{name} <= 0) THEN 'invalid_{name}'")
    connection.execute(f"""CREATE OR REPLACE TABLE history_work AS
        WITH raw AS (SELECT row_number() OVER ()::BIGINT AS history_record,
            *, struct_pack({raw_struct}) AS history_fields
            FROM read_csv({literal(str(source))}, header=true, auto_detect=false,
                columns={{{columns}}}, delim=',', quote='"', escape='"',
                parallel=false, strict_mode=true, ignore_errors=false, null_padding=false,
                force_not_null=[{','.join(literal(h) for h in header)}])),
        typed AS (SELECT *, {casts}, {parsed_time} AS history_time FROM raw)
        SELECT *, CASE {' '.join(reasons)} ELSE NULL END AS history_reason,
            coalesce(history_longitude BETWEEN {BOUNDS[0]} AND {BOUNDS[2]} AND
                     history_latitude BETWEEN {BOUNDS[1]} AND {BOUNDS[3]},false) AS history_in_region
        FROM typed""")
    total, valid, rejected, region, outside = connection.execute("""SELECT count(*),
        count(*) FILTER(WHERE history_reason IS NULL), count(*) FILTER(WHERE history_reason IS NOT NULL),
        count(*) FILTER(WHERE history_reason IS NULL AND history_in_region),
        count(*) FILTER(WHERE history_reason IS NULL AND NOT history_in_region) FROM history_work""").fetchone()
    output = stage / f"year={year}"
    output.mkdir()
    projection = f"""SELECT {literal(SCHEMA)} AS historical_schema,
        {literal(PRODUCT)} AS source_product_declared, {year}::INTEGER AS source_file_year,
        {literal(receipt['sha256'])} AS source_sha256, history_record AS source_record_number,
        {literal(entry['url'])} AS source_url,
        {literal(receipt['received_at_utc'])}::TIMESTAMPTZ AS received_at_utc,
        history_latitude AS latitude, history_longitude AS longitude,
        history_time::DATE AS acquisition_date, lpad(trim(acq_time),4,'0') AS acquisition_time_utc,
        history_time AT TIME ZONE 'UTC' AS acquired_at_utc,
        trim(satellite) AS satellite_source, 'NOAA-20' AS platform,
        {field('instrument')} AS instrument_source, {field('version')} AS source_version,
        lower(trim(confidence)) AS confidence_category, history_frp AS frp_mw,
        history_scan AS scan_km, history_track AS track_km,
        history_bright_ti4 AS bright_ti4_k, history_bright_ti5 AS bright_ti5_k,
        {field('daynight')} AS daynight_source, {field('type')} AS detection_type_source,
        history_fields AS source_fields FROM history_work
        WHERE history_reason IS NULL AND history_in_region"""
    connection.execute(f"COPY ({projection}) TO {literal(str(output / 'observations.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)")
    connection.execute(f"""COPY (SELECT history_record AS source_record_number,
        {literal(receipt['sha256'])} AS source_sha256, history_reason AS reason, history_fields AS source_fields
        FROM history_work WHERE history_reason IS NOT NULL)
        TO {literal(str(output / 'rejected.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)""")
    ranges = {}
    for scope, predicate in (("brazil_valid", "history_reason IS NULL"),
                             ("region", "history_reason IS NULL AND history_in_region")):
        lo, hi = connection.execute(f"SELECT min(history_time)::VARCHAR,max(history_time)::VARCHAR FROM history_work WHERE {predicate}").fetchone()
        ranges[scope] = {"min": lo, "max": hi}
    distributions = {}
    for name in ("instrument", "satellite", "version", "type", "confidence", "daynight"):
        if name in header:
            distributions[name] = [{"value": value, "brazil_rows": n, "region_rows": nr}
                                   for value, n, nr in connection.execute(f"SELECT {identifier(name)},count(*),count(*) FILTER(WHERE history_reason IS NULL AND history_in_region) FROM history_work GROUP BY 1 ORDER BY 1").fetchall()]
    rejection_counts = dict(connection.execute("SELECT history_reason,count(*) FROM history_work WHERE history_reason IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall())
    output_columns = [{"name": row[0], "type": row[1]} for row in connection.execute(f"DESCRIBE SELECT * FROM read_parquet({literal(str(output / 'observations.parquet'))})").fetchall()]
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Raw source changed during preparation")
    return {"year": year, "raw_path": entry["raw_path"], "source_sha256": receipt["sha256"],
            "source_size_bytes": receipt["size_bytes"], "source_columns": header,
            "source_sql_types": {name: "VARCHAR" for name in header}, "output_columns": output_columns,
            "metadata_fields_available": {name: name in header for name in ("instrument", "satellite", "version", "type", "daynight")},
            "counts": {"brazil_raw": total, "brazil_valid": valid, "rejected": rejected,
                       "region": region, "outside_region_valid": outside},
            "count_reconciliation": total == rejected + region + outside and valid == region + outside,
            "acquisition_ranges_utc": ranges, "source_value_counts": distributions,
            "rejection_counts": rejection_counts}


def prepare(input_dir: str | Path, output_dir: str | Path, *, memory_limit: str = "512MB",
            threads: int = 2) -> dict:
    if threads < 1:
        raise ValueError("threads must be positive")
    root = Path(input_dir).resolve(strict=True)
    acquisition = json.loads((root / "acquisition.json").read_text())
    if acquisition["product_declared"] != PRODUCT or acquisition["country"] != "Brazil":
        raise ValueError("Expected separate NOAA-20 SP Brazil acquisition")
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError("Prepared output must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".history-", dir=target.parent))
    try:
        with duckdb.connect(str(stage / "work.duckdb")) as connection:
            connection.execute(f"SET memory_limit={literal(memory_limit)}")
            connection.execute(f"SET threads={threads}")
            connection.execute(f"SET temp_directory={literal(str(stage / 'spill'))}")
            connection.execute("SET TimeZone='UTC'")
            connection.execute("SET preserve_insertion_order=true")
            per_year = [_prepare_year(connection, entry, root, stage) for entry in acquisition["files"]]
            effective = connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        (stage / "work.duckdb").unlink()
        shutil.rmtree(stage / "spill", ignore_errors=True)
        outputs = [{"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in sorted(stage.rglob("*.parquet"))]
        manifest = {"schema": SCHEMA, "created_at_utc": _now(), "product_declared": PRODUCT,
                    "processing_declared": "standard", "source_acquisition_manifest": str(root / "acquisition.json"),
                    "source_acquisition_manifest_sha256": sha256(root / "acquisition.json"),
                    "geography": {"country": "Brazil", "region_bounds_wsen": list(BOUNDS),
                                  "core_bounds_wsen": [-48, -17, -46, -15], "halo_degrees": 2,
                                  "edges": "inclusive", "crs": "EPSG:4326"},
                    "counts": {key: sum(row["counts"][key] for row in per_year) for key in per_year[0]["counts"]},
                    "years": per_year, "outputs": outputs,
                    "resources": {"engine": "duckdb", "version": duckdb.__version__, "memory_limit": memory_limit,
                                  "effective_memory_limit": effective, "threads": threads, "row_group_size": 16384,
                                  "strategy": "one annual on-disk table at a time; explicit VARCHAR input; disk spill"},
                    "units": {"coordinates": "degrees", "frp_mw": "MW", "scan_km": "km", "track_km": "km", "brightness": "K"},
                    "historical_availability": "unknown; not acquisition time or receipt time",
                    "metadata_policy": "Only available source fields are exposed; raw source_fields retain every input column; platform derives from checked N20 satellite.",
                    "limitations": LIMITATIONS}
        _json_new(stage / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError("Output appeared during preparation")
        stage.rename(target)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("acquire", help="Preserve public Brazil country CSVs and receipts")
    download.add_argument("--output-dir", required=True)
    download.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    download.add_argument("--max-bytes", type=int, default=1_000_000_000)
    project = commands.add_parser("prepare", help="Prepare distinct historical observations, not NRT normalization")
    project.add_argument("--input-dir", required=True)
    project.add_argument("--output-dir", required=True)
    project.add_argument("--memory-limit", default="512MB")
    project.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        if args.command == "acquire":
            result = acquire(args.output_dir, years=tuple(args.years), max_bytes=args.max_bytes)
            print(json.dumps({"years": result["years"], "raw_bytes": result["total_raw_bytes"]}))
        else:
            result = prepare(args.input_dir, args.output_dir, memory_limit=args.memory_limit, threads=args.threads)
            print(json.dumps(result["counts"]))
    except (ValueError, OSError, duckdb.Error) as error:
        print(f"History operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
