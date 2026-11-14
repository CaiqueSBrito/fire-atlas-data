"""Bounded-memory, SQL-based FIRMS CSV normalization (RF11/CA09)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb

SCHEMA_VERSION = "1.3.0"
METHOD_VERSION = "1.6.0"
REQUIRED = {"latitude", "longitude", "scan", "track", "acq_date", "acq_time", "satellite", "confidence", "version", "frp", "daynight"}
PRODUCTS = {
    "VIIRS_NOAA20_NRT": ("VIIRS", "N20"),
    "VIIRS_NOAA21_NRT": ("VIIRS", "N21"),
    "VIIRS_SNPP_NRT": ("VIIRS", "N"),
    "VIIRS_SNPP_SP": ("VIIRS", "SNPP"),
    "VIIRS_NOAA20_SP": ("VIIRS", "N20"),
    "VIIRS_NOAA21_SP": ("VIIRS", "N21"),
    "MODIS_NRT": ("MODIS", None),
    "MODIS_SP": ("MODIS", None),
}


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_expression(name: str, header: list[str]) -> str:
    return f"nullif(trim({identifier(name)}),'')" if name in header else "NULL::VARCHAR"


def normalize(input_path: str | Path, output_dir: str | Path, *, source_product: str,
              source_url: str | None = None, memory_limit: str = "1GB", threads: int = 2,
              temp_dir: str | Path | None = None) -> dict:
    """Publish Parquet and manifest atomically; never modify the input CSV.

    Python handles metadata/files only. All observation transformations run in
    DuckDB vectorized SQL with an on-disk intermediate and spill directory.
    """
    source = Path(input_path).resolve(strict=True)
    target = Path(output_dir).absolute()
    if source_product not in PRODUCTS:
        raise ValueError(f"Unsupported source product; choose {', '.join(PRODUCTS)}")
    if source_url is not None and not source_url.strip():
        raise ValueError("source_url must be nonempty when supplied")
    if threads < 1:
        raise ValueError("threads must be positive")
    if target.exists():
        raise FileExistsError(f"Output must not already exist: {target}")
    if source == target or target in source.parents:
        raise ValueError("Output cannot contain the raw input")
    # Only the header is parsed in Python, never observation rows.
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, strict=True), [])
    if not header or len(set(header)) != len(header) or len({h.lower() for h in header}) != len(header):
        raise ValueError("CSV requires unique, nonempty column names")
    if any(not h for h in header) or not REQUIRED.issubset(header):
        raise ValueError(f"Missing required CSV columns: {sorted(REQUIRED - set(header))}")
    reserved = {"source_record_number", "source_fields", "parsed_time", "reason"}
    if any(h.lower() in reserved or h.lower().startswith(("n_", "z_")) for h in header):
        raise ValueError("CSV column name collides with internal normalization fields")
    if not ({"brightness", "bright_ti4"} & set(header)):
        raise ValueError("Missing brightness or bright_ti4 column")
    sensor, expected_platform = PRODUCTS[source_product]
    if sensor == "MODIS" and not {"brightness", "bright_t31"}.issubset(header):
        raise ValueError("MODIS requires brightness and bright_t31 columns")
    before = source.stat()
    source_hash = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    spill_parent = Path(temp_dir).resolve() if temp_dir else stage
    spill_parent.mkdir(parents=True, exist_ok=True)
    spill = Path(tempfile.mkdtemp(prefix="normalization-spill-", dir=spill_parent))
    connection = None
    try:
        connection = duckdb.connect(str(stage / "work.duckdb"))
        connection.execute(f"SET memory_limit={literal(memory_limit)}")
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET temp_directory={literal(str(spill))}")
        connection.execute("SET TimeZone='UTC'")
        connection.execute("SET preserve_insertion_order=true")
        connection.execute("SET partitioned_write_max_open_files=2")
        connection.execute("SET partitioned_write_flush_threshold=16384")
        columns = ",".join(f"{literal(h)}:'VARCHAR'" for h in header)
        raw_struct = ",".join(f"{identifier(h)} := {identifier(h)}" for h in header)
        numeric = ["latitude", "longitude", "scan", "track", "frp", "brightness", "bright_t31", "bright_ti4", "bright_ti5"]
        casts = ",".join(f"try_cast({_raw_expression(n, header)} AS DOUBLE) AS n_{n}" for n in numeric)
        confidence_category = ("CASE lower(trim(confidence)) WHEN 'low' THEN 'l' "
                               "WHEN 'nominal' THEN 'n' WHEN 'high' THEN 'h' "
                               "ELSE lower(trim(confidence)) END")
        inferred_instrument = "" if "instrument" in header else f", {literal(sensor)} AS instrument"
        # FIRMS SNPP files (archive and, as observed in 2026, NRT) label both
        # instrument and satellite as SNPP. The alias is accepted only for the
        # SNPP products, never for NOAA-20/21 or MODIS.
        archive_snpp = source_product == "VIIRS_SNPP_SP"
        snpp_alias = source_product in {"VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"}
        instrument_values = "('VIIRS','SNPP')" if snpp_alias else f"({literal(sensor)})"
        reasons = [
            "WHEN n_latitude IS NULL OR NOT isfinite(n_latitude) OR n_latitude NOT BETWEEN -90 AND 90 THEN 'invalid_latitude'",
            "WHEN n_longitude IS NULL OR NOT isfinite(n_longitude) OR n_longitude NOT BETWEEN -180 AND 180 THEN 'invalid_longitude'",
            "WHEN parsed_time IS NULL THEN 'invalid_acquisition_datetime'",
            f"WHEN upper(trim(instrument)) NOT IN {instrument_values} THEN 'sensor_product_mismatch'",
            "WHEN upper(trim(daynight)) NOT IN ('D','N') THEN 'invalid_daynight'",
            "WHEN nullif(trim(version),'') IS NULL THEN 'missing_source_version'",
        ]
        if source_product == "VIIRS_SNPP_NRT":
            # Only the observed pairing instrument=SNPP with satellite=SNPP is an alias.
            reasons.append("WHEN upper(trim(instrument)) = 'SNPP' AND upper(trim(satellite)) <> 'SNPP' THEN 'sensor_product_mismatch'")
        platforms = "('SNPP')" if archive_snpp else "('N','SNPP')" if snpp_alias else "('N','N20','N21')" if sensor == "VIIRS" else "('T','A','TERRA','AQUA')"
        reasons.append(f"WHEN upper(trim(satellite)) NOT IN {platforms} THEN 'invalid_platform'")
        if expected_platform and not snpp_alias:
            reasons.append(f"WHEN upper(trim(satellite)) <> {literal(expected_platform)} THEN 'platform_product_mismatch'")
        if sensor == "VIIRS":
            reasons.append(f"WHEN ({confidence_category}) NOT IN ('l','n','h') THEN 'invalid_confidence'")
        else:
            reasons.append("WHEN try_cast(confidence AS DOUBLE) IS NULL OR NOT isfinite(try_cast(confidence AS DOUBLE)) OR try_cast(confidence AS DOUBLE) NOT BETWEEN 0 AND 100 OR try_cast(confidence AS DOUBLE) <> trunc(try_cast(confidence AS DOUBLE)) THEN 'invalid_confidence'")
        # Physically impossible FRP (<0) and brightness (<=0 K) are sentinel/garbage
        # values: keep the detection, drop only that measurement (raw kept in source_fields).
        sentinel = {"frp": "< 0", "brightness": "<= 0", "bright_t31": "<= 0", "bright_ti4": "<= 0", "bright_ti5": "<= 0"}
        for name in numeric[2:]:
            condition = "" if name in sentinel else f" OR n_{name} <= 0"
            reasons.append(f"WHEN {_raw_expression(name, header)} IS NOT NULL AND (n_{name} IS NULL OR NOT isfinite(n_{name}){condition}) THEN 'invalid_{name}'")
        nullified = {name: f"n_{name} {rule}" for name, rule in sentinel.items()}
        cleaned = ",".join(f"CASE WHEN {test} THEN NULL ELSE n_{name} END AS n_{name}" for name, test in nullified.items())
        if sensor == "VIIRS" and "bright_ti4" in header and "brightness" in header:
            reasons.append("WHEN n_bright_ti4 IS NOT NULL AND n_brightness IS NOT NULL AND n_bright_ti4 <> n_brightness THEN 'conflicting_i4_aliases'")
        # FIRMS `type` exists only in archive/SP files. Checked last so that no
        # previously reported rejection reason changes for any row.
        hotspot_raw = _raw_expression("type", header)
        if "type" in header:
            reasons.append(f"WHEN {hotspot_raw} IS NOT NULL AND {hotspot_raw} NOT IN ('0','1','2','3') THEN 'invalid_type'")
        date_text = "trim(acq_date) || ' ' || lpad(trim(acq_time),4,'0')"
        parsed = f"CASE WHEN regexp_full_match(trim(acq_date),'[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}') AND left(trim(acq_date),4) <> '0000' AND regexp_full_match(trim(acq_time),'[0-9]{{1,4}}') AND try_cast(left(lpad(trim(acq_time),4,'0'),2) AS INTEGER) BETWEEN 0 AND 23 AND try_cast(right(trim(acq_time),2) AS INTEGER) BETWEEN 0 AND 59 THEN try_strptime({date_text}, '%Y-%m-%d %H%M') END"
        connection.execute(f"""CREATE TABLE observations AS
            WITH raw AS (
              SELECT row_number() OVER ()::BIGINT AS source_record_number,
                     *, struct_pack({raw_struct}) AS source_fields{inferred_instrument}
              FROM read_csv({literal(str(source))}, header=true, auto_detect=false,
                columns={{{columns}}}, delim=',', quote='"', escape='"',
                parallel=true, strict_mode=true, ignore_errors=false,
                null_padding=false, force_not_null=[{','.join(literal(h) for h in header)}], skip=0)
            ), typed AS (SELECT *, {casts}, {parsed} AS parsed_time FROM raw)
            SELECT * REPLACE ({cleaned}), {', '.join(f'coalesce({t}, false) AS z_{n}' for n, t in nullified.items())},
                   CASE {' '.join(reasons)} ELSE NULL END AS reason FROM typed""")
        connection.execute("SET preserve_insertion_order=false")
        effective_memory = connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        total, accepted, rejected = connection.execute("SELECT count(*), count(*) FILTER (WHERE reason IS NULL), count(*) FILTER (WHERE reason IS NOT NULL) FROM observations").fetchone()
        platform_sql = "CASE upper(trim(satellite)) WHEN 'T' THEN 'Terra' WHEN 'TERRA' THEN 'Terra' WHEN 'A' THEN 'Aqua' WHEN 'AQUA' THEN 'Aqua' WHEN 'N' THEN 'Suomi-NPP' WHEN 'SNPP' THEN 'Suomi-NPP' WHEN 'N20' THEN 'NOAA-20' WHEN 'N21' THEN 'NOAA-21' END"
        i4 = "coalesce(n_bright_ti4,n_brightness)" if sensor == "VIIRS" else "NULL::DOUBLE"
        i5 = "n_bright_ti5" if sensor == "VIIRS" else "NULL::DOUBLE"
        category = confidence_category if sensor == "VIIRS" else "NULL::VARCHAR"
        percent = "NULL::SMALLINT" if sensor == "VIIRS" else "cast(confidence AS SMALLINT)"
        connection.execute(f"""CREATE VIEW accepted AS SELECT
            '{SCHEMA_VERSION}' AS schema_version,
            {literal(source_hash + ':')} || source_record_number::VARCHAR AS record_id,
            {literal(source_hash)} AS source_sha256, source_record_number,
            {literal(source_product)} AS source_product, {literal(source_url) if source_url is not None else 'NULL::VARCHAR'} AS source_url,
            n_latitude AS latitude, n_longitude AS longitude,
            parsed_time::DATE AS acquisition_date, parsed_time AT TIME ZONE 'UTC' AS acquired_at,
            NULL::TIMESTAMPTZ AS available_at, '{sensor}' AS sensor,
            {platform_sql} AS platform, trim(version) AS source_version,
            {category} AS confidence_category, {percent} AS confidence_percent,
            {i4} AS brightness_i4_k, {i5} AS brightness_i5_k,
            n_brightness AS brightness_k, n_bright_t31 AS bright_t31_k,
            n_frp AS frp_mw, n_scan AS scan_km, n_track AS track_km,
            upper(trim(daynight)) AS daynight,
            'active_fire_detection' AS observation_type,
            try_cast({hotspot_raw} AS SMALLINT) AS hotspot_type, source_fields,
            year(parsed_time)::INTEGER AS acquisition_year,
            month(parsed_time)::INTEGER AS acquisition_month
            FROM observations WHERE reason IS NULL""")
        accepted_path = stage / "accepted"
        # File size is a target (not a hard bound); row groups limit writer batches.
        connection.execute(f"COPY accepted TO {literal(str(accepted_path))} (FORMAT PARQUET, PARTITION_BY(acquisition_year,acquisition_month), COMPRESSION ZSTD, ROW_GROUP_SIZE 16384, FILE_SIZE_BYTES '128MB')")
        if not accepted:
            accepted_path.mkdir(exist_ok=True)
            connection.execute(f"COPY accepted TO {literal(str(accepted_path / 'empty.parquet'))} (FORMAT PARQUET)")
        rejected_path = stage / "rejected"
        rejection_query = f"SELECT source_record_number, {literal(source_hash)} AS source_sha256, reason, source_fields FROM observations WHERE reason IS NOT NULL"
        if rejected:
            connection.execute(f"COPY ({rejection_query}) TO {literal(str(rejected_path))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384, FILE_SIZE_BYTES '128MB', PER_THREAD_OUTPUT true)")
        else:
            rejected_path.mkdir()
            connection.execute(f"COPY ({rejection_query}) TO {literal(str(rejected_path / 'empty.parquet'))} (FORMAT PARQUET)")
        dates = connection.execute("SELECT min(acquired_at)::VARCHAR,max(acquired_at)::VARCHAR FROM accepted").fetchone()
        groups = connection.execute("SELECT sensor,platform,count(*) FROM accepted GROUP BY sensor,platform ORDER BY sensor,platform").fetchall()
        rejection_counts = connection.execute("SELECT reason,count(*) FROM observations WHERE reason IS NOT NULL GROUP BY reason ORDER BY reason").fetchall()
        nullified_counts = connection.execute("SELECT " + ",".join(f"count(*) FILTER (WHERE z_{n})" for n in nullified) + " FROM observations WHERE reason IS NULL").fetchone()
        hotspot_types = connection.execute("SELECT hotspot_type,count(*) FROM accepted GROUP BY hotspot_type ORDER BY hotspot_type NULLS LAST").fetchall()
        connection.close()
        connection = None
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError("Source changed during normalization; refusing publication")
        (stage / "work.duckdb").unlink()
        shutil.rmtree(spill)
        outputs = [{"path": str(p.relative_to(stage)), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(stage.rglob("*.parquet"))]
        manifest = {
            "schema_version": SCHEMA_VERSION, "method_version": METHOD_VERSION,
            "format": "parquet", "engine": {"name": "duckdb", "version": duckdb.__version__},
            "resources": {"memory_limit": memory_limit, "effective_memory_limit": effective_memory, "threads": threads, "row_group_size": 16384, "file_size_target": "128MB", "partitioned_write_max_open_files": 2, "partitioned_write_flush_threshold": 16384, "spill_location": "operator_selected_volume" if temp_dir else "staging_volume"},
            "source": {"path": str(source), "size_bytes": before.st_size, "sha256": source_hash, "product": source_product, "url": source_url, "url_verification": "user_declared_not_verified" if source_url else "unknown", "columns": header},
            "sensor_origin": ("source_instrument_or_snpp_archive_alias_checked_against_product_and_platform" if archive_snpp else "source_instrument_checked_against_product") if "instrument" in header else "operator_declared_product_checked_against_platform",
            "counts": {"total": total, "accepted": accepted, "rejected": rejected},
            "acquisition_range_utc": {"min": dates[0], "max": dates[1]},
            "sensor_platform_counts": [{"sensor": s, "platform": p, "count": n} for s,p,n in groups],
            "hotspot_type_counts": [{"hotspot_type": t, "count": n} for t,n in hotspot_types],
            "rejection_counts": dict(rejection_counts),
            "nullified_value_counts": {f"{n}_{'negative' if n == 'frp' else 'non_positive'}": c for n, c in zip(nullified, nullified_counts)},
            "outputs": outputs,
            "units": {"coordinates": "degrees EPSG:4326", "brightness": "K", "frp": "MW", "scan": "km", "track": "km"},
            "limitations": ["Engineering normalization, not sensor harmonization or model training.", "Detections may include thermal anomalies; zero observations does not establish no fire.", "available_at is unknown (null); acquisition time is not availability time.", "No deduplication or confidence filtering; confidence is sensor-specific, not comparable probability.", "VIIRS bright_t31 is preserved without asserting I-5 equivalence.", "FIRMS type is an inferred hotspot class present only in archive/SP streams; a NULL hotspot_type means unknown, not vegetation fire.", "Out-of-core design is not evidence of a billion-row benchmark; disk capacity must cover intermediate and output plus spill.", "SHA256 requires one extra source pass; output hashing requires additional bounded-memory I/O."]
        }
        with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        if target.exists():
            raise FileExistsError(f"Output appeared during processing: {target}")
        # Same-filesystem rename makes a completed dataset visible as a unit.
        os.rename(stage, target)
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(spill, ignore_errors=True)
        raise
