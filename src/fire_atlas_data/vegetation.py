"""Public ORNL MOD13Q1 center-patch covariates, explicitly retrospective (RF10/CA08)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import h3

from .normalization import literal, sha256

API = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"
PREFIX = "250m_16_days_"
NDVI = PREFIX + "NDVI"
EVI = PREFIX + "EVI"
RELIABILITY = PREFIX + "pixel_reliability"
QUALITY = PREFIX + "VI_Quality"
DAY = PREFIX + "composite_day_of_the_year"
REQUIRED = (NDVI, EVI, RELIABILITY, QUALITY, DAY)
VEGETATION_FEATURES = ["veg_ndvi", "veg_evi", "veg_valid_fraction", "veg_age_days", "veg_source_age_days"]
MAX_BYTES = 4 * 1024 * 1024
RESERVED_START = date(2026, 11, 13)
SOURCES = [
    "https://modis.ornl.gov/data/modis_webservice.html",
    "https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061",
    "https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD13Q1",
]


def _json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _modis_date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"A\d{7}", value):
        raise ValueError("Malformed MODIS composite date")
    parsed = datetime.strptime(value, "A%Y%j").date()
    if parsed.strftime("A%Y%j") != value:
        raise ValueError("MODIS day is outside its year")
    return parsed


def _production(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{13}", value):
        raise ValueError("Malformed MODIS production timestamp")
    parsed = datetime.strptime(value, "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    if parsed.strftime("%Y%j%H%M%S") != value:
        raise ValueError("Production day is outside its year")
    return parsed


def _metadata(payload: dict) -> dict:
    bands = payload.get("bands")
    if not isinstance(bands, list):
        raise ValueError("Missing ORNL bands metadata")
    result = {}
    for band in bands:
        name = band.get("band")
        if name in result:
            raise ValueError("Duplicate ORNL band metadata")
        result[name] = band
    expected = {NDVI: (-2000, 10000, -3000), EVI: (-2000, 10000, -3000),
                RELIABILITY: (0, 3, -1), QUALITY: (0, 65534, -1), DAY: (1, 366, -1)}
    for name, (low, high, fill) in expected.items():
        band = result.get(name, {})
        match = re.fullmatch(r"(-?\d+) to (-?\d+)", str(band.get("valid_range", "")))
        if (not match or tuple(map(int, match.groups())) != (low, high)
                or str(band.get("fill_value")) != str(fill)):
            raise ValueError(f"Unexpected MOD13Q1 range/fill metadata: {name}")
        if name in (NDVI, EVI) and (float(band.get("scale_factor", "nan")) != 0.0001
                                       or float(band.get("add_offset", "nan")) != 0):
            raise ValueError(f"Unexpected MOD13Q1 scale metadata: {name}")
    return result


def _dates(payload: dict, earliest: date, latest: date) -> list[str]:
    rows = payload.get("dates")
    if not isinstance(rows, list):
        raise ValueError("Missing ORNL date inventory")
    result = []
    seen = set()
    for row in rows:
        value = row.get("modis_date")
        parsed = _modis_date(value)
        if row.get("calendar_date") != parsed.isoformat() or value in seen:
            raise ValueError("Inconsistent or duplicate ORNL composite date")
        seen.add(value)
        if earliest <= parsed <= latest:
            result.append(value)
    result.sort()
    if not result:
        raise ValueError("No MOD13Q1 dates in requested warmup/issue interval")
    return result


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _patch(payload: dict, requested: list[str], latitude: float, longitude: float,
           metadata: dict) -> tuple[list[dict], dict]:
    """Validate aligned arrays, then retain only jointly reliable sampled pixels."""
    nrows, ncols = payload.get("nrows"), payload.get("ncols")
    if (not _integer(nrows) or not _integer(ncols) or not 1 <= nrows <= 11
            or not 1 <= ncols <= 11 or payload.get("band") != "all"):
        raise ValueError("Unexpected ORNL patch dimensions/band selection")
    size = nrows * ncols
    try:
        resolution = float(payload["cellsize"])
        x, y = float(payload["xllcorner"]), float(payload["yllcorner"])
        lat, lon = float(payload["latitude"]), float(payload["longitude"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Missing ORNL patch georeferencing") from None
    if (not all(math.isfinite(v) for v in (resolution, x, y, lat, lon))
            or not math.isclose(resolution, 231.656358264, abs_tol=0.001)
            or abs(lat - latitude) > 1e-6 or abs(lon - longitude) > 1e-6):
        raise ValueError("Different ORNL resolution or requested center metadata")
    grouped = {value: {} for value in requested}
    entries = payload.get("subset")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Missing ORNL subset arrays")
    for entry in entries:
        value, band = entry.get("modis_date"), entry.get("band")
        if value not in grouped or band not in metadata or band in grouped[value]:
            raise ValueError("Unexpected/duplicate ORNL date or band")
        if entry.get("calendar_date", _modis_date(value).isoformat()) != _modis_date(value).isoformat():
            raise ValueError("Different ORNL calendar/composite date")
        data = entry.get("data")
        if not isinstance(data, list) or len(data) != size or not all(_integer(v) for v in data):
            raise ValueError("Malformed ORNL pixel array")
        _production(entry.get("proc_date"))
        grouped[value][band] = entry
    composites = []
    for value, bands in grouped.items():
        if not set(REQUIRED).issubset(bands):
            raise ValueError(f"Missing MOD13Q1 date or required band: {value}")
        identities = {(b.get("proc_date"), b.get("tile", b.get("title")), b.get("version"))
                      for b in bands.values()}
        if len(identities) != 1:
            raise ValueError("Different production/tile/version metadata across bands")
        proc, tile, version = identities.pop()
        if not isinstance(tile, str) or not re.fullmatch(r"h\d{2}v\d{2}", tile):
            raise ValueError("Missing MODIS tile identity")
        if version is not None and not re.fullmatch(r"\d{3}", str(version)):
            raise ValueError("Malformed MODIS collection version")
        start = _modis_date(value)
        end = start + timedelta(days=15)
        valid_ndvi, valid_evi, observation_ordinals = [], [], []
        for ndvi, evi, reliability, quality, day in zip(*(bands[b]["data"] for b in REQUIRED)):
            if reliability not in (-1, 0, 1, 2, 3):
                raise ValueError("Unknown MOD13Q1 pixel reliability rank")
            if quality != -1 and not 0 <= quality <= 65534:
                raise ValueError("Out-of-range MOD13Q1 VI Quality")
            # Out-of-range/fill measurements are missing, never clipped or replaced by zero.
            if (reliability != 0 or quality == -1 or not -2000 <= ndvi <= 10000
                    or not -2000 <= evi <= 10000 or not 1 <= day <= 366):
                continue
            observed = date(start.year, 1, 1) + timedelta(days=day - 1)
            if not start <= observed <= end:
                continue
            valid_ndvi.append(ndvi * float(metadata[NDVI]["scale_factor"]))
            valid_evi.append(evi * float(metadata[EVI]["scale_factor"]))
            observation_ordinals.append(observed.toordinal())
        count = len(valid_ndvi)
        composites.append({"start": start, "end": end, "production": _production(proc),
                           "proc_date": proc, "tile": tile, "version": version,
                           "ndvi": sum(valid_ndvi) / count if count else None,
                           "evi": sum(valid_evi) / count if count else None,
                           "valid_fraction": count / size, "valid_pixels": count,
                           "sampled_pixels": size,
                           "mean_observation_ordinal": sum(observation_ordinals) / count if count else None})
    bounds = {"latitude": lat, "longitude": lon, "nrows": nrows, "ncols": ncols,
              "cellsize_m": resolution, "crs": "MODIS sinusoidal; sphere radius 6371007.181m",
              "bounds_m": [x, y, x + ncols * resolution, y + nrows * resolution],
              "collection_versions_returned": sorted({c["version"] for c in composites if c["version"] is not None}),
              "collection_version_missing": any(c["version"] is None for c in composites)}
    return sorted(composites, key=lambda c: c["start"]), bounds


def _select(composites: list[dict], issue: date) -> tuple[list, dict | None]:
    cutoff = datetime.combine(issue, time.max, timezone.utc)
    eligible = [c for c in composites if c["end"] <= issue
                and (c["production"] is None or c["production"] <= cutoff)]
    if not eligible:
        return [None] * len(VEGETATION_FEATURES), None
    selected = max(eligible, key=lambda c: c["start"])
    source_age = (issue.toordinal() - selected["mean_observation_ordinal"]
                  if selected["mean_observation_ordinal"] is not None else None)
    return [selected["ndvi"], selected["evi"], selected["valid_fraction"],
            float((issue - selected["end"]).days), source_age], selected


def _fetch(url: str, stage: Path, name: str) -> tuple[dict, dict]:
    """Preserve bounded raw bytes and receipt even if JSON validation subsequently fails."""
    raw = stage / "raw" / (name + ".json")
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "fire-atlas-data/0.1"})
    with urlopen(request, timeout=180) as response:
        body = response.read(MAX_BYTES + 1)
        status, final_url = response.status, response.url
        headers = dict(response.headers.items())
    raw.write_bytes(body)
    receipt = {"url": url, "response_url": final_url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
               "http_status": status, "headers": headers, "path": str(raw.relative_to(stage)),
               "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body),
               "available_at": None}
    _json(stage / "receipts" / (name + ".json"), receipt)
    if status != 200 or len(body) > MAX_BYTES:
        raise ValueError("ORNL returned unsuccessful or oversized response")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("ORNL JSON response must be an object")
    return payload, receipt


def _fetch_cell(job: tuple) -> tuple:
    cell, latitude, longitude, dates, stage, metadata = job
    chunks, receipts, composites, patch_bounds = [], [], [], None
    for offset in range(0, len(dates), 10):
        chunk = dates[offset:offset + 10]
        params = {"latitude": format(latitude, ".10f"), "longitude": format(longitude, ".10f"),
                  "startDate": chunk[0], "endDate": chunk[-1], "kmAboveBelow": 1, "kmLeftRight": 1}
        url = f"{API}/{PRODUCT}/subset?{urlencode(params)}"
        payload, receipt = _fetch(url, stage, f"subset-{cell}-{offset // 10:02d}")
        parsed, bounds = _patch(payload, chunk, latitude, longitude, metadata)
        if patch_bounds is not None and bounds != patch_bounds:
            raise ValueError("Different patch metadata across request chunks")
        patch_bounds = bounds
        composites.extend(parsed)
        receipts.append(receipt)
        chunks.append(chunk)
    return cell, composites, {"cell_id": cell, **patch_bounds, "requested_dates": chunks}, receipts


def acquire(features: str | Path, output_dir: str | Path, *, concurrency: int = 3) -> dict:
    """Acquire all distinct center/date keys without reading labels or touching future holdout."""
    if not _integer(concurrency) or not 1 <= concurrency <= 3:
        raise ValueError("Concurrency must be 1..3")
    source = Path(features).resolve(strict=True)
    target = Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output must not exist: {target}")
    source_hash = sha256(source)
    connection = duckdb.connect()
    try:
        keys = connection.execute(f"SELECT DISTINCT cell_id,issue_date FROM read_parquet({literal(str(source))}) ORDER BY 1,2").fetchall()
    finally:
        connection.close()
    if not keys or any(not isinstance(c, str) or not isinstance(d, date) or isinstance(d, datetime) for c, d in keys):
        raise ValueError("Features require nonempty VARCHAR cell_id and DATE issue_date keys")
    issues = sorted({d for _, d in keys})
    if max(issues) >= RESERVED_START:
        raise ValueError("Reserved future issue dates must not be fetched or scored")
    cells = sorted({c for c, _ in keys})
    if len(cells) > 500 or (max(issues) - min(issues)).days > 120:
        raise ValueError("Acquisition is bounded to 500 centers and a 120-day issue interval")
    centers = [(c, *h3.cell_to_latlng(c)) for c in cells]
    earliest = min(issues).replace(day=1) - timedelta(days=1)
    earliest = earliest.replace(day=1)
    latest = max(issues) - timedelta(days=15)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    (stage / "raw").mkdir()
    (stage / "receipts").mkdir()
    reserved = False
    linked = []
    directories = []
    try:
        bands, receipt = _fetch(f"{API}/{PRODUCT}/bands", stage, "bands")
        metadata = _metadata(bands)
        inventory, date_receipt = _fetch(f"{API}/{PRODUCT}/dates?{urlencode({'latitude': centers[0][1], 'longitude': centers[0][2]})}", stage, "dates")
        dates = _dates(inventory, earliest, latest)
        receipts = [receipt, date_receipt]
        jobs = [(c, lat, lon, dates, stage, metadata) for c, lat, lon in centers]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(_fetch_cell, jobs))
        by_cell, patches, composite_metadata = {}, [], []
        for cell, composites, bounds, cell_receipts in results:
            by_cell[cell] = composites
            patches.append(bounds)
            receipts.extend(cell_receipts)
            composite_metadata.extend({"cell_id": cell, **{k: (v.isoformat() if isinstance(v, (date, datetime)) else v)
                                                           for k, v in c.items()}} for c in composites)
        rows, missing_selection, unknown_production, selected_counts = [], 0, 0, {}
        for cell, issue in keys:
            values, selected = _select(by_cell[cell], issue)
            rows.append((cell, issue, *values))
            if selected is None:
                missing_selection += 1
            else:
                unknown_production += selected["production"] is None
                identity = selected["start"].isoformat()
                selected_counts[identity] = selected_counts.get(identity, 0) + 1
        if all(row[2] is None for row in rows):
            raise ValueError("No valid vegetation measurements survive date/production/QA guards; inspect retained raw evidence")
        _json(stage / "composites.json", {"composites": composite_metadata})
        _json(stage / "patches.json", {"patches": patches})
        connection = duckdb.connect()
        try:
            connection.execute("CREATE TABLE covariates(cell_id VARCHAR, issue_date DATE," + ",".join(c + " DOUBLE" for c in VEGETATION_FEATURES) + ", PRIMARY KEY(cell_id,issue_date))")
            connection.executemany("INSERT INTO covariates VALUES (?,?,?,?,?,?,?)", rows)
            connection.execute(f"COPY (SELECT * FROM covariates ORDER BY cell_id,issue_date) TO {literal(str(stage / 'covariates.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            connection.close()
        if sha256(source) != source_hash:
            raise ValueError("Input features changed during acquisition")
        output_files = ["covariates.parquet", "composites.json", "patches.json"]
        output_files.extend(str(p.relative_to(stage)) for folder in ("raw", "receipts") for p in sorted((stage / folder).iterdir()))
        manifest = {"schema_version": "1.0.0", "method": "ornl_mod13q1_center_patch_production_guarded_retrospective_v1",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": {"product": PRODUCT, "kmAboveBelow": 1, "kmLeftRight": 1,
                               "concurrency": concurrency, "max_response_bytes": MAX_BYTES,
                               "max_dates_per_request": 10, "warmup_start": earliest.isoformat(),
                               "issue_start": min(issues).isoformat(), "issue_end": max(issues).isoformat(),
                               "requested_composites": dates, "issue_cutoff": "end of UTC issue_date"},
                    "columns": VEGETATION_FEATURES,
                    "units": {"veg_ndvi": "dimensionless ratio", "veg_evi": "dimensionless ratio",
                              "veg_valid_fraction": "fraction of sampled pixels jointly valid, not whole H3 or NOAA20 cloud coverage",
                              "veg_age_days": "days since composite end (start+15 days)",
                              "veg_source_age_days": "days since mean accepted pixel observation day"},
                    "inputs_hashes": [{"path": str(source), "sha256": source_hash},
                                      {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}],
                    "outputs": [{"path": p, "sha256": sha256(stage / p), "size_bytes": (stage / p).stat().st_size} for p in output_files],
                    "source": {"provider": "ORNL DAAC TESViS", "api": API, "product": PRODUCT,
                               "processing": "scientific composite, not NOAA20 NRT", "available_at": None,
                               "collection_version": "per-record if returned; unknown when API omits version; never inferred from product shortname",
                               "official_references": SOURCES, "band_metadata": bands, "requests": receipts},
                    "quality_policy": "pixel_reliability==0; non-fill VI_Quality; both VI values within [-2000,10000]; pixel day within composite; common mask; raw scale 0.0001",
                    "selection_policy": "latest start with start+15 days<=D and proc_date<=endD when present; missing proc_date explicitly counted; never assume production time is availability",
                    "counts": {"rows": len(rows), "cells": len(cells), "issue_dates": len(issues),
                               "composites": len(composite_metadata), "requests": len(receipts),
                               "missing_selection": missing_selection, "selected_unknown_production": unknown_production,
                               "selected_composite_counts": selected_counts,
                               "null_counts": {c: sum(r[index + 2] is None for r in rows) for index, c in enumerate(VEGETATION_FEATURES)}},
                    "limitations": ["Retrospective experiment only: actual historical available_at is unknown, even when production predates issuance.",
                                    "A roughly 2km center patch is sampled, not the H3 cell; valid fraction is not NOAA20 cloud/observation coverage.",
                                    "NDVI/EVI describe greenness, not direct fuel moisture or measured fuel load.",
                                    "The dates inventory is queried at one center; every requested date and required band is validated at each center.",
                                    "Collection version missing from current API responses remains unknown; raw response and production metadata are preserved.",
                                    "Latest eligible composite is selected independently of QA; no backfill with older greener/cloud-free composites.",
                                    "Missing values remain null and do not imply no fire; no labels are read for acquisition."]}
        _json(stage / "manifest.json", manifest)
        # Reserve destination, then atomically link each complete artifact; manifest is commit marker.
        target.mkdir()
        reserved = True
        for name in ("raw", "receipts"):
            directory = target / name
            directory.mkdir()
            directories.append(directory)
        for name in output_files + ["manifest.json"]:
            os.link(stage / name, target / name)
            linked.append(target / name)
        shutil.rmtree(stage, ignore_errors=True)
        return manifest
    except Exception as error:
        # Do not destroy acquired evidence or publish a partial covariate manifest.
        if reserved:
            for path in reversed(linked):
                path.unlink(missing_ok=True)
            for directory in reversed(directories):
                try:
                    directory.rmdir()
                except OSError:
                    pass  # Keep any unrelated file introduced by another writer.
            try:
                target.rmdir()
            except OSError:
                pass
        _json(stage / "failure.json", {"error": str(error), "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "features": str(source), "features_sha256": source_hash})
        raise RuntimeError(f"Vegetation acquisition failed; evidence retained at {stage}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    try:
        manifest = acquire(args.features, args.output_dir, concurrency=args.concurrency)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({"counts": manifest["counts"], "columns": manifest["columns"],
                      "outputs": manifest["outputs"][:3],
                      "manifest_sha256": sha256(Path(args.output_dir) / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
