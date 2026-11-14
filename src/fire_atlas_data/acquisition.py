"""Preserve FIRMS CSV bytes and acquisition receipts without changing normalization."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


PRODUCTS = (
    "MODIS_NRT", "MODIS_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT", "VIIRS_NOAA21_SP", "VIIRS_SNPP_NRT", "VIIRS_SNPP_SP",
)
AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CHUNK_BYTES = 1024 * 1024


class AcquisitionError(ValueError):
    """A safe-to-display acquisition error, never including remote response bodies."""


def _https_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or any(c.isspace() for c in url)):
            raise ValueError
        parsed.port  # Validate malformed ports before any request.
    except ValueError:
        raise AcquisitionError("An explicit HTTPS URL without embedded credentials is required") from None
    return url


def _sanitized_url(url: str) -> str:
    parsed = urlsplit(url)
    # Archive paths as well as queries may carry signed download credentials.
    return urlunsplit(("https", parsed.netloc, "/[redacted]", "", ""))


def _processing(product: str, declared: str | None = None) -> str:
    if product not in PRODUCTS:
        raise AcquisitionError("Unsupported FIRMS product")
    expected = "scientific" if product.endswith("_SP") else "nrt"
    if declared is not None and declared != expected:
        raise AcquisitionError("Declared processing must agree with the FIRMS product suffix (_SP or _NRT)")
    return expected


def _area_value(area: str) -> str:
    if area == "world":
        return area
    try:
        coordinates = [float(value) for value in area.split(",")]
    except ValueError:
        raise AcquisitionError("Area must be 'world' or west,south,east,north") from None
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        raise AcquisitionError("Area must have four finite coordinates")
    west, south, east, north = coordinates
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise AcquisitionError("Area bounds must be ordered and within longitude/latitude limits")
    return ",".join(format(value, ".15g") for value in coordinates)


class _HTTPSRedirects(HTTPRedirectHandler):
    def __init__(self, allow_redirects: bool):
        super().__init__()
        self.allow_redirects = allow_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.allow_redirects:
            raise AcquisitionError("Area API redirects are refused to protect the MAP_KEY")
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, *, allow_redirects: bool):
    request = Request(url, headers={"Accept": "text/csv, text/plain, application/octet-stream",
                                    "User-Agent": "fire-atlas-data/0.1"})
    return build_opener(_HTTPSRedirects(allow_redirects)).open(request, timeout=120)


def _validate_csv(path: Path) -> tuple[list[str], int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if (not header or any(not value for value in header)
                    or len(set(header)) != len(header)
                    or not {"latitude", "longitude", "acq_date", "acq_time"}.issubset(header)):
                raise AcquisitionError("Response is not a FIRMS CSV with observation columns")
            count = 0
            for row in reader:
                if len(row) != len(header) or any("\x00" in value for value in row):
                    raise AcquisitionError("Malformed FIRMS CSV row; no files published")
                count += 1
    except (UnicodeError, csv.Error):
        raise AcquisitionError("Response is not valid UTF-8 CSV; no files published") from None
    return header, count


def _copy_bytes(source, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("xb") as stream:
        while chunk := source.read(CHUNK_BYTES):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest(), size


def _acquire(output_dir: str | Path, *, product: str, processing: str, source_url: str | None,
             method: str, url: str | None = None, source: Path | None = None,
             request_details: dict | None = None) -> dict:
    target = Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Acquisition output must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".firms-acquisition-", dir=target.parent))
    published = False
    reserved = False
    linked: list[Path] = []
    try:
        raw = stage / "raw.csv"
        if url is not None:
            try:
                with _open_url(url, allow_redirects=method == "history_download") as response:
                    if response.status != 200:
                        raise AcquisitionError("FIRMS download did not return HTTP 200")
                    content_type = response.headers.get_content_type()
                    if content_type not in {"text/csv", "application/csv", "text/plain", "application/octet-stream", "binary/octet-stream"}:
                        raise AcquisitionError("FIRMS download returned a non-CSV content type")
                    digest, size = _copy_bytes(response, raw)
                    expected_length = response.headers.get("Content-Length")
                    if expected_length is not None and int(expected_length) != size:
                        raise AcquisitionError("Incomplete FIRMS download; no files published")
            except Exception:
                # urllib exceptions and remote error bodies can embed the MAP_KEY or signed URL.
                raise AcquisitionError("FIRMS HTTPS download failed or returned a non-CSV response; check access, product and availability") from None
        else:
            with source.open("rb") as stream:
                digest, size = _copy_bytes(stream, raw)
        received_at = datetime.now(timezone.utc).isoformat()
        header, count = _validate_csv(raw)
        manifest = {
            "schema_version": "1.0.0", "method": method,
            "received_at_utc": received_at,
            "downloaded_at_utc": received_at if url is not None else None,
            "source": {"product": product, "processing": processing,
                       "processing_verification": "operator_declared_not_scientifically_verified",
                       "url": source_url, "url_sanitization": "credentials_query_fragment_removed; archive_path_redacted",
                       "local_filename": source.name if source is not None else None},
            "request": request_details,
            "raw": {"path": "raw.csv", "sha256": digest, "size_bytes": size,
                    "columns": header, "rows": count},
            "limitations": [
                "Receipt time is the actual ingestion time, not acquisition time or historical availability.",
                "Raw bytes are preserved; no scientific validation or normalization is performed.",
                "NRT products can include RT/URT detections. Standard processing is not silently mapped to NRT.",
            ],
        }
        with (stage / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # mkdir reserves the destination without replacing even an empty directory.
        # Each complete file becomes visible atomically; manifest is the commit marker.
        target.mkdir()
        reserved = True
        os.link(raw, target / "raw.csv")
        linked.append(target / "raw.csv")
        os.link(stage / "manifest.json", target / "manifest.json")
        linked.append(target / "manifest.json")
        published = True
        return manifest
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if reserved and not published:
            for path in reversed(linked):
                path.unlink(missing_ok=True)
            try:
                target.rmdir()
            except OSError:
                pass  # Never remove unrelated files introduced by another writer.


def download_history(url: str, output_dir: str | Path, *, product: str, processing: str) -> dict:
    """Download an operator-supplied authorized archive URL; never invent archive endpoints."""
    _https_url(url)
    quality = _processing(product, processing)
    return _acquire(output_dir, product=product, processing=quality,
                    source_url=_sanitized_url(url), method="history_download", url=url)


def fetch_area(output_dir: str | Path, *, product: str, area: str, days: int = 1,
               start_date: str | None = None, key_env: str = "FIRMS_MAP_KEY") -> dict:
    """Fetch the official Area CSV endpoint; the secret is read only from the environment."""
    quality = _processing(product)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 5:
        raise AcquisitionError("Area API days must be an integer from 1 to 5")
    area = _area_value(area)
    if start_date is not None:
        try:
            if date.fromisoformat(start_date).isoformat() != start_date:
                raise ValueError
        except ValueError:
            raise AcquisitionError("Date must be YYYY-MM-DD") from None
    key = os.environ.get(key_env, "").strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise AcquisitionError("MAP_KEY environment variable is missing or malformed")
    suffix = f"/{product}/{quote(area, safe=',-')}/{days}"
    if start_date:
        suffix += f"/{start_date}"
    return _acquire(output_dir, product=product, processing=quality,
                    source_url=AREA_BASE + "/[redacted]" + suffix, method="area_api",
                    url=AREA_BASE + "/" + key + suffix,
                    request_details={"area": area, "days": days, "date": start_date,
                                     "date_semantics": "forward_from_date" if start_date else "today_backwards"})


def import_local(input_path: str | Path, output_dir: str | Path, *, product: str,
                 processing: str, source_url: str | None = None) -> dict:
    """Copy a previously obtained CSV with a receipt for now, never a backdated receipt."""
    quality = _processing(product, processing)
    if source_url is not None:
        _https_url(source_url)
        source_url = _sanitized_url(source_url)
    return _acquire(output_dir, product=product, processing=quality, source_url=source_url,
                    method="local_import", source=Path(input_path).resolve(strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Preserve FIRMS CSV downloads and sanitized receipts")
    commands = parser.add_subparsers(dest="command", required=True)
    history = commands.add_parser("history", help="Download an explicit authorized HTTPS CSV URL")
    history.add_argument("--url", required=True)
    local = commands.add_parser("import", help="Preserve an already obtained local CSV")
    local.add_argument("input")
    local.add_argument("--source-url")
    for command in (history, local):
        command.add_argument("--processing", required=True, choices=["scientific", "nrt"])
    area = commands.add_parser("area", help="Fetch the official Area API using an environment MAP_KEY")
    area.add_argument("--area", required=True, help="world or west,south,east,north; use --area=-85,-57,-32,14")
    area.add_argument("--days", type=int, default=1)
    area.add_argument("--date")
    area.add_argument("--key-env", default="FIRMS_MAP_KEY")
    for command in (history, local, area):
        command.add_argument("--product", required=True, choices=PRODUCTS)
        command.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        if args.command == "history":
            manifest = download_history(args.url, args.output_dir, product=args.product, processing=args.processing)
        elif args.command == "import":
            manifest = import_local(args.input, args.output_dir, product=args.product,
                                    processing=args.processing, source_url=args.source_url)
        else:
            manifest = fetch_area(args.output_dir, product=args.product, area=args.area,
                                  days=args.days, start_date=args.date, key_env=args.key_env)
    except Exception:
        # Even filesystem errors can contain a credential-bearing input filename.
        print("Acquisition failed; check HTTPS access, input CSV, product, request bounds and a new output directory.", file=sys.stderr)
        return 1
    print(json.dumps({"rows": manifest["raw"]["rows"], "sha256": manifest["raw"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
