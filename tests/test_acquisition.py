"""Offline acquisition contracts; mocked HTTP is not proof of NASA archive access."""
from datetime import datetime, timezone
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from fire_atlas_data import acquisition


CSV = (b"\xef\xbb\xbflatitude,longitude,acq_date,acq_time,version\r\n"
       b"-15.12,-47.75,2026-09-01,0007,2.0NRT\r\n")


class Response(io.BytesIO):
    def __init__(self, data=CSV, content_type="text/csv", status=200, length=None):
        super().__init__(data)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(data) if length is None else length)


def mock_download(monkeypatch, data=CSV, content_type="text/csv", status=200, length=None):
    requests = []

    def open_url(url, *, allow_redirects):
        requests.append((url, allow_redirects))
        return Response(data, content_type, status, length)

    monkeypatch.setattr(acquisition, "_open_url", open_url)
    return requests


def test_history_preserves_bytes_and_sanitizes_signed_url(monkeypatch, tmp_path):
    requests = mock_download(monkeypatch)
    url = "https://example.org/private-token/fire.csv?signature=private-signature#private-fragment"
    before = datetime.now(timezone.utc)
    receipt = acquisition.download_history(url, tmp_path / "archive", product="MODIS_SP", processing="scientific")
    after = datetime.now(timezone.utc)
    assert requests == [(url, True)]
    assert (tmp_path / "archive/raw.csv").read_bytes() == CSV
    assert receipt == json.loads((tmp_path / "archive/manifest.json").read_text())
    assert receipt["raw"]["sha256"] == hashlib.sha256(CSV).hexdigest()
    assert receipt["raw"]["size_bytes"] == len(CSV)
    assert receipt["raw"]["rows"] == 1
    assert receipt["source"]["processing"] == "scientific"
    assert receipt["source"]["url"] == "https://example.org/[redacted]"
    assert before <= datetime.fromisoformat(receipt["received_at_utc"]) <= after
    assert receipt["downloaded_at_utc"] == receipt["received_at_utc"]
    assert "private-" not in json.dumps(receipt)
    assert not list(tmp_path.glob(".firms-acquisition-*"))


def test_area_uses_environment_key_and_exact_documented_path(monkeypatch, tmp_path):
    requests = mock_download(monkeypatch)
    monkeypatch.setenv("CUSTOM_FIRMS_KEY", "secret-key-123")
    receipt = acquisition.fetch_area(tmp_path / "area", product="VIIRS_NOAA20_NRT",
                                     area="-48,-17,-46,-15", days=5,
                                     start_date="2026-10-01", key_env="CUSTOM_FIRMS_KEY")
    assert requests == [(acquisition.AREA_BASE + "/secret-key-123/VIIRS_NOAA20_NRT/-48,-17,-46,-15/5/2026-10-01", False)]
    assert "secret-key-123" not in json.dumps(receipt)
    assert receipt["source"]["processing"] == "nrt"
    assert receipt["request"]["date_semantics"] == "forward_from_date"
    assert receipt["source"]["url"].endswith("/[redacted]/VIIRS_NOAA20_NRT/-48,-17,-46,-15/5/2026-10-01")


def test_recent_area_and_header_only_csv_are_valid(monkeypatch, tmp_path):
    mock_download(monkeypatch, b"latitude,longitude,acq_date,acq_time\n")
    monkeypatch.setenv("FIRMS_MAP_KEY", "secret")
    receipt = acquisition.fetch_area(tmp_path / "empty", product="MODIS_SP", area="world")
    assert receipt["raw"]["rows"] == 0
    assert receipt["request"] == {"area": "world", "days": 1, "date": None, "date_semantics": "today_backwards"}
    assert receipt["source"]["processing"] == "scientific"


@pytest.mark.parametrize("days", [0, 6, -1, 1.5, True])
def test_invalid_days_never_request(monkeypatch, tmp_path, days):
    requests = mock_download(monkeypatch)
    with pytest.raises(acquisition.AcquisitionError, match="1 to 5"):
        acquisition.fetch_area(tmp_path / "bad", product="MODIS_NRT", area="world", days=days)
    assert requests == []
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("area", ["south", "1,2,3", "1,2,3,4,5", "nan,0,1,2", "-181,0,2,3", "0,-91,2,3", "2,0,1,3"])
def test_invalid_area_never_requests(monkeypatch, tmp_path, area):
    requests = mock_download(monkeypatch)
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.fetch_area(tmp_path / "bad", product="MODIS_NRT", area=area)
    assert requests == []


@pytest.mark.parametrize("start", ["20261001", "2026-02-30", "2026-1-01", "2026-10-01T00:00:00"])
def test_invalid_date_never_requests(monkeypatch, tmp_path, start):
    requests = mock_download(monkeypatch)
    with pytest.raises(acquisition.AcquisitionError, match="YYYY-MM-DD"):
        acquisition.fetch_area(tmp_path / "bad", product="MODIS_NRT", area="world", start_date=start)
    assert requests == []


def test_missing_key_is_not_network_access(monkeypatch, tmp_path):
    requests = mock_download(monkeypatch)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(acquisition.AcquisitionError, match="environment"):
        acquisition.fetch_area(tmp_path / "bad", product="MODIS_NRT", area="world")
    assert requests == []


@pytest.mark.parametrize("data,content_type,status,length", [
    (b"<html>Earthdata login</html>", "text/html", 200, None),
    (b"Invalid MAP_KEY: secret-key", "text/plain", 200, None),
    (b'{"error":"invalid key"}', "application/json", 200, None),
    (CSV, "text/csv", 403, None),
    (CSV, "text/csv", 200, len(CSV) + 1),
    (b"latitude,longitude,acq_date,acq_time\n1,2\n", "text/csv", 200, None),
    (b"latitude,longitude,acq_date,acq_time\n\xff,2,2026-01-01,1200\n", "text/csv", 200, None),
])
def test_non_csv_and_http_errors_publish_nothing(monkeypatch, tmp_path, data, content_type, status, length):
    mock_download(monkeypatch, data, content_type, status, length)
    with pytest.raises(acquisition.AcquisitionError) as failure:
        acquisition.download_history("https://example.org/secret-key", tmp_path / "bad", product="MODIS_SP", processing="scientific")
    assert "secret-key" not in str(failure.value)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".firms-acquisition-*"))


def test_stream_failure_does_not_expose_url_or_leave_partial_output(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("https://example.org/secret-key-123?token=signed-secret")

    monkeypatch.setattr(acquisition, "_open_url", fail)
    with pytest.raises(acquisition.AcquisitionError) as failure:
        acquisition.download_history("https://example.org/secret-key-123", tmp_path / "bad", product="MODIS_NRT", processing="nrt")
    assert "secret-key" not in str(failure.value)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".firms-acquisition-*"))


def test_local_import_records_now_and_leaves_original_untouched(tmp_path):
    source = tmp_path / "historical.csv"
    source.write_bytes(CSV)
    before = datetime.now(timezone.utc)
    receipt = acquisition.import_local(source, tmp_path / "copy", product="VIIRS_NOAA20_NRT", processing="nrt")
    assert source.read_bytes() == (tmp_path / "copy/raw.csv").read_bytes() == CSV
    assert before <= datetime.fromisoformat(receipt["received_at_utc"]) <= datetime.now(timezone.utc)
    assert receipt["downloaded_at_utc"] is None
    assert receipt["method"] == "local_import"


def test_existing_destination_is_never_changed_or_requested(monkeypatch, tmp_path):
    requests = mock_download(monkeypatch)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "original").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        acquisition.download_history("https://example.org/archive.csv", target, product="MODIS_SP", processing="scientific")
    assert (target / "original").read_bytes() == b"preserve"
    assert requests == []


def test_destination_appearing_during_download_is_not_replaced(monkeypatch, tmp_path):
    target = tmp_path / "raced"

    def open_url(*args, **kwargs):
        target.mkdir()
        return Response()

    monkeypatch.setattr(acquisition, "_open_url", open_url)
    with pytest.raises(FileExistsError):
        acquisition.download_history("https://example.org/archive.csv", target, product="MODIS_SP", processing="scientific")
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_failed_manifest_publication_removes_only_own_artifacts(monkeypatch, tmp_path):
    mock_download(monkeypatch)
    real_link = acquisition.os.link

    def fail_manifest(source, target):
        if Path(target).name == "manifest.json":
            raise OSError("disk full")
        real_link(source, target)

    monkeypatch.setattr(acquisition.os, "link", fail_manifest)
    with pytest.raises(OSError):
        acquisition.download_history("https://example.org/archive.csv", tmp_path / "bad", product="MODIS_SP", processing="scientific")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("url", ["http://example.org/a.csv", "https://user:password@example.org/a.csv", "https:///a.csv", "https://example.org:bad/a.csv"])
def test_only_explicit_credential_free_https(monkeypatch, tmp_path, url):
    requests = mock_download(monkeypatch)
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.download_history(url, tmp_path / "bad", product="MODIS_SP", processing="scientific")
    assert requests == []


def test_scientific_products_cannot_be_declared_nrt(monkeypatch, tmp_path):
    requests = mock_download(monkeypatch)
    with pytest.raises(acquisition.AcquisitionError, match="processing"):
        acquisition.download_history("https://example.org/a.csv", tmp_path / "bad", product="MODIS_SP", processing="nrt")
    assert requests == []


def test_redirects_cannot_downgrade_https_or_forward_area_key():
    request = Request("https://firms.modaps.eosdis.nasa.gov/api/area/csv/secret")
    with pytest.raises(acquisition.AcquisitionError, match="HTTPS"):
        acquisition._HTTPSRedirects(True).redirect_request(request, None, 302, "Found", {}, "http://example.org/file.csv")
    with pytest.raises(acquisition.AcquisitionError, match="MAP_KEY"):
        acquisition._HTTPSRedirects(False).redirect_request(request, None, 302, "Found", {}, "https://example.org/file.csv")


def test_cli_failure_does_not_log_signed_url(monkeypatch, tmp_path, capsys):
    def fail(*args, **kwargs):
        raise OSError("private-key and signed-secret")

    monkeypatch.setattr(acquisition, "_open_url", fail)
    monkeypatch.setattr("sys.argv", ["acquisition", "history", "--url", "https://example.org/private-key?token=signed-secret",
                                   "--product", "MODIS_SP", "--processing", "scientific", "--output-dir", str(tmp_path / "bad")])
    assert acquisition.main() == 1
    captured = capsys.readouterr()
    assert "private-key" not in captured.err + captured.out
    assert "signed-secret" not in captured.err + captured.out
