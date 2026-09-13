"""All HTTPS boundaries are replaced locally; this suite never opens a socket."""

from __future__ import annotations

import http.client
import io
import signal
import socket
import ssl
import subprocess
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from crypto_ai.sentiment import live_transport
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.live_deadline import hard_deadline
from crypto_ai.sentiment.live_transport import RealGSGTransport
from crypto_ai.sentiment.providers.gdelt_gsg import expected_gsg_source_locator, plan_retrieval

PLAN = plan_retrieval("2026-09-13T00:00:00Z", "2026-09-14T00:00:00Z")
URL = expected_gsg_source_locator(PLAN.intervals[0])
RAW = b"\x1f\x8b\x08\x00opaque-gzip-body\x00\xff"


class _Socket:
    def __init__(self, wire: bytes) -> None:
        self.stream = io.BytesIO(wire)
        self.timeouts: list[float] = []

    def makefile(self, *_args, **_kwargs):
        return self.stream

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _Connection:
    def __init__(self, wire: bytes) -> None:
        self.sock = _Socket(wire)
        self.request = Mock()
        self.close = Mock()
        self.response = None

    def getresponse(self):
        # Exercise Python's actual HTTP header parser, not an invented facsimile.
        self.response = http.client.HTTPResponse(self.sock)
        self.response.begin()
        return self.response


@pytest.fixture(autouse=True)
def _no_live_connections(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("offline transport test attempted a live socket")

    monkeypatch.setattr(socket, "create_connection", forbidden)


def _connection(monkeypatch, *, body=RAW, headers=None, status="200 OK"):
    if headers is None:
        headers = [("Content-Length", str(len(body)))]
    wire = (
        f"HTTP/1.1 {status}\r\n"
        + "".join(f"{name}: {value}\r\n" for name, value in headers)
        + "\r\n"
    ).encode("ascii") + body
    connection = _Connection(wire)
    factory = Mock(return_value=connection)
    monkeypatch.setattr(live_transport.http.client, "HTTPSConnection", factory)
    return connection, factory


def test_allowlisted_https_preserves_exact_bytes_and_requires_actual_eof(monkeypatch):
    connection, factory = _connection(monkeypatch)
    transport = RealGSGTransport(PLAN)
    response = transport.open(URL, timeout=10.0)
    assert transport.mock_only is False
    assert transport.incremental_cost_usd == 0.0
    assert response.status == 200
    assert response.url == URL
    assert response.read(65536, timeout=3.0) == RAW
    assert response.transfer_complete is False
    assert response.read(65536, timeout=2.0) == b""
    assert response.transfer_complete is True
    assert response.read(65536, timeout=1.0) == b""
    assert connection.sock.timeouts == [3.0, 2.0]
    assert connection.sock.suppress_ragged_eofs is False
    args, kwargs = factory.call_args
    assert args == ("data.gdeltproject.org",)
    assert kwargs["port"] == 443
    assert kwargs["timeout"] == 10.0
    assert kwargs["context"].verify_mode == ssl.CERT_REQUIRED
    assert kwargs["context"].check_hostname is True
    request_args, request_kwargs = connection.request.call_args
    assert request_args == ("GET", "/gdeltv3/gsg/20260913000000.gsg.json.gz")
    assert request_kwargs["headers"]["Accept-Encoding"] == "identity"
    assert request_kwargs["headers"]["Connection"] == "close"
    assert not {"Authorization", "Cookie", "Proxy-Authorization"} & set(request_kwargs["headers"])
    response.close()
    response.close()
    connection.close.assert_called_once()
    with pytest.raises(NetworkSafetyError, match="after close"):
        response.read(1, timeout=1.0)


@pytest.mark.parametrize(
    "url",
    [
        URL.replace("https://", "http://"),
        URL.replace("data.gdeltproject.org", "evil.example"),
        URL.replace("data.gdeltproject.org", "data.gdeltproject.org:443"),
        URL.replace("data.gdeltproject.org", "user:secret@data.gdeltproject.org"),
        URL + "?token=secret",
        URL + "#fragment",
        URL.replace("20260913000000", "20260912000000"),
        URL.replace("20260913000000", "20260914000000"),
        URL.replace("20260913000000", "20260913000001"),
        URL.replace("/gsg/", "/gsg/../gsg/"),
        URL.replace("/gsg/", "/gsg/%2e/"),
        "https://api.gdeltproject.org/api/v2/doc/doc",
        None,
        [],
    ],
)
def test_locator_rejected_before_connection_creation(monkeypatch, url):
    _, factory = _connection(monkeypatch)
    with pytest.raises(NetworkSafetyError, match="outside"):
        RealGSGTransport(PLAN).open(url, timeout=1.0)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "timeout", [0, -1, 10.0001, 10**1000, float("inf"), float("nan"), True, "10"]
)
def test_invalid_timeout_rejected_before_connect(monkeypatch, timeout):
    _, factory = _connection(monkeypatch)
    with pytest.raises(NetworkSafetyError, match="timeout"):
        RealGSGTransport(PLAN).open(URL, timeout=timeout)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "plan", [None, {}, replace(PLAN, plan_id="0" * 64), replace(PLAN, intervals=None)]
)
def test_plan_is_verified_before_transport_exists(plan):
    with pytest.raises(NetworkSafetyError, match="authoritative"):
        RealGSGTransport(plan)


def test_short_plan_is_not_a_live_pilot():
    short = plan_retrieval("2026-09-13T00:00:00Z", "2026-09-13T00:01:00Z")
    with pytest.raises(NetworkSafetyError, match="24-hour"):
        RealGSGTransport(short)


@pytest.mark.parametrize("status", ["301 Moved", "302 Found", "307 Temporary", "308 Permanent"])
def test_redirects_are_not_followed(monkeypatch, status):
    connection, factory = _connection(
        monkeypatch, status=status, headers=[("Location", "https://secret@evil.example")]
    )
    with pytest.raises(NetworkSafetyError, match="failed closed") as caught:
        RealGSGTransport(PLAN).open(URL, timeout=1.0)
    assert "secret" not in str(caught.value)
    assert factory.call_count == 1
    connection.close.assert_called_once()
    assert connection.response.fp is None


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Length", "1"), ("content-length", "2")],
        [("X-Test", "a"), ("x-test", "b")],
        [("Transfer-Encoding", "chunked")],
        [("Transfer-Encoding", "identity")],
        [("Content-Encoding", "gzip")],
        [("Content-Length", "-1")],
        [("Content-Length", "01")],
        [("Content-Length", "1, 1")],
        [("Content-Length", "500000001")],
        [("X-Fold", "first\r\n continuation")],
    ],
)
def test_unsupported_or_ambiguous_headers_fail_before_body_read(monkeypatch, headers):
    connection, _ = _connection(monkeypatch, headers=headers)
    with pytest.raises(NetworkSafetyError, match="failed closed"):
        RealGSGTransport(PLAN).open(URL, timeout=1.0)
    connection.close.assert_called_once()


def test_http_parser_cannot_silently_drop_malformed_header_lines(monkeypatch):
    connection = _Connection(b"HTTP/1.1 200 OK\r\nMalformed header\r\n\r\nbody")
    monkeypatch.setattr(
        live_transport.http.client, "HTTPSConnection", Mock(return_value=connection)
    )
    with pytest.raises(NetworkSafetyError, match="failed closed"):
        RealGSGTransport(PLAN).open(URL, timeout=1.0)
    connection.close.assert_called_once()


@pytest.mark.parametrize("difference", [-1, 1, 100])
def test_declared_length_never_hides_truncation_or_surplus(monkeypatch, difference):
    _connection(monkeypatch, headers=[("Content-Length", str(len(RAW) + difference))])
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    # An ordinary HTTPResponse.read() silently clips the surplus-byte case.
    assert response.read(65536, timeout=1.0) == RAW
    assert response.read(65536, timeout=1.0) == b""
    assert response.transfer_complete is False
    response.close()


def test_close_delimited_unknown_length_attests_peer_eof(monkeypatch):
    _connection(monkeypatch, headers=[])
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    assert response.read(65536, timeout=1.0) == RAW
    assert response.transfer_complete is False
    assert response.read(65536, timeout=1.0) == b""
    assert response.transfer_complete is True
    response.close()


@pytest.mark.parametrize("status", ["400 Bad", "404 Missing", "429 Busy", "500 Error", "599 Error"])
def test_nonredirect_http_errors_are_returned_to_signed_retry_policy(monkeypatch, status):
    _connection(monkeypatch, status=status)
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    assert response.status == int(status[:3])
    assert response.read(65536, timeout=1.0) == RAW
    assert response.read(65536, timeout=1.0) == b""
    assert response.transfer_complete is True
    response.close()


@pytest.mark.parametrize("maximum_bytes", [0, -1, 65537, True, 1.5])
def test_read_bound_is_enforced(monkeypatch, maximum_bytes):
    _connection(monkeypatch)
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    with pytest.raises(NetworkSafetyError, match="read size"):
        response.read(maximum_bytes, timeout=1.0)
    response.close()


def test_partial_bytes_are_delivered_before_sanitized_stream_failure(monkeypatch):
    connection, _ = _connection(monkeypatch)
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    stream = Mock()
    stream.read1.side_effect = http.client.IncompleteRead(b"partial", 100)
    connection.response.fp = stream
    assert response.read(65536, timeout=1.0) == b"partial"
    assert response.transfer_complete is False
    with pytest.raises(NetworkSafetyError, match="prematurely"):
        response.read(65536, timeout=1.0)
    assert stream.read1.call_count == 1
    connection.close.assert_called_once()


def test_stream_error_is_sanitized_and_closes_connection(monkeypatch):
    connection, _ = _connection(monkeypatch)
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    stream = Mock()
    stream.read1.side_effect = OSError("credential=secret; sensitive transport diagnostics")
    connection.response.fp = stream
    with pytest.raises(NetworkSafetyError, match="body read failed") as caught:
        response.read(65536, timeout=1.0)
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__ is True
    assert response.transfer_complete is False
    connection.close.assert_called_once()


@pytest.mark.parametrize("failure", [ssl.SSLCertVerificationError("secret"), OSError("secret")])
def test_tls_or_connect_failure_closes_without_leaking_diagnostics(monkeypatch, failure):
    connection, _ = _connection(monkeypatch)
    connection.request.side_effect = failure
    with pytest.raises(NetworkSafetyError, match="failed closed") as caught:
        RealGSGTransport(PLAN).open(URL, timeout=1.0)
    assert "secret" not in str(caught.value)
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "verify_mode,check_hostname", [(ssl.CERT_NONE, False), (ssl.CERT_REQUIRED, False)]
)
def test_tls_verification_cannot_be_disabled(monkeypatch, verify_mode, check_hostname):
    _, factory = _connection(monkeypatch)
    context = Mock(verify_mode=verify_mode, check_hostname=check_hostname)
    monkeypatch.setattr(live_transport.ssl, "create_default_context", Mock(return_value=context))
    with pytest.raises(NetworkSafetyError, match="failed closed"):
        RealGSGTransport(PLAN).open(URL, timeout=1.0)
    factory.assert_not_called()


def test_proxy_environment_does_not_change_destination(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@evil.example:8080")
    monkeypatch.setenv("ALL_PROXY", "http://evil.example:8080")
    connection, factory = _connection(monkeypatch)
    response = RealGSGTransport(PLAN).open(URL, timeout=1.0)
    assert factory.call_args.args == ("data.gdeltproject.org",)
    assert not hasattr(connection, "set_tunnel")
    response.close()


@pytest.mark.parametrize("seconds", [0, -1, 900.001, 10**1000, float("nan"), True, "10"])
def test_invalid_process_deadline_fails_before_watchdog_spawn(monkeypatch, seconds):
    spawn = Mock()
    monkeypatch.setattr("crypto_ai.sentiment.live_deadline.subprocess.Popen", spawn)
    with pytest.raises(NetworkSafetyError, match="deadline"), hard_deadline(seconds):
        pytest.fail("invalid deadline entered")
    spawn.assert_not_called()


@pytest.mark.parametrize(
    "operation",
    [
        "with hard_deadline(0.05):\n    time.sleep(1)",
        "with hard_deadline(0.05):\n    with hard_deadline(0.5):\n        time.sleep(1)",
        "\n".join(
            [
                "from crypto_ai.sentiment import live_transport",
                "from crypto_ai.sentiment.providers.gdelt_gsg import plan_retrieval",
                "def blocked_context():",
                "    time.sleep(1)",
                "live_transport.ssl.create_default_context = blocked_context",
                "plan = plan_retrieval('2026-09-13T00:00:00Z', '2026-09-14T00:00:00Z')",
                "transport = live_transport.RealGSGTransport(plan)",
                f"transport.open({URL!r}, timeout=0.05)",
            ]
        ),
    ],
)
def test_watchdog_hard_kills_blocked_process_and_cannot_extend_outer_deadline(operation):
    script = (
        "import time\nfrom crypto_ai.sentiment.live_deadline import hard_deadline\n" + operation
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=5)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()


def test_watchdog_is_cancelled_after_successful_operation():
    script = (
        "import time\nfrom crypto_ai.sentiment.live_deadline import hard_deadline\n"
        "with hard_deadline(0.5):\n    pass\ntime.sleep(0.6)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode()
