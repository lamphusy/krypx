"""Narrow, certificate-verified GSG HTTPS transport with no credential discovery.

This module provides transport only, never network authorization or a scheduler. The
pilot runner must enforce its independent process deadline around DNS, TLS, headers,
and streaming. Tests replace the connection boundary and perform no network I/O.
"""

from __future__ import annotations

import errno
import http.client
import math
import re
import ssl
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.exceptions import (
    NetworkFailureDiagnostic,
    NetworkSafetyError,
    NetworkStageError,
)
from crypto_ai.sentiment.live_deadline import hard_deadline
from crypto_ai.sentiment.providers.gdelt_gsg import (
    RetrievalPlan,
    expected_gsg_source_locator,
    plan_retrieval,
)

_HOST = "data.gdeltproject.org"
_ORIGIN = f"https://{_HOST}"
_MAX_TIMEOUT_SECONDS = 10.0
_MAX_READ_BYTES = 64 * 1024
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_CONTENT_LENGTH = re.compile(r"(?:0|[1-9][0-9]{0,8})\Z")
# These response metadata fields have comma-list semantics. GCS sends one
# x-goog-hash field per checksum algorithm; rejecting the second field would
# discard a legitimate completed HTTP response. Framing and interpretation-
# critical fields remain singular below, including Content-Length.
_REPEATABLE_HEADERS = frozenset({"cache-control", "vary", "x-goog-hash"})


def _body_error_category(exc: Exception) -> str:
    """Classify a stream failure without retaining its text or provider metadata."""
    if isinstance(exc, ssl.SSLEOFError):
        return "tls_eof"
    if isinstance(exc, http.client.IncompleteRead):
        return "length_mismatch"
    if isinstance(exc, ConnectionResetError) or getattr(exc, "errno", None) == errno.ECONNRESET:
        return "connection_reset"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "stream_error"


def _timeout(value: float) -> float:
    if (
        type(value) not in (int, float)
        or not 0 < value <= _MAX_TIMEOUT_SECONDS
        or not math.isfinite(value)
    ):
        raise NetworkSafetyError("HTTPS operation timeout is outside its hard bound")
    return float(value)


def _verified_ssl_context() -> ssl.SSLContext:
    """Use the Darwin system CA bundle without relaxing TLS verification.

    Some Python installations on macOS have no usable default issuer chain even
    though the OS bundle is available at ``/etc/ssl/cert.pem``. Loading that
    bundle supplements the default context; it never disables hostname checks.
    An empty trust store without an available system bundle fails before I/O.
    """
    context = ssl.create_default_context()
    no_default_ca = context.cert_store_stats().get("x509_ca", 0) == 0
    if sys.platform == "darwin" or no_default_ca:
        bundle = Path("/etc/ssl/cert.pem")
        if bundle.is_file():
            context.load_verify_locations(cafile=str(bundle))
        elif no_default_ca:
            raise NetworkSafetyError("HTTPS system CA bundle is unavailable")
    if context.verify_mode != ssl.CERT_REQUIRED or context.check_hostname is not True:
        raise NetworkSafetyError("HTTPS certificate verification cannot be disabled")
    return context


def _headers(response: http.client.HTTPResponse) -> tuple[Mapping[str, str], int | None]:
    if response.headers is None or response.headers.defects:
        raise NetworkSafetyError("HTTPS response header parsing was incomplete")
    headers: dict[str, str] = {}
    for name, value in response.getheaders():
        if (
            type(name) is not str
            or type(value) is not str
            or not _HEADER_NAME.fullmatch(name)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise NetworkSafetyError("HTTPS response headers are malformed or duplicated")
        lowered = name.lower()
        if lowered in headers:
            if lowered not in _REPEATABLE_HEADERS:
                raise NetworkSafetyError("HTTPS response headers are malformed or duplicated")
            headers[lowered] = f"{headers[lowered]}, {value}"
        else:
            headers[lowered] = value
    if "transfer-encoding" in headers:
        raise NetworkSafetyError("HTTPS transfer encoding is not supported by this pilot")
    if headers.get("content-encoding", "identity").lower() != "identity":
        raise NetworkSafetyError("HTTPS content transformation is prohibited")
    length = headers.get("content-length")
    if length is not None and (not _CONTENT_LENGTH.fullmatch(length) or int(length) > 500_000_000):
        raise NetworkSafetyError("HTTPS Content-Length is outside its strict contract")
    return MappingProxyType(headers), None if length is None else int(length)


class RealGSGResponse:
    """Exact undecoded response-body stream, terminated by an actual peer EOF.

    ``HTTPResponse.read`` stops at Content-Length and can conceal surplus bytes. The
    frozen pilot requests ``Connection: close`` and reads the buffered raw body
    stream directly instead. Chunked framing is rejected before exposing a body.
    A declared-length response can tolerate peer EOF without TLS close_notify
    only after every declared byte was captured and a further bounded read
    found no surplus bytes.
    """

    def __init__(
        self,
        connection: http.client.HTTPSConnection,
        response: http.client.HTTPResponse,
        connected_socket: Any,
        url: str,
        headers: Mapping[str, str],
        content_length: int | None,
    ) -> None:
        self.status = response.status
        self.headers = headers
        self.url = url
        self.transfer_complete = False
        self._connection = connection
        self._response = response
        self._socket = connected_socket
        self._content_length = content_length
        self._received = 0
        self._closed = False
        self._eof = False
        self._pending_failure = False
        self._failure_diagnostic: NetworkFailureDiagnostic | None = None

    @property
    def failure_diagnostic(self) -> NetworkFailureDiagnostic | None:
        """Return only bounded local fields, never headers or exception text."""
        return self._failure_diagnostic

    def _mark_failure(self, category: str) -> NetworkFailureDiagnostic:
        diagnostic = NetworkFailureDiagnostic(
            http_status=self.status,
            content_length=self._content_length,
            bytes_received=self._received,
            error_category=category,
        )
        self._failure_diagnostic = diagnostic
        return diagnostic

    def read(self, maximum_bytes: int, *, timeout: float) -> bytes:
        seconds = _timeout(timeout)
        if type(maximum_bytes) is not int or not 0 < maximum_bytes <= _MAX_READ_BYTES:
            raise NetworkSafetyError("HTTPS body read size is outside its hard bound")
        if self._closed:
            raise NetworkSafetyError("HTTPS body cannot be read after close")
        if self._pending_failure:
            self.close()
            raise NetworkStageError(
                "response_body",
                "HTTPS response stream terminated prematurely",
                diagnostic=self._failure_diagnostic,
            )
        if self._eof:
            return b""
        read_started = False
        try:
            self._socket.settimeout(seconds)
            if self._response.fp is None:
                raise NetworkSafetyError("HTTPS body stream is unavailable")
            read_started = True
            chunk = self._response.fp.read1(maximum_bytes)
        except Exception as exc:
            # Some transport errors expose bytes already consumed. Return those
            # first so the persistent byte ledger counts them, then fail on the
            # next read. No exception text, URL, credentials, or headers escape.
            partial = getattr(exc, "partial", None)
            if type(partial) is bytes and 0 < len(partial) <= maximum_bytes:
                self._received += len(partial)
                self._pending_failure = True
                self._mark_failure(_body_error_category(exc))
                return partial
            if (
                read_started
                and isinstance(exc, ssl.SSLEOFError)
                and self._content_length is not None
                and self._received == self._content_length
                and (partial is None or (type(partial) is bytes and not partial))
            ):
                # Exact declared framing proves no body byte is missing. The
                # bounded read above still exposes buffered surplus bytes;
                # only an actual TLS EOF with no partial bytes is accepted.
                self._eof = True
                self.transfer_complete = True
                return b""
            diagnostic = self._mark_failure(_body_error_category(exc))
            self.close()
            raise NetworkStageError(
                "response_body", "HTTPS response body read failed", diagnostic=diagnostic
            ) from None
        if type(chunk) is not bytes or len(chunk) > maximum_bytes:
            self.close()
            raise NetworkSafetyError("HTTPS response stream violated its read bound")
        self._received += len(chunk)
        if not chunk:
            self._eof = True
            self.transfer_complete = (
                self._content_length is None or self._received == self._content_length
            )
            if not self.transfer_complete:
                self._mark_failure("length_mismatch")
        elif self._content_length is not None and self._received > self._content_length:
            self._mark_failure("length_mismatch")
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (self._response, self._connection):
            try:
                resource.close()
            except Exception:
                # Closing must not leak arbitrary transport diagnostics or mask
                # the original integrity failure. The runner also owns a deadline.
                pass

    def __enter__(self) -> RealGSGResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class RealGSGTransport:
    """Allowlisted HTTPS only; no redirects, proxy, netrc, cookies, or decoding."""

    mock_only = False
    incremental_cost_usd = 0.0

    def __init__(self, plan: RetrievalPlan) -> None:
        try:
            if (
                type(plan) is not RetrievalPlan
                or len(plan.intervals) not in (720, 1440)
                or plan != plan_retrieval(plan.start_at, plan.end_at_exclusive)
            ):
                raise NetworkSafetyError("live HTTPS requires an authoritative 12- or 24-hour plan")
        except (CryptoAIError, TypeError, ValueError):
            raise NetworkSafetyError(
                "live HTTPS requires an authoritative 12- or 24-hour plan"
            ) from None
        self._allowed_urls = frozenset(
            expected_gsg_source_locator(interval) for interval in plan.intervals
        )

    def open(self, url: str, *, timeout: float) -> RealGSGResponse:
        seconds = _timeout(timeout)
        if type(url) is not str or url not in self._allowed_urls:
            raise NetworkSafetyError("HTTPS locator is outside the approved retrieval plan")
        # The independent child enforces one aggregate bound across certificate
        # setup, DNS, TLS, sending, and headers; socket timeouts alone cannot.
        with hard_deadline(seconds):
            return self._open_verified(url, seconds)

    def _open_verified(self, url: str, seconds: float) -> RealGSGResponse:
        connection = None
        response = None
        stage = "tls_context"
        try:
            context = _verified_ssl_context()
            stage = "request_dispatch"
            connection = http.client.HTTPSConnection(
                _HOST, port=443, timeout=seconds, context=context
            )
            connection.request(
                "GET",
                url[len(_ORIGIN) :],
                headers={
                    "Accept": "application/gzip, application/octet-stream",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "User-Agent": "KrypX-Batch-B-Pilot/1.0 (GDELT research; no redistribution)",
                },
            )
            connected_socket = connection.sock
            if connected_socket is None:
                raise NetworkSafetyError("HTTPS connection lacks a verified stream")
            # An abrupt TLS disconnect is not proof of a complete unknown-length
            # response. Require TLS close_notify rather than silently translating
            # a ragged TLS EOF to an ordinary HTTP stream EOF.
            connected_socket.suppress_ragged_eofs = False
            stage = "response_headers"
            response = connection.getresponse()
            stage = "response_validation"
            if type(response.status) is not int or not 200 <= response.status <= 599:
                raise NetworkSafetyError("HTTPS response status is invalid")
            if 300 <= response.status <= 399:
                raise NetworkSafetyError("HTTPS redirects are prohibited")
            stage = "response_headers"
            headers, content_length = _headers(response)
            return RealGSGResponse(
                connection, response, connected_socket, url, headers, content_length
            )
        except Exception as exc:
            for resource in (response, connection):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass
            if isinstance(exc, ssl.SSLCertVerificationError):
                stage = "tls_verification"
            raise NetworkStageError(stage, "allowlisted HTTPS request failed closed") from None
