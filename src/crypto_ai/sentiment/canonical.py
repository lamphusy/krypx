"""Dependency-free RFC 8785 JSON Canonicalization Scheme support.

The implementation deliberately accepts only the interoperable data model used by
KrypX contracts: JSON objects with string keys, arrays, strings, booleans, null,
IEEE-754 finite numbers, and integers in the exact binary64 range.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from crypto_ai.exceptions import CanonicalizationError

MAX_SAFE_INTEGER = (1 << 53) - 1


def canonicalize(value: Any) -> bytes:
    """Return the exact RFC 8785/JCS UTF-8 representation of ``value``."""
    return _serialize(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the exact JCS bytes for ``value`` with SHA-256."""
    return hashlib.sha256(canonicalize(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for an exact byte string."""
    if not isinstance(value, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(value).hexdigest()


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalizationError("integer is outside the RFC 8785 interoperable range")
        return str(value)
    if isinstance(value, float):
        return _serialize_float(value)
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, Mapping):
        entries: list[tuple[bytes, str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("object keys must be strings")
            _validate_unicode(key)
            entries.append((key.encode("utf-16-be"), key, item))
        entries.sort(key=lambda entry: entry[0])
        return (
            "{"
            + ",".join(f"{_serialize_string(key)}:{_serialize(item)}" for _, key, item in entries)
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    raise CanonicalizationError(f"unsupported canonical JSON type: {type(value).__name__}")


def _validate_unicode(value: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError("strings must not contain lone surrogate code points") from exc


def _serialize_string(value: str) -> str:
    _validate_unicode(value)
    short_escapes = {
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
        '"': '\\"',
        "\\": "\\\\",
    }
    pieces = ['"']
    for character in value:
        escaped = short_escapes.get(character)
        if escaped is not None:
            pieces.append(escaped)
        elif ord(character) <= 0x1F:
            pieces.append(f"\\u{ord(character):04x}")
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces)


def _serialize_float(value: float) -> str:
    if not math.isfinite(value):
        raise CanonicalizationError("non-finite numbers are forbidden")
    if value == 0.0:
        return "0"

    sign = "-" if value < 0 else ""
    rendered = repr(abs(value)).lower()
    if "e" not in rendered:
        if rendered.endswith(".0"):
            rendered = rendered[:-2]
        return sign + rendered

    mantissa, raw_exponent = rendered.split("e", maxsplit=1)
    exponent = int(raw_exponent)
    digits = mantissa.replace(".", "")

    # ECMAScript Number::toString, which JCS adopts, uses decimal form in this range.
    if -6 <= exponent < 21:
        decimal_position = 1 + exponent
        if decimal_position <= 0:
            number = "0." + ("0" * -decimal_position) + digits
        elif decimal_position >= len(digits):
            number = digits + ("0" * (decimal_position - len(digits)))
        else:
            number = digits[:decimal_position] + "." + digits[decimal_position:]
        return sign + number

    normalized_mantissa = digits[0]
    if len(digits) > 1:
        normalized_mantissa += "." + digits[1:]
    exponent_text = f"+{exponent}" if exponent >= 0 else str(exponent)
    return f"{sign}{normalized_mantissa}e{exponent_text}"
