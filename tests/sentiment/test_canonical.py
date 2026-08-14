"""RFC 8785 canonical serialization tests."""

import pytest

from crypto_ai.exceptions import CanonicalizationError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize


def test_canonicalizes_key_order_strings_and_numbers_to_exact_bytes() -> None:
    value = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27, -0.0],
        "literals": [None, True, False],
        "string": "€\u000f\nA'B\"\\\\/",
    }

    expected = (
        '{"literals":[null,true,false],'
        '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0],'
        '"string":"€\\u000f\\nA\'B\\"\\\\\\\\/"}'
    )
    assert canonicalize(value) == expected.encode()


def test_float_format_uses_ecmascript_decimal_boundaries() -> None:
    assert canonicalize([1e-7, 1e-6, 1e20, 1e21]) == (
        b"[1e-7,0.000001,100000000000000000000,1e+21]"
    )


def test_rejects_nonfinite_unsafe_integer_and_lone_surrogate() -> None:
    for value in (float("nan"), float("inf"), 1 << 53, "\ud800"):
        with pytest.raises(CanonicalizationError):
            canonicalize(value)


def test_hash_is_over_exact_canonical_bytes() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
