"""Google error classification — typed exceptions first (2.2.0, audit §1-4).

2.1.x classified by substring over ``str(e)``: ``'429' in msg`` etc.
That heuristic misroutes any message that *echoes* a number — the audit
called out a 500 whose body contains "400" landing in BAD_REQUEST
(fatal, no retry) instead of SERVER_ERROR (recoverable). The installed
SDK (google-genai) raises typed errors carrying ``.code`` (HTTP int) and
``.status`` (gRPC-style string); classification now reads those, with
``google.api_core`` types honoured when that package is present and the
substring path demoted to a genuinely-last resort for structureless
exceptions (transport errors, asyncio timeouts).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from google.genai import errors as genai_errors  # noqa: E402

from geny_executor.core.errors import APIError, ErrorCategory  # noqa: E402
from geny_executor.llm_client.google import GoogleClient  # noqa: E402


def _client() -> GoogleClient:
    return GoogleClient(api_key="sk-mock")


def _genai_error(
    code: int, status: str, message: str, *, server: bool = False
) -> genai_errors.APIError:
    """Synthesize the SDK's typed error exactly as ``raise_for_response``
    builds it: ClientError for 4xx, ServerError for 5xx."""
    cls = genai_errors.ServerError if server else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "status": status, "message": message}})


# ---------------------------------------------------------------------------
# google-genai typed errors (what the installed SDK raises)
# ---------------------------------------------------------------------------


def test_resource_exhausted_is_rate_limited() -> None:
    err = _client()._classify_error(
        _genai_error(429, "RESOURCE_EXHAUSTED", "quota exceeded")
    )
    assert err.category is ErrorCategory.RATE_LIMITED
    assert err.status_code == 429


def test_unauthenticated_is_auth() -> None:
    err = _client()._classify_error(
        _genai_error(401, "UNAUTHENTICATED", "API key not valid")
    )
    assert err.category is ErrorCategory.AUTH


def test_permission_denied_is_auth() -> None:
    err = _client()._classify_error(
        _genai_error(403, "PERMISSION_DENIED", "caller lacks permission")
    )
    assert err.category is ErrorCategory.AUTH


def test_invalid_argument_is_bad_request() -> None:
    err = _client()._classify_error(
        _genai_error(400, "INVALID_ARGUMENT", "contents must not be empty")
    )
    assert err.category is ErrorCategory.BAD_REQUEST


def test_deadline_exceeded_is_timeout() -> None:
    err = _client()._classify_error(
        _genai_error(504, "DEADLINE_EXCEEDED", "deadline exceeded", server=True)
    )
    assert err.category is ErrorCategory.TIMEOUT


def test_unavailable_is_server_error() -> None:
    err = _client()._classify_error(
        _genai_error(503, "UNAVAILABLE", "service unavailable", server=True)
    )
    assert err.category is ErrorCategory.SERVER_ERROR


def test_internal_500_echoing_400_is_server_error_not_bad_request() -> None:
    """THE audit case: the old ``'400' in str(e)`` substring check ran
    before any 5xx check, so a 500 whose body echoes a nested 400 was
    classified fatal (BAD_REQUEST) and never retried."""
    err = _client()._classify_error(
        _genai_error(
            500, "INTERNAL",
            "internal error while proxying upstream response code 400",
            server=True,
        )
    )
    assert err.category is ErrorCategory.SERVER_ERROR
    assert err.status_code == 500


def test_novel_4xx_partitions_to_bad_request() -> None:
    """A code/status pair the explicit table doesn't know (e.g. 422)
    still lands on the right side via the ClientError/4xx partition."""
    err = _client()._classify_error(
        _genai_error(422, "SOME_FUTURE_STATUS", "unprocessable")
    )
    assert err.category is ErrorCategory.BAD_REQUEST


def test_novel_5xx_partitions_to_server_error() -> None:
    err = _client()._classify_error(
        _genai_error(599, "SOME_FUTURE_STATUS", "weird gateway thing", server=True)
    )
    assert err.category is ErrorCategory.SERVER_ERROR


def test_status_string_wins_when_code_is_nonstandard() -> None:
    """gRPC transports sometimes surface a generic code with a precise
    status string — the status must still route correctly."""
    err = _client()._classify_error(
        _genai_error(400, "RESOURCE_EXHAUSTED", "per-minute quota")
    )
    # 429-style status on a 400 code: RESOURCE_EXHAUSTED is checked first.
    assert err.category is ErrorCategory.RATE_LIMITED


# ---------------------------------------------------------------------------
# google.api_core typed exceptions (present only with Vertex/grpc extras)
# ---------------------------------------------------------------------------


def _install_fake_api_core(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """google.api_core is NOT a google-genai dependency and is absent
    from this venv — inject a shape-compatible stand-in so the
    isinstance chain is exercised without bloating dev deps."""
    exc_mod = types.ModuleType("google.api_core.exceptions")
    for name in (
        "ResourceExhausted",
        "Unauthenticated",
        "PermissionDenied",
        "InvalidArgument",
        "DeadlineExceeded",
        "ServiceUnavailable",
        "InternalServerError",
    ):
        setattr(exc_mod, name, type(name, (Exception,), {}))
    pkg = types.ModuleType("google.api_core")
    pkg.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "google.api_core", pkg)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", exc_mod)
    return exc_mod


@pytest.mark.parametrize(
    "exc_name,expected",
    [
        ("ResourceExhausted", ErrorCategory.RATE_LIMITED),
        ("Unauthenticated", ErrorCategory.AUTH),
        ("PermissionDenied", ErrorCategory.AUTH),
        ("InvalidArgument", ErrorCategory.BAD_REQUEST),
        ("DeadlineExceeded", ErrorCategory.TIMEOUT),
        ("ServiceUnavailable", ErrorCategory.SERVER_ERROR),
        ("InternalServerError", ErrorCategory.SERVER_ERROR),
    ],
)
def test_api_core_typed_exceptions_classified(
    monkeypatch: pytest.MonkeyPatch, exc_name: str, expected: ErrorCategory
) -> None:
    exc_mod = _install_fake_api_core(monkeypatch)
    exc = getattr(exc_mod, exc_name)("synthetic")
    assert _client()._classify_error(exc).category is expected


def test_api_core_absent_degrades_to_message_fallback() -> None:
    """Without the package the guarded import must not blow up — a
    structureless exception falls through to the substring path."""
    err = _client()._classify_error(RuntimeError("entirely unstructured"))
    assert err.category is ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Last-resort substring fallback (structureless exceptions only)
# ---------------------------------------------------------------------------


def test_fallback_timeout_substring() -> None:
    err = _client()._classify_error(TimeoutError("timeout waiting for response"))
    assert err.category is ErrorCategory.TIMEOUT


def test_fallback_auth_substring() -> None:
    err = _client()._classify_error(ValueError("api key not provided"))
    assert err.category is ErrorCategory.AUTH


def test_fallback_5xx_checked_before_400_needle() -> None:
    """Even in the fallback, server-side evidence outranks the '400'
    needle — a plain exception describing a 500 that mentions 400 in
    the same breath stays recoverable."""
    err = _client()._classify_error(
        RuntimeError("HTTP 500 from upstream while handling code 400")
    )
    assert err.category is ErrorCategory.SERVER_ERROR


def test_existing_api_error_passes_through() -> None:
    original = APIError("already classified", category=ErrorCategory.NETWORK)
    assert _client()._classify_error(original) is original


# ---------------------------------------------------------------------------
# Provenance (APIResponse.raw)
# ---------------------------------------------------------------------------


def test_parse_response_raw_carries_provenance() -> None:
    import google.genai as genai_sdk

    client = _client()
    fake = SimpleNamespace(candidates=[], usage_metadata=None)
    resp = client._parse_response(fake, "gemini-mock")
    assert resp.raw["provider"] == "google"
    assert resp.raw["sdk_version"] == genai_sdk.__version__
    assert resp.raw["response"] is fake
