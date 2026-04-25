"""Default reviewers for Stage 11: Tool Review (S9b.1).

Each reviewer inspects the pending tool calls (and optionally the
results from the most recent tool execution) and emits zero or more
:class:`ToolReviewFlag` records. The default chain runs them in this
order::

    Schema → Sensitive → Destructive → Network → Size

Reviewers are intentionally simple and conservative. Hosts can plug
their own implementations into the slot chain and override the
defaults. The patterns here are starting points — production
deployments should tune them via host-side config.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review.interface import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
    Reviewer,
    ToolReviewFlag,
)


def _tool_call_id(call: Dict[str, Any]) -> str:
    return str(call.get("id", "") or call.get("tool_use_id", ""))


def _tool_name(call: Dict[str, Any]) -> str:
    return str(call.get("name", "") or "")


def _tool_input(call: Dict[str, Any]) -> Dict[str, Any]:
    raw = call.get("input")
    return dict(raw) if isinstance(raw, dict) else {}


# ── SchemaReviewer ──────────────────────────────────────────────────────


class SchemaReviewer(Reviewer):
    """Flag tool calls whose ``input`` is missing required fields.

    Required fields are looked up by tool name in
    ``required_fields[tool_name]``. Tools with no entry in the map
    are treated as schema-free (no flags emitted).
    """

    def __init__(self, required_fields: Dict[str, List[str]] | None = None) -> None:
        self._required = {
            str(k): tuple(str(f) for f in v) for k, v in (required_fields or {}).items()
        }

    @property
    def name(self) -> str:
        return "schema"

    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        flags: List[ToolReviewFlag] = []
        for call in tool_calls:
            name = _tool_name(call)
            required = self._required.get(name)
            if not required:
                continue
            args = _tool_input(call)
            missing = [f for f in required if f not in args]
            if missing:
                flags.append(
                    ToolReviewFlag(
                        tool_call_id=_tool_call_id(call),
                        reviewer=self.name,
                        severity=SEVERITY_ERROR,
                        reason=f"missing required fields: {', '.join(missing)}",
                        details={"tool": name, "missing": list(missing)},
                    )
                )
        return flags


# ── SensitivePatternReviewer ───────────────────────────────────────────


_DEFAULT_SENSITIVE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # name, regex
    ("api_key_assignment", r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[=:]"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ("bearer_header", r"(?i)bearer\s+[a-z0-9._\-+/=]{16,}"),
)


class SensitivePatternReviewer(Reviewer):
    """Flag tool inputs that look like they contain secrets.

    Patterns are regex-matched against the JSON-serialised tool
    input. Hosts can override the default pattern set via the
    ``patterns`` kwarg (list of ``(label, regex)`` tuples).
    """

    def __init__(self, patterns: List[Tuple[str, str]] | None = None) -> None:
        if patterns is None:
            patterns = list(_DEFAULT_SENSITIVE_PATTERNS)
        self._compiled = [(label, re.compile(rx)) for label, rx in patterns]

    @property
    def name(self) -> str:
        return "sensitive"

    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        import json

        flags: List[ToolReviewFlag] = []
        for call in tool_calls:
            payload = json.dumps(_tool_input(call), default=str, sort_keys=True)
            for label, rx in self._compiled:
                if rx.search(payload):
                    flags.append(
                        ToolReviewFlag(
                            tool_call_id=_tool_call_id(call),
                            reviewer=self.name,
                            severity=SEVERITY_WARN,
                            reason=f"sensitive pattern matched: {label}",
                            details={"tool": _tool_name(call), "pattern": label},
                        )
                    )
                    # Don't double-flag the same call for multiple patterns.
                    break
        return flags


# ── DestructiveResultReviewer ──────────────────────────────────────────


_DEFAULT_DESTRUCTIVE_TOOLS: Tuple[str, ...] = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Write",
    "Edit",
    "MultiEdit",
    "Delete",
    "Remove",
)


class DestructiveResultReviewer(Reviewer):
    """Flag tool *results* whose source tool is known to mutate state.

    Inspects ``tool_results`` (post-execution). Useful for surfacing
    "this turn touched the filesystem" or similar dashboards. Severity
    defaults to ``info`` so the flag is advisory rather than blocking.
    """

    def __init__(
        self,
        destructive_tools: List[str] | None = None,
        *,
        severity: str = SEVERITY_INFO,
    ) -> None:
        self._destructive = frozenset(destructive_tools or _DEFAULT_DESTRUCTIVE_TOOLS)
        self._severity = severity

    @property
    def name(self) -> str:
        return "destructive"

    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        # Build {tool_use_id → tool_name} from this turn's pending calls.
        name_by_id: Dict[str, str] = {}
        for call in tool_calls:
            tcid = _tool_call_id(call)
            if tcid:
                name_by_id[tcid] = _tool_name(call)

        flags: List[ToolReviewFlag] = []
        for result in tool_results:
            tcid = str(result.get("tool_use_id", ""))
            tool_name = name_by_id.get(tcid, "")
            if tool_name and tool_name in self._destructive:
                flags.append(
                    ToolReviewFlag(
                        tool_call_id=tcid,
                        reviewer=self.name,
                        severity=self._severity,
                        reason=f"destructive tool ran: {tool_name}",
                        details={"tool": tool_name},
                    )
                )
        return flags


# ── NetworkAuditReviewer ───────────────────────────────────────────────


_DEFAULT_NETWORK_TOOLS: Tuple[str, ...] = (
    "WebFetch",
    "WebSearch",
    "WebRead",
    "Curl",
    "Http",
)


class NetworkAuditReviewer(Reviewer):
    """Audit tool calls that perform network egress.

    Records an ``info`` flag for each network call so hosts can build
    egress dashboards. Hosts that want a strict allowlist can pass
    ``allowed_hosts`` — calls whose ``input.url`` host falls outside
    the allowlist are flagged ``error`` instead of ``info``.
    """

    def __init__(
        self,
        network_tools: List[str] | None = None,
        *,
        allowed_hosts: List[str] | None = None,
    ) -> None:
        self._network = frozenset(network_tools or _DEFAULT_NETWORK_TOOLS)
        self._allowed = frozenset(h.lower() for h in (allowed_hosts or []))

    @property
    def name(self) -> str:
        return "network"

    @staticmethod
    def _extract_host(url_value: Any) -> str:
        if not isinstance(url_value, str):
            return ""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url_value)
            return (parsed.hostname or "").lower()
        except Exception:  # noqa: BLE001 — parse failures are advisory
            return ""

    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        flags: List[ToolReviewFlag] = []
        for call in tool_calls:
            name = _tool_name(call)
            if name not in self._network:
                continue
            args = _tool_input(call)
            host = self._extract_host(args.get("url"))
            if self._allowed and host and host not in self._allowed:
                flags.append(
                    ToolReviewFlag(
                        tool_call_id=_tool_call_id(call),
                        reviewer=self.name,
                        severity=SEVERITY_ERROR,
                        reason=f"network egress to disallowed host: {host}",
                        details={"tool": name, "host": host, "allowed": sorted(self._allowed)},
                    )
                )
            else:
                flags.append(
                    ToolReviewFlag(
                        tool_call_id=_tool_call_id(call),
                        reviewer=self.name,
                        severity=SEVERITY_INFO,
                        reason=f"network egress: {host or '(unknown host)'}",
                        details={"tool": name, "host": host},
                    )
                )
        return flags


# ── SizeReviewer ───────────────────────────────────────────────────────


class SizeReviewer(Reviewer):
    """Flag tool *results* whose serialised content exceeds a byte limit.

    Severity scales with the breach: between ``warn_threshold_bytes``
    and ``error_threshold_bytes`` → ``warn``; beyond
    ``error_threshold_bytes`` → ``error``.
    """

    def __init__(
        self,
        *,
        warn_threshold_bytes: int = 50_000,
        error_threshold_bytes: int = 250_000,
    ) -> None:
        if warn_threshold_bytes < 0 or error_threshold_bytes < 0:
            raise ValueError("size thresholds must be non-negative")
        if error_threshold_bytes < warn_threshold_bytes:
            raise ValueError("error_threshold_bytes must be >= warn_threshold_bytes")
        self._warn = warn_threshold_bytes
        self._error = error_threshold_bytes

    @property
    def name(self) -> str:
        return "size"

    @staticmethod
    def _content_size(result: Dict[str, Any]) -> int:
        content = result.get("content")
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        if isinstance(content, bytes):
            return len(content)
        # Fallback: stringify.
        return len(str(content).encode("utf-8"))

    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        flags: List[ToolReviewFlag] = []
        for result in tool_results:
            size = self._content_size(result)
            if size < self._warn:
                continue
            severity = SEVERITY_ERROR if size >= self._error else SEVERITY_WARN
            flags.append(
                ToolReviewFlag(
                    tool_call_id=str(result.get("tool_use_id", "")),
                    reviewer=self.name,
                    severity=severity,
                    reason=f"tool result size {size} bytes",
                    details={
                        "bytes": size,
                        "warn_threshold": self._warn,
                        "error_threshold": self._error,
                    },
                )
            )
        return flags


__all__ = [
    "DestructiveResultReviewer",
    "NetworkAuditReviewer",
    "SchemaReviewer",
    "SensitivePatternReviewer",
    "SizeReviewer",
]
