"""WebFetch — pull an HTTP(S) URL and return its text body.

Cycle 20260424 executor uplift — Phase 3 Week 5.

Fetches a URL with an async HTTPX client, strips HTML markup down to a
plain text body, and returns a bounded snippet suitable for an LLM's
context window. Safe to fan out (``concurrency_safe=True``) — the tool
performs a read-only network round-trip and makes no local state
changes.

Limits that keep this tool "LLM-safe" without an explicit budget yet:

* Redirect chain capped at 5 hops.
* Response body capped at ``max_bytes`` (default 1 MiB). Anything beyond
  is truncated with a marker so the model sees the notice.
* Plain-text output capped at ``max_chars`` (default 80 000 — sits
  below the 100 000-char ``max_result_chars`` default so persistence
  only kicks in for pathological pages).
* Timeout defaults to 30 seconds, overridable per call.

Fetch failures surface as a ``ToolResult`` with ``is_error=True``; a
compact one-line message is returned to the model so it can decide
whether to retry with a different URL.

HTML → text strategy is intentionally stdlib-only (``html.parser``) so
no new dependencies ship with 0.33.x. Future work can add a
``markdownify`` opt-in for richer output.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 4-5 Web 계열).
"""

from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 1_048_576  # 1 MiB
_DEFAULT_MAX_CHARS = 80_000
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_USER_AGENT = "geny-executor-webfetch/1.0"

# Tags whose textual content should never reach the model — they add
# noise (script source, style rules) and occasionally leak secrets.
_STRIP_TAGS = {"script", "style", "noscript", "template", "svg"}

# Tags that should introduce a newline in the rendered text so
# paragraphs / list items / headings stay visually separated.
_BLOCK_TAGS = {
    "p",
    "br",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
    "main",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ul",
    "ol",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
    "blockquote",
    "pre",
    "hr",
}


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain-text converter.

    * Skips the contents of ``<script>`` / ``<style>`` / ``<noscript>``
      / ``<template>`` / ``<svg>``.
    * Inserts a newline after block-level tags so paragraphs and list
      items stay separated.
    * Decodes HTML entities.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0
        self._title: Optional[str] = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in _STRIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _STRIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title and self._title is None:
            self._title = data.strip()
        self._parts.append(data)

    @property
    def title(self) -> Optional[str]:
        return self._title

    def render(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of whitespace inside each line while keeping
        # paragraph breaks.
        lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
        # Drop empty lines that now cluster together; leave one blank
        # line between paragraphs for readability.
        rendered: List[str] = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank and rendered:
                    rendered.append("")
                prev_blank = True
            else:
                rendered.append(line)
                prev_blank = False
        return "\n".join(rendered).strip()


def _extract_text(body: str, content_type: str) -> Tuple[str, Optional[str]]:
    """Return ``(text, title)`` from a raw response body.

    Non-HTML content types are returned as-is (trimmed) with no title.
    """
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct and not (
        ct.startswith("text/html") or ct.endswith("+xml") or ct == "application/xhtml+xml"
    ):
        return body.strip(), None

    extractor = _TextExtractor()
    try:
        extractor.feed(body)
        extractor.close()
    except Exception as exc:
        logger.warning("WebFetch HTML parse failed: %s — returning raw body", exc)
        return body.strip(), None

    title = extractor.title
    if title is not None:
        title = html.unescape(title).strip() or None
    return extractor.render(), title


def _validate_url(raw: str) -> str:
    """Normalise + guard the URL against obvious misuse.

    * Empty / whitespace-only → error.
    * Missing scheme → prepend ``https://``.
    * Scheme must be http or https — ``file://`` / ``ftp://`` / data
      URIs are rejected so WebFetch can't be coaxed into reading local
      files or sending credentials to arbitrary endpoints.
    """
    if not raw or not raw.strip():
        raise ValueError("url must not be empty")
    url = raw.strip()
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"URL missing host: {raw!r}")
    return url


class WebFetchTool(Tool):
    """Fetch a URL and return its text body.

    Returns a ``ToolResult.content`` that starts with a ``Fetched: <url>``
    header, followed by the optional HTML ``<title>`` and the extracted
    text. Useful for quickly reading a doc page, README, or small API
    response without spinning up a full browser automation tool.
    """

    @property
    def name(self) -> str:
        return "WebFetch"

    @property
    def description(self) -> str:
        return (
            "Fetch an HTTP(S) URL and return its plain-text body. "
            "HTML is stripped to text; other content types are returned "
            "as-is. Follows up to 5 redirects. Use this for reading "
            "documentation pages, README files, or small API responses."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Absolute URL to fetch. Schemes http:// and https:// "
                        "only. If no scheme is given, https:// is assumed."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": f"Request timeout in seconds. Default {_DEFAULT_TIMEOUT}.",
                    "exclusiveMinimum": 0,
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        f"Cap on the text returned to the model. Default "
                        f"{_DEFAULT_MAX_CHARS}. Excess is truncated with a "
                        f"marker."
                    ),
                    "exclusiveMinimum": 0,
                },
                "max_bytes": {
                    "type": "integer",
                    "description": (
                        f"Cap on the raw response bytes read. Default "
                        f"{_DEFAULT_MAX_BYTES}. Protects against very "
                        f"large downloads."
                    ),
                    "exclusiveMinimum": 0,
                },
                "headers": {
                    "type": "object",
                    "description": (
                        "Optional request headers (e.g. Accept, "
                        "User-Agent). Supplied in addition to the "
                        "executor's default User-Agent."
                    ),
                },
            },
            "required": ["url"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Read-only network fetch — safe to parallelise, performs egress.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,  # server may return different bodies on refetch
            network_egress=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            url = _validate_url(input.get("url", ""))
        except ValueError as exc:
            return ToolResult(content=f"invalid URL: {exc}", is_error=True)

        timeout = float(input.get("timeout", _DEFAULT_TIMEOUT))
        max_chars = int(input.get("max_chars", _DEFAULT_MAX_CHARS))
        max_bytes = int(input.get("max_bytes", _DEFAULT_MAX_BYTES))
        user_headers = input.get("headers") or {}

        request_headers: Dict[str, str] = {
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
        }
        for k, v in user_headers.items():
            if isinstance(k, str) and isinstance(v, str):
                request_headers[k] = v

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=timeout,
            ) as client:
                response = await client.get(url, headers=request_headers)
        except httpx.TimeoutException:
            return ToolResult(
                content=f"timeout after {timeout}s fetching {url}",
                is_error=True,
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                content=f"network error fetching {url}: {exc}",
                is_error=True,
            )

        if response.status_code >= 400:
            return ToolResult(
                content=(
                    f"HTTP {response.status_code} {response.reason_phrase} for {response.url}"
                ),
                is_error=True,
                metadata={
                    "status_code": response.status_code,
                    "final_url": str(response.url),
                },
            )

        raw = response.content or b""
        truncated_bytes = False
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated_bytes = True

        # Decode with the response's detected encoding; fall back to
        # utf-8 with replacement rather than crashing.
        encoding = response.encoding or "utf-8"
        try:
            body = raw.decode(encoding, errors="replace")
        except LookupError:
            body = raw.decode("utf-8", errors="replace")

        content_type = response.headers.get("content-type", "")
        text, title = _extract_text(body, content_type)

        truncated_chars = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated_chars = True

        header_lines = [f"Fetched: {response.url}"]
        if title:
            header_lines.append(f"Title: {title}")
        if str(response.url) != url:
            header_lines.append(f"Redirected from: {url}")
        if truncated_bytes:
            header_lines.append(f"[body truncated at {max_bytes} bytes]")
        if truncated_chars:
            header_lines.append(f"[text truncated at {max_chars} chars]")

        return ToolResult(
            content="\n".join(header_lines) + "\n\n" + text,
            metadata={
                "status_code": response.status_code,
                "final_url": str(response.url),
                "content_type": content_type,
                "text_chars": len(text),
                "truncated_bytes": truncated_bytes,
                "truncated_chars": truncated_chars,
            },
        )
