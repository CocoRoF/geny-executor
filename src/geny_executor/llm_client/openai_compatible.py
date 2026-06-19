"""OpenAI-compatible (local) clients generated from :class:`ProviderProfile`.

Each branded local backend (``ollama`` / ``lmstudio`` / ``custom``) is a
thin :class:`~geny_executor.llm_client.openai.OpenAIClient` subclass that
pulls its provider name + capabilities from a profile and adds the
local-backend quirks:

* resolve ``base_url`` from the profile default when the host gives none
  (and raise when the profile requires one and none is resolvable);
* default the API key to ``"EMPTY"`` so ``AsyncOpenAI`` constructs against
  a keyless local server;
* send a ``max_tokens`` floor so Ollama doesn't collapse to
  ``num_predict=128`` and truncate;
* thread ``num_ctx`` / ``think`` into ``extra_body`` (Ollama's native
  context-window + reasoning toggles).

This module is imported lazily (see
:func:`geny_executor.llm_client.profiles.get_profiled_client_class`) so the
SDK path is not pulled merely by registering these providers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any, Dict, Optional

from geny_executor.llm_client.openai import OpenAIClient
from geny_executor.llm_client.profiles import (
    CUSTOM_PROFILE,
    LMSTUDIO_PROFILE,
    OLLAMA_PROFILE,
    ProviderProfile,
)


logger = logging.getLogger(__name__)


# Trailing comma before a closing brace/bracket: ``{"a": 1,}`` / ``[1,]``.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# Bare Python literals a non-OpenAI server may emit instead of JSON ones.
# Word-boundaried so ``None``/``True``/``False`` inside string values are
# left alone as much as a regex can manage (this is a last-resort repair,
# only reached after strict parsing already failed).
_PY_LITERALS_RE = re.compile(r"\b(None|True|False)\b")
_PY_TO_JSON = {"None": "null", "True": "true", "False": "false"}


def _repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort repair of malformed tool-call argument JSON.

    Handles the failure modes seen from local OpenAI-compatible servers
    (Ollama / llama.cpp / GLM-family): markdown code fences, leading or
    trailing prose around the object, trailing commas, and bare Python
    literals (``None`` / ``True`` / ``False``). Returns the parsed object
    on success or ``None`` when it still cannot be salvaged — never raises.
    Only object (``dict``) results are accepted; a repaired scalar/list is
    treated as unsalvageable for a tool-argument slot.
    """
    s = raw.strip()
    if not s:
        return None

    # Strip a ```json … ``` (or bare ```) fence.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # If prose surrounds the object, isolate the outermost {...}.
    first, last = s.find("{"), s.rfind("}")
    candidate = s[first : last + 1] if 0 <= first < last else s

    attempts = [
        candidate,
        _TRAILING_COMMA_RE.sub(r"\1", candidate),
    ]
    # Python-literal fix applied on top of the comma-stripped form.
    attempts.append(_PY_LITERALS_RE.sub(lambda m: _PY_TO_JSON[m.group(0)], attempts[1]))

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class OpenAICompatibleClient(OpenAIClient):
    """Profile-driven OpenAI-compatible client. Subclasses bind ``_profile``."""

    #: Bound by each concrete subclass below. The base class is never
    #: registered/instantiated directly.
    _profile: ProviderProfile

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_sink: Optional[Any] = None,
        num_ctx: Optional[int] = None,
        think: Optional[bool] = None,
    ) -> None:
        profile = self._profile
        resolved_base_url = base_url or profile.default_base_url
        if not resolved_base_url and profile.requires_base_url:
            raise ValueError(
                f"{profile.name!r} requires base_url (the OpenAI-compatible "
                "server endpoint). Example: "
                "base_url='http://localhost:8080/v1'"
            )
        # AsyncOpenAI rejects an empty api_key; local servers are keyless.
        super().__init__(
            api_key=api_key or "EMPTY",
            base_url=resolved_base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._num_ctx = num_ctx
        self._think = think

    def configure_capabilities(self, **overrides: bool) -> None:
        """Upgrade/downgrade capability flags for the deployed model.

        A local endpoint serving a model without tool support opts out::

            client.configure_capabilities(supports_tools=False,
                                           supports_tool_choice=False)

        ``capabilities.drops`` is interpreted against the upgraded flags
        (see ``BaseClient._apply_declared_drops``), so toggling
        ``supports_tools`` really does gate whether ``tools`` reach the
        server — no need to rewrite the drops tuple here.
        """
        # type-ignore: dataclasses.replace stubs can't type heterogeneous
        # **kwargs; the dataclass validates values at construction.
        self.capabilities = replace(self.capabilities, **overrides)  # type: ignore[arg-type]

    def _parse_tool_arguments(self, raw: Any) -> Any:
        """Strict parse, then repair the malformed JSON local servers emit.

        Local OpenAI-compatible backends frequently return tool-call
        arguments that ``json.loads`` rejects (trailing commas, ``None`` /
        ``True`` literals, markdown fences). Rather than silently collapse
        those to ``{}`` — which drops the model's real tool arguments and
        looks like the model "called the tool with nothing" — try a
        conservative repair first. A successful repair is reported (WARNING
        + ``llm_client.tool_args_repaired`` event) so it stays visible
        instead of masking a flaky server.
        """
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        if isinstance(raw, str):
            repaired = _repair_json(raw)
            if repaired is not None:
                self._report_tool_args_repaired(raw)
                return repaired
        return {}

    def _report_tool_args_repaired(self, raw: str) -> None:
        logger.warning(
            "%s: repaired malformed tool-call arguments from a local "
            "server (%d chars). The model's JSON was not strictly valid; "
            "a conservative repair recovered it.",
            self.provider,
            len(raw),
        )
        if self._event_sink is not None:
            self._event_sink(
                {
                    "type": "llm_client.tool_args_repaired",
                    "provider": self.provider,
                    "raw_length": len(raw),
                }
            )

    def _build_kwargs(self, request: Any) -> Dict[str, Any]:
        """OpenAI kwargs + local quirks (token-cap floor, num_ctx, think)."""
        kwargs = super()._build_kwargs(request)

        # Token-cap floor: a missing/zero cap makes Ollama fall back to
        # num_predict=128 and silently truncate. Only fills when neither
        # token kwarg carries a positive value — an explicit small cap is
        # the operator's choice and is honoured.
        floor = self._profile.default_max_tokens
        if floor and not kwargs.get("max_tokens") and not kwargs.get("max_completion_tokens"):
            kwargs["max_tokens"] = floor

        # num_ctx / think → extra_body (Ollama options + reasoning toggle).
        # Emitted only when explicitly configured, so the default request
        # is byte-identical to a plain OpenAI-compatible call and servers
        # that don't read these fields are never sent surprising body keys.
        extra_body: Dict[str, Any] = dict(kwargs.get("extra_body") or {})
        if self._num_ctx is not None:
            options = dict(extra_body.get("options") or {})
            options.setdefault("num_ctx", self._num_ctx)
            extra_body["options"] = options
        if self._think is not None:
            extra_body["think"] = self._think
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs


class OllamaClient(OpenAICompatibleClient):
    """Ollama via its OpenAI-compatible ``/v1`` endpoint."""

    _profile = OLLAMA_PROFILE
    provider = OLLAMA_PROFILE.name
    capabilities = OLLAMA_PROFILE.capabilities


class LMStudioClient(OpenAICompatibleClient):
    """LM Studio local OpenAI-compatible server."""

    _profile = LMSTUDIO_PROFILE
    provider = LMSTUDIO_PROFILE.name
    capabilities = LMSTUDIO_PROFILE.capabilities


class CustomOpenAIClient(OpenAICompatibleClient):
    """Generic OpenAI-compatible endpoint (requires ``base_url``)."""

    _profile = CUSTOM_PROFILE
    provider = CUSTOM_PROFILE.name
    capabilities = CUSTOM_PROFILE.capabilities


#: Primary profile name → generated client class. ``get_profiled_client_class``
#: resolves aliases to a primary name before indexing this map.
CLIENT_CLASSES: Dict[str, type] = {
    OLLAMA_PROFILE.name: OllamaClient,
    LMSTUDIO_PROFILE.name: LMStudioClient,
    CUSTOM_PROFILE.name: CustomOpenAIClient,
}


__all__ = [
    "OpenAICompatibleClient",
    "OllamaClient",
    "LMStudioClient",
    "CustomOpenAIClient",
    "CLIENT_CLASSES",
]
