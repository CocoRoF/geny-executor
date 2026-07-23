"""Stage 3: System — concrete stage implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.interface import PromptBuilder
from geny_executor.stages.s03_system.artifact.default.builders import (
    ComposablePromptBuilder,
    MutablePromptBuilder,
    StaticPromptBuilder,
)
from geny_executor.stages.s03_system.persona import DynamicPersonaPromptBuilder
from geny_executor.tools.registry import ToolRegistry


class SystemStage(Stage[Any, Any]):
    """Stage 3: System.

    Dual abstraction:
      - Level 2 builder: how to construct the system prompt
    """

    def __init__(
        self,
        builder: Optional[PromptBuilder] = None,
        *,
        prompt: str = "",
        template_vars: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        volatile_placement: str = "turn_context",
    ):
        if builder is None:
            builder = StaticPromptBuilder(prompt) if prompt else StaticPromptBuilder()

        self._slots: Dict[str, StrategySlot] = {
            "builder": StrategySlot(
                name="builder",
                strategy=builder,
                registry={
                    "static": StaticPromptBuilder,
                    "mutable": MutablePromptBuilder,
                    "composable": ComposablePromptBuilder,
                    # Phase 7 S7.1 — host-attached PersonaProvider
                    # drives this. Manifests can name it; the actual
                    # provider instance must arrive via
                    # ``Pipeline.attach_runtime(system_builder=...)``.
                    "dynamic_persona": DynamicPersonaPromptBuilder,
                },
                description="System prompt builder strategy",
            ),
        }
        self._tool_registry = tool_registry
        self._prompt = prompt
        self._template_vars: Dict[str, Any] = dict(template_vars or {})
        self._volatile_placement = (
            volatile_placement if volatile_placement in ("turn_context", "system") else "turn_context"
        )
        # Deferred-tool catalog cache (progressive disclosure) — rebuilt
        # only when the registry version moves; see _deferred_catalog_text.
        self._catalog_cache: str = ""
        self._catalog_version: Optional[int] = -1

    @property
    def _builder(self) -> PromptBuilder:
        return self._slots["builder"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "system"

    @property
    def order(self) -> int:
        return 3

    @property
    def category(self) -> str:
        return "ingress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="system",
            fields=[
                ConfigField(
                    name="prompt",
                    type="string",
                    label="System Prompt",
                    description="Static system prompt injected before the conversation.",
                    default="",
                    ui_widget="textarea",
                ),
                ConfigField(
                    name="template_vars",
                    type="object",
                    label="Template Variables",
                    description=(
                        "Key-value pairs substituted into the built system "
                        "prompt: every {name} placeholder is replaced "
                        "post-build, whichever builder produced the prompt. "
                        "Placeholders without a matching key are left intact."
                    ),
                    default={},
                ),
                ConfigField(
                    name="volatile_placement",
                    type="select",
                    label="Volatile block placement",
                    description=(
                        "Where per-turn prompt blocks (clock, retrieved "
                        "memory) go. 'turn_context' (default) keeps them "
                        "out of the system prompt and injects them next to "
                        "the latest user message so the system+history "
                        "prompt-cache prefix stays byte-stable; 'system' "
                        "keeps the pre-2.50 in-system layout."
                    ),
                    default="turn_context",
                    options=[
                        {"value": "turn_context", "label": "Turn context (cache-friendly)"},
                        {"value": "system", "label": "System prompt (legacy)"},
                    ],
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "prompt": self._prompt,
            "template_vars": dict(self._template_vars),
            "volatile_placement": self._volatile_placement,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "prompt" in config:
            prompt = str(config["prompt"])
            self._prompt = prompt
            builder = self._slots["builder"].strategy
            if isinstance(builder, StaticPromptBuilder):
                builder.configure({"prompt": prompt})
        if "template_vars" in config:
            tv = config["template_vars"] or {}
            self._template_vars = dict(tv)
        if "volatile_placement" in config:
            vp = str(config["volatile_placement"])
            if vp in ("turn_context", "system"):
                self._volatile_placement = vp

    def _apply_template_vars(
        self, system: Union[str, List[Dict[str, Any]]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Substitute ``{name}`` placeholders into the built prompt.

        Why post-build instead of a builder kwarg (2.2.0 wave 4, config
        liveness): the :class:`PromptBuilder` contract is ``build(state)``
        — adding a ``template_vars`` parameter would break every custom
        builder hosts attach via ``Pipeline.attach_runtime``. Substituting
        on the *output* keeps the contract intact and works uniformly for
        static, composable and persona builders.

        Substitution is a literal ``{key}`` → ``str(value)`` replacement,
        NOT ``str.format``: prompts routinely contain literal braces (JSON
        examples, code snippets) that ``format`` would choke on. Unknown
        placeholders are left untouched.
        """

        def _substitute(text: str) -> str:
            for key, value in self._template_vars.items():
                text = text.replace("{" + str(key) + "}", str(value))
            return text

        if isinstance(system, str):
            return _substitute(system)
        if isinstance(system, list):
            blocks: List[Dict[str, Any]] = []
            for block in system:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block = {**block, "text": _substitute(block["text"])}
                blocks.append(block)
            return blocks
        return system

    def _assemble_system(self, state: PipelineState) -> tuple[Any, str]:
        """Build the system prompt, separating the volatile tail.

        TTFT program (2.50.0): provider prompt caches (Anthropic
        cache_control, OpenAI/vLLM automatic prefix caching) key on a
        byte-stable request prefix. Per-turn blocks (clock, retrieved
        memory) rendered INTO the system prompt used to re-prefill
        system + all history every turn. When the builder can expose its
        stable/volatile structure via ``build_parts``, the volatile tail
        is pulled out here and either

        - ``turn_context`` (default): handed to Stage 6, which attaches
          it next to the latest user message at request-build time —
          never persisted to history, always after every cache
          breakpoint; or
        - ``system``: kept in the system string (legacy layout), with
          the split recorded in ``state.shared['system_parts']`` so the
          Stage 5 cache strategy can place its breakpoint before it.

        Returns ``(system, volatile_text)`` where ``volatile_text`` is
        only non-empty in ``turn_context`` mode.
        """
        parts = None
        build_parts = getattr(self._builder, "build_parts", None)
        if callable(build_parts):
            try:
                parts = build_parts(state)
            except Exception:  # noqa: BLE001 — structure is an optimization, never a failure
                parts = None

        if not parts:
            system = self._builder.build(state)
            if self._template_vars:
                system = self._apply_template_vars(system)
            state.shared.pop("system_parts", None)
            return system, ""

        texts: List[str] = []
        for part in parts:
            text = str(part.get("text", ""))
            if self._template_vars:
                text = self._apply_template_vars(text)  # type: ignore[assignment]
            texts.append(text)

        # Split at the FIRST volatile part: everything before it is the
        # cacheable prefix; everything from it on (including any stable
        # block ordered after a volatile one — prefix caching can't reach
        # past the first changed byte anyway) is the volatile tail.
        first_volatile = next(
            (i for i, part in enumerate(parts) if part.get("volatile")),
            len(parts),
        )
        stable_text = "\n\n".join(t for t in texts[:first_volatile] if t)
        volatile_text = "\n\n".join(t for t in texts[first_volatile:] if t)

        if not volatile_text:
            state.shared.pop("system_parts", None)
            return stable_text, ""

        if self._volatile_placement == "system":
            joined = "\n\n".join(t for t in (stable_text, volatile_text) if t)
            state.shared["system_parts"] = {
                "stable_text": stable_text,
                "volatile_text": volatile_text,
            }
            return joined, ""

        state.shared.pop("system_parts", None)
        return stable_text, volatile_text

    # ── Deferred-tool catalog (progressive disclosure, tier 0) ─────────
    #
    # Only exposed tools ship schemas, so without help the model cannot
    # know what ELSE exists — and you can't ToolSearch for a tool you've
    # never heard of. This appends a compact, cache-stable catalog of the
    # DEFERRED tools (name + first-line one-liner) plus a one-line usage
    # rule to the system prompt. Derived from the registry's *core flag*
    # (not the live activation state), so the text does not change when a
    # ToolSearch activation happens mid-session — the prompt-cache prefix
    # stays intact. Rebuilt only when the registry version moves
    # (register/unregister/MCP re-seed), the same trigger that already
    # rebuilds ``state.tools``.
    _CATALOG_ONE_LINER_CHARS = 72
    _CATALOG_MAX_CHARS = 4_000

    def _deferred_catalog_text(self) -> str:
        registry = self._tool_registry
        if registry is None:
            return ""
        is_core = getattr(registry, "is_core", None)
        get = getattr(registry, "get", None)
        list_names = getattr(registry, "list_names", None)
        if not (callable(is_core) and callable(get) and callable(list_names)):
            return ""
        entries: List[Tuple[str, str]] = []
        for name in sorted(list_names()):
            try:
                if is_core(name):
                    continue
                tool = get(name)
                desc = str(getattr(tool, "description", "") or "").strip()
                line = desc.splitlines()[0] if desc else ""
                if len(line) > self._CATALOG_ONE_LINER_CHARS:
                    line = line[: self._CATALOG_ONE_LINER_CHARS - 1] + "…"
                entries.append((name, line))
            except Exception:  # noqa: BLE001 — a broken tool never breaks the prompt
                continue
        if not entries:
            return ""
        header = (
            "## Additional tools (hidden — not in your tool list)\n"
            f"{len(entries)} more tools exist. To use one, call "
            'ToolSearch("<keyword or exact name>") — its schema arrives on '
            "your next step. ToolSearch with no query browses this catalog."
        )
        lines = [f"- {n} — {d}" if d else f"- {n}" for n, d in entries]
        body = "\n".join(lines)
        if len(header) + len(body) > self._CATALOG_MAX_CHARS:
            # Degrade gracefully: names only on one wrapped line.
            body = ", ".join(n for n, _ in entries)[: self._CATALOG_MAX_CHARS]
        return header + "\n" + body

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Build system prompt (stable prefix + volatile tail separation)
        system, volatile_text = self._assemble_system(state)

        # Progressive disclosure: make the hidden catalog discoverable.
        # Cache-stable text (see _deferred_catalog_text) appended AFTER the
        # builder output; rebuilt only on registry-version change.
        if self._tool_registry is not None:
            reg_version = getattr(self._tool_registry, "version", None)
            if self._catalog_version != reg_version:
                self._catalog_cache = self._deferred_catalog_text()
                self._catalog_version = reg_version
            catalog = self._catalog_cache
            if catalog:
                if isinstance(system, str):
                    parts_rec = state.shared.get("system_parts")
                    if (isinstance(parts_rec, dict)
                            and parts_rec.get("stable_text")
                            and parts_rec.get("volatile_text")):
                        # volatile_placement="system": the catalog is
                        # CACHE-STABLE, so it belongs in the stable region —
                        # BEFORE the volatile tail. Appending it after the
                        # joined string used to break the Stage-5 split
                        # (`system == stable + "\n\n" + volatile` no longer
                        # held), which silently caching the volatile tail too
                        # → a full system re-prefill every turn.
                        stable = parts_rec["stable_text"]
                        stable = f"{stable}\n\n{catalog}" if stable else catalog
                        parts_rec["stable_text"] = stable
                        system = f"{stable}\n\n{parts_rec['volatile_text']}"
                    else:
                        system = (system + "\n\n" + catalog) if system else catalog
                elif isinstance(system, list):
                    system = [*system, {"type": "text", "text": catalog}]

        state.system = system
        if volatile_text:
            state.shared["turn_context_text"] = volatile_text
        else:
            state.shared.pop("turn_context_text", None)

        # Register tools in state if registry provided. Snapshotted on the first
        # turn, then rebuilt only when the live registry's version moves — so a
        # tool/skill enabled/disabled/created mid-session (self-modifying
        # environment), an MCP re-seed, or a ToolSearch activation takes effect
        # on the next iteration. The version guard keeps the steady-state cost
        # at one int compare per turn.
        #
        # Only *exposed* tools ship to the model: core tools plus deferred
        # tools a ToolSearch hit activated. Deferred tools stay registered
        # (Stage 10 can still dispatch them) but their schemas stay out of
        # the request payload until discovered — that's the token contract.
        if self._tool_registry is not None:
            reg_version = getattr(self._tool_registry, "version", None)
            if not state.tools or (
                reg_version is not None and reg_version != state.tools_version
            ):
                try:
                    state.tools = self._tool_registry.to_api_format(exposed_only=True)
                except TypeError:
                    # Host passed a registry-alike without exposure support.
                    state.tools = self._tool_registry.to_api_format()
                if reg_version is not None:
                    state.tools_version = reg_version

        state.add_event(
            "system.built",
            {
                "prompt_type": "content_blocks" if isinstance(system, list) else "string",
                "prompt_length": (
                    sum(len(b.get("text", "")) for b in system)
                    if isinstance(system, list)
                    else len(str(system))
                ),
                "tools_count": len(state.tools),
                "volatile_placement": self._volatile_placement,
                "turn_context_chars": len(volatile_text),
            },
        )

        return input
