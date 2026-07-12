"""Pipeline state — the mutable context flowing through all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None

    @property
    def total_tokens(self) -> int:
        # Deliberately input + output only. Cache tokens can't be folded
        # in provider-agnostically: Anthropic's ``input_tokens`` EXCLUDES
        # cache reads (so they'd be additive) while OpenAI's INCLUDES them
        # (so adding cache_read double-counts). Without the provider here
        # there is no safe sum, so we report the unambiguous base and let
        # cache accounting live on the explicit cache_* fields.
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cost_usd = _sum_optional(self.cost_usd, other.cost_usd)
        self.duration_ms = _sum_optional(self.duration_ms, other.duration_ms)
        return self

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
            cost_usd=_sum_optional(self.cost_usd, other.cost_usd),
            duration_ms=_sum_optional(self.duration_ms, other.duration_ms),
        )


def _sum_optional(a, b):  # type: ignore[no-untyped-def]
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


@dataclass
class CacheMetrics:
    """Prompt caching efficiency metrics."""

    total_cache_writes: int = 0
    total_cache_reads: int = 0
    estimated_savings_usd: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class PipelineState:
    """Pipeline execution state — readable/writable by all stages.

    Lifecycle contract (2.2.0, audit 2026-06-09 §3.3)
    -------------------------------------------------
    A state object may serve a single ``run()`` (the "fresh state per
    turn" model — Geny) or be reused across many runs (the "long-lived
    state" model — GAPT). Both are supported. One *run* of the pipeline
    is one conversational **turn**; the agent loop may iterate many
    times inside that turn. Every field falls into exactly one of three
    lifetime classes, and :meth:`begin_turn` is the single place the
    per-turn class is reset — :meth:`Pipeline._init_state` calls it
    automatically when it detects a reused state, so hosts holding a
    state across turns get a clean turn boundary without doing anything.

    **Per-turn (reset by** :meth:`begin_turn` **):**
    ``iteration``, ``current_stage``, ``stage_history``,
    ``loop_decision``, ``completion_signal``, ``completion_detail``,
    ``final_text``, ``final_output``, ``pending_tool_calls``,
    ``tool_results``, ``delegate_requests``, ``agent_results``,
    ``evaluation_score``, ``evaluation_feedback``, ``events``,
    ``turn_token_usage``, ``total_cost_usd``, ``last_api_response``.
    Earlier releases never reset these, so a long-lived state carried a
    prior turn's ``loop_decision="error"`` into the next turn's success
    verdict and accumulated ``iteration`` until MAX_ITERATIONS fired
    spuriously — the exact GAPT incident class this contract closes.
    ``events``/``stage_history`` reset per turn because unbounded growth
    was the audit complaint; ``run_stream`` re-emits everything live, so
    hosts that want a session-long event log already have one.

    **Sticky (survive across turns until the host changes them):**
    ``session_id``, ``pipeline_id``, ``system``, ``messages``,
    ``shared``, ``metadata``, ``memory_refs``, ``thinking_history``,
    ``cache_metrics``, ``llm_client``, ``credentials``,
    ``subagent_registry``, ``session_runtime``, ``tools``,
    ``tool_choice``, and every model/limit knob stomped by
    ``PipelineConfig.apply_to_state`` at each run start (``model``,
    ``max_tokens``, ``temperature``, …, ``max_iterations``,
    ``cost_budget_usd``). NOTE: earlier docstrings claimed ``shared``
    "resets to {} at the start of each run" — that reset never existed
    in code and hosts now rely on cross-turn persistence (e.g. Geny
    creature state), so the *documentation* was fixed, not the
    behaviour.

    **Session-cumulative (grow for the lifetime of the state object):**
    ``token_usage`` (sums every API call ever made on this state) and
    ``session_cost_usd`` (each turn's ``total_cost_usd`` is folded in at
    turn end). ``total_cost_usd`` itself is the PER-TURN accumulator —
    it is what ``cost_budget_usd`` guards (Stage 4 / Stage 16 / the
    pipeline's hard limit) compare against, so budgets bound a single
    turn, not the whole session. Use ``session_cost_usd`` for
    session-level spend reporting.

    Two free-form buckets are available for stage-authored data:

    - ``shared`` — cross-stage communication. Any stage may read and write.
      Persists across turns on a reused state (see lifetime table above).
      Keys are free-form strings; writers and readers cooperate by
      convention (see :class:`~geny_executor.core.shared_keys.SharedKeys`).
      Not an event channel — use :meth:`add_event` for that.
    - ``metadata`` — general scratch. Historically used for pipeline
      signals (``needs_reflection``, ``L0_tail``, ``cost_breakdown``) and
      backs :meth:`Stage.local_state`. New cross-stage data should prefer
      ``shared``; ``metadata`` remains for per-stage bookkeeping and legacy
      signals. Persists across turns.

    Within a single loop turn, stages run sequentially, so ``shared`` has one
    writer at a time. If a future cycle introduces parallel sub-stages,
    readers/writers of the same key will need to add explicit coordination.
    """

    # ── Identity ──
    session_id: str = ""
    pipeline_id: str = ""

    # ── Messages (Anthropic API format) ──
    system: Union[str, List[Dict[str, Any]]] = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # ── Execution tracking ──
    iteration: int = 0
    max_iterations: int = 50
    current_stage: str = ""
    stage_history: List[str] = field(default_factory=list)

    # ── Behavior (from PipelineConfig) ──
    stream: bool = True
    single_turn: bool = False

    # ── Model config ──
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    # Self-modifying environment: the ToolRegistry.version baked into the
    # current ``tools`` snapshot. Stage 3 rebuilds ``tools`` whenever the live
    # registry's version moves past this (a tool/skill enabled/disabled/created
    # mid-session, or an MCP re-seed). ``-1`` = no snapshot yet (first turn).
    tools_version: int = -1

    # ── Extended Thinking ──
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000
    thinking_type: str = "enabled"  # "enabled" | "disabled" | "adaptive"
    thinking_display: Optional[str] = None  # "summarized" | "omitted" | None
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)

    # ── Token & Cost tracking ──
    # token_usage: session-cumulative — sums every API call made on this
    # state object, across turns. turn_token_usage / total_cost_usd:
    # per-turn — reset by begin_turn(); total_cost_usd is the accumulator
    # the cost_budget_usd guards read, so budgets bound one turn.
    # session_cost_usd: session-cumulative — the pipeline folds each
    # turn's total_cost_usd into it at turn end (2.2.0, audit §3.3).
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_token_usage: List[TokenUsage] = field(default_factory=list)
    total_cost_usd: float = 0.0
    session_cost_usd: float = 0.0
    cost_budget_usd: Optional[float] = None

    # ── Cache tracking ──
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)

    # ── Context ──
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    context_window_budget: int = 200_000

    # ── Loop control ──
    loop_decision: str = "continue"  # continue | complete | error | escalate
    completion_signal: Optional[str] = None
    completion_detail: Optional[str] = None

    # ── Tool execution ──
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # ── Tool dispatch handle (2.3.0) ──
    # Installed by ``Pipeline._init_state`` whenever a Tool stage is
    # registered: a :class:`~geny_executor.stages.s10_tool.dispatcher.
    # ToolDispatcher` wrapping that stage's registry + permission
    # machinery. Consumed by Stage 6's ``tool_loop="internal"`` strategy
    # so internally-resolved tool calls take the EXACT dispatch path
    # Stage 10 would. Lifetime: refreshed every run (sticky across the
    # run only); never serialized.
    tool_dispatcher: Optional[Any] = field(default=None, repr=False, compare=False)

    # ── Agent orchestration ──
    delegate_requests: List[Dict[str, Any]] = field(default_factory=list)
    agent_results: List[Dict[str, Any]] = field(default_factory=list)

    # ── Evaluation ──
    evaluation_score: Optional[float] = None
    evaluation_feedback: Optional[str] = None

    # ── Output ──
    final_text: str = ""
    final_output: Optional[Any] = None

    # ── Raw API response (for debugging/passthrough) ──
    last_api_response: Optional[Any] = None

    # ── Metadata (free-form per-run scratch; legacy signals + stage-local storage) ──
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Shared (cross-stage communication for one run) ──
    shared: Dict[str, Any] = field(default_factory=dict)

    # ── Event log ──
    events: List[Dict[str, Any]] = field(default_factory=list)

    # ── Event listener (legacy per-state callback) ──
    # Pre-2.2.0 this was how run_stream saw state events; the pipeline
    # no longer installs it (the bus bridge below replaced it) but the
    # hook is kept functional for hosts that set it directly.
    _event_listener: Optional[Any] = field(default=None, repr=False)

    # ── Event bus bridge (set by Pipeline._init_state each run) ──
    # Callable[[dict], None] that forwards every add_event() dict into
    # the owning pipeline's event channel (journal → events() taps →
    # EventBus). This is the 2.2.0 channel unification (audit §3.2):
    # before it, ``pipeline.on('*')`` subscribers never saw text.delta /
    # api.* events because state and bus were disjoint worlds. The
    # bridge closure carries the per-run correlation ids; a state reused
    # across pipelines is re-pointed at the current owner every run.
    _bus_emitter: Optional[Any] = field(default=None, repr=False)

    # ── Run correlation id (set by Pipeline._init_state each run) ──
    # uuid hex minted per run()/run_stream() invocation; stamped onto
    # every PipelineEvent this state produces so multi-session hosts
    # can attribute events without wrapping emit sites. Private: the
    # pipeline owns it; hosts read it off events, not the state.
    _run_id: str = field(default="", repr=False)

    # ── Deferred run-start events (set by Pipeline._init_state) ──
    # (event_type, data) pairs queued by THIS run's per-run overrides
    # (``config.override_applied``) and flushed by the owning run's
    # ``_run_phases``. Per-state on purpose (2.2.0 review B2): a
    # pipeline-global FIFO was flushed by whichever run started next,
    # so overlapping runs on a shared pipeline misattributed override
    # events to the wrong run_id. Overlapping runs each get their own
    # state, hence their own queue. Private: the pipeline owns it.
    _pending_run_events: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list, repr=False)

    # ── Turn accounting (set by Pipeline._init_state) ──
    # Number of runs this state object has entered. >0 marks the state
    # as "reused" so the next run knows to call begin_turn(). Private:
    # hosts should not write it.
    _run_count: int = field(default=0, repr=False)

    # ── Client generation (set by Pipeline._init_state) ──
    # Records Pipeline._client_generation at the moment the pipeline
    # resolved llm_client INTO this state. None means the client (if
    # any) was placed here by the host directly and the pipeline must
    # never clobber it. A mismatch against the pipeline's current
    # generation means Pipeline.invalidate_client() ran since capture
    # (credential rotation / provider swap) → _init_state re-resolves.
    _client_generation: Optional[int] = field(default=None, repr=False)

    # ── LLM client (injected by Pipeline.attach_runtime; None for non-LLM pipelines) ──
    llm_client: Optional[Any] = field(default=None, repr=False)

    # ── Credential bundle (set by Pipeline._init_state from the bundle
    # passed to from_manifest_async). Stage code uses this for
    # ``provider_override`` to build a stage-local client of a different
    # provider. ``None`` when the pipeline was not constructed via
    # ``from_manifest`` (manual ``register_stage`` style). ──
    credentials: Optional[Any] = field(default=None, repr=False)

    # ── Sub-agent registry (set by Pipeline._init_state from the value
    # passed to ``Pipeline.attach_runtime(subagent_registry=...)``). The
    # Stage 12 ``subagent_type`` orchestrator consumes this; factories
    # may also reach it for nested wiring (intentionally rare). ──
    subagent_registry: Optional[Any] = field(default=None, repr=False)

    # ── Plugin-supplied session runtime container (v0.30.0) ──
    # Free-shape carrier for host-side session-scoped objects (e.g.
    # creature state, persona providers, emitter chains). The executor
    # does not constrain its type — the host (or third-party plugin)
    # decides what attributes live on it. Stages that read from it
    # should duck-type and tolerate ``None`` for hosts that never
    # attach one. See ``Pipeline.attach_runtime(session_runtime=...)``.
    session_runtime: Optional[Any] = field(default=None, repr=False)

    def begin_turn(self) -> None:
        """Reset every per-turn field for a new conversational turn.

        Called automatically by ``Pipeline._init_state`` when a reused
        state (prior messages or a prior run on record) enters
        ``run()`` / ``run_stream()`` — hosts holding a long-lived state
        do not need to call this themselves, but MAY call it manually
        when constructing a state from a checkpoint whose per-turn
        fields should not leak into the next run.

        Why this exists (audit 2026-06-09 §3.3): the class docstring
        used to *claim* a per-run reset that no code performed. A
        long-lived state therefore carried ``loop_decision="error"``
        from a failed turn into the next turn's result (poisoning its
        ``success``), kept ``iteration`` climbing toward
        MAX_ITERATIONS across turns, and grew ``events`` without
        bound. This method is the reset contract, and the class
        docstring's lifetime table is the authoritative list of what
        it touches.

        Sticky and session-cumulative fields (``messages``, ``shared``,
        ``metadata``, ``token_usage``, ``session_cost_usd``, clients,
        registries) are deliberately NOT touched — conversation
        continuity is the whole point of reusing a state.
        """
        # Loop machinery
        self.iteration = 0
        self.current_stage = ""
        self.stage_history = []
        self.loop_decision = "continue"
        self.completion_signal = None
        self.completion_detail = None
        # Output of the previous turn
        self.final_text = ""
        self.final_output = None
        self.last_api_response = None
        # In-flight tool / agent work. Stale pending_tool_calls from a
        # turn that died mid-loop would otherwise be re-executed by
        # Stage 10 at the start of the next turn — a silent replay of
        # side-effectful tools.
        self.pending_tool_calls = []
        self.tool_results = []
        self.delegate_requests = []
        self.agent_results = []
        # Per-turn judgments
        self.evaluation_score = None
        self.evaluation_feedback = None
        # Per-turn accounting. total_cost_usd is the budget-guard
        # accumulator; the previous turn's value was already folded
        # into session_cost_usd at turn end by the pipeline.
        self.turn_token_usage = []
        self.total_cost_usd = 0.0
        # Event log — per-turn to bound growth (run_stream re-emits
        # everything live; PipelineResult.events carries this turn's).
        self.events = []

        # Bound sticky per-session lists so a very long-lived reused state
        # can't grow without limit (audit C2). Caps are generous — a
        # normal session never reaches them; they only clip pathological
        # accumulation (a months-old always-on VTuber session).
        _MAX_STICKY = 2000
        if len(self.thinking_history) > _MAX_STICKY:
            del self.thinking_history[:-_MAX_STICKY]
        if len(self.memory_refs) > _MAX_STICKY:
            del self.memory_refs[:-_MAX_STICKY]
        hitl = self.shared.get("hitl_history")
        if isinstance(hitl, list) and len(hitl) > _MAX_STICKY:
            del hitl[:-_MAX_STICKY]

        # Drop per-turn transient scratch keys from shared so they never
        # leak across turns (audit C2 / R2): the TTFT anchor + latch, the
        # prompt-token memo, the volatile turn-context text, and the
        # per-turn tool-call counter are all recomputed each turn.
        for _k in (
            "_api_call_t0",
            "_api_ttft_emitted",
            "_prompt_tokens_memo",
            "turn_context_text",
            "system_parts",
            "executor.tool_calls_total",
        ):
            self.shared.pop(_k, None)

    def add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Append an event to the log and forward it to the pipeline.

        Event names should come from the published catalogue
        (:class:`geny_executor.events.EventTypes`) — a grep-driven test
        keeps the catalogue in lockstep with the emit sites.

        Forwarding order (2.2.0): the bus bridge installed by
        ``Pipeline._init_state`` runs first (journal → ``events()``
        taps → ``EventBus`` subscribers incl. ``run_stream``), then the
        legacy per-state ``_event_listener`` if a host set one. Both
        are best-effort observability hooks; neither can fail the
        emitting stage.
        """
        event_dict = {
            "type": event_type,
            "stage": self.current_stage,
            "iteration": self.iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        self.events.append(event_dict)
        self.updated_at = datetime.now(timezone.utc)

        # Forward into the pipeline event channel (2.2.0 unification).
        if self._bus_emitter is not None:
            try:
                self._bus_emitter(event_dict)
            except Exception:  # noqa: BLE001 — observability must not kill the run
                import logging

                logging.getLogger(__name__).warning(
                    "state event bridge failed for %s (ignored)", event_type, exc_info=True
                )

        # Forward to the legacy per-state listener (host-installed).
        if self._event_listener is not None:
            self._event_listener(event_dict)

    def add_message(self, role: str, content: Any) -> None:
        """Append a message in Anthropic API format."""
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_use_id: str, content: Any, is_error: bool = False) -> None:
        """Append a tool result message."""
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        self.tool_results.append(result)

    def accumulate_cost(self, cost_usd: float) -> None:
        """Add cost to the per-turn running total (``total_cost_usd``).

        The pipeline folds the per-turn total into the
        session-cumulative ``session_cost_usd`` at turn end — callers
        only ever add to the turn accumulator.
        """
        self.total_cost_usd += cost_usd

    @property
    def is_over_budget(self) -> bool:
        """Check if the PER-TURN cost budget is exceeded.

        Compares ``total_cost_usd`` (per-turn accumulator) against
        ``cost_budget_usd`` — budgets bound a single turn, not the
        session (see class docstring lifetime table).
        """
        if self.cost_budget_usd is None:
            return False
        return self.total_cost_usd >= self.cost_budget_usd

    @property
    def is_over_iterations(self) -> bool:
        """Check if max iterations is exceeded."""
        return self.iteration >= self.max_iterations
