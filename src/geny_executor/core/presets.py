"""Pipeline presets — pre-configured pipeline patterns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from geny_executor.core.environment import EnvironmentManager

logger = logging.getLogger(__name__)

#: Entry-point group scanned for third-party presets. A package publishes
#: ``[project.entry-points."geny_executor.presets"]`` entries where each value
#: resolves to a callable returning a :class:`Pipeline`.
PRESET_ENTRY_POINT_GROUP = "geny_executor.presets"


@dataclass
class PresetInfo:
    """Metadata about a preset (built-in / user / plugin)."""

    name: str
    description: str = ""
    preset_type: str = "built_in"  # "built_in" | "user" | "plugin"
    tags: List[str] = field(default_factory=list)
    environment_id: Optional[str] = None


@dataclass
class _PresetRecord:
    """Internal record bound to a registered preset factory."""

    name: str
    factory: Callable[..., Pipeline]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = "plugin"  # "plugin" | "built_in"


class PresetRegistry:
    """Global registry for pipeline presets contributed by plugins.

    Plugins register presets either programmatically via
    :func:`register_preset` or declaratively through the
    ``geny_executor.presets`` entry-point group.

    The registry is thread-safe and idempotent: re-registering the same
    name replaces the prior factory and logs a debug message.
    """

    _lock = RLock()
    _records: Dict[str, _PresetRecord] = {}
    _discovered: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., Pipeline],
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
        source: str = "plugin",
    ) -> None:
        """Register *factory* under *name*.

        Raises ``ValueError`` if *name* is empty.
        """
        if not name:
            raise ValueError("Preset name must be non-empty")
        resolved_desc = description
        if not resolved_desc and factory.__doc__:
            first_line = factory.__doc__.strip().splitlines()
            resolved_desc = first_line[0] if first_line else ""
        with cls._lock:
            if name in cls._records:
                logger.debug("Preset %r re-registered (source=%s)", name, source)
            cls._records[name] = _PresetRecord(
                name=name,
                factory=factory,
                description=resolved_desc,
                tags=list(tags or []),
                source=source,
            )

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Remove *name* from the registry. Returns ``True`` if removed."""
        with cls._lock:
            return cls._records.pop(name, None) is not None

    @classmethod
    def get(cls, name: str) -> Optional[_PresetRecord]:
        with cls._lock:
            return cls._records.get(name)

    @classmethod
    def list(cls) -> List[_PresetRecord]:
        with cls._lock:
            return list(cls._records.values())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered presets (useful in tests)."""
        with cls._lock:
            cls._records.clear()
            cls._discovered = False

    @classmethod
    def discover(cls, *, force: bool = False) -> int:
        """Scan ``geny_executor.presets`` entry-points and register plugins.

        Safe to call multiple times — results are cached until :meth:`clear`
        or ``force=True``. Returns the number of presets registered in this
        call. Import or load failures are logged and skipped rather than
        raised, so a broken plugin cannot take down the host process.
        """
        with cls._lock:
            if cls._discovered and not force:
                return 0
            cls._discovered = True

        added = 0
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover — Python 3.8 fallback
            return 0

        try:
            eps = entry_points()
            group_eps = (
                eps.select(group=PRESET_ENTRY_POINT_GROUP)
                if hasattr(eps, "select")
                else eps.get(PRESET_ENTRY_POINT_GROUP, [])
            )
        except Exception as exc:  # pragma: no cover — metadata backend variance
            logger.warning("Failed to enumerate preset entry-points: %s", exc)
            return 0

        for ep in group_eps:
            try:
                target: Any = ep.load()
            except Exception as exc:
                logger.warning("Failed to load preset entry-point %s: %s", ep.name, exc)
                continue

            factory: Optional[Callable[..., Pipeline]] = None
            description = ""
            tags: List[str] = []

            if callable(target) and not isinstance(target, type):
                factory = target
            elif isinstance(target, dict):
                factory = target.get("factory") or target.get("builder")
                description = str(target.get("description", ""))
                raw_tags = target.get("tags") or []
                tags = list(raw_tags) if isinstance(raw_tags, (list, tuple)) else []
            elif hasattr(target, "factory"):
                factory = getattr(target, "factory")
                description = getattr(target, "description", "") or ""
                tags = list(getattr(target, "tags", []) or [])

            if factory is None:
                logger.warning("Preset entry-point %s did not expose a callable factory", ep.name)
                continue

            cls.register(
                ep.name,
                factory,
                description=description,
                tags=tags,
                source="plugin",
            )
            added += 1

        return added


def register_preset(
    name: str,
    *,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., Pipeline]], Callable[..., Pipeline]]:
    """Decorator that registers *func* as a preset factory.

    Usage::

        @register_preset("research-agent", description="Tool-heavy agent preset")
        def research_agent(api_key: str) -> Pipeline:
            return PipelineBuilder("research", api_key=api_key).with_tools().build()
    """

    def _decorator(func: Callable[..., Pipeline]) -> Callable[..., Pipeline]:
        PresetRegistry.register(name, func, description=description, tags=tags, source="plugin")
        return func

    return _decorator


class PipelinePresets:
    """Pre-configured pipeline patterns for common use cases."""

    @staticmethod
    def minimal(api_key: str, model: str = "claude-sonnet-4-20250514") -> Pipeline:
        """Minimal pipeline — simple Q&A.

        Active stages: Input → API → Parse → Yield
        """
        return PipelineBuilder("minimal", api_key=api_key, model=model).build()

    @staticmethod
    def chat(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[ToolRegistry] = None,
    ) -> Pipeline:
        """Chat pipeline — history, system prompt, optional tools.

        Active stages: Input → Context → System → Guard → Cache
                       → API → Token → Parse → Tool → Loop → Memory → Yield
        """
        builder = (
            PipelineBuilder("chat", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=system_prompt)
            .with_guard()
            .with_cache(strategy="system")
            .with_loop(max_turns=20)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()

    @staticmethod
    def agent(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str = "You are an autonomous agent. Complete the task step by step.",
        tools: Optional[ToolRegistry] = None,
        max_turns: int = 50,
    ) -> Pipeline:
        """Agent pipeline — full autonomous agent with all stages.

        Active stages: All 16 stages
        """
        builder = (
            PipelineBuilder("agent", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=system_prompt)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_evaluate()
            .with_loop(max_turns=max_turns)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()

    @staticmethod
    def evaluator(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        evaluation_prompt: str = "Evaluate the following response for quality, accuracy, and completeness.",
    ) -> Pipeline:
        """Evaluator pipeline — lightweight evaluation for Generator/Evaluator pattern.

        Active stages: Input → System → API → Parse → Evaluate → Yield
        """
        return (
            PipelineBuilder("evaluator", api_key=api_key, model=model)
            .with_system(prompt=evaluation_prompt)
            .with_evaluate()
            .build()
        )

    @staticmethod
    def geny_vtuber(
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        persona: str = "You are Geny, a friendly AI VTuber.",
        tools: Optional[ToolRegistry] = None,
    ) -> Pipeline:
        """Geny VTuber pipeline — full Geny system reproduction.

        Active stages: All 16 stages + VTuber/TTS emitters
        """
        builder = (
            PipelineBuilder("geny-vtuber", api_key=api_key, model=model)
            .with_context()
            .with_system(prompt=persona)
            .with_guard()
            .with_cache(strategy="aggressive")
            .with_think()
            .with_evaluate()
            .with_loop(max_turns=50)
            .with_memory()
        )

        if tools:
            builder = builder.with_tools(registry=tools)

        return builder.build()


# ═══════════════════════════════════════════════════════════
#  PresetManager — built-in + user presets
# ═══════════════════════════════════════════════════════════


class PresetManager:
    """Manages pipeline presets (built-in + plugin + user-defined)."""

    _BUILT_IN_DESCRIPTIONS: Dict[str, str] = {
        "minimal": "Minimal Q&A pipeline — Input → API → Parse → Yield",
        "chat": "Chat pipeline — history, system prompt, optional tools",
        "agent": "Agent pipeline — full autonomous agent with all stages",
        "evaluator": "Evaluator pipeline — lightweight evaluation",
        "geny_vtuber": "Geny VTuber pipeline — full Geny system reproduction",
    }

    def __init__(
        self,
        env_manager: EnvironmentManager,
        *,
        auto_discover: bool = True,
    ) -> None:
        self._env_manager = env_manager
        self._built_in_factories: Dict[str, Callable[..., Pipeline]] = {
            "minimal": PipelinePresets.minimal,
            "chat": PipelinePresets.chat,
            "agent": PipelinePresets.agent,
            "evaluator": PipelinePresets.evaluator,
            "geny_vtuber": PipelinePresets.geny_vtuber,
        }
        if auto_discover:
            PresetRegistry.discover()

    def list_all(self) -> List[PresetInfo]:
        """List built-in, plugin, and user-defined presets."""
        presets: List[PresetInfo] = []

        # Built-in
        for name in self._built_in_factories:
            presets.append(
                PresetInfo(
                    name=name,
                    description=self._BUILT_IN_DESCRIPTIONS.get(name, ""),
                    preset_type="built_in",
                )
            )

        # Plugin presets (from entry-points + explicit register_preset calls)
        for record in PresetRegistry.list():
            if record.name in self._built_in_factories:
                # Plugin re-registrations intentionally shadow; still expose once.
                continue
            presets.append(
                PresetInfo(
                    name=record.name,
                    description=record.description,
                    preset_type="plugin",
                    tags=list(record.tags),
                )
            )

        # User presets (environments tagged "preset")
        for env in self._env_manager.list_all():
            if "preset" in env.tags:
                presets.append(
                    PresetInfo(
                        name=f"user:{env.id}",
                        description=env.description,
                        preset_type="user",
                        tags=env.tags,
                        environment_id=env.id,
                    )
                )

        return presets

    def create(self, name: str, **kwargs: Any) -> Pipeline:
        """Instantiate a preset by name.

        Resolution order: built-in → plugin registry. User-defined presets
        (environment-based) are constructed via the :class:`EnvironmentManager`
        and are not handled here.
        """
        factory = self._built_in_factories.get(name)
        if factory is not None:
            return factory(**kwargs)

        record = PresetRegistry.get(name)
        if record is not None:
            return record.factory(**kwargs)

        raise KeyError(f"Unknown preset: {name!r}")

    def refresh_plugins(self) -> int:
        """Force re-discovery of entry-point presets. Returns count added."""
        return PresetRegistry.discover(force=True)

    def save_as_preset(self, env_id: str) -> None:
        """Mark an environment as a reusable preset."""
        manifest = self._env_manager.load(env_id)
        if "preset" not in manifest.metadata.tags:
            manifest.metadata.tags.append("preset")
            self._env_manager.update(env_id, {"metadata": {"tags": manifest.metadata.tags}})

    def remove_preset_flag(self, env_id: str) -> None:
        """Un-mark an environment as a preset."""
        manifest = self._env_manager.load(env_id)
        if "preset" in manifest.metadata.tags:
            manifest.metadata.tags.remove("preset")
            self._env_manager.update(env_id, {"metadata": {"tags": manifest.metadata.tags}})
