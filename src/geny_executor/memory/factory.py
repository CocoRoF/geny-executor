"""MemoryProviderFactory — name-keyed registry for provider builds.

Consumers (geny-executor-web, the CLI, the Pipeline factory itself)
should never reach for a concrete provider class directly. Instead
they pass a config dict to `MemoryProviderFactory.build(config)` and
receive a fully-wired `MemoryProvider`. This is the integration point
that lets the same JSON manifest swap a session between file and SQL
storage without code changes.

Built-in builders ship for `ephemeral`, `file`, `sql`, and
`composite`. The composite builder defers to `factory.build` for
each named sub-provider so the recursion stays single-source.

Config shape (per provider):

    {"provider": "ephemeral", "scope": "session"}

    {"provider": "file", "root": "/path/to/dir",
     "embedding": {"provider": "local", "model": "...",
                   "dimension": 384}}

    {"provider": "sql", "dsn": "/path/to/db.sqlite",
     "embedding": {...}}

    # Postgres dialect — auto-detected from DSN scheme
    {"provider": "sql",
     "dsn": "postgresql://user:pw@host:5432/dbname",
     "embedding": {...}}

    # Or override explicitly
    {"provider": "sql", "dsn": "postgresql://...",
     "dialect": "postgres"}

    {"provider": "composite",
     "session_id": "session-abc",
     "user_id": "alice",                       # surfaced on CuratedHandle.user_id
     "providers": {
        "session": {"provider": "file",
                    "root": "/storage/sessions/session-abc",
                    "embedding": {"provider": "openai", ...}},
        "user_curated": {"provider": "file",
                         "root": "/storage/curated/alice",
                         "embedding": {"provider": "openai", ...},
                         "scope": "user"},
     },
     "layers": {
        "stm": "session", "ltm": "session", "notes": "session",
        "vector": "session", "index": "session",
     },
     "scope_providers": {
        "session": "session",                  # explicit so promote_from_session knows the source
        "user": "user_curated",                # `provider.curated()` resolves to this delegate
     }}

The `providers` block under composite is named so two layers can
share the same underlying provider instance — that's how a single
file root ends up serving STM + LTM + Notes + Vector + Index for one
session, while a second file root sits at a separate (`scope=user`)
root for the curated knowledge plane. A future SQL setup mirrors the
same shape with `dsn` / `dialect` instead of `root`.

`scope_providers["user"]` (and `"global"`) are the canonical hook for
the curated / global handle resolution. The composite wraps the
delegate's `notes()` + `vector()` into a `CuratedHandle` /
`GlobalHandle` automatically — no separate provider class needs to
implement those handles natively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from geny_executor.memory.composite.provider import CompositeMemoryProvider
from geny_executor.memory.composite.routing import LayerRouting
from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.embedding.registry import create_embedding_client
from geny_executor.memory.provider import Layer, MemoryProvider, Scope
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file import FileMemoryProvider

# `geny_executor.memory.providers.sql` lazily imports `psycopg` for
# Postgres dialects, but the Postgres SDK lives in the optional
# `[postgres]` extra. Import the SQL provider lazily inside `_build_sql`
# so a default `pip install geny-executor` (no extras) does not pay
# the import-time cost or reach for a dep it does not need. SQLite
# stays available because the connection module imports `sqlite3` from
# the standard library only when an SQLite DSN is constructed.


Builder = Callable[["MemoryProviderFactory", Mapping[str, Any]], MemoryProvider]


class MemoryProviderFactory:
    """Registry + dispatcher for provider construction.

    The factory is stateless w.r.t. provider instances — every call
    to `build()` produces a fresh provider tree. Builder functions
    are cheap to swap, so tests can register a stub builder under a
    well-known name and get deterministic construction.
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Builder] = {}
        self._register_builtins()

    # ── registration ────────────────────────────────────────────────

    def register(self, name: str, builder: Builder) -> None:
        if not name:
            raise ValueError("provider name must be a non-empty string")
        self._builders[name] = builder

    def has(self, name: str) -> bool:
        return name in self._builders

    def names(self) -> list[str]:
        return sorted(self._builders.keys())

    # ── dispatch ────────────────────────────────────────────────────

    def build(self, config: Mapping[str, Any]) -> MemoryProvider:
        name = _require_str(config, "provider")
        builder = self._builders.get(name)
        if builder is None:
            available = ", ".join(self.names())
            raise ValueError(f"unknown memory provider {name!r}; registered: {available}")
        return builder(self, config)

    # ── built-in builders ───────────────────────────────────────────

    def _register_builtins(self) -> None:
        self._builders.update(
            {
                "ephemeral": _build_ephemeral,
                "file": _build_file,
                "sql": _build_sql,
                "composite": _build_composite,
            }
        )


# ── builder implementations ─────────────────────────────────────────


def _build_ephemeral(_: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    return EphemeralMemoryProvider(scope=_resolve_scope(config))


def _build_file(_: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    root = _require_path(config, "root")
    embedding_client = _build_embedding(config.get("embedding"))
    return FileMemoryProvider(
        root=root,
        scope=_resolve_scope(config),
        session_id=str(config.get("session_id", "")),
        timezone_name=_optional_str(config.get("timezone")),
        embedding_client=embedding_client,
    )


def _build_sql(_: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    dsn = config.get("dsn")
    if dsn in (None, ""):
        raise ValueError("sql provider config requires non-empty 'dsn'")
    # Defer the SQL provider import until a caller actually asks for
    # it. This keeps geny-executor importable without `psycopg`
    # installed (postgres DSNs raise inside `connection.py` only when
    # a connection is opened); SQLite DSNs work via stdlib `sqlite3`.
    from geny_executor.memory.providers.sql import SQLMemoryProvider
    from geny_executor.memory.providers.sql.schema import Dialect  # noqa: F401  (resolves at runtime)

    embedding_client = _build_embedding(config.get("embedding"))
    dialect = _resolve_dialect(config.get("dialect"))
    return SQLMemoryProvider(
        dsn=dsn,
        scope=_resolve_scope(config),
        session_id=str(config.get("session_id", "")),
        timezone_name=_optional_str(config.get("timezone")),
        embedding_client=embedding_client,
        dialect=dialect,
    )


def _build_composite(factory: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    providers_cfg = config.get("providers")
    if not isinstance(providers_cfg, Mapping) or not providers_cfg:
        raise ValueError(
            "composite provider config requires a non-empty 'providers' "
            "mapping of name → sub-config"
        )

    built: Dict[str, MemoryProvider] = {}
    for name, sub in providers_cfg.items():
        if not isinstance(sub, Mapping):
            raise TypeError(
                f"composite providers[{name!r}] must be a mapping, got {type(sub).__name__}"
            )
        built[str(name)] = factory.build(sub)

    layers_cfg = config.get("layers")
    if not isinstance(layers_cfg, Mapping):
        raise ValueError(
            "composite provider config requires a 'layers' mapping of layer-name → provider-name"
        )

    layers: MutableMapping[Layer, MemoryProvider] = {}
    for layer_key, provider_name in layers_cfg.items():
        layer = Layer(layer_key)
        delegate = built.get(str(provider_name))
        if delegate is None:
            raise ValueError(
                f"composite layers[{layer_key!r}] references unknown provider {provider_name!r}"
            )
        layers[layer] = delegate

    scope_routes: MutableMapping[Scope, MemoryProvider] = {}
    for scope_key, provider_name in (config.get("scope_providers") or {}).items():
        scope = Scope(scope_key)
        delegate = built.get(str(provider_name))
        if delegate is None:
            raise ValueError(
                f"composite scope_providers[{scope_key!r}] references unknown provider "
                f"{provider_name!r}"
            )
        scope_routes[scope] = delegate

    routing = LayerRouting(layers=dict(layers), scope_providers=dict(scope_routes))
    return CompositeMemoryProvider(
        routing=routing,
        scope=_resolve_scope(config),
        session_id=str(config.get("session_id", "")),
        user_id=str(config.get("user_id", "")),
    )


# ── helpers ─────────────────────────────────────────────────────────


def _build_embedding(spec: Optional[Mapping[str, Any]]) -> Optional[EmbeddingClient]:
    if not spec:
        return None
    if not isinstance(spec, Mapping):
        raise TypeError(f"embedding config must be a mapping, got {type(spec).__name__}")
    provider = _require_str(spec, "provider")
    kwargs = {k: v for k, v in spec.items() if k != "provider"}
    return create_embedding_client(provider, **kwargs)


def _resolve_scope(config: Mapping[str, Any]) -> Scope:
    raw = config.get("scope", Scope.SESSION.value)
    if isinstance(raw, Scope):
        return raw
    return Scope(str(raw))


def _resolve_dialect(raw: Any) -> Optional[Any]:
    """Map config ``dialect`` value to a `Dialect` enum, or ``None``
    so the provider falls back to DSN-scheme detection.

    Imports the `Dialect` enum lazily to keep parity with `_build_sql`
    — callers that never trigger the SQL path never import the SQL
    provider tree.
    """
    if raw is None or raw == "":
        return None
    from geny_executor.memory.providers.sql.schema import Dialect

    if isinstance(raw, Dialect):
        return raw
    return Dialect(str(raw).lower())


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config key {key!r} must be a non-empty string")
    return value


def _require_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"config key {key!r} is required")
    return Path(value).expanduser()


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


__all__ = [
    "MemoryProviderFactory",
    "Builder",
]
