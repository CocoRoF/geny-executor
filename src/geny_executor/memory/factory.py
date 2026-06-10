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
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, MutableMapping, Optional

from geny_executor.memory.composite.provider import CompositeMemoryProvider
from geny_executor.memory.composite.routing import LayerRouting
from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.embedding.registry import create_embedding_client
from geny_executor.memory.provider import Layer, MemoryProvider, Scope
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file import FileMemoryProvider

if TYPE_CHECKING:
    from geny_executor.llm_client.credentials import CredentialBundle

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

    Credentials (2.2.0, audit §2.6)
    -------------------------------
    Pass ``credentials=`` (a :class:`CredentialBundle`) to source
    embedding API keys from the bundle's ``'embedding'`` provider
    entry instead of the legacy env-var ladder. The bundle entry's
    ``api_key`` fills any embedding config that didn't set one
    explicitly, and its ``extras`` may carry ``provider`` / ``model``
    / ``base_url`` defaults. This closes the parallel credential
    channel that caused the live 401-spam incident — the embedding
    key now flows through the same object as every other provider
    secret, so hosts rotate it in one place. The env ladder inside
    the clients survives as a DEPRECATED fallback (one-time warning).

    The bundle is factory-level rather than per-``build()`` because
    the composite builder recurses through ``factory.build`` for each
    sub-provider — per-call plumbing would have to thread through
    every third-party builder signature, breaking the ``Builder``
    Protocol for a value that is session-constant anyway.
    """

    def __init__(self, *, credentials: Optional["CredentialBundle"] = None) -> None:
        self._builders: Dict[str, Builder] = {}
        self._credentials = credentials
        self._register_builtins()

    @property
    def credentials(self) -> Optional["CredentialBundle"]:
        """The bundle embedding clients are constructed from (if any).

        Exposed read-only so builder functions (including third-party
        ones registered via :meth:`register`) can reach the same
        credential source the built-in builders use.
        """
        return self._credentials

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


# ── manifest glue (2.2.0 Wave 3, audit §1-1) ────────────────────────
#
# The manifest's ``memory`` block is ``{"provider": <name>, "config":
# {...}}`` — the same per-provider keys ``build()`` reads, with the
# provider name lifted out so env editors can render the selector
# without parsing the config body.

#: Config keys each built-in builder consumes (``provider`` included —
#: it is legal inside ``config`` too, though redundant there).
#: ``validate_manifest`` warns on ``memory.config`` keys outside the
#: named builder's set; builders registered by hosts at runtime are
#: absent here and skip the key check entirely.
MEMORY_PROVIDER_CONFIG_KEYS: Dict[str, frozenset] = {
    "ephemeral": frozenset({"provider", "scope"}),
    "file": frozenset({"provider", "root", "embedding", "scope", "session_id", "timezone"}),
    "sql": frozenset(
        {"provider", "dsn", "dialect", "embedding", "scope", "session_id", "timezone"}
    ),
    "composite": frozenset(
        {
            "provider",
            "providers",
            "layers",
            "scope_providers",
            "scope",
            "session_id",
            "user_id",
        }
    ),
}


def provider_from_manifest_memory(
    memory: Mapping[str, Any],
    *,
    credentials: Optional["CredentialBundle"] = None,
) -> MemoryProvider:
    """Build a :class:`MemoryProvider` from a manifest ``memory`` block.

    The single translation point between the manifest shape
    (``{"provider": ..., "config": {...}}``) and the factory's flat
    config dict — ``Pipeline.from_manifest`` calls this when the block
    is non-empty, passing the session's :class:`CredentialBundle` so
    embedding keys flow through the bundle's ``'embedding'`` entry
    (the 2.2.0 single credential channel) instead of env vars.

    Raises:
        ValueError: Missing/unknown provider name, or per-provider
            required keys absent (e.g. ``file`` without ``root``) —
            the same errors :meth:`MemoryProviderFactory.build` raises.
    """
    name = memory.get("provider")
    if not isinstance(name, str) or not name:
        raise ValueError(
            "manifest.memory requires a non-empty 'provider' string "
            "(e.g. 'file', 'sql', 'ephemeral', 'composite')"
        )
    config: Dict[str, Any] = dict(memory.get("config") or {})
    config["provider"] = name
    return MemoryProviderFactory(credentials=credentials).build(config)


# ── builder implementations ─────────────────────────────────────────


def _build_ephemeral(_: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    return EphemeralMemoryProvider(scope=_resolve_scope(config))


def _build_file(factory: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    root = _require_path(config, "root")
    embedding_client = _build_embedding(config.get("embedding"), factory.credentials)
    return FileMemoryProvider(
        root=root,
        scope=_resolve_scope(config),
        session_id=str(config.get("session_id", "")),
        timezone_name=_optional_str(config.get("timezone")),
        embedding_client=embedding_client,
    )


def _build_sql(factory: MemoryProviderFactory, config: Mapping[str, Any]) -> MemoryProvider:
    dsn = config.get("dsn")
    if dsn in (None, ""):
        raise ValueError("sql provider config requires non-empty 'dsn'")
    # Defer the SQL provider import until a caller actually asks for
    # it. This keeps geny-executor importable without `psycopg`
    # installed (postgres DSNs raise inside `connection.py` only when
    # a connection is opened); SQLite DSNs work via stdlib `sqlite3`.
    from geny_executor.memory.providers.sql import SQLMemoryProvider
    from geny_executor.memory.providers.sql.schema import Dialect  # noqa: F401  (resolves at runtime)

    embedding_client = _build_embedding(config.get("embedding"), factory.credentials)
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


def _build_embedding(
    spec: Optional[Mapping[str, Any]],
    credentials: Optional["CredentialBundle"] = None,
) -> Optional[EmbeddingClient]:
    """Construct the embedding client for a provider config.

    The config ``spec`` stays authoritative for *what* to build
    (provider / model / dimension) — it is the manifest-editable
    surface. The :class:`CredentialBundle`'s ``'embedding'`` entry is
    authoritative for *how to authenticate*: its ``api_key`` fills in
    whenever the spec didn't set one, and its ``extras`` may carry
    ``provider`` / ``model`` / ``base_url`` defaults for specs that
    omit them. Precedence per field is explicit-spec > bundle >
    backend default, so a host that already passes keys in config is
    untouched, while bundle-only hosts stop depending on env vars
    (the env ladder inside the clients is the DEPRECATED last resort
    — audit §2.6's 401-spam channel).
    """
    embedding_creds = credentials.get("embedding") if credentials is not None else None
    if embedding_creds is not None and embedding_creds.is_empty():
        embedding_creds = None

    if not spec:
        if embedding_creds is None:
            return None
        # Bundle-only construction: the host supplied embedding
        # credentials but the provider config has no embedding block.
        # Without a provider name we cannot build anything — config
        # remains the opt-in switch for the vector layer.
        extras = dict(embedding_creds.extras or {})
        provider = extras.pop("provider", None)
        if not provider:
            return None
        spec = {"provider": str(provider)}

    if not isinstance(spec, Mapping):
        raise TypeError(f"embedding config must be a mapping, got {type(spec).__name__}")

    kwargs: Dict[str, Any] = {k: v for k, v in spec.items() if k != "provider"}
    provider_name = spec.get("provider")

    if embedding_creds is not None:
        extras = dict(embedding_creds.extras or {})
        if not provider_name:
            provider_name = extras.get("provider")
        if not kwargs.get("api_key") and embedding_creds.api_key:
            kwargs["api_key"] = embedding_creds.api_key
        if not kwargs.get("model") and extras.get("model"):
            kwargs["model"] = str(extras["model"])
        base_url = embedding_creds.base_url or extras.get("base_url")
        if base_url:
            options = dict(kwargs.get("options") or {})
            options.setdefault("base_url", str(base_url))
            kwargs["options"] = options

    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("config key 'provider' must be a non-empty string")
    return create_embedding_client(provider_name, **kwargs)


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
    "MEMORY_PROVIDER_CONFIG_KEYS",
    "provider_from_manifest_memory",
]
