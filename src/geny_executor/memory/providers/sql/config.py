"""ConfigSchema factory for SQLMemoryProvider.

Mirrors the file provider's schema (same 21 fields from
`MEMORY_SPEC.yaml::config_fields`) plus one SQL-specific field —
`dsn` — so the web mirror can introspect the connection target.
The duplication is deliberate: provider schemas are *self-contained*
so a UI can render one provider without knowing about the others.
"""

from __future__ import annotations

from geny_executor.core.schema import ConfigField, ConfigSchema


def sql_provider_config_schema() -> ConfigSchema:
    return ConfigSchema(
        name="sql_memory",
        fields=[
            ConfigField(
                name="dsn",
                type="string",
                label="Database DSN",
                default="",
                description=(
                    "SQLite path or `:memory:`. A future Postgres backend "
                    "will accept `postgresql://…` here."
                ),
            ),
            # master
            ConfigField(
                name="master_enabled",
                type="boolean",
                label="Memory enabled",
                default=True,
            ),
            # embedding
            ConfigField(
                name="embedding_provider",
                type="select",
                label="Embedding provider",
                default="openai",
                description="Embedding backend used when Vector is enabled.",
            ),
            ConfigField(
                name="embedding_model",
                type="string",
                label="Embedding model",
                default="text-embedding-3-small",
            ),
            ConfigField(
                name="embedding_api_key",
                type="string",
                label="Embedding API key",
                default="",
                description="Secret — stored as an opaque string.",
            ),
            # chunking
            ConfigField(
                name="chunk_size",
                type="integer",
                label="Chunk size (tokens)",
                default=800,
                min_value=1,
            ),
            ConfigField(
                name="chunk_overlap",
                type="integer",
                label="Chunk overlap (tokens)",
                default=100,
                min_value=0,
            ),
            # retrieval
            ConfigField(
                name="retrieval_top_k",
                type="integer",
                label="Retrieval top-k",
                default=5,
                min_value=1,
            ),
            ConfigField(
                name="retrieval_threshold",
                type="number",
                label="Retrieval similarity threshold",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
            ),
            ConfigField(
                name="retrieval_max_inject_chars",
                type="integer",
                label="Max characters injected into context",
                default=8000,
                min_value=0,
            ),
            # curated
            ConfigField(
                name="curated_enabled",
                type="boolean",
                label="Curated layer enabled",
                default=False,
            ),
            ConfigField(
                name="curated_vector_enabled",
                type="boolean",
                label="Curated vector search",
                default=False,
            ),
            ConfigField(
                name="curated_inject_budget",
                type="integer",
                label="Curated inject budget (chars)",
                default=2000,
                min_value=0,
            ),
            ConfigField(
                name="curated_max_results",
                type="integer",
                label="Curated max results",
                default=3,
                min_value=0,
            ),
            # auto-curation
            ConfigField(
                name="auto_curation_enabled",
                type="boolean",
                label="Auto-curation enabled",
                default=False,
            ),
            ConfigField(
                name="auto_curation_use_llm",
                type="boolean",
                label="Auto-curation uses LLM",
                default=True,
            ),
            ConfigField(
                name="auto_curation_quality_threshold",
                type="number",
                label="Auto-curation quality threshold",
                default=0.7,
                min_value=0.0,
                max_value=1.0,
            ),
            ConfigField(
                name="auto_curation_schedule",
                type="select",
                label="Auto-curation schedule",
                default="interval",
            ),
            ConfigField(
                name="auto_curation_interval_minutes",
                type="integer",
                label="Auto-curation interval (minutes)",
                default=60,
                min_value=1,
            ),
            ConfigField(
                name="auto_curation_max_per_run",
                type="integer",
                label="Auto-curation max notes per run",
                default=10,
                min_value=0,
            ),
            ConfigField(
                name="auto_curation_last_run_iso",
                type="string",
                label="Last auto-curation run (ISO-8601)",
                default="",
            ),
            ConfigField(
                name="user_obsidian_index_enabled",
                type="boolean",
                label="Obsidian index enabled",
                default=False,
            ),
        ],
    )
