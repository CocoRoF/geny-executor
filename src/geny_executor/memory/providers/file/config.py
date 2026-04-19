"""ConfigSchema factory for FileMemoryProvider.

Exposes all 21 fields from `MEMORY_SPEC.yaml::config_fields` so the
web mirror's introspection UI can drive the provider end-to-end.
Fields that this provider does not act on yet (embedding_*, curated_*,
auto_curation_*, user_obsidian_*) are still declared — the contract
is that the descriptor *advertises* everything; Phase 2b+ wires them
to real behaviour.
"""

from __future__ import annotations

from geny_executor.core.schema import ConfigField, ConfigSchema


def file_provider_config_schema() -> ConfigSchema:
    return ConfigSchema(
        name="file_memory",
        fields=[
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
            # obsidian
            ConfigField(
                name="user_obsidian_index_enabled",
                type="boolean",
                label="Obsidian index enabled",
                default=False,
            ),
        ],
    )
