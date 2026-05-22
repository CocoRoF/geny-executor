# Memory (Stage 2 / 18)

> Status: current for geny-executor 2.1.0.

Memory in geny-executor is split across two stages:

| Stage | Role |
|---|---|
| **Stage 2 — Context** | *Read*: load conversation history + relevant memory blocks into the per-turn `PipelineState` before the LLM sees the prompt. |
| **Stage 18 — Memory** | *Write*: persist new facts / artifacts / summaries from the turn's results back into long-term storage. |

The two are decoupled — a session can read from a vector store and write to a SQLite file, or skip persistence entirely and let the host manage history out-of-band. The default `chat` preset wires both stages to file-backed storage.

## Stage 2 — Context loading strategies

| Strategy | Behaviour |
|---|---|
| `simple_load` | Loads the session's stored conversation messages verbatim. Cheap, deterministic. |
| `progressive_disclosure` | Loads the most recent N turns + a per-session summary + a "vault map" (titles / categories of stored memories without their bodies). The LLM reaches for bodies on demand via memory tools. **Recommended for long-lived sessions.** |
| `vector_search` | Retrieves top-k memory blocks by embedding similarity against the current user message. Best when the corpus is large and recency-only loading would miss relevant context. |

Custom strategies plug in via `core/stage.Strategy`. Geny ships a `dynamic_persona` strategy that layers a per-session persona override on top of the static system prompt — it's a Stage 3 strategy, not Stage 2, but the pattern is the same.

### Reading the vault map (progressive disclosure)

```python
state.shared["vault_map"]  # → list of {id, title, category, importance, tags}
state.shared["session_summary"]  # → distilled summary of older turns
```

Tools like `memory_search` / `memory_read` resolve a `vault_map` entry's `id` to its body, so the LLM can pull only what it needs into the next turn's prompt.

## Stage 18 — Memory write strategies

| Strategy | Behaviour |
|---|---|
| `append_only` | Append every assistant message + tool result to the session's history file. Simple, exhaustive. |
| `reflective` | Run a sub-LLM pass to distill "what's worth remembering" from the turn; store only that. Smaller footprint, lossy. |
| `vault` | Reflective-style distillation + categorisation. Writes structured entries (`title` / `body` / `category` / `importance` / `tags`) so `progressive_disclosure` retrieval can preview them. |

Backends are pluggable via `MemoryProvider`:

| Backend | When |
|---|---|
| File (`MemoryProvider.file`) | Local dev, single-process. JSONL per session. |
| SQLite (`MemoryProvider.sqlite`) | Multi-session host, queryable. |
| Custom | Implement the `MemoryProvider` protocol — `read_session`, `write_entry`, `search`, `delete`. Geny wires a Postgres-backed one for production. |

## Vector retrieval (optional)

Install the `[memory]` extra (`pip install geny-executor[memory]`) for numpy-backed cosine similarity. The default embedder is host-supplied — the executor ships the retrieval contract but not a specific embedding model, so you can route through Anthropic, OpenAI, or a local model without locking the pipeline in.

```python
from geny_executor.memory import VectorMemoryProvider

provider = VectorMemoryProvider(
    backend="sqlite",
    db_path="/var/lib/agent/memory.db",
    embedder=my_embedder,           # async callable: text → vector
)
```

## Memory tools

Built-in tools in `geny_executor.tools.built_in.memory_tools` give the LLM first-class memory access:

| Tool | Surface |
|---|---|
| `memory_write` | Persist a new entry with title + body + category + tags + importance. |
| `memory_read` | Fetch a specific entry by id. |
| `memory_search` | Vector / lexical search against the session's vault. |
| `memory_list` | List entries in a category, paginated. |
| `memory_update` / `memory_delete` | CRUD over existing entries. |
| `memory_distill` | Trigger a reflective pass to consolidate recent turns. |
| `memory_link` / `memory_pin` | Cross-reference entries; pin to always-loaded set. |

Geny's `chat` and `agent` presets register these by default. Manifests can opt in/out by editing `tools.built_in[]`.

## Inspecting memory state at runtime

```python
pipeline.attach_runtime(memory_provider=my_provider)

# After a turn:
result = await pipeline.run("Remember that I prefer Python over Rust")
entries = await my_provider.search(session_id="s1", query="preferences", top_k=5)
for e in entries:
    print(e.title, e.importance)
```

## When to use which strategy

- **Short Q&A session, single user:** `simple_load` + `append_only`. Don't over-engineer.
- **Long-lived chat session (hours/days):** `progressive_disclosure` + `vault`. Keeps context window bounded without losing long-tail facts.
- **RAG-style retrieval over a curated corpus:** `vector_search` + custom provider that points at the corpus, not the session history.
- **Multi-agent (VTuber ↔ Sub-Worker):** Geny pattern — each session gets its own `progressive_disclosure` + `vault`, plus cross-session `memory_link` for shared facts.

## See also

- [architecture.md](architecture.md) — where Stage 2 / 18 sit in the 21-stage pipeline
- [hooks.md](hooks.md) — using `POST_TOOL_USE` to track memory writes for audit
- [Geny's memory v2 docs](https://github.com/CocoRoF/Geny/tree/main/docs/llm-backend-upgrade-plan) — production reference implementation
