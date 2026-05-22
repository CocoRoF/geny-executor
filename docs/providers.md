# LLM Providers

> Status: current for geny-executor 2.1.0. Five providers shipped.

geny-executor abstracts the LLM call site behind a single contract: `BaseClient` (`llm_client/base.py`). A host supplies credentials via one `CredentialBundle`; the manifest picks which provider Stage 6 calls. Switching providers is a manifest edit, not a code change.

## The five providers

| Provider id | Client class | Backend | Strengths | Notes |
|---|---|---|---|---|
| `anthropic` | `AnthropicClient` | Anthropic Messages API | streaming, `tool_use`, thinking blocks, prompt caching, cost telemetry | Default for Claude family. Hard dependency. |
| `openai` | `OpenAIClient` | OpenAI Responses / Chat Completions | streaming, tools, JSON-schema structured output, reasoning models | |
| `google` | `GoogleClient` | Google GenAI (Gemini) | streaming, function calling, thinking blocks | |
| `vllm` | `VLLMClient` | OpenAI-compatible local endpoint | streaming, free-form model id | Inherits `OpenAIClient`; tool support is opt-in via `configure_capabilities()`. |
| `claude_code_cli` | `ClaudeCodeCLIClient` | `claude` CLI subprocess | full agentic loop **internally**, host MCP wrap, OAuth + API-key auth, file/shell built-ins | See [claude_code_cli.md](claude_code_cli.md) — non-trivial integration. |

The legacy `copilot_cli` provider was removed in 2.0.6 — `gh copilot` is text-only with no streaming / tools / MCP, and could not host Stage 10 dispatch.

## ClientCapabilities — the honest contract

Every client advertises its capability set via `ClientCapabilities` (`llm_client/base.py`):

```python
class ClientCapabilities:
    supports_streaming: bool
    supports_tools: bool
    supports_thinking: bool
    supports_tool_choice: bool
    supports_mcp_passthrough: bool
    supports_token_usage: bool
    supports_json_schema: bool
    dropped_fields: tuple[str, ...]     # silently-discarded request fields
```

Hosts read the capability set up-front so the UI can grey out features the chosen provider can't honour. `dropped_fields` documents what's silently ignored (e.g. `top_k` on OpenAI, `temperature` on `claude_code_cli`).

## CredentialBundle + ProviderCredentials

```python
from geny_executor import CredentialBundle, ProviderCredentials

bundle = CredentialBundle(by_provider={
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
    "openai":    ProviderCredentials(api_key="sk-..."),
    "google":    ProviderCredentials(api_key="AIza..."),
    "vllm":      ProviderCredentials(base_url="http://localhost:8000/v1"),
    "claude_code_cli": ProviderCredentials(
        api_key="sk-ant-...",
        binary_path="/usr/local/bin/claude",
        extras={
            "bare_mode": True,
            "allow_tools": (),         # empty → CLI defaults
            "default_permission_mode": "default",
            "settings_path": '{"permissions":{"allow":["mcp__myhost","Bash"]}}',
            "mcp_config": {"mcpServers": {...}},
        },
    ),
})
```

`ProviderCredentials` fields:
- `api_key: str` — API providers + claude_code_cli's API-key auth path
- `base_url: str | None` — vLLM endpoint
- `default_headers: Mapping[str, str] | None` — per-call HTTP header injection
- `binary_path: str` — CLI binary location (claude_code_cli only)
- `extras: dict` — provider-specific knobs (see [claude_code_cli.md](claude_code_cli.md) for the full extras catalog)

The bundle is single-channel: every legacy `api_key=` kwarg path is auto-wrapped into a `CredentialBundle` so existing call sites still work.

## ClientRegistry — adding a custom provider

```python
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.base import BaseClient

class MyProviderClient(BaseClient):
    capabilities = ClientCapabilities(...)
    async def create_message(self, *, model_config, messages, **_): ...
    async def create_message_stream(self, *, model_config, messages, **_): ...

ClientRegistry.register("my_provider", lambda: MyProviderClient)
```

After registration, manifest `stages[6].config["provider"] = "my_provider"` routes Stage 6 calls into your client. The pipeline picks credentials for `"my_provider"` from the bundle automatically.

## Stage 6 provider resolution

The pipeline reads provider only from `stages[6].config["provider"]`. Manifests that try to set it on `strategies["provider"]` are rejected at strict-load (`core/pipeline.py:_validate_manifest_provider_locations`). Single source of truth → no silent divergence.

```json
{
  "stages": [
    {"order": 6, "name": "api", "config": {"provider": "claude_code_cli"}, "strategies": {}},
    ...
  ]
}
```

## Per-provider tips

### `anthropic`
Streaming uses Anthropic's SDK `messages.stream()` context manager. Tool calls are accumulated by the SDK and emitted on `message_complete`. Thinking blocks surface as `ContentBlock(type="thinking")`.

### `openai`
Tool calls also accumulate at `message_complete` (per OpenAI's protocol). Set `response_format={"type": "json_schema", "json_schema": {...}}` on the request for structured output.

### `google`
Function calls map to/from Anthropic-shaped `tool_use` / `tool_result` blocks via the canonical translator (`translators/_canonical.py`). Thinking blocks supported.

### `vllm`
Inherits `OpenAIClient`. Set `base_url` to your local vLLM `/v1` endpoint. Most vLLM deployments default to `supports_tools=False`; flip via `configure_capabilities()` if your model handles them.

### `claude_code_cli`
Most complex of the five. The CLI runs its own agentic loop internally, so:
- The host wraps its tool registry through an MCP bridge → CLI's LLM calls `mcp__<bridge>__<tool>` natively.
- `tool_use` blocks observed in the CLI's stream-json output are intentionally **dropped** from the assembled `APIResponse` (since 2.0.6) — the CLI dispatched them already; Stage 10 must not re-dispatch.
- Argv has automatic compat handling: `--verbose` injected for stream-json, `--bare` stripped on OAuth path, `--strict-mcp-config` emitted when a host MCP config is attached.

Full integration guide: [claude_code_cli.md](claude_code_cli.md).
