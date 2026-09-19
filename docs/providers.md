# LLM Providers

> Status: current for geny-executor 2.68.0. Seven providers shipped.

geny-executor abstracts the LLM call site behind a single contract: `BaseClient` (`llm_client/base.py`). A host supplies credentials via one `CredentialBundle`; the manifest picks which provider Stage 6 calls. Switching providers is a manifest edit, not a code change.

## The vendor providers

| Provider id | Client class | Backend | Strengths | Notes |
|---|---|---|---|---|
| `anthropic` | `AnthropicClient` | Anthropic Messages API | streaming, `tool_use`, thinking blocks, prompt caching, cost telemetry | Default for Claude family. Hard dependency. |
| `openai` | `OpenAIClient` | OpenAI Responses / Chat Completions | streaming, tools, JSON-schema structured output, reasoning models | |
| `google` | `GoogleClient` | Google GenAI (Gemini) | streaming, function calling, thinking blocks | |
| `vllm` | `VLLMClient` | OpenAI-compatible local endpoint | streaming, free-form model id | Inherits `OpenAIClient`; tools and vision are opt-in — `capabilities={...}` at construction, or `configure_capabilities()` later. |

## The routed providers (2.66.0)

A route is an ordered list of accounts. `geny_router` is the provider a manifest names; it decides per call which real backend answers, so one conversation can move between a Claude subscription, a second Claude login and a ChatGPT plan without losing its tools, memory or permission policy.

| Provider id | Client class | Backend | Notes |
|---|---|---|---|
| `geny_router` | `RouterClient` | the accounts in its route | Fails over only **before the first token** and only for failures another account can fix (rate limit, auth, outage, missing CLI). A bad request is the request's fault and reaches the caller unchanged. A failed account cools down process-wide so sibling agents skip it. |
| `geny_claude_code` | `ClaudeCodeTokenClient` | `claude` CLI via the Agent SDK, tools off | Claude Code as a **token generator**: `tools=[] max_turns=1`, MCP and host settings locked out, `<tool_call>` text parsed back into canonical `tool_use` blocks that Stage 10 executes. Each account owns a `CLAUDE_CONFIG_DIR`, so any number of logins coexist. |
| `geny_codex` | `CodexResponsesClient` | `chatgpt.com/backend-api/codex` | A ChatGPT plan over the Responses API. Native `function_call` → `tool_use`. No `codex` binary; tokens live in the host's store and rotate through `notify` (the refresh token is single-use). |

These three read their whole constructor surface from `ProviderCredentials.extras` (`ROUTED_PROVIDERS`) — the route, an account id, a Claude config dir, Codex tokens are host state with no vendor-shaped equivalent, so a host can add an account channel without a library release.

The legacy `copilot_cli` provider was removed in 2.0.6 — `gh copilot` is text-only with no streaming / tools / MCP, and could not host Stage 10 dispatch.

The legacy `claude_code_cli` provider was removed in 2.68.0. It handed the whole agentic loop to the `claude` subprocess: the CLI ran its own Read/Write/Bash under its own permission model, and the 21 stages saw an announcement of what had already happened rather than a request they could answer. That is a *different agent* behind the pipeline, not a model inside it — it could not share a conversation with another provider, could not be sandboxed by this library's rules, and could not be held to its permission ladder. Use `geny_claude_code`, which drives the same binary as a pure token generator.

**A provider may be a model. It may not be an agent.** No shipped client runs its own tool loop, and the machinery that once allowed it is gone: `ClientCapabilities.is_subprocess`, `.supports_mcp_passthrough`, CLI MCP passthrough, `attach_runtime(containerize_cli=)`, and the `api.cli_tool_call` event.

## ClientCapabilities — the honest contract

Every client advertises its capability set via `ClientCapabilities` (`llm_client/base.py`):

```python
class ClientCapabilities:
    supports_streaming: bool
    supports_tools: bool
    supports_thinking: bool
    supports_tool_choice: bool
    supports_token_usage: bool
    supports_json_schema: bool
    dropped_fields: tuple[str, ...]     # silently-discarded request fields
```

Hosts read the capability set up-front so the UI can grey out features the chosen provider can't honour. `dropped_fields` documents what's silently ignored (e.g. `top_k` on OpenAI, `temperature` and `max_tokens` on `geny_claude_code` — `claude -p` takes neither). The declaration is enforced: `_build_request` strips every listed field and emits one `llm_client.field_dropped` event, so a pinned setting that does nothing is visible instead of silent.

## CredentialBundle + ProviderCredentials

```python
from geny_executor import CredentialBundle, ProviderCredentials

bundle = CredentialBundle(by_provider={
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
    "openai":    ProviderCredentials(api_key="sk-..."),
    "google":    ProviderCredentials(api_key="AIza..."),
    "vllm":      ProviderCredentials(base_url="http://localhost:8000/v1"),
    "geny_claude_code": ProviderCredentials(
        binary_path="/usr/local/bin/claude",
        extras={"config_dir": "/data/llm-accounts/claude/work"},
    ),
})
```

`ProviderCredentials` fields:
- `api_key: str` — API providers
- `base_url: str | None` — vLLM endpoint
- `default_headers: Mapping[str, str] | None` — per-call HTTP header injection
- `binary_path: str` — `claude` binary location (`geny_claude_code`)
- `extras: dict` — provider-specific knobs; for the three routed providers `extras` **is** the constructor surface

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
    {"order": 6, "name": "api", "config": {"provider": "geny_router"}, "strategies": {}},
    ...
  ]
}
```

## How much context a model has

`PipelineConfig.context_window_budget` sizes Stage-2 proactive compaction and Stage-4's headroom guard. It ships at 200_000, which is right for the frontier vendors and wrong by a factor of six for a local server launched at 32k — there it means compaction never fires and the request overflows before anything notices.

`llm_client/context_window.py` resolves the number without guessing:

```python
from geny_executor.llm_client import resolve_context_window, binding_context_window

window = resolve_context_window(
    declared=account_setting,      # the operator says
    discovered=from_v1_models,     # the endpoint says
    model="anthropic/claude-sonnet-5",
    provider="custom",
)
budget = binding_context_window(w for w in every_hops_window)  # a route holds its smallest hop
```

`discover_models` captures what the endpoint states — `context_length` (aggregators; `top_provider`'s lower figure wins, because that is who serves the call), `max_model_len` (vLLM: what the server was *launched* with), Google's `inputTokenLimit`. `local_probe.resolve_local_context_window` covers Ollama, which states it on `/api/show` rather than `/v1/models`.

`None` means no source knows. It is deliberately not a fallback guess: a host that knows the number is unknown can say so to its user, and a number cannot.

## Per-provider tips

### `anthropic`
Streaming uses Anthropic's SDK `messages.stream()` context manager. Tool calls are accumulated by the SDK and emitted on `message_complete`. Thinking blocks surface as `ContentBlock(type="thinking")`.

### `openai`
Tool calls also accumulate at `message_complete` (per OpenAI's protocol). Set `response_format={"type": "json_schema", "json_schema": {...}}` on the request for structured output.

### `google`
Function calls map to/from Anthropic-shaped `tool_use` / `tool_result` blocks via the canonical translator (`translators/_canonical.py`). Thinking blocks supported.

### `vllm`
Inherits `OpenAIClient`. Set `base_url` to your local vLLM `/v1` endpoint. The class declares `supports_tools=False` because a vLLM server is whatever model it loaded; a deployment running a tool-calling model says so with `capabilities={"supports_tools": True, "supports_tool_choice": True}` at construction (or `configure_capabilities()` later). See *Declaring what an endpoint serves* below.

### Declaring what an endpoint serves

`custom` / `local` / `ollama` / `lmstudio` / `vllm` each serve endpoints that have nothing in common but the wire format: the same `custom` class reaches a cloud aggregator in front of Claude and a keyless llama.cpp on a laptop. A class can only declare the weakest of them, so the **host** amends its instance:

```python
client = ClientRegistry.get("custom")(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
    capabilities={"supports_vision": True},
)
```

Through the pipeline the same declaration rides on the credentials — `extras["capabilities"]` — and `_creds_to_client_kwargs` threads it to the constructor. Unknown keys are logged and ignored rather than raised (the value comes from config), and each instance is independent, so two accounts on one class never inherit each other's declaration.

A declaration is enforced, not advisory: `supports_tools=False` strips the tools from the request (and the `tool_choice` with them), the same way `supports_top_k` and `supports_stop_sequences` already worked. `supports_vision` is the one that matters most: it is the only capability whose absence is **silent**. An image a client cannot see is replaced with a note saying one was there (both on the message and nested inside a `tool_result`), so an undeclared endpoint answers confidently about a picture it never received.

### `geny_claude_code`
The `claude` binary as a pure token generator, driven through the official **Claude Agent SDK** (`claude-agent-sdk`): `tools=[]` (no built-in tools), `max_turns=1` (one generation, never an agent loop), `strict_mcp_config` + `mcp_servers={}` (no MCP), `setting_sources=[]` (no host `settings.json`). Tool calls come back as `<tool_call>` text (`llm_client/text_tool_protocol.py`), become canonical `tool_use` blocks, and Stage 10 executes them under this pipeline's jail, permissions and hooks. `temperature`, `top_p`, `top_k`, `stop_sequences`, `max_tokens` and `tool_choice` are declared drops — the CLI takes none of them.

Auth is the CLI's own, because the SDK drives the same binary: a Pro/Max login, a setup-token, or a Console key. Each account owns a `CLAUDE_CONFIG_DIR` (passed via the SDK's `env`), so any number of logins coexist without touching the user's own `~/.claude`.

### `geny_codex`
A ChatGPT plan over the Responses API — no `codex` binary. Native `function_call` blocks translate to `tool_use`. Tokens live in the host's store; the refresh token is single-use, so a rotation must be persisted through `notify` or the account is locked out.
