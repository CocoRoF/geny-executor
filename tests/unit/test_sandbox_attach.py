"""attach_runtime(sandbox=) wraps a resolved claude_code_cli client in a
ContainerCLIRunner, reusing the host's resolved client kwargs. SDK providers
ignore the sandbox.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.pipeline import Pipeline
from geny_executor.llm_client import CredentialBundle, ProviderCredentials
from geny_executor.llm_client._cli_runtime import ContainerCLIRunner


class _FakeSandbox:
    container_name = "gapt-ws-abc"

    async def ensure(self) -> None:  # pragma: no cover - not spawned here
        return None


def test_attach_runtime_stores_sandbox() -> None:
    p = Pipeline()
    assert p._attached_sandbox is None
    p.attach_runtime(sandbox=_FakeSandbox())
    assert isinstance(p._attached_sandbox, _FakeSandbox)


def test_build_client_for_wraps_cli_in_container_runner() -> None:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={
            "claude_code_cli": ProviderCredentials(api_key="sk-test")
        }
    )
    p._attached_sandbox = _FakeSandbox()

    client = p._build_client_for("claude_code_cli")
    assert client.provider == "claude_code_cli"
    # Every spawn (incl. the --version probe) routes through the container.
    runner = client._make_runner()
    assert isinstance(runner, ContainerCLIRunner)
    assert runner.sandbox.container_name == "gapt-ws-abc"
    # The API key resolved by the host flows into the container env.
    assert runner.env_extras.get("ANTHROPIC_API_KEY") == "sk-test"


def test_build_client_for_sdk_provider_ignores_sandbox() -> None:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={
            "anthropic": ProviderCredentials(api_key="sk-test")
        }
    )
    p._attached_sandbox = _FakeSandbox()

    client = p._build_client_for("anthropic")
    # Not a CLI client → not wrapped; sandbox is irrelevant for SDK providers.
    assert type(client).__name__ == "AnthropicClient"


def test_no_sandbox_builds_plain_cli_client() -> None:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={
            "claude_code_cli": ProviderCredentials(api_key="sk-test", binary_path="/bin/sh")
        }
    )
    # No sandbox attached → default in-process runner path.
    client = p._build_client_for("claude_code_cli")
    runner = client._make_runner()
    assert not isinstance(runner, ContainerCLIRunner)


def test_containerize_cli_false_keeps_cli_on_host_but_attaches_sandbox() -> None:
    """Decouple: a sandbox attached with containerize_cli=False is used for TOOL
    execution (ctx.sandbox) but the claude_code_cli client stays on the host —
    so an OAuth (rotating-token) session can use sandboxed GAPT/forge tools
    without the in-container OAuth rotation problem."""
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={
            "claude_code_cli": ProviderCredentials(api_key="sk-test", binary_path="/bin/sh")
        }
    )
    p.attach_runtime(sandbox=_FakeSandbox(), containerize_cli=False)
    # Sandbox is attached (tools get ctx.sandbox)…
    assert isinstance(p._attached_sandbox, _FakeSandbox)
    assert p._containerize_cli is False
    # …but the CLI client is NOT wrapped in a container runner.
    client = p._build_client_for("claude_code_cli")
    runner = client._make_runner()
    assert not isinstance(runner, ContainerCLIRunner)


def test_containerize_cli_default_true_still_wraps() -> None:
    p = Pipeline()
    p._credentials = CredentialBundle(
        by_provider={"claude_code_cli": ProviderCredentials(api_key="sk-test")}
    )
    p.attach_runtime(sandbox=_FakeSandbox())  # default containerize_cli=True
    assert p._containerize_cli is True
    runner = p._build_client_for("claude_code_cli")._make_runner()
    assert isinstance(runner, ContainerCLIRunner)
