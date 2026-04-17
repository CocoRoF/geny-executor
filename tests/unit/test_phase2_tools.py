"""Phase 2 tests — Ad-hoc tools, Composer, Scope, Sandbox."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.registry import ToolRegistry
from geny_executor.tools.adhoc import (
    AdhocTool,
    AdhocToolDefinition,
    AdhocToolFactory,
    HttpToolConfig,
    ScriptToolConfig,
    TemplateToolConfig,
    CompositeToolConfig,
    CompositeStep,
)
from geny_executor.tools.composer import ToolComposer, ToolPreset
from geny_executor.tools.scope import ToolScope, ToolScopeRule, ToolScopeManager
from geny_executor.tools.sandbox import ToolSandbox, SandboxConfig, SandboxPolicy


# ── Helpers ──────────────────────────────────────────────


def _ctx(working_dir: str = "/tmp") -> ToolContext:
    return ToolContext(session_id="test", working_dir=working_dir)


class DummyTool(Tool):
    def __init__(self, name_str="dummy"):
        self._name = name_str

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return f"Dummy tool: {self._name}"

    @property
    def input_schema(self):
        return {"type": "object", "properties": {"x": {"type": "string"}}}

    async def execute(self, input, context):
        return ToolResult(content=f"result:{input.get('x', '')}")


class SlowTool(Tool):
    @property
    def name(self):
        return "slow"

    @property
    def description(self):
        return "Sleeps forever"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def execute(self, input, context):
        await asyncio.sleep(100)
        return ToolResult(content="done")


class BigOutputTool(Tool):
    @property
    def name(self):
        return "big_output"

    @property
    def description(self):
        return "Returns huge output"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def execute(self, input, context):
        return ToolResult(content="x" * 5_000_000)


# ══════════════════════════════════════════════════════════
# AdhocToolDefinition serialization
# ══════════════════════════════════════════════════════════


class TestAdhocToolDefinition:
    def test_template_roundtrip(self):
        defn = AdhocToolDefinition(
            name="greet",
            description="Greeting tool",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            executor_type="template",
            template_config=TemplateToolConfig(template="Hello, $name!"),
            tags=["test"],
        )
        d = defn.to_dict()
        restored = AdhocToolDefinition.from_dict(d)
        assert restored.name == "greet"
        assert restored.template_config.template == "Hello, $name!"
        assert restored.tags == ["test"]

    def test_script_roundtrip(self):
        defn = AdhocToolDefinition(
            name="calc",
            description="Calculator",
            input_schema={"type": "object"},
            executor_type="script",
            script_config=ScriptToolConfig(
                code="async def execute(input, ctx): return 42",
                timeout=10,
            ),
        )
        d = defn.to_dict()
        restored = AdhocToolDefinition.from_dict(d)
        assert restored.script_config.code == "async def execute(input, ctx): return 42"
        assert restored.script_config.timeout == 10

    def test_http_roundtrip(self):
        defn = AdhocToolDefinition(
            name="api_call",
            description="Call API",
            input_schema={"type": "object"},
            executor_type="http",
            http_config=HttpToolConfig(
                url="https://api.example.com/{endpoint}",
                method="GET",
                timeout=15,
            ),
        )
        d = defn.to_dict()
        restored = AdhocToolDefinition.from_dict(d)
        assert restored.http_config.url == "https://api.example.com/{endpoint}"
        assert restored.http_config.method == "GET"

    def test_composite_roundtrip(self):
        defn = AdhocToolDefinition(
            name="chain",
            description="Chain tools",
            input_schema={"type": "object"},
            executor_type="composite",
            composite_config=CompositeToolConfig(
                steps=[
                    CompositeStep(
                        tool_name="Read",
                        input_mapping={"path": "input['file']"},
                        output_key="content",
                    ),
                    CompositeStep(
                        tool_name="Grep", input_mapping={"text": "content"}, output_key="matches"
                    ),
                ]
            ),
        )
        d = defn.to_dict()
        restored = AdhocToolDefinition.from_dict(d)
        assert len(restored.composite_config.steps) == 2
        assert restored.composite_config.steps[0].tool_name == "Read"


# ══════════════════════════════════════════════════════════
# AdhocToolFactory & Executors
# ══════════════════════════════════════════════════════════


class TestTemplateExecutor:
    @pytest.mark.asyncio
    async def test_basic_template(self):
        defn = AdhocToolDefinition(
            name="greet",
            description="Greet",
            input_schema={"type": "object"},
            executor_type="template",
            template_config=TemplateToolConfig(template="Hello, $name! You are $age."),
        )
        tool = AdhocToolFactory.create(defn)
        result = await tool.execute({"name": "Alice", "age": "30"}, _ctx())
        assert "Alice" in result.content
        assert "30" in result.content

    @pytest.mark.asyncio
    async def test_safe_substitute_missing_key(self):
        defn = AdhocToolDefinition(
            name="safe",
            description="Safe",
            input_schema={"type": "object"},
            executor_type="template",
            template_config=TemplateToolConfig(template="Hi $name, $missing"),
        )
        tool = AdhocToolFactory.create(defn)
        result = await tool.execute({"name": "Bob"}, _ctx())
        assert "Bob" in result.content
        assert "$missing" in result.content  # safe_substitute keeps it


class TestScriptExecutor:
    @pytest.mark.asyncio
    async def test_basic_script(self):
        defn = AdhocToolDefinition(
            name="add",
            description="Add",
            input_schema={"type": "object"},
            executor_type="script",
            script_config=ScriptToolConfig(
                code='async def execute(input, ctx): return input["a"] + input["b"]',
                sandbox=True,
            ),
        )
        tool = AdhocToolFactory.create(defn)
        result = await tool.execute({"a": 3, "b": 4}, _ctx())
        assert result.content == "7"

    @pytest.mark.asyncio
    async def test_sandbox_blocks_os(self):
        defn = AdhocToolDefinition(
            name="evil",
            description="Evil",
            input_schema={"type": "object"},
            executor_type="script",
            script_config=ScriptToolConfig(
                code='import os\nasync def execute(input, ctx): return os.listdir("/")',
                sandbox=True,
                allowed_modules=["json"],
            ),
        )
        tool = AdhocToolFactory.create(defn)
        result = await tool.execute({}, _ctx())
        assert result.is_error

    @pytest.mark.asyncio
    async def test_missing_execute_fn(self):
        defn = AdhocToolDefinition(
            name="noop",
            description="Noop",
            input_schema={"type": "object"},
            executor_type="script",
            script_config=ScriptToolConfig(code="x = 42"),
        )
        tool = AdhocToolFactory.create(defn)
        result = await tool.execute({}, _ctx())
        assert result.is_error
        assert "execute" in result.content


class TestAdhocToolFactory:
    def test_unknown_executor_raises(self):
        defn = AdhocToolDefinition(
            name="bad",
            description="Bad",
            input_schema={},
            executor_type="unknown",
        )
        with pytest.raises(ValueError, match="Unknown executor"):
            AdhocToolFactory.create(defn)

    def test_from_dict(self):
        d = {
            "name": "test",
            "description": "Test",
            "input_schema": {"type": "object"},
            "executor_type": "template",
            "template_config": {"template": "hello"},
        }
        tool = AdhocToolFactory.from_dict(d)
        assert isinstance(tool, AdhocTool)
        assert tool.name == "test"


# ══════════════════════════════════════════════════════════
# ToolComposer
# ══════════════════════════════════════════════════════════


class TestToolComposer:
    def _make_composer(self):
        registry = ToolRegistry()
        registry.register(DummyTool("Read"))
        registry.register(DummyTool("Write"))
        registry.register(DummyTool("Bash"))
        return ToolComposer(registry)

    def test_register_adhoc(self):
        composer = self._make_composer()
        defn = AdhocToolDefinition(
            name="custom",
            description="Custom",
            input_schema={"type": "object"},
            executor_type="template",
            template_config=TemplateToolConfig(template="hi"),
        )
        tool = composer.register_adhoc(defn)
        assert tool.name == "custom"
        assert composer.get_adhoc("custom") is not None

    def test_unregister_adhoc(self):
        composer = self._make_composer()
        defn = AdhocToolDefinition(
            name="tmp",
            description="Tmp",
            input_schema={"type": "object"},
            executor_type="template",
            template_config=TemplateToolConfig(template="x"),
        )
        composer.register_adhoc(defn)
        assert composer.unregister_adhoc("tmp") is True
        assert composer.unregister_adhoc("tmp") is False

    def test_list_all_tools(self):
        composer = self._make_composer()
        infos = composer.list_all_tools()
        assert len(infos) == 3  # Read, Write, Bash
        names = {i.name for i in infos}
        assert "Read" in names

    def test_build_registry_filter(self):
        composer = self._make_composer()
        reg = composer.build_registry(include_built_in={"Read", "Write"})
        assert len(reg) == 2

    def test_build_registry_from_preset(self):
        composer = self._make_composer()
        reg = composer.build_registry_from_preset("readonly")
        names = set(reg.list_names())
        assert "Read" in names
        assert "Write" not in names

    def test_preset_crud(self):
        composer = self._make_composer()
        preset = ToolPreset(name="test_preset", tools=["Read"])
        composer.save_preset(preset)
        assert composer.load_preset("test_preset") is not None
        assert composer.delete_preset("test_preset") is True
        assert composer.load_preset("test_preset") is None


class TestToolPreset:
    def test_roundtrip(self):
        preset = ToolPreset(
            name="dev",
            description="Dev tools",
            tools=["Read", "Write"],
            tags=["dev"],
        )
        d = preset.to_dict()
        restored = ToolPreset.from_dict(d)
        assert restored.name == "dev"
        assert restored.tools == ["Read", "Write"]


# ══════════════════════════════════════════════════════════
# ToolScope
# ══════════════════════════════════════════════════════════


class TestToolScopeRule:
    def test_always(self):
        rule = ToolScopeRule(tool_name="X", action="add", condition_type="always")
        # Create a mock state with enough attrs
        state = _make_mock_state()
        assert rule.matches(state) is True

    def test_iteration_threshold(self):
        rule = ToolScopeRule(
            tool_name="X", action="add", condition_type="iteration", condition_value=">= 3"
        )
        state = _make_mock_state(iteration=5)
        assert rule.matches(state) is True
        state2 = _make_mock_state(iteration=1)
        assert rule.matches(state2) is False

    def test_numeric_threshold(self):
        rule = ToolScopeRule(
            tool_name="X", action="add", condition_type="iteration", condition_value=3
        )
        state = _make_mock_state(iteration=5)
        assert rule.matches(state) is True


class TestToolScope:
    def test_include_filter(self):
        scope = ToolScope(include={"Read", "Write"})
        state = _make_mock_state()
        result = scope.resolve(["Read", "Write", "Bash"], state)
        assert result == {"Read", "Write"}

    def test_exclude_filter(self):
        scope = ToolScope(exclude={"Bash"})
        state = _make_mock_state()
        result = scope.resolve(["Read", "Write", "Bash"], state)
        assert "Bash" not in result

    def test_rules_add_remove(self):
        scope = ToolScope(
            rules=[
                ToolScopeRule(tool_name="Secret", action="add", condition_type="always"),
                ToolScopeRule(
                    tool_name="Read",
                    action="remove",
                    condition_type="iteration",
                    condition_value=">= 3",
                ),
            ]
        )
        state = _make_mock_state(iteration=5)
        result = scope.resolve(["Read", "Write"], state)
        assert "Secret" in result
        assert "Read" not in result


class TestToolScopeManager:
    def test_global_scope(self):
        mgr = ToolScopeManager()
        mgr.set_global_scope(ToolScope(exclude={"Bash"}))
        state = _make_mock_state()
        result = mgr.resolve_for_stage(10, ["Read", "Bash"], state)
        assert "Bash" not in result

    def test_stage_override(self):
        mgr = ToolScopeManager()
        mgr.set_global_scope(ToolScope())
        mgr.set_stage_scope(10, ToolScope(include={"Read"}))
        state = _make_mock_state()
        result = mgr.resolve_for_stage(10, ["Read", "Write", "Bash"], state)
        assert result == {"Read"}


# ══════════════════════════════════════════════════════════
# ToolSandbox
# ══════════════════════════════════════════════════════════


class TestToolSandbox:
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        sandbox = ToolSandbox()
        tool = DummyTool()
        result = await sandbox.execute_tool(tool, {"x": "hello"}, _ctx())
        assert result.content == "result:hello"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_timeout(self):
        sandbox = ToolSandbox(SandboxConfig(max_execution_time=1))
        tool = SlowTool()
        result = await sandbox.execute_tool(tool, {}, _ctx())
        assert result.is_error
        assert "timed out" in result.content

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        sandbox = ToolSandbox(SandboxConfig(max_output_size=100))
        tool = BigOutputTool()
        result = await sandbox.execute_tool(tool, {}, _ctx())
        assert len(result.content) <= 120  # 100 + "... (truncated)"
        assert "truncated" in result.content

    @pytest.mark.asyncio
    async def test_path_validation_blocked(self):
        sandbox = ToolSandbox(SandboxConfig(allowed_paths=["/safe/dir"]))
        tool = DummyTool()
        result = await sandbox.execute_tool(
            tool,
            {"path": "/etc/passwd"},
            _ctx(working_dir="/safe/dir"),
        )
        assert result.is_error
        assert "outside" in result.content

    @pytest.mark.asyncio
    async def test_path_validation_allowed(self):
        sandbox = ToolSandbox(SandboxConfig(allowed_paths=["/tmp"]))
        tool = DummyTool()
        result = await sandbox.execute_tool(
            tool,
            {"path": "/tmp/test.txt"},
            _ctx(working_dir="/tmp"),
        )
        assert not result.is_error


class TestSandboxPolicy:
    def test_strict(self):
        sb = SandboxPolicy.strict("/tmp")
        assert sb.config.max_execution_time == 30
        assert sb.config.network_policy == "deny"

    def test_standard(self):
        sb = SandboxPolicy.standard("/tmp")
        assert sb.config.max_execution_time == 120

    def test_permissive(self):
        sb = SandboxPolicy.permissive()
        assert sb.config.allowed_paths is None
        assert sb.config.max_execution_time == 600


# ── Mock state helper ────────────────────────────────────


class _MockState:
    """Minimal mock for PipelineState used in scope tests."""

    def __init__(self, iteration=0, total_cost_usd=0.0):
        self.iteration = iteration
        self.total_cost_usd = total_cost_usd


def _make_mock_state(iteration=0, total_cost_usd=0.0):
    return _MockState(iteration=iteration, total_cost_usd=total_cost_usd)
