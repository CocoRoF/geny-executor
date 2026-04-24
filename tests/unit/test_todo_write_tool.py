"""Phase 3 Week 6 — TodoWrite tests."""

from __future__ import annotations

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.todo_write_tool import TodoWriteTool


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="")


class TestSchemaAndCapabilities:
    def test_name(self):
        assert TodoWriteTool().name == "TodoWrite"

    def test_capabilities_are_serial(self):
        # Two concurrent TodoWrite calls would race — must serialise.
        caps = TodoWriteTool().capabilities({})
        assert caps.concurrency_safe is False
        assert caps.idempotent is True

    def test_schema_requires_todos(self):
        schema = TodoWriteTool().input_schema
        assert "todos" in schema["required"]
        assert schema["properties"]["todos"]["type"] == "array"


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_single_pending_todo(self):
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "write tests"}]}, _ctx()
        )
        assert not result.is_error
        assert "1 total" in result.content
        assert "1 pending" in result.content
        assert "write tests" in result.content
        assert result.metadata["total"] == 1
        assert result.metadata["counts"] == {
            "pending": 1,
            "in_progress": 0,
            "completed": 0,
        }

    @pytest.mark.asyncio
    async def test_mixed_statuses(self):
        result = await TodoWriteTool().execute(
            {
                "todos": [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "in_progress", "activeForm": "Doing b"},
                    {"content": "c", "status": "pending"},
                ]
            },
            _ctx(),
        )
        assert not result.is_error
        assert result.metadata["counts"] == {
            "pending": 1,
            "in_progress": 1,
            "completed": 1,
        }
        assert "Doing b" in result.content
        # Markdown checklist marker reflects status
        assert "- [x]" in result.content
        assert "- [~]" in result.content
        assert "- [ ]" in result.content

    @pytest.mark.asyncio
    async def test_empty_list_allowed(self):
        result = await TodoWriteTool().execute({"todos": []}, _ctx())
        assert not result.is_error
        assert "0 total" in result.content
        assert "(no todos)" in result.content
        assert result.metadata["total"] == 0

    @pytest.mark.asyncio
    async def test_auto_generates_stable_ids(self):
        r1 = await TodoWriteTool().execute(
            {"todos": [{"content": "task A"}, {"content": "task B"}]}, _ctx()
        )
        r2 = await TodoWriteTool().execute(
            {"todos": [{"content": "task A"}, {"content": "task B"}]}, _ctx()
        )
        ids_1 = [t["id"] for t in r1.metadata["todos"]]
        ids_2 = [t["id"] for t in r2.metadata["todos"]]
        # Same content + same position → same ids
        assert ids_1 == ids_2
        # Different positions → different ids even for the same content
        assert ids_1[0] != ids_1[1]

    @pytest.mark.asyncio
    async def test_custom_id_preserved(self):
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "x", "id": "manual-id"}]}, _ctx()
        )
        assert result.metadata["todos"][0]["id"] == "manual-id"

    @pytest.mark.asyncio
    async def test_state_mutations_propose_shared_key(self):
        """state_mutations hints the future Stage 10 wiring — when hosts
        apply them, ``state.shared['executor.todos']`` carries the list."""
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "x"}]}, _ctx()
        )
        assert "executor.todos" in result.state_mutations
        assert isinstance(result.state_mutations["executor.todos"], list)

    @pytest.mark.asyncio
    async def test_content_and_activeform_trimmed(self):
        result = await TodoWriteTool().execute(
            {
                "todos": [
                    {
                        "content": "  leading / trailing  ",
                        "status": "in_progress",
                        "activeForm": "  Working  ",
                    }
                ]
            },
            _ctx(),
        )
        todo = result.metadata["todos"][0]
        assert todo["content"] == "leading / trailing"
        assert todo["activeForm"] == "Working"


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_missing_todos_field(self):
        result = await TodoWriteTool().execute({}, _ctx())
        assert result.is_error
        assert "'todos'" in result.content

    @pytest.mark.asyncio
    async def test_non_list_todos(self):
        result = await TodoWriteTool().execute({"todos": "oops"}, _ctx())
        assert result.is_error
        assert "must be a list" in result.content

    @pytest.mark.asyncio
    async def test_over_limit(self):
        many = [{"content": f"t{i}"} for i in range(101)]
        result = await TodoWriteTool().execute({"todos": many}, _ctx())
        assert result.is_error
        assert "too many" in result.content

    @pytest.mark.asyncio
    async def test_bad_status(self):
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "x", "status": "weird"}]}, _ctx()
        )
        assert result.is_error
        assert "invalid status" in result.content

    @pytest.mark.asyncio
    async def test_empty_content(self):
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "   "}]}, _ctx()
        )
        assert result.is_error
        assert "content" in result.content

    @pytest.mark.asyncio
    async def test_non_dict_todo(self):
        result = await TodoWriteTool().execute(
            {"todos": ["just a string"]}, _ctx()
        )
        assert result.is_error
        assert "must be an object" in result.content

    @pytest.mark.asyncio
    async def test_duplicate_ids_rejected(self):
        result = await TodoWriteTool().execute(
            {
                "todos": [
                    {"content": "a", "id": "same"},
                    {"content": "b", "id": "same"},
                ]
            },
            _ctx(),
        )
        assert result.is_error
        assert "duplicate" in result.content

    @pytest.mark.asyncio
    async def test_non_string_id_rejected(self):
        result = await TodoWriteTool().execute(
            {"todos": [{"content": "x", "id": 123}]}, _ctx()
        )
        assert result.is_error
        assert "id" in result.content


class TestRegistry:
    def test_registered_in_built_in_tool_classes(self):
        from geny_executor.tools.built_in import (
            BUILT_IN_TOOL_CLASSES,
            BUILT_IN_TOOL_FEATURES,
            TodoWriteTool,
            get_builtin_tools,
        )

        assert BUILT_IN_TOOL_CLASSES["TodoWrite"] is TodoWriteTool
        assert "TodoWrite" in BUILT_IN_TOOL_FEATURES["workflow"]
        tools = get_builtin_tools(features=["workflow"])
        assert set(tools.keys()) == {"TodoWrite"}
