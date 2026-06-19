"""Generalized sub-agent catalog (2.8.0)."""
from geny_executor.stages.s12_agent import (
    BUILTIN_SUBAGENT_TYPES,
    DEFAULT_PERSISTENT_SUBAGENT_PROMPT,
    SubagentTypeSpec,
    default_subagent_specs,
    specs_to_descriptors,
)


def test_builtin_catalog_is_generalized_and_app_neutral():
    names = {s.agent_type for s in BUILTIN_SUBAGENT_TYPES}
    assert names == {"worker", "researcher", "summarizer", "critic"}
    # app-specific types must NOT leak into the generalized catalog
    assert "vtuber-narrator" not in names
    for s in BUILTIN_SUBAGENT_TYPES:
        assert isinstance(s, SubagentTypeSpec)
        assert s.system_prompt and len(s.system_prompt) > 40


def test_default_persistent_prompt_is_strong():
    assert DEFAULT_PERSISTENT_SUBAGENT_PROMPT
    assert "companion" in DEFAULT_PERSISTENT_SUBAGENT_PROMPT.lower()


def test_specs_to_descriptors_attaches_factory():
    def f(ctx):
        return None

    descs = specs_to_descriptors(f, default_subagent_specs())
    assert len(descs) == 4
    worker = next(d for d in descs if d.agent_type == "worker")
    assert worker.factory is f
    assert worker.system_prompt
    assert worker.description
