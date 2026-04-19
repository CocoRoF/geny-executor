"""Meta-test — guards `docs/MEMORY_SPEC.yaml` structural invariants.

This is the *only* test in the completeness suite that is green from
day 1. Its job is to make sure the spec the rest of the suite reads
hasn't been silently mutated. If a future change drops a layer or
collapses the 21 config fields, this test fails loudly so we don't
ship a provider that satisfies a degraded spec.
"""

from __future__ import annotations

EXPECTED_TOP_LEVEL = {
    "version",
    "spec_date",
    "source_doc",
    "layers",
    "capabilities",
    "scopes",
    "backends",
    "retrieval",
    "events",
    "config_fields",
    "requirements",
    "completeness_criteria",
}

REQUIRED_LAYER_IDS = {"stm", "ltm", "notes", "vector", "index", "curated", "global"}
REQUIRED_CAPABILITIES = {
    "read",
    "write",
    "search",
    "link",
    "promote",
    "reindex",
    "snapshot",
    "reflect",
    "summarize",
}
REQUIRED_SCOPES = {"ephemeral", "session", "user", "tenant", "global"}
REQUIRED_EVENTS = {
    "context.built",
    "context.compacted",
    "memory.turn_recorded",
    "memory.execution_recorded",
    "memory.insight",
    "memory.promoted",
    "memory.reindexed",
    "memory.cost",
    "memory.snapshot",
}


def test_spec_has_expected_top_level_sections(spec):
    missing = EXPECTED_TOP_LEVEL - set(spec.keys())
    assert not missing, f"MEMORY_SPEC.yaml missing sections: {missing}"


def test_spec_layers_cover_4_axis_model(spec):
    assert set(spec["layers"].keys()) >= REQUIRED_LAYER_IDS


def test_spec_capabilities_cover_4_axis_model(spec):
    assert set(spec["capabilities"].keys()) >= REQUIRED_CAPABILITIES


def test_spec_scopes_cover_multitenancy_axis(spec):
    assert set(spec["scopes"].keys()) >= REQUIRED_SCOPES


def test_spec_event_schema_is_complete(spec):
    assert set(spec["events"].keys()) >= REQUIRED_EVENTS


def test_spec_has_exactly_21_config_fields(spec):
    """R-F gate: LTMConfig has 21 fields; descriptor.config_schema
    must expose at least these 21. Spec must list them so descriptor
    coverage is checkable."""
    assert len(spec["config_fields"]) == 21


def test_spec_config_field_ids_are_unique(spec):
    ids = [f["id"] for f in spec["config_fields"]]
    assert len(ids) == len(set(ids)), "duplicate config field id"


def test_spec_requirements_reference_known_layers_and_capabilities(spec):
    layer_ids = set(spec["layers"].keys())
    capability_ids = set(spec["capabilities"].keys())
    bad: list[str] = []
    for req in spec["requirements"]:
        if req["layer"] not in layer_ids:
            bad.append(f"{req['id']}: unknown layer {req['layer']!r}")
        if req["capability"] not in capability_ids:
            bad.append(f"{req['id']}: unknown capability {req['capability']!r}")
    assert not bad, "requirements reference unknown axes:\n" + "\n".join(bad)


def test_spec_has_seven_completeness_criteria(spec):
    crits = spec["completeness_criteria"]
    assert len(crits) == 7
    ids = [c["id"] for c in crits]
    assert ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    for c in crits:
        assert c["title"], f"{c['id']} missing title"
        assert c["acceptance"].strip(), f"{c['id']} missing acceptance text"
        assert isinstance(c["activates_in_phase"], int)


def test_spec_retrieval_layer_order_uses_known_layers(spec):
    layer_ids = set(spec["layers"].keys())
    for entry in spec["retrieval"]["layer_order"]:
        assert entry["from"] in layer_ids, f"retrieval entry {entry['id']!r} from unknown layer"
