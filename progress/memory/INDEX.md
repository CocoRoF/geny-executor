# Memory Initiative — Progress Index

> **Charter** — `geny-executor` must, *without any Geny code*, express
> and execute every memory semantics that Geny's `SessionMemoryManager`
> currently provides. Adapters are validation fixtures, not the
> operating path. `geny-executor-web` is a thin mirror; it can only
> show what executor exposes. `Geny` is the *final* product, rewritten
> on top of executor + web after both reach v1.0.0.
>
> Source spec: `geny-executor-web/docs/MEMORY_ARCHITECTURE.md`
> (rev 2026-04-19).

---

## Phase status

| Phase | Title | Tracker | Gate | Status |
|---|---|---|---|---|
| 0 | Spec Freeze | [phase_0_spec_freeze.md](phase_0_spec_freeze.md) | G0 — spec YAML + red pytest exist | **closed 2026-04-19** |
| 1 | Interface | [phase_1_interface.md](phase_1_interface.md) | G1 — provider contract + EphemeralMemoryProvider + contract suite green | **closed 2026-04-19** (v0.14.0) |
| 2 | Native Providers | [phase_2_native_providers.md](phase_2_native_providers.md) | G2 — C1-C3·C5-C6 green, no adapter dependency | in progress (2a v0.15.0, 2b v0.16.0 on 2026-04-19) |
| 3 | Completeness Validation | [phase_3_validation.md](phase_3_validation.md) | **G3 — C1-C7 all green + adapter parity + perf ±20%** | pending |
| 4 | Web Mirror | [phase_4_web_mirror.md](phase_4_web_mirror.md) | G4 — endpoints + UI scenarios | pending |
| 5 | Hardening | [phase_5_hardening.md](phase_5_hardening.md) | G5 — retention/cost/migration | pending |
| G | Geny Rewrite (out of scope) | — | GG — provider contract freeze | not started |

The G3 gate is the *executor completeness* line. Phase 4 web work
must not begin until G3 is green.

---

## Completeness criteria (C1–C7)

These seven scenarios are the **definition of done** for the executor
side. They run in `tests/completeness/` and start red.

- C1 — session → query → 6-layer retrieval → response → STM record.
- C2 — execution end → dated LTM + structured note + vector incremental.
- C3 — LLM reflection → Insight → auto-promote on high importance.
- C4 — REST CRUD/wikilink/graph/tags/importance/keyword·vector search.
- C5 — embedding provider swap → dimension mismatch → reindex plan
       → background reindex → `memory.reindexed` event.
- C6 — server restart → session resumes with prior context.
- C7 — `GenyManagerAdapter` and native `FileMemoryProvider` produce
       semantically equivalent outputs across the shared assertion
       suite.

---

## How this folder is used

- One markdown per phase. Sections: **Summary / Changes / Tests /
  Compatibility / Version bumps / Follow-up** (mirrors the
  `progress/stage_required_flag.md` convention already in this repo).
- Status updates land in the phase file *and* in the table above.
- When a phase closes, its file gains a `Closed:` date header and the
  next phase file is created.
- Decisions that change scope reference §6 of
  `MEMORY_ARCHITECTURE.md`.
