"""Storage growth policies — the caps that keep a long-lived session's disk
footprint bounded (prod evidence: 270 MB transcript with 2,110 lines; 1,482
checkpoint files / 367 MB in one session).

Effect-proving doctrine: each test asserts the MEASURED bound, not just that
code ran.
"""

from __future__ import annotations

import json

import pytest

from geny_executor.memory.provider import MemoryHooks, Turn
from geny_executor.memory.providers.file.stm_store import (
    MAX_RECORD_BYTES,
    MAX_STM_BYTES,
    _bound_record_line,
    _JSONLSTMStore,
)
from geny_executor.stages.s20_persist.artifact.default.persisters import (
    FilePersister,
)


def _mk_store(tmp_path):
    return _JSONLSTMStore(tmp_path / "transcripts" / "session.jsonl",
                          tz=None, hooks=MemoryHooks())


# ── record-level cap ──────────────────────────────────────────────────


def test_fat_record_truncated_at_append():
    huge = json.dumps({"type": "message", "role": "assistant",
                       "content": "글" * 300_000, "ts": "t"}, ensure_ascii=False)
    bounded = _bound_record_line(huge)
    assert len(bounded.encode("utf-8")) <= MAX_RECORD_BYTES + 1024
    rec = json.loads(bounded)
    assert "truncated at record cap" in rec["content"]
    assert rec["content"].startswith("글" * 100)  # head preserved


def test_normal_record_untouched():
    line = json.dumps({"type": "message", "role": "user",
                       "content": "짧은 메시지", "ts": "t"}, ensure_ascii=False)
    assert _bound_record_line(line) == line


@pytest.mark.asyncio
async def test_fat_event_payload_dropped(tmp_path):
    """EFFECT PROOF: a 500 KB observation-style event line (the production
    270 MB transcript's fat-line shape) is reduced to a small envelope."""
    store = _mk_store(tmp_path)
    await store.append_event("observation.frame",
                             {"image_b64": "A" * 500_000})
    raw = (tmp_path / "transcripts" / "session.jsonl").read_text()
    assert len(raw) < 2_000
    rec = json.loads(raw.strip())
    assert rec["data"]["truncated"] is True


# ── file-level byte budget ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_byte_cap_bounds_whole_file(tmp_path):
    """EFFECT PROOF: even under the 2,000-line cap, fat lines must not push
    the file past MAX_STM_BYTES — oldest lines are dropped first, newest
    survive."""
    store = _mk_store(tmp_path)
    path = tmp_path / "transcripts" / "session.jsonl"
    path.parent.mkdir(parents=True)
    # 600 lines × ~48 KB ≈ 28 MB — over budget while far under the line cap.
    chunk = json.dumps({"type": "message", "role": "assistant",
                        "content": "데" * 24_000, "ts": "t"}, ensure_ascii=False)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(600):
            fh.write(chunk[:-1] + f'{i}"' + "}"[0:0] + "\n") if False else fh.write(chunk + "\n")
    assert path.stat().st_size > MAX_STM_BYTES

    dropped = await store.enforce_byte_cap()
    assert dropped > 0
    assert path.stat().st_size <= MAX_STM_BYTES
    # the newest lines survive (tail-biased retention)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 600 - dropped


@pytest.mark.asyncio
async def test_byte_cap_noop_under_budget(tmp_path):
    store = _mk_store(tmp_path)
    await store.append(Turn(role="user", content="hello"))
    assert await store.enforce_byte_cap() == 0


# ── checkpoint retention ──────────────────────────────────────────────


class _Rec:
    def __init__(self, sid, cid):
        self.session_id = sid
        self.checkpoint_id = cid

    def to_dict(self):
        return {"session_id": self.session_id, "checkpoint_id": self.checkpoint_id}


def test_checkpoint_retention_bounds_file_count(tmp_path):
    """EFFECT PROOF: 300 writes leave at most KEEP_LAST files, newest kept
    (prod had 1,482 files because nothing ever pruned)."""
    p = FilePersister(base_dir=tmp_path)
    for i in range(300):
        p._write_sync(_Rec("sess", f"ck{i:04d}"))
    files = sorted((tmp_path / "sess").glob("*.json"))
    assert len(files) == FilePersister.KEEP_LAST
    names = {f.stem for f in files}
    assert "ck0299" in names and "ck0000" not in names


# ── parsed-line cache (whole-file re-read fix) ────────────────────────


@pytest.mark.asyncio
async def test_line_cache_hits_until_file_changes(tmp_path, monkeypatch):
    """EFFECT PROOF: repeat recent() calls parse the file ONCE; an append
    (mtime/size change) invalidates and re-parses exactly once more."""
    store = _mk_store(tmp_path)
    for i in range(50):
        await store.append(Turn(role="user", content=f"메시지 {i} " + "글" * 500))

    opens = {"n": 0}
    real_open = type(store._path).open

    def counting_open(self, *a, **k):
        if self == store._path:
            opens["n"] += 1
        return real_open(self, *a, **k)

    monkeypatch.setattr(type(store._path), "open", counting_open)

    store._lines_cache = None  # cold start
    r1 = await store.recent(10)
    assert opens["n"] == 1 and len(r1) == 10
    for _ in range(20):
        await store.recent(10)
        await store.search("메시지", limit=3)
    assert opens["n"] == 1, "repeat reads must be served from the cache"

    await store.append(Turn(role="user", content="새 메시지"))
    opens["n"] = 0
    r2 = await store.recent(1)
    assert r2[0].content == "새 메시지"
    assert opens["n"] == 1, "append must invalidate exactly once"
