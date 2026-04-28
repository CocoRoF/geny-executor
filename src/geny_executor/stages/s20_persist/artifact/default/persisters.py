"""Default persisters for Stage 20 (S9b.5)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s20_persist.interface import Persister
from geny_executor.stages.s20_persist.types import CheckpointRecord


class NoPersister(Persister):
    """Default. Writes nothing — kept so the stage is a zero-cost no-op
    until a host opts in."""

    @property
    def name(self) -> str:
        return "no_persist"

    @property
    def description(self) -> str:
        return "No-op persister"

    async def write(self, record: CheckpointRecord, state: PipelineState) -> None:
        return None


class FilePersister(Persister):
    """JSON-file checkpoint persister.

    One file per checkpoint, named ``<checkpoint_id>.json``. Files are
    grouped under ``base_dir/<session_id>/`` so listing by session is
    cheap. Writes are atomic via tempfile + ``os.replace``; the
    directory is created on first :meth:`write`.

    Hosts that need encryption or stronger durability should plug their
    own :class:`Persister` — this implementation is plaintext-on-disk.
    """

    DEFAULT_BASE_DIR = ".geny/checkpoints"

    def __init__(self, base_dir: str | os.PathLike[str] = DEFAULT_BASE_DIR) -> None:
        # Default lets the registry instantiate via cls() during a
        # manifest swap; configure() then overrides with the manifest's
        # actual base_dir.
        self._base = Path(base_dir)
        self._lock = Lock()

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "JSON-file checkpoint persister (one file per checkpoint)"

    @property
    def base_dir(self) -> Path:
        return self._base

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="file",
            fields=[
                ConfigField(
                    name="base_dir",
                    type="string",
                    label="Base directory",
                    description="Filesystem root for checkpoint files. Per-session subfolders are created automatically.",
                    default=cls.DEFAULT_BASE_DIR,
                    required=True,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        base = config.get("base_dir")
        if isinstance(base, str) and base.strip():
            self._base = Path(base)

    def get_config(self) -> Dict[str, Any]:
        return {"base_dir": str(self._base)}

    def _path_for(self, session_id: str, checkpoint_id: str) -> Path:
        return self._session_dir(session_id) / f"{checkpoint_id}.json"

    def _session_dir(self, session_id: str) -> Path:
        bucket = session_id or "_unknown"
        return self._base / bucket

    async def write(self, record: CheckpointRecord, state: PipelineState) -> None:
        # Run blocking IO in a thread so we don't stall the event loop.
        await asyncio.to_thread(self._write_sync, record)

    def _write_sync(self, record: CheckpointRecord) -> None:
        path = self._path_for(record.session_id, record.checkpoint_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record.to_dict(), fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)

    async def read(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        return await asyncio.to_thread(self._read_sync, checkpoint_id)

    def _read_sync(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        with self._lock:
            for session_dir in sorted(self._base.glob("*")):
                candidate = session_dir / f"{checkpoint_id}.json"
                if candidate.exists():
                    return self._record_from_path(candidate)
        return None

    @staticmethod
    def _record_from_path(path: Path) -> CheckpointRecord:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        from datetime import datetime

        created = data.get("created_at")
        created_dt = datetime.fromisoformat(created) if isinstance(created, str) else None
        kwargs: Dict[str, Any] = {
            "checkpoint_id": str(data.get("checkpoint_id", "")),
            "session_id": str(data.get("session_id", "")),
            "iteration": int(data.get("iteration", 0)),
            "payload": dict(data.get("payload") or {}),
        }
        if created_dt is not None:
            kwargs["created_at"] = created_dt
        return CheckpointRecord(**kwargs)

    async def list_checkpoints(self, session_id: str = "") -> List[CheckpointRecord]:
        return await asyncio.to_thread(self._list_sync, session_id)

    def _list_sync(self, session_id: str) -> List[CheckpointRecord]:
        with self._lock:
            session_dir = self._session_dir(session_id) if session_id else None
            paths: List[Path] = []
            if session_dir is not None:
                if session_dir.exists():
                    paths.extend(sorted(session_dir.glob("*.json")))
            else:
                for sd in sorted(self._base.glob("*")):
                    if sd.is_dir():
                        paths.extend(sorted(sd.glob("*.json")))
            out: List[CheckpointRecord] = []
            for p in paths:
                try:
                    out.append(self._record_from_path(p))
                except Exception:  # noqa: BLE001 — best-effort listing
                    continue
            return out


__all__ = [
    "FilePersister",
    "NoPersister",
]
