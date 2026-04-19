"""Tarball snapshot + restore for FileMemoryProvider.

Captures the entire `root` directory (STM JSONL, LTM markdown,
notes, index cache, optional vectordb) into a `tar.gz` payload plus
a SHA-256 checksum. Restore replaces the existing tree atomically
(new dir swapped in, old dir removed).

Tarball is in memory (`bytes`) rather than on disk because:
  - `MemorySnapshot.payload` is free-form and consumers may want to
    round-trip via a DB row, an S3 upload, or a web-API response.
  - The web mirror (Phase 4) will stream snapshots — keeping the
    payload out of the file tree simplifies the transport story.

For Very Large sessions this may exceed comfortable memory. Phase 5
hardening will add an optional disk-streaming mode. For Phase 2a the
format choice is explicit and documented.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
from pathlib import Path
from typing import Tuple


def build_tarball(root: Path) -> Tuple[bytes, str]:
    """Archive every file under `root` into an in-memory `.tar.gz`.
    Returns `(payload_bytes, sha256_hex)`.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        if root.exists():
            tar.add(root, arcname=".", recursive=True)
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def restore_tarball(root: Path, payload: bytes, checksum: str) -> None:
    """Replace `root` with the contents of the given `.tar.gz` bytes.

    Raises `ValueError` if the checksum does not match. The write is
    staged under a sibling `.restore-tmp` directory and swapped into
    place, so a mid-restore crash cannot leave the original tree in a
    partial state.
    """
    actual = hashlib.sha256(payload).hexdigest()
    if checksum and actual != checksum:
        raise ValueError(f"snapshot checksum mismatch: expected {checksum}, got {actual}")
    root = root.resolve()
    staging = root.with_name(root.name + ".restore-tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        _safe_extract(tar, staging)

    backup = root.with_name(root.name + ".restore-old")
    if backup.exists():
        shutil.rmtree(backup)
    if root.exists():
        root.rename(backup)
    staging.rename(root)
    if backup.exists():
        shutil.rmtree(backup)


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    """Guard against absolute paths / path traversal in archive
    entries. Every member must resolve *inside* `target`.
    """
    target = target.resolve()
    for member in tar.getmembers():
        member_path = (target / member.name).resolve()
        if target not in member_path.parents and member_path != target:
            raise ValueError(f"snapshot member {member.name!r} escapes the target directory")
    tar.extractall(target, filter="data")
