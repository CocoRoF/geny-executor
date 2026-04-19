"""Composite snapshot encoder/decoder.

A composite snapshot is a JSON envelope that nests one
`MemorySnapshot` per distinct underlying provider. The envelope shape
is::

    {
        "format": "composite",
        "version": "1.0.0",
        "delegates": {
            "<provider_id>": {
                "provider": "...",
                "version": "...",
                "layers": [...],
                "size_bytes": N,
                "checksum": "...",
                "payload_kind": "bytes" | "json",
                "payload": <base64 str | json value>
            }, ...
        }
    }

`payload_kind` records whether the wrapped payload was bytes (file /
sql tarballs) or a JSON value (ephemeral). The composite snapshot
itself carries its own SHA-256 checksum over the JSON document so
tampering is detectable at the envelope level too.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from geny_executor.memory.provider import Layer, MemorySnapshot


COMPOSITE_FORMAT = "composite"
COMPOSITE_VERSION = "1.0.0"


def encode_snapshot(by_id: Dict[str, MemorySnapshot]) -> Tuple[bytes, str]:
    """Pack per-delegate snapshots into one canonical JSON document.

    Returns (payload_bytes, sha256_hex).
    """
    delegates: Dict[str, Dict[str, Any]] = {}
    for provider_id, snap in by_id.items():
        payload_kind, payload_value = _encode_payload(snap.payload)
        delegates[provider_id] = {
            "provider": snap.provider,
            "version": snap.version,
            "layers": [layer.value for layer in snap.layers],
            "size_bytes": snap.size_bytes,
            "checksum": snap.checksum,
            "created_at": snap.created_at.isoformat() if snap.created_at else None,
            "payload_kind": payload_kind,
            "payload": payload_value,
        }
    document = {
        "format": COMPOSITE_FORMAT,
        "version": COMPOSITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "delegates": delegates,
    }
    blob = json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return blob, hashlib.sha256(blob).hexdigest()


def decode_snapshot(payload: bytes, expected_checksum: str) -> Dict[str, MemorySnapshot]:
    """Unpack the JSON envelope back into MemorySnapshot objects keyed
    by their provider_id. Raises `ValueError` on checksum mismatch or
    malformed envelope.
    """
    actual = hashlib.sha256(payload).hexdigest()
    if expected_checksum and actual != expected_checksum:
        raise ValueError(
            f"composite snapshot checksum mismatch: expected {expected_checksum!r}, got {actual!r}"
        )
    document = json.loads(payload.decode("utf-8"))
    if document.get("format") != COMPOSITE_FORMAT:
        raise ValueError(
            f"composite snapshot format must be {COMPOSITE_FORMAT!r}, got {document.get('format')!r}"
        )
    out: Dict[str, MemorySnapshot] = {}
    for provider_id, sub in document.get("delegates", {}).items():
        out[provider_id] = MemorySnapshot(
            provider=str(sub["provider"]),
            version=str(sub["version"]),
            layers=[Layer(value) for value in sub.get("layers", [])],
            payload=_decode_payload(sub.get("payload_kind", "json"), sub.get("payload")),
            size_bytes=int(sub.get("size_bytes", 0)),
            checksum=str(sub.get("checksum", "")),
        )
    return out


def _encode_payload(payload: Any) -> Tuple[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        return "bytes", base64.b64encode(bytes(payload)).decode("ascii")
    return "json", payload


def _decode_payload(kind: str, value: Any) -> Any:
    if kind == "bytes":
        if value is None:
            return b""
        return base64.b64decode(str(value).encode("ascii"))
    return value


__all__ = [
    "COMPOSITE_FORMAT",
    "COMPOSITE_VERSION",
    "encode_snapshot",
    "decode_snapshot",
]
