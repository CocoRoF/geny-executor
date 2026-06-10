"""Environment Diff — deep comparison of two environment snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set


def _order_keyed(items: List[Any]) -> Optional[Dict[int, Dict[str, Any]]]:
    """Return ``order → entry`` when *items* is an order-keyed entity list.

    A list qualifies when it is non-empty, every element is a dict
    carrying an int-able ``order`` key, and the orders are unique.
    Stage manifest entries are the canonical case; the shape test is
    structural so any future order-keyed list (and hosts' own payloads)
    benefits without a schema registry.

    Returns ``None`` when the list does not qualify — callers fall back
    to the positional strategies.
    """
    if not items or not all(isinstance(x, dict) and "order" in x for x in items):
        return None
    keyed: Dict[int, Dict[str, Any]] = {}
    for entry in items:
        try:
            order = int(entry["order"])
        except (TypeError, ValueError):
            return None
        if order in keyed:
            return None  # duplicate orders — positional fallback is honest
        keyed[order] = entry
    return keyed


@dataclass
class DiffEntry:
    """A single difference between two environment configs."""

    path: str  # JSON path (e.g., "model.temperature")
    change_type: str  # "added" | "removed" | "changed"
    old_value: Any = None
    new_value: Any = None

    def human_readable(self) -> str:
        if self.change_type == "added":
            return f"+ {self.path}: {self.new_value}"
        elif self.change_type == "removed":
            return f"- {self.path}: {self.old_value}"
        else:
            return f"~ {self.path}: {self.old_value} → {self.new_value}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiffEntry:
        return cls(
            path=data["path"],
            change_type=data["change_type"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
        )


@dataclass
class EnvironmentDiff:
    """Result of comparing two environment configs."""

    entries: List[DiffEntry] = field(default_factory=list)

    # ── Computed properties ────────────────────────────────

    @property
    def identical(self) -> bool:
        return len(self.entries) == 0

    @property
    def summary(self) -> Dict[str, int]:
        """Count of each change type."""
        counts = {"added": 0, "removed": 0, "changed": 0}
        for e in self.entries:
            counts[e.change_type] = counts.get(e.change_type, 0) + 1
        return counts

    @property
    def paths_changed(self) -> Set[str]:
        return {e.path for e in self.entries}

    # ── Filtering ──────────────────────────────────────────

    def filter_by_type(self, change_type: str) -> EnvironmentDiff:
        return EnvironmentDiff(entries=[e for e in self.entries if e.change_type == change_type])

    def filter_by_prefix(self, prefix: str) -> EnvironmentDiff:
        return EnvironmentDiff(entries=[e for e in self.entries if e.path.startswith(prefix)])

    # ── Computation ────────────────────────────────────────

    # Keys that are expected to differ (metadata IDs, timestamps)
    IGNORE_KEYS: ClassVar[Set[str]] = {
        "metadata.id",
        "metadata.created_at",
        "metadata.updated_at",
    }

    @classmethod
    def compute(
        cls,
        a: Dict[str, Any],
        b: Dict[str, Any],
        prefix: str = "",
        ignore_keys: Optional[Set[str]] = None,
    ) -> EnvironmentDiff:
        """Compute a deep diff between two dicts.

        Recursively walks both dicts, comparing values at each path.
        """
        if ignore_keys is None:
            ignore_keys = cls.IGNORE_KEYS

        entries: List[DiffEntry] = []
        all_keys = sorted(set(a.keys()) | set(b.keys()))

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key

            if path in ignore_keys:
                continue

            if key not in a:
                entries.append(DiffEntry(path, "added", new_value=b[key]))
            elif key not in b:
                entries.append(DiffEntry(path, "removed", old_value=a[key]))
            elif isinstance(a[key], dict) and isinstance(b[key], dict):
                sub = cls.compute(a[key], b[key], path, ignore_keys)
                entries.extend(sub.entries)
            elif isinstance(a[key], list) and isinstance(b[key], list):
                if a[key] != b[key]:
                    # Order-keyed entity lists (stage manifest entries)
                    # diff per-order rather than positionally. Before
                    # 2.2.0 (audit §3.1), two stage lists of unequal
                    # length collapsed into ONE opaque "changed" blob —
                    # a 16-stage stored manifest diffed against the
                    # 21-stage canonical layout reported nothing usable.
                    # Keying by ``order`` makes added / removed /
                    # changed stages individually addressable
                    # (``stages[order=N].…`` paths), and stays correct
                    # when one side reordered its array.
                    a_keyed = _order_keyed(a[key])
                    b_keyed = _order_keyed(b[key])
                    if a_keyed is not None and b_keyed is not None:
                        for order in sorted(set(a_keyed) | set(b_keyed)):
                            entry_path = f"{path}[order={order}]"
                            if order not in a_keyed:
                                entries.append(
                                    DiffEntry(entry_path, "added", new_value=b_keyed[order])
                                )
                            elif order not in b_keyed:
                                entries.append(
                                    DiffEntry(entry_path, "removed", old_value=a_keyed[order])
                                )
                            else:
                                sub = cls.compute(
                                    a_keyed[order], b_keyed[order], entry_path, ignore_keys
                                )
                                entries.extend(sub.entries)
                    # Positional fallback: same length + all dicts.
                    elif (
                        len(a[key]) == len(b[key])
                        and all(isinstance(x, dict) for x in a[key])
                        and all(isinstance(x, dict) for x in b[key])
                    ):
                        for i, (ai, bi) in enumerate(zip(a[key], b[key])):
                            sub = cls.compute(ai, bi, f"{path}[{i}]", ignore_keys)
                            entries.extend(sub.entries)
                    else:
                        entries.append(
                            DiffEntry(path, "changed", old_value=a[key], new_value=b[key])
                        )
            elif a[key] != b[key]:
                entries.append(DiffEntry(path, "changed", old_value=a[key], new_value=b[key]))

        return cls(entries=entries)

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> EnvironmentDiff:
        return cls(entries=[DiffEntry.from_dict(d) for d in data])
