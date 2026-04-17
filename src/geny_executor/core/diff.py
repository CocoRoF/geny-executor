"""Environment Diff — deep comparison of two environment snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set


@dataclass
class DiffEntry:
    """A single difference between two environment configs."""

    path: str               # JSON path (e.g., "model.temperature")
    change_type: str        # "added" | "removed" | "changed"
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
        return EnvironmentDiff(
            entries=[e for e in self.entries if e.change_type == change_type]
        )

    def filter_by_prefix(self, prefix: str) -> EnvironmentDiff:
        return EnvironmentDiff(
            entries=[e for e in self.entries if e.path.startswith(prefix)]
        )

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
                    # For lists, compare element-by-element if same length and dicts
                    if (
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
                entries.append(
                    DiffEntry(path, "changed", old_value=a[key], new_value=b[key])
                )

        return cls(entries=entries)

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> EnvironmentDiff:
        return cls(entries=[DiffEntry.from_dict(d) for d in data])
