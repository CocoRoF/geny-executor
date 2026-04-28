"""Default summarizers for Stage 19 (S9b.4)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance
from geny_executor.stages.s19_summarize.interface import Summarizer
from geny_executor.stages.s19_summarize.types import SummaryRecord


def _turn_id(state: PipelineState) -> str:
    sid = state.session_id or "session"
    return f"{sid}:{state.iteration}"


def _last_assistant_text(state: PipelineState) -> str:
    """Return the most recent assistant message text, or "" if none."""
    for msg in reversed(state.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        return text
        return ""
    return ""


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[-' ][A-Z][A-Za-z0-9]+)*\b")


class NoSummarizer(Summarizer):
    """Default. Skips the turn — Stage 19 publishes nothing."""

    @property
    def name(self) -> str:
        return "no_summary"

    @property
    def description(self) -> str:
        return "Skip summarisation this turn"

    async def summarize(self, state: PipelineState) -> Optional[SummaryRecord]:
        return None


class RuleBasedSummarizer(Summarizer):
    """Cheap rule-based summary built from the most recent assistant turn.

    Strategy:
      * abstract = first ``max_sentences`` sentences of the assistant
        text (or the full text if shorter).
      * key_facts = each remaining sentence, deduped, capped at
        ``max_facts``.
      * entities = capitalised tokens (single word or hyphen-joined),
        deduped, capped at ``max_entities``.
      * tags = caller-supplied static tags + 'rule_based'.

    Hosts that want a tighter summary should swap in an LLM-based
    summarizer; this one is for quick local extraction without a
    second API call.
    """

    DEFAULT_TAGS: tuple[str, ...] = ("rule_based",)

    def __init__(
        self,
        *,
        max_sentences: int = 3,
        max_facts: int = 5,
        max_entities: int = 8,
        extra_tags: Optional[List[str]] = None,
    ) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        if max_facts < 0 or max_entities < 0:
            raise ValueError("max_facts / max_entities must be non-negative")
        self._max_sentences = max_sentences
        self._max_facts = max_facts
        self._max_entities = max_entities
        self._extra_tags = list(extra_tags or [])

    @property
    def name(self) -> str:
        return "rule_based"

    @property
    def description(self) -> str:
        return "Sentence-split + capitalised-token extraction"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="rule_based",
            fields=[
                ConfigField(
                    name="max_sentences",
                    type="integer",
                    label="Max abstract sentences",
                    description="First N sentences of the assistant text become the summary abstract.",
                    default=3,
                    min_value=1,
                ),
                ConfigField(
                    name="max_facts",
                    type="integer",
                    label="Max key facts",
                    description="Cap on key_facts list (deduped sentences after the abstract).",
                    default=5,
                    min_value=0,
                ),
                ConfigField(
                    name="max_entities",
                    type="integer",
                    label="Max entities",
                    description="Cap on extracted capitalised tokens.",
                    default=8,
                    min_value=0,
                ),
                ConfigField(
                    name="extra_tags",
                    type="array",
                    item_type="string",
                    label="Extra tags",
                    description="Additional tags appended after the default 'rule_based' tag.",
                    default=[],
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        v = config.get("max_sentences")
        if isinstance(v, int) and v >= 1:
            self._max_sentences = v
        v = config.get("max_facts")
        if isinstance(v, int) and v >= 0:
            self._max_facts = v
        v = config.get("max_entities")
        if isinstance(v, int) and v >= 0:
            self._max_entities = v
        tags = config.get("extra_tags")
        if isinstance(tags, list):
            self._extra_tags = [str(t) for t in tags]

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_sentences": self._max_sentences,
            "max_facts": self._max_facts,
            "max_entities": self._max_entities,
            "extra_tags": list(self._extra_tags),
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for v in values:
            seen.setdefault(v, None)
        return list(seen.keys())

    def _entities(self, text: str) -> List[str]:
        candidates = _ENTITY_RE.findall(text or "")
        # Drop short tokens and pure single-letter matches.
        clean = [c for c in candidates if len(c) >= 2]
        deduped = self._dedupe(clean)
        return deduped[: self._max_entities]

    async def summarize(self, state: PipelineState) -> Optional[SummaryRecord]:
        text = _last_assistant_text(state)
        if not text.strip():
            return None
        sentences = self._split_sentences(text)
        if not sentences:
            return None

        abstract = " ".join(sentences[: self._max_sentences])
        remaining = sentences[self._max_sentences :]
        key_facts = self._dedupe(remaining)[: self._max_facts]
        entities = self._entities(text)
        tags = list(self.DEFAULT_TAGS) + list(self._extra_tags)

        return SummaryRecord(
            turn_id=_turn_id(state),
            abstract=abstract,
            key_facts=key_facts,
            entities=entities,
            tags=self._dedupe(tags),
            importance=Importance.MEDIUM,
        )


__all__ = [
    "NoSummarizer",
    "RuleBasedSummarizer",
]
