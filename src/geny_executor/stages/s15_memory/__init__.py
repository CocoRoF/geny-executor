"""Stage 15: Memory — update and persist memory."""

from geny_executor.stages.s15_memory.stage import MemoryStage
from geny_executor.stages.s15_memory.strategies import (
    MemoryUpdateStrategy,
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
    StructuredReflectiveStrategy,
)
from geny_executor.stages.s15_memory.persistence import (
    ConversationPersistence,
    InMemoryPersistence,
    FilePersistence,
)
from geny_executor.stages.s15_memory.insight import (
    INSIGHTS_KEY,
    PENDING_INSIGHTS_KEY,
    coerce_insight,
    drain_pending_insights,
    insights_to_dicts,
    list_recorded_insights,
    record_insight,
)

__all__ = [
    "MemoryStage",
    "MemoryUpdateStrategy",
    "AppendOnlyStrategy",
    "NoMemoryStrategy",
    "ReflectiveStrategy",
    "StructuredReflectiveStrategy",
    "ConversationPersistence",
    "InMemoryPersistence",
    "FilePersistence",
    "INSIGHTS_KEY",
    "PENDING_INSIGHTS_KEY",
    "coerce_insight",
    "drain_pending_insights",
    "insights_to_dicts",
    "list_recorded_insights",
    "record_insight",
]
