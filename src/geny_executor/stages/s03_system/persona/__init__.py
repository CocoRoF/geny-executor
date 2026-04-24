"""Built-in dynamic-persona system-prompt builder.

Cycle 20260424 executor uplift — Phase 7 Sprint S7.1.

Pre-Phase-7 the executor only shipped two ``PromptBuilder``
implementations: ``StaticPromptBuilder`` (a fixed string) and
``ComposablePromptBuilder`` (a fixed list of blocks). Hosts that
needed per-turn persona resolution (Geny's VTuber characters, mood
state, time-of-day prompts) had to implement their own
``DynamicPersonaSystemBuilder`` against the executor's interfaces
and re-do the same plumbing for every host.

Phase 7 S7.1 promotes that pattern into the executor. Hosts now
import :class:`PersonaProvider` (a Protocol) and
:class:`DynamicPersonaPromptBuilder` and only have to write the
provider — the builder + resolution lifecycle is shared.

The builder is registered in ``SystemStage`` under the
``dynamic_persona`` strategy name so manifests can swap it in.
"""

from geny_executor.stages.s03_system.persona.builder import (
    DynamicPersonaPromptBuilder,
)
from geny_executor.stages.s03_system.persona.provider import (
    PersonaProvider,
    PersonaResolution,
)

__all__ = [
    "DynamicPersonaPromptBuilder",
    "PersonaProvider",
    "PersonaResolution",
]
