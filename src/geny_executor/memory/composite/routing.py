"""Per-layer routing table for `CompositeMemoryProvider`.

The composite holds a `LayerRouting` mapping each declared `Layer`
to the underlying `MemoryProvider` that owns it. The same provider
may serve more than one layer (the common case is a single SQL
provider serving STM + LTM + Notes + Vector + Index, with the
composite layer existing only to attach a separate scope-promotion
target).

Keeping this as a dataclass (instead of a bare dict) buys us:
  - explicit validation at construction time (every required layer
    must be claimed by some provider);
  - a stable iteration order over distinct providers, which the
    composite's snapshot/restore relies on for round-tripping;
  - a single place to grow scope-routing in the future without
    touching the provider call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple

from geny_executor.memory.provider import Layer, MemoryProvider, Scope


REQUIRED_LAYERS: Tuple[Layer, ...] = (
    Layer.STM,
    Layer.LTM,
    Layer.NOTES,
    Layer.INDEX,
)
OPTIONAL_LAYERS: Tuple[Layer, ...] = (
    Layer.VECTOR,
    Layer.CURATED,
    Layer.GLOBAL,
)


@dataclass(frozen=True)
class LayerRouting:
    """Maps each layer to the provider that owns it.

    `scope_providers` is an *additional* axis: the keys are scopes
    (SESSION / USER / TENANT / GLOBAL) and the values are providers
    where notes belonging to that scope live. The composite uses this
    table inside `promote(ref, to)` to copy a note from the source
    scope's provider into the target scope's provider.
    """

    layers: Mapping[Layer, MemoryProvider]
    scope_providers: Mapping[Scope, MemoryProvider] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [layer for layer in REQUIRED_LAYERS if layer not in self.layers]
        if missing:
            names = ", ".join(layer.value for layer in missing)
            raise ValueError(
                f"LayerRouting missing required layers: {names}. Required: STM, LTM, NOTES, INDEX."
            )

    def provider_for(self, layer: Layer) -> Optional[MemoryProvider]:
        return self.layers.get(layer)

    def has_layer(self, layer: Layer) -> bool:
        return layer in self.layers

    def scope_provider(self, scope: Scope) -> Optional[MemoryProvider]:
        return self.scope_providers.get(scope)

    def declared_layers(self) -> Set[Layer]:
        return set(self.layers.keys())

    def distinct_providers(self) -> List[MemoryProvider]:
        """Return each unique underlying provider exactly once,
        in the order they were first declared across (layers, scope_providers).
        Snapshot/restore iterate this list so the on-disk layout is stable.
        """
        seen_ids: Set[int] = set()
        out: List[MemoryProvider] = []
        for layer in (*REQUIRED_LAYERS, *OPTIONAL_LAYERS):
            prov = self.layers.get(layer)
            if prov is None:
                continue
            if id(prov) in seen_ids:
                continue
            seen_ids.add(id(prov))
            out.append(prov)
        for scope in (Scope.EPHEMERAL, Scope.SESSION, Scope.USER, Scope.TENANT, Scope.GLOBAL):
            prov = self.scope_providers.get(scope)
            if prov is None or id(prov) in seen_ids:
                continue
            seen_ids.add(id(prov))
            out.append(prov)
        return out

    def layers_owned_by(self, provider: MemoryProvider) -> List[Layer]:
        """Layers this specific provider owns inside the composite —
        used by retrieve() so we don't ask the same backend for the
        same layer twice.
        """
        return [layer for layer, owner in self.layers.items() if owner is provider]

    def provider_id(self, provider: MemoryProvider) -> str:
        """Stable identifier used in the composite snapshot payload.
        We tag by `<NAME>#<index>` so two providers of the same class
        but different DSNs stay distinguishable through a round-trip.
        """
        index = 0
        for candidate in self.distinct_providers():
            if candidate is provider:
                name = getattr(candidate, "NAME", type(candidate).__name__.lower())
                return f"{name}#{index}"
            index += 1
        raise KeyError("provider not registered in this routing table")

    def by_id(self) -> Dict[str, MemoryProvider]:
        out: Dict[str, MemoryProvider] = {}
        for index, prov in enumerate(self.distinct_providers()):
            name = getattr(prov, "NAME", type(prov).__name__.lower())
            out[f"{name}#{index}"] = prov
        return out


__all__ = [
    "LayerRouting",
    "REQUIRED_LAYERS",
    "OPTIONAL_LAYERS",
]
