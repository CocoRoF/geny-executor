"""``/model`` — show or change the session model."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)

# Light-weight allow list to catch obvious typos before they hit the
# API. Hosts that ship custom IDs override via ctx.extras["model_allow"].
_DEFAULT_ALLOW_PREFIXES = ("claude-",)


class ModelCommand(SlashCommand):
    name = "model"
    description = "Show or set the session model. Usage: /model [<model_id>]"
    category = SlashCategory.CONTROL

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        manifest = getattr(ctx.pipeline, "manifest", None)
        current = getattr(manifest, "model", None) if manifest else None
        if not args:
            return SlashResult(
                content=(f"Current model: `{current}`" if current else "No model set.")
            )
        new_model = args[0]
        allow_prefixes = ctx.extras.get("model_allow") or _DEFAULT_ALLOW_PREFIXES
        if not any(new_model.startswith(p) for p in allow_prefixes):
            return SlashResult(
                content=(
                    f"Model id `{new_model}` does not match allowed prefixes "
                    f"({', '.join(allow_prefixes)}). Pass model_allow via ctx.extras to override."
                ),
                success=False,
            )
        for method in ("set_model", "switch_model"):
            fn = getattr(ctx.pipeline, method, None)
            if callable(fn):
                try:
                    result = fn(new_model)
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:  # noqa: BLE001
                    return SlashResult(content=f"Model switch failed: {exc}", success=False)
                return SlashResult(content=f"✓ Model switched to `{new_model}`.")
        # Fall back: mutate manifest in place if no setter available.
        if manifest is not None:
            try:
                setattr(manifest, "model", new_model)
                return SlashResult(
                    content=f"✓ Manifest.model set to `{new_model}` (no setter exposed; mutation in place)."
                )
            except Exception as exc:  # noqa: BLE001
                return SlashResult(content=f"Manifest mutation failed: {exc}", success=False)
        return SlashResult(content="Pipeline does not expose set_model.", success=False)


__all__ = ["ModelCommand"]
