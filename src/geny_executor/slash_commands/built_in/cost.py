"""``/cost`` — show current session token usage and estimated cost."""

from __future__ import annotations

from geny_executor.slash_commands.built_in._helpers import find_strategy, need_pipeline
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)


class CostCommand(SlashCommand):
    name = "cost"
    description = "Show current session token usage and estimated cost."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        early = need_pipeline(ctx)
        if early is not None:
            return early
        # Common slot names hosts wire for token bookkeeping.
        accountant = (
            find_strategy(ctx.pipeline, "token_accountant")
            or find_strategy(ctx.pipeline, "token_account")
        )
        if accountant is None:
            return SlashResult(
                content="No token accountant strategy configured.",
                success=False,
            )
        snapshot = _snapshot(accountant)
        if snapshot is None:
            return SlashResult(
                content="Token accountant did not expose a snapshot method.",
                success=False,
            )
        body_lines = ["**Session cost**", ""]
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = snapshot.get(key)
            if value is not None:
                body_lines.append(f"- {key.replace('_', ' ').title()}: {value:,}")
        usd = snapshot.get("estimated_usd")
        if usd is not None:
            body_lines.append(f"- Estimated USD: ${float(usd):.4f}")
        return SlashResult(content="\n".join(body_lines), metadata=snapshot)


def _snapshot(accountant) -> dict | None:
    for method in ("snapshot", "summary", "report", "to_dict"):
        fn = getattr(accountant, method, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                continue
            if isinstance(value, dict):
                return value
            # Pydantic / dataclass.
            for attr in ("model_dump", "dict", "__dict__"):
                conv = getattr(value, attr, None)
                if callable(conv):
                    try:
                        out = conv()
                        if isinstance(out, dict):
                            return out
                    except Exception:
                        pass
                elif isinstance(conv, dict):
                    return conv
    return None


__all__ = ["CostCommand"]
