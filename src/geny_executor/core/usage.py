"""One line of what a turn's model calls cost — for hosts and logs.

A turn's cost was a single dollar figure. Whether the prompt cache was
working — the difference between a turn that re-reads its history at a
tenth of the price and one that pays full price on every call — was nowhere
in it. The summary rides on the run's terminal event.
"""

from __future__ import annotations

from typing import Any, Dict


def prompt_tokens_of(usage: Any) -> int:
    """What the model read for one call, cache included, counted once."""
    prompt = getattr(usage, "prompt_tokens", None)
    if isinstance(prompt, int):
        return prompt
    return (
        int(getattr(usage, "input_tokens", 0) or 0)
        + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )


def turn_usage_summary(state: Any) -> Dict[str, Any]:
    """Calls, prompt sizes and cache share for the turn so far."""
    usages = list(getattr(state, "turn_token_usage", None) or [])
    prompts = []
    cache_read = cache_write = output = 0
    for u in usages:
        cache_read += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        output += int(getattr(u, "output_tokens", 0) or 0)
        prompts.append(prompt_tokens_of(u))
    total = sum(prompts)
    return {
        "calls": len(usages),
        "input_tokens": total,
        "first_prompt_tokens": prompts[0] if prompts else 0,
        "max_prompt_tokens": max(prompts) if prompts else 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output,
        "cache_read_share": round(cache_read / total, 3) if total else 0.0,
    }


__all__ = ["prompt_tokens_of", "turn_usage_summary"]
