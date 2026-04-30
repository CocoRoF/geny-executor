---
name: Lorem Ipsum
description: Generate placeholder text — paragraphs, lists, code snippets, or markdown stubs.
category: utility
effort: low
when_to_use: When a draft, mock-up, demo, or test fixture needs filler text. Pick paragraph / list / code style based on the surrounding context. Don't use for real user-facing copy.
arguments:
  - count
  - style
argument_hint: "[count=3] [style=paragraph|list|code|markdown]"
examples:
  - 3 paragraphs of placeholder text
  - 5-item bullet list with realistic-looking labels
  - JSON snippet with example fields
  - Markdown blog post stub with sections
version: 1.0.0
---

# Lorem Ipsum — placeholder generator

You are generating filler / placeholder text. The user does not want
real content — they want something that *looks* like content so they
can wireframe a layout, demo a flow, or fill out a test fixture.

## Inputs

- `${count}` — how many units to produce. Defaults to **3** when the
  caller leaves it blank. Interpret "unit" based on the chosen style:
  paragraphs, list items, code blocks, etc.
- `${style}` — one of:
  - `paragraph` (default) — flowing prose, 3–5 sentences each.
  - `list` — bulleted items, each one short (≤ 12 words).
  - `code` — code-shaped placeholder. Pick a language that fits the
    surrounding context (TS / Python / JSON / SQL); use realistic
    identifiers (`fetchUserById`, `users.email`) instead of `foo`.
  - `markdown` — small markdown block with a heading, a paragraph,
    a list, and one code fence.

## How

1. Read the count + style. If style is missing, default to
   `paragraph`. If count is missing, default to **3**.
2. Generate `count` units in the chosen style.
3. Make the text *look real* — vary sentence length, use plausible
   technical terms when relevant, avoid the literal phrase "Lorem
   ipsum dolor sit amet" unless the caller asked for classic Latin
   filler.
4. No headings or commentary around the output — just the filler.

## Why this exists

Plain `lorem ipsum` text gives away that something is unfinished but
adds no signal about *what* the finished content will look like.
This skill produces context-aware filler so demos and wireframes
read more like the real thing.
