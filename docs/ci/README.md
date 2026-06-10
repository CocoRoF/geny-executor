# Pending CI workflow updates (2.2.0)

`2.2.0-workflow-updates.patch` carries the wave-3 CI changes that could not be
pushed from the automation environment: the GitHub OAuth token used by the
release tooling has `repo` scope but not `workflow` scope, and GitHub rejects
any push that modifies `.github/workflows/*` without it.

Contents of the patch:
- `ci.yml`: ruff now genuinely lints `tests/` (the old config silently
  force-excluded it), a mypy job over `core` + `llm_client`, and an
  events-docs `--check` step (`scripts/gen_event_docs.py`).
- `publish.yml`: a release gate asserting the pyproject version has a
  matching `## [<version>]` CHANGELOG header before publishing.

Apply from a checkout with workflow-scoped credentials:

    git apply docs/ci/2.2.0-workflow-updates.patch
    git add .github && git commit -m "ci: apply 2.2.0 workflow updates"
