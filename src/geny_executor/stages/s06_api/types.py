"""Back-compat re-export shim — canonical types live in llm_client.types.

Existing code that imports from ``geny_executor.stages.s06_api.types``
(tests, vendored providers, external stages) keeps working unchanged.
New code should import from :mod:`geny_executor.llm_client.types`.

This shim is deleted in PR-4 along with the ``s06_api/artifact/*``
provider directories.
"""

from __future__ import annotations

from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

__all__ = ["APIRequest", "APIResponse", "ContentBlock"]
