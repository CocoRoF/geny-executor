"""Speech-to-text: provider Protocol + factory/registry + built-ins."""

from geny_executor.audio.stt.provider import (  # noqa: F401
    STT_ERROR_CATEGORIES,
    STTError,
    STTProvider,
    STTResult,
    STTSegment,
)
from geny_executor.audio.stt.registry import (  # noqa: F401
    create_stt_client,
    register_stt_provider,
    unregister_stt_provider,
)

__all__ = [
    "STT_ERROR_CATEGORIES",
    "STTError",
    "STTProvider",
    "STTResult",
    "STTSegment",
    "create_stt_client",
    "register_stt_provider",
    "unregister_stt_provider",
]
