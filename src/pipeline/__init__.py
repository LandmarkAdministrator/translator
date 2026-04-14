"""Pipeline modules for the Church Audio Translator."""

# Import modules as they become available
try:
    from .asr import ASRService, TranscriptionResult
except ImportError:
    ASRService = None
    TranscriptionResult = None

try:
    from .translation import TranslationService, TranslationResult
except ImportError:
    TranslationService = None
    TranslationResult = None

try:
    from .tts import TTSService, SpeechResult
except ImportError:
    TTSService = None
    SpeechResult = None

__all__ = [
    'ASRService', 'TranscriptionResult',
    'TranslationService', 'TranslationResult',
    'TTSService', 'SpeechResult',
]
