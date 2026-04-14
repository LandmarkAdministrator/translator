"""
Configuration Loader Module

Handles loading and validation of YAML configuration files for the
Church Audio Translator system.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/config to project root
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Get the configuration directory."""
    return get_project_root() / "config"


@dataclass
class AudioInputConfig:
    """Audio input configuration."""
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0
    chunk_overlap: float = 0.5
    buffer_size: float = 30.0


@dataclass
class ASRConfig:
    """ASR (Speech Recognition) configuration."""
    model: str = "base.en"
    language: str = "en"
    use_gpu: bool = True
    compute_type: str = "float16"
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.25
    max_speech_duration: float = 30.0


@dataclass
class TranslationConfig:
    """Translation configuration."""
    use_gpu: bool = True
    compute_type: str = "float16"
    beam_size: int = 4
    max_length: int = 256


@dataclass
class TTSConfig:
    """TTS (Text-to-Speech) configuration."""
    sample_rate: int = 22050
    speed: float = 1.0
    use_gpu: bool = True


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    max_pipelines: int = 4
    queue_size: int = 100
    timeout: float = 10.0
    shutdown_timeout: float = 5.0


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    asr_threads: int = 4
    translation_threads: int = 2
    gpu_memory_fraction: float = 0.8
    prewarm_models: bool = True


@dataclass
class SystemConfig:
    """System configuration."""
    name: str = "Church Audio Translator"
    version: str = "0.1.0"
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"


@dataclass
class Config:
    """Main configuration container."""
    system: SystemConfig = field(default_factory=SystemConfig)
    audio_input: AudioInputConfig = field(default_factory=AudioInputConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class LanguageTranslationConfig:
    """Language-specific translation configuration."""
    model: str
    fallback_models: List[str] = field(default_factory=list)


@dataclass
class LanguageTTSConfig:
    """Language-specific TTS configuration."""
    model: str
    model_url: Optional[str] = None
    config_url: Optional[str] = None
    speaker_id: int = -1
    rate: float = 1.0
    fallback_voice: Optional[Dict[str, str]] = None
    alternative_voices: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class LanguageAudioConfig:
    """Language-specific audio output configuration."""
    device: Optional[str] = None
    sample_rate: int = 22050
    channels: int = 1


@dataclass
class LanguageInfo:
    """Language metadata."""
    code: str
    name: str
    native_name: str


@dataclass
class LanguageConfig:
    """Complete language configuration."""
    language: LanguageInfo
    translation: LanguageTranslationConfig
    tts: LanguageTTSConfig
    audio_output: LanguageAudioConfig
    notes: List[str] = field(default_factory=list)


@dataclass
class PresetLanguageConfig:
    """Language configuration within a preset."""
    enabled: bool = True
    output_device: str = "default"


@dataclass
class PresetInfo:
    """Preset metadata."""
    name: str
    description: str = ""


@dataclass
class Preset:
    """Preset configuration."""
    preset: PresetInfo
    audio_input: Dict[str, Any] = field(default_factory=dict)
    languages: Dict[str, PresetLanguageConfig] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)


def _dict_to_dataclass(data: Dict[str, Any], cls: type) -> Any:
    """Convert a dictionary to a dataclass instance."""
    if data is None:
        return cls()

    # Get the field names and types from the dataclass
    field_names = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to only include valid fields
    filtered_data = {k: v for k, v in data.items() if k in field_names}

    return cls(**filtered_data)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load the main configuration file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Config dataclass with all settings.
    """
    if config_path is None:
        config_path = get_config_dir() / "config.yaml"

    data = load_yaml(config_path)

    return Config(
        system=_dict_to_dataclass(data.get('system', {}), SystemConfig),
        audio_input=_dict_to_dataclass(data.get('audio_input', {}), AudioInputConfig),
        asr=_dict_to_dataclass(data.get('asr', {}), ASRConfig),
        translation=_dict_to_dataclass(data.get('translation', {}), TranslationConfig),
        tts=_dict_to_dataclass(data.get('tts', {}), TTSConfig),
        pipeline=_dict_to_dataclass(data.get('pipeline', {}), PipelineConfig),
        performance=_dict_to_dataclass(data.get('performance', {}), PerformanceConfig),
    )


def load_language_config(language_code: str) -> LanguageConfig:
    """
    Load a language configuration file.

    Args:
        language_code: Language code (e.g., 'spanish', 'haitian_creole')

    Returns:
        LanguageConfig dataclass with language settings.
    """
    # Map common codes to file names
    code_map = {
        'es': 'spanish',
        'spanish': 'spanish',
        'ht': 'haitian_creole',
        'haitian_creole': 'haitian_creole',
        'haitian': 'haitian_creole',
    }

    file_name = code_map.get(language_code.lower(), language_code.lower())
    config_path = get_config_dir() / "languages" / f"{file_name}.yaml"

    data = load_yaml(config_path)

    # Parse language info
    lang_data = data.get('language', {})
    language_info = LanguageInfo(
        code=lang_data.get('code', ''),
        name=lang_data.get('name', ''),
        native_name=lang_data.get('native_name', ''),
    )

    # Parse translation config
    trans_data = data.get('translation', {})
    translation_config = LanguageTranslationConfig(
        model=trans_data.get('model', ''),
        fallback_models=trans_data.get('fallback_models', []),
    )

    # Parse TTS config
    tts_data = data.get('tts', {})
    tts_config = LanguageTTSConfig(
        model=tts_data.get('model', ''),
        model_url=tts_data.get('model_url'),
        config_url=tts_data.get('config_url'),
        speaker_id=tts_data.get('speaker_id', -1),
        rate=tts_data.get('rate', 1.0),
        fallback_voice=tts_data.get('fallback_voice'),
        alternative_voices=tts_data.get('alternative_voices', []),
    )

    # Parse audio output config
    audio_data = data.get('audio_output', {})
    audio_config = LanguageAudioConfig(
        device=audio_data.get('device'),
        sample_rate=audio_data.get('sample_rate', 22050),
        channels=audio_data.get('channels', 1),
    )

    return LanguageConfig(
        language=language_info,
        translation=translation_config,
        tts=tts_config,
        audio_output=audio_config,
        notes=data.get('notes', []),
    )


def load_preset(preset_name: str) -> Preset:
    """
    Load a preset configuration file.

    Args:
        preset_name: Name of the preset (e.g., 'default', 'sunday_service')

    Returns:
        Preset dataclass with preset settings.
    """
    config_path = get_config_dir() / "presets" / f"{preset_name}.yaml"
    data = load_yaml(config_path)

    # Parse preset info
    preset_data = data.get('preset', {})
    preset_info = PresetInfo(
        name=preset_data.get('name', preset_name),
        description=preset_data.get('description', ''),
    )

    # Parse languages
    languages = {}
    for lang_name, lang_data in data.get('languages', {}).items():
        if isinstance(lang_data, dict):
            languages[lang_name] = PresetLanguageConfig(
                enabled=lang_data.get('enabled', True),
                output_device=lang_data.get('output_device', 'default'),
            )

    return Preset(
        preset=preset_info,
        audio_input=data.get('audio_input', {}),
        languages=languages,
        overrides=data.get('overrides', {}),
    )


def get_available_languages() -> List[str]:
    """
    Get list of available language configurations.

    Returns:
        List of language names (without .yaml extension)
    """
    languages_dir = get_config_dir() / "languages"
    if not languages_dir.exists():
        return []

    return [
        f.stem for f in languages_dir.glob("*.yaml")
        if f.is_file() and not f.name.startswith('_')
    ]


def get_available_presets() -> List[str]:
    """
    Get list of available presets.

    Returns:
        List of preset names (without .yaml extension)
    """
    presets_dir = get_config_dir() / "presets"
    if not presets_dir.exists():
        return []

    return [
        f.stem for f in presets_dir.glob("*.yaml")
        if f.is_file() and not f.name.startswith('_')
    ]


def merge_config_with_preset(config: Config, preset: Preset) -> Config:
    """
    Merge base configuration with preset overrides.

    Args:
        config: Base configuration
        preset: Preset with overrides

    Returns:
        New Config with preset overrides applied
    """
    import copy

    # Create a deep copy of the config
    merged = copy.deepcopy(config)

    # Apply overrides
    for section, values in preset.overrides.items():
        if hasattr(merged, section) and isinstance(values, dict):
            section_obj = getattr(merged, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    # Apply audio input device from preset
    if 'device' in preset.audio_input:
        merged.audio_input.device = preset.audio_input['device']

    return merged
