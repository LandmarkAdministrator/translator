"""Configuration management for Church Audio Translator."""

from .loader import (
    load_config,
    load_language_config,
    load_preset,
    get_available_languages,
    get_available_presets,
    Config,
)

from .settings import (
    SettingsManager,
    get_settings_manager,
    load_settings,
    save_settings,
)

__all__ = [
    'load_config',
    'load_language_config',
    'load_preset',
    'get_available_languages',
    'get_available_presets',
    'Config',
    'SettingsManager',
    'get_settings_manager',
    'load_settings',
    'save_settings',
]
