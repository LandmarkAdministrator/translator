"""
Settings Manager

Handles persistent configuration for the translator.
Saves and loads settings from config/settings.yaml.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


# Default settings
DEFAULT_SETTINGS = {
    "input_device": "default",
    "languages": [
        {
            "code": "es",
            "name": "Spanish",
            "output_device": "default",
            "output_channel": None,  # 0=left, 1=right, None=both/mono
            "enabled": True,
        },
        {
            "code": "ht",
            "name": "Haitian Creole",
            "output_device": "default",
            "output_channel": None,  # 0=left, 1=right, None=both/mono
            "enabled": True,
        },
    ],
    "asr_model": "large-v3",
}


class SettingsManager:
    """
    Manages persistent settings for the translator.

    Settings are stored in config/settings.yaml and use device names
    (not indices) for audio devices to survive reboots.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the settings manager.

        Args:
            config_dir: Directory for config files (default: project/config)
        """
        if config_dir is None:
            # Default to project_root/config
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self.settings_file = self.config_dir / "settings.yaml"
        self._settings: Dict[str, Any] = {}

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """
        Load settings from file.

        Returns:
            Settings dictionary (defaults if file doesn't exist)
        """
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    self._settings = yaml.safe_load(f) or {}
                print(f"Loaded settings from {self.settings_file}")
            except Exception as e:
                print(f"Warning: Could not load settings: {e}")
                self._settings = {}
        else:
            self._settings = {}

        # Merge with defaults for any missing keys
        self._settings = self._merge_defaults(self._settings)
        return self._settings

    def save(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """
        Save settings to file.

        Args:
            settings: Settings to save (uses current settings if None)
        """
        if settings is not None:
            self._settings = settings

        try:
            with open(self.settings_file, 'w') as f:
                yaml.dump(self._settings, f, default_flow_style=False, sort_keys=False)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _merge_defaults(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge settings with defaults — user values override, unknown keys preserved."""
        result = DEFAULT_SETTINGS.copy()
        if settings:
            result.update(settings)
        return result

    @property
    def settings(self) -> Dict[str, Any]:
        """Get current settings."""
        if not self._settings:
            self.load()
        return self._settings

    def get_input_device(self) -> str:
        """Get the configured input device name."""
        return self.settings.get("input_device", "default")

    def set_input_device(self, device_name: str) -> None:
        """Set the input device by name."""
        self._settings["input_device"] = device_name

    def get_languages(self) -> List[Dict[str, Any]]:
        """Get configured languages."""
        return self.settings.get("languages", DEFAULT_SETTINGS["languages"])

    def set_languages(self, languages: List[Dict[str, Any]]) -> None:
        """Set the language configurations."""
        self._settings["languages"] = languages

    def get_asr_model(self) -> str:
        """Get the ASR model size."""
        return self.settings.get("asr_model", "large-v3")

    def set_asr_model(self, model: str) -> None:
        """Set the ASR model size."""
        self._settings["asr_model"] = model


# Global settings manager instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get or create the global settings manager."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def load_settings() -> Dict[str, Any]:
    """Load settings from file."""
    return get_settings_manager().load()


def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to file."""
    get_settings_manager().save(settings)
