#!/usr/bin/env python3
"""
Tests for the configuration system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    load_config,
    load_language_config,
    load_preset,
    get_available_languages,
    get_available_presets,
)


def test_load_config():
    """Test loading the main configuration."""
    print("Testing main config loading...")

    config = load_config()

    assert config.system.name == "Church Audio Translator"
    assert config.audio_input.sample_rate == 16000
    assert config.asr.model == "base.en"
    assert config.translation.use_gpu is True
    assert config.tts.sample_rate == 22050

    print(f"  System name: {config.system.name}")
    print(f"  ASR model: {config.asr.model}")
    print(f"  GPU enabled: {config.asr.use_gpu}")
    print("  ✓ Main config loaded successfully")


def test_load_spanish():
    """Test loading Spanish language config."""
    print("\nTesting Spanish language config...")

    lang = load_language_config('spanish')

    assert lang.language.code == "es"
    assert lang.language.name == "Spanish"
    assert "opus-mt-en-es" in lang.translation.model
    assert "davefx" in lang.tts.model

    print(f"  Language: {lang.language.name} ({lang.language.native_name})")
    print(f"  Translation model: {lang.translation.model}")
    print(f"  TTS model: {lang.tts.model}")
    print("  ✓ Spanish config loaded successfully")


def test_load_haitian_creole():
    """Test loading Haitian Creole language config."""
    print("\nTesting Haitian Creole language config...")

    lang = load_language_config('haitian_creole')

    assert lang.language.code == "ht"
    assert lang.language.name == "Haitian Creole"
    assert "opus-mt-en-ht" in lang.translation.model

    print(f"  Language: {lang.language.name} ({lang.language.native_name})")
    print(f"  Translation model: {lang.translation.model}")
    print(f"  TTS model: {lang.tts.model}")
    if lang.notes:
        print(f"  Notes: {len(lang.notes)} special notes")
    print("  ✓ Haitian Creole config loaded successfully")


def test_load_preset():
    """Test loading a preset."""
    print("\nTesting preset loading...")

    preset = load_preset('default')

    assert preset.preset.name == "Default"
    assert 'spanish' in preset.languages
    assert 'haitian_creole' in preset.languages
    assert preset.languages['spanish'].enabled is True

    print(f"  Preset name: {preset.preset.name}")
    print(f"  Languages: {list(preset.languages.keys())}")
    for lang, cfg in preset.languages.items():
        print(f"    {lang}: enabled={cfg.enabled}, device={cfg.output_device}")
    print("  ✓ Preset loaded successfully")


def test_available_configs():
    """Test listing available configurations."""
    print("\nTesting available configurations...")

    languages = get_available_languages()
    presets = get_available_presets()

    print(f"  Available languages: {languages}")
    print(f"  Available presets: {presets}")

    assert 'spanish' in languages
    assert 'haitian_creole' in languages
    assert 'default' in presets

    print("  ✓ Available configurations listed successfully")


def main():
    """Run all configuration tests."""
    print("=" * 60)
    print("Configuration System Tests")
    print("=" * 60)

    try:
        test_load_config()
        test_load_spanish()
        test_load_haitian_creole()
        test_load_preset()
        test_available_configs()

        print("\n" + "=" * 60)
        print("✓ All configuration tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
