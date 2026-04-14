#!/usr/bin/env python3
"""
Interactive Setup Script for Church Audio Translator

This script provides an interactive CLI for:
- Selecting audio input/output devices (by name for persistence)
- Configuring target languages
- Testing the translation pipeline
- Launching the translator

Settings are automatically saved to config/settings.yaml
"""

import os
import sys
from pathlib import Path

# Load GPU environment from .env.rocm if present (written by install.sh for AMD)
PROJECT_ROOT = Path(__file__).parent.parent
_env_file = PROJECT_ROOT / ".env.rocm"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("export "):
                _line = _line[7:]
            if "=" in _line and not _line.startswith("#"):
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from typing import List, Dict, Optional

from audio.device_manager import AudioDeviceManager, AudioDevice
from config.settings import SettingsManager, load_settings, save_settings


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_menu(options: List[str], title: str = "Options"):
    """Print a numbered menu."""
    print(f"\n{title}:")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print()


def get_choice(prompt: str, max_val: int, allow_zero: bool = False) -> int:
    """Get a numeric choice from user."""
    min_val = 0 if allow_zero else 1
    while True:
        try:
            choice = input(f"{prompt} [{min_val}-{max_val}]: ").strip()
            if not choice:
                return min_val
            val = int(choice)
            if min_val <= val <= max_val:
                return val
            print(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no response from user."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not response:
        return default
    return response in ('y', 'yes')


def list_audio_devices(manager: AudioDeviceManager):
    """List available audio devices."""
    print_header("Audio Devices")

    print("INPUT DEVICES:")
    print("-" * 50)
    for dev in manager.get_input_devices():
        default = " (default)" if dev.is_default_input else ""
        print(f"  [{dev.index}] {dev.name}{default}")

    print("\nOUTPUT DEVICES:")
    print("-" * 50)
    for dev in manager.get_output_devices():
        default = " (default)" if dev.is_default_output else ""
        print(f"  [{dev.index}] {dev.name}{default}")


def select_input_device(manager: AudioDeviceManager, current: str = "default") -> str:
    """
    Select an input device.

    Returns:
        Device name (not index) for persistence across reboots
    """
    devices = manager.get_input_devices()
    default_dev = manager.get_default_input()

    # Find current device by name
    current_dev = manager.resolve_device(current, 'input')
    current_name = current_dev.name if current_dev else "default"

    print(f"\nSelect INPUT device (current: {current_name}):")
    for i, dev in enumerate(devices):
        marker = " *" if dev.name == current_name else ""
        default_marker = " (system default)" if dev.is_default_input else ""
        print(f"  {i + 1}. {dev.name}{default_marker}{marker}")
    print(f"  0. Use system default ({default_dev.name if default_dev else 'none'})")

    choice = get_choice("Choice", len(devices), allow_zero=True)

    if choice == 0:
        return "default"
    return devices[choice - 1].name


def select_output_device(manager: AudioDeviceManager, language: str, current: str = "default") -> str:
    """
    Select an output device for a language.

    Returns:
        Device name (not index) for persistence across reboots
    """
    devices = manager.get_output_devices()
    default_dev = manager.get_default_output()

    # Find current device by name
    current_dev = manager.resolve_device(current, 'output')
    current_name = current_dev.name if current_dev else "default"

    print(f"\nSelect OUTPUT device for {language} (current: {current_name}):")
    for i, dev in enumerate(devices):
        marker = " *" if dev.name == current_name else ""
        default_marker = " (system default)" if dev.is_default_output else ""
        print(f"  {i + 1}. {dev.name}{default_marker}{marker}")
    print(f"  0. Use system default ({default_dev.name if default_dev else 'none'})")

    choice = get_choice("Choice", len(devices), allow_zero=True)

    if choice == 0:
        return "default"
    return devices[choice - 1].name


def select_output_channel(language: str, current_channel: Optional[int] = None) -> Optional[int]:
    """
    Select stereo channel for output (left/right/both).

    This allows multiple languages to share a single stereo device
    by outputting to different channels.

    Returns:
        0 for left, 1 for right, None for both/mono
    """
    current_str = "both channels"
    if current_channel == 0:
        current_str = "left channel"
    elif current_channel == 1:
        current_str = "right channel"

    print(f"\nSelect stereo channel for {language} (current: {current_str}):")
    print("  1. Both channels (mono output to both L+R)")
    print("  2. Left channel only")
    print("  3. Right channel only")

    choice = get_choice("Choice", 3)

    if choice == 1:
        return None  # Both channels
    elif choice == 2:
        return 0  # Left
    else:
        return 1  # Right


def get_available_languages() -> List[tuple]:
    """Return language pairs whose models are actually downloaded."""
    from pipeline.translation import TranslationService

    all_languages = [
        ("es", "Spanish"),
        ("ht", "Haitian Creole"),
        ("fr", "French"),
        ("pt", "Portuguese"),
        ("de", "German"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("ar", "Arabic"),
        ("ru", "Russian"),
    ]

    models_dir = PROJECT_ROOT / "models" / "translation"
    available = []
    for code, name in all_languages:
        key = ("en", code)
        if key not in TranslationService.MODEL_MAP:
            continue
        model_id = TranslationService.MODEL_MAP[key]
        # HuggingFace cache uses "models--org--name" directory naming
        cache_dir_name = "models--" + model_id.replace("/", "--")
        if (models_dir / cache_dir_name).exists():
            available.append((code, name))
        else:
            print(f"  (skipping {name} — model not downloaded)")

    return available


def configure_languages(manager: AudioDeviceManager, current_languages: List[Dict]) -> List[Dict]:
    """Configure target languages."""
    available_languages = get_available_languages()

    # Build lookup for current settings
    current_by_code = {lang["code"]: lang for lang in current_languages}

    print_header("Language Configuration")
    print("Select languages to enable:\n")

    selected = []
    for code, name in available_languages:
        # Check if currently enabled
        current = current_by_code.get(code, {})
        default_enabled = current.get("enabled", code in ['es', 'ht'])
        current_output = current.get("output_device", "default")
        current_channel = current.get("output_channel", None)

        if get_yes_no(f"Enable {name} ({code})?", default=default_enabled):
            output_device = select_output_device(manager, name, current_output)

            # Ask about stereo channel assignment
            if get_yes_no(f"Assign {name} to a specific stereo channel?", default=current_channel is not None):
                output_channel = select_output_channel(name, current_channel)
            else:
                output_channel = None

            selected.append({
                "code": code,
                "name": name,
                "output_device": output_device,
                "output_channel": output_channel,
                "enabled": True,
            })

    return selected


def resolve_device_for_display(manager: AudioDeviceManager, device_spec: str, direction: str) -> str:
    """Resolve device spec to display name."""
    if device_spec == "default":
        return "system default"
    dev = manager.resolve_device(device_spec, direction)
    if dev:
        return f"{dev.name} (index {dev.index})"
    return f"{device_spec} (not found!)"


def test_pipeline(config: Dict, manager: AudioDeviceManager):
    """Test the translation pipeline with sample text."""
    from pipeline.coordinator import PipelineConfig

    print_header("Testing Pipeline")
    print("This will test the translation and TTS for each enabled language.\n")

    # Resolve input device
    input_device_spec = config.get("input_device", "default")
    input_dev = manager.resolve_device(input_device_spec, 'input')
    if input_dev:
        input_device = str(input_dev.index)
        print(f"Input device: {input_dev.name} (index {input_dev.index})")
    else:
        input_device = "default"
        print(f"Input device: system default")

    # Build pipeline configs with resolved device indices
    pipeline_configs = []
    for lang in config.get("languages", []):
        if lang.get("enabled", False):
            output_dev = manager.resolve_device(lang.get("output_device", "default"), 'output')
            output_channel = lang.get("output_channel", None)
            channel_str = ""
            if output_channel == 0:
                channel_str = " [LEFT channel]"
            elif output_channel == 1:
                channel_str = " [RIGHT channel]"

            if output_dev:
                output_device = str(output_dev.index)
                print(f"{lang['name']} output: {output_dev.name} (index {output_dev.index}){channel_str}")
            else:
                output_device = "default"
                print(f"{lang['name']} output: system default{channel_str}")

            pipeline_configs.append(PipelineConfig(
                language_code=lang["code"],
                language_name=lang["name"],
                output_device=output_device,
                output_channel=output_channel,
                enabled=True,
            ))

    if not pipeline_configs:
        print("No languages enabled!")
        return

    print("\nLoading models (this may take a moment)...")

    from pipeline.translation import TranslationService
    from pipeline.tts import TTSService
    from audio.output_stream import AudioOutputStream

    test_text = "Hello and welcome to our church service today. God bless you."

    for lang_config in pipeline_configs:
        print(f"\nTesting {lang_config.language_name}...")

        try:
            # Translate
            translator = TranslationService(
                source_language="en",
                target_language=lang_config.language_code,
            )
            translator.load()
            translation = translator.translate(test_text)
            print(f"  Translation: {translation.translated_text}")

            # TTS
            tts = TTSService(language=lang_config.language_code)
            tts.load()
            speech = tts.synthesize(translation.translated_text)
            print(f"  Audio duration: {speech.duration:.2f}s")

            # Play audio
            if get_yes_no("  Play audio?", default=True):
                output = AudioOutputStream(
                    device=lang_config.output_device,
                    sample_rate=speech.sample_rate,
                    stereo_channel=lang_config.output_channel,
                )
                output.start()
                output.play(speech.audio, sample_rate=speech.sample_rate, block=True)
                output.stop()

            translator.unload()
            tts.unload()
            print(f"  ✓ {lang_config.language_name} working!")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nPipeline test complete!")


def run_translator(config: Dict, manager: AudioDeviceManager):
    """Run the translator with the given configuration."""
    from pipeline.coordinator import TranslationCoordinator, PipelineConfig
    from utils.logger import setup_logger
    setup_logger(log_dir="logs", log_level="INFO")

    print_header("Starting Translator")

    # Resolve input device name to index
    input_device_spec = config.get("input_device", "default")
    input_dev = manager.resolve_device(input_device_spec, 'input')
    if input_dev:
        input_device = str(input_dev.index)
        print(f"Input device: {input_dev.name} (index {input_dev.index})")
    else:
        input_device = "default"
        print(f"Input device: system default (could not resolve '{input_device_spec}')")

    # Build pipeline configs with resolved device indices
    pipeline_configs = []
    for lang in config.get("languages", []):
        if lang.get("enabled", False):
            output_dev = manager.resolve_device(lang.get("output_device", "default"), 'output')
            output_channel = lang.get("output_channel", None)
            channel_str = ""
            if output_channel == 0:
                channel_str = " [LEFT channel]"
            elif output_channel == 1:
                channel_str = " [RIGHT channel]"

            if output_dev:
                output_device = str(output_dev.index)
                print(f"{lang['name']} output: {output_dev.name} (index {output_dev.index}){channel_str}")
            else:
                output_device = "default"
                print(f"{lang['name']} output: system default (could not resolve '{lang.get('output_device')}'){channel_str}")

            pipeline_configs.append(PipelineConfig(
                language_code=lang["code"],
                language_name=lang["name"],
                output_device=output_device,
                output_channel=output_channel,
                enabled=True,
            ))

    if not pipeline_configs:
        print("No languages enabled!")
        return

    print()

    # Ask about ASR mode
    print("\nASR Mode:")
    print("  1. Batch (silence-based chunks, original behaviour)")
    print("  2. Streaming (rolling re-transcription, better sentence coherence)")
    mode_choice = get_choice("Select mode", 2)
    streaming = (mode_choice == 2)

    # Ask about GPU ASR
    gpu_asr = get_yes_no("\nRun Whisper on GPU? (requires ROCm/CUDA, recommended with large-v3)", default=False)
    asr_device = "cuda" if gpu_asr else "cpu"

    # ASR model from saved config
    asr_model = config.get("asr_model", "base.en")
    print(f"ASR model: {asr_model}  (change with --setup → model config, or --model flag)")

    # Create and run coordinator
    coordinator = TranslationCoordinator(
        input_device=input_device,
        languages=pipeline_configs,
        streaming=streaming,
        asr_device=asr_device,
        asr_model=asr_model,
    )

    print("Starting translation system...")
    print("Press Ctrl+C to stop.\n")

    coordinator.run()


def main_menu():
    """Main interactive menu."""
    # Initialize device manager
    manager = AudioDeviceManager()

    # Load saved settings
    config = load_settings()
    print(f"Loaded settings from config/settings.yaml")

    while True:
        clear_screen()
        print_header("Church Audio Translator - Setup")

        # Refresh device manager for accurate display
        manager.refresh()

        print("Current Configuration:")
        input_display = resolve_device_for_display(manager, config.get('input_device', 'default'), 'input')
        print(f"  Input device: {input_display}")
        print(f"  Languages:")
        for lang in config.get('languages', []):
            status = "✓" if lang.get('enabled', False) else "✗"
            output = resolve_device_for_display(manager, lang.get('output_device', 'default'), 'output')
            channel = lang.get('output_channel', None)
            channel_str = ""
            if channel == 0:
                channel_str = " [LEFT]"
            elif channel == 1:
                channel_str = " [RIGHT]"
            print(f"    {status} {lang['name']} -> {output}{channel_str}")

        asr_model = config.get("asr_model", "base.en")
        print(f"  ASR model:  {asr_model}")

        options = [
            "List audio devices",
            "Configure audio input",
            "Configure languages",
            "Configure ASR model",
            "Test pipeline",
            "Run translator",
            "Exit",
        ]

        print_menu(options, "Menu")
        choice = get_choice("Select option", len(options))

        if choice == 1:
            list_audio_devices(manager)
            input("\nPress Enter to continue...")

        elif choice == 2:
            manager.refresh()
            config["input_device"] = select_input_device(manager, config.get("input_device", "default"))
            save_settings(config)
            print("\nConfiguration saved!")
            input("Press Enter to continue...")

        elif choice == 3:
            manager.refresh()
            config["languages"] = configure_languages(manager, config.get("languages", []))
            save_settings(config)
            print("\nConfiguration saved!")
            input("Press Enter to continue...")

        elif choice == 4:
            models = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
            print("\nSelect ASR model:")
            for i, m in enumerate(models, 1):
                current = " (current)" if m == config.get("asr_model", "base.en") else ""
                print(f"  {i}. {m}{current}")
            model_choice = get_choice("Choice", len(models))
            config["asr_model"] = models[model_choice - 1]
            save_settings(config)
            print(f"\nASR model set to: {config['asr_model']}")
            input("Press Enter to continue...")

        elif choice == 5:
            manager.refresh()
            test_pipeline(config, manager)
            input("\nPress Enter to continue...")

        elif choice == 6:
            manager.refresh()
            run_translator(config, manager)

        elif choice == 7:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
