#!/usr/bin/env python3
"""
Church Audio Translator - Main Entry Point

Usage:
    python run.py              # Run with saved settings
    python run.py --setup      # Run interactive setup
    python run.py --test       # Test GPU and components
    python run.py --help       # Show help
"""

import os
import sys
from pathlib import Path

# Load ROCm environment variables if present (written by install.sh for AMD iGPUs).
# This sets HSA_OVERRIDE_GFX_VERSION and PATH for AMD GPUs before any GPU libs load.
# On NVIDIA or CPU-only systems this file won't exist and nothing is set.
_env_file = Path(__file__).parent / ".env.rocm"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("export "):
                _line = _line[7:]
            if "=" in _line and not _line.startswith("#"):
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Church Audio Translator - Real-time speech translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Run with saved settings
  python run.py --setup            Interactive setup wizard
  python run.py --test             Test GPU and pipeline components
  python run.py -l es fr           Override languages (Spanish and French)
  python run.py -i "Onyx Producer" Override input device by name
        """
    )

    parser.add_argument("--setup", action="store_true",
                        help="Run interactive setup wizard")
    parser.add_argument("--test", action="store_true",
                        help="Test GPU and pipeline components")
    parser.add_argument("-i", "--input",
                        help="Override input audio device (name or index)")
    parser.add_argument("-l", "--languages", nargs="+",
                        help="Override target languages (es, ht, fr, pt, de)")
    parser.add_argument("-m", "--model",
                        help="Override ASR model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices")

    args = parser.parse_args()

    # List devices
    if args.list_devices:
        from audio.device_manager import AudioDeviceManager
        manager = AudioDeviceManager()
        manager.print_devices()
        return 0

    # Run setup wizard
    if args.setup:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "setup", PROJECT_ROOT / "scripts" / "setup.py"
        )
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)
        setup_module.main_menu()
        return 0

    # Run tests
    if args.test:
        print("Running GPU test...")
        os.execv(sys.executable, [sys.executable, str(PROJECT_ROOT / "scripts" / "test_gpu.py")])
        return 0

    # Load saved settings
    from config.settings import load_settings
    from audio.device_manager import AudioDeviceManager
    from pipeline.coordinator import TranslationCoordinator, PipelineConfig

    config = load_settings()
    manager = AudioDeviceManager()

    language_names = {
        "es": "Spanish",
        "ht": "Haitian Creole",
        "fr": "French",
        "pt": "Portuguese",
        "de": "German",
    }

    print("=" * 60)
    print("Church Audio Translator")
    print("=" * 60)

    # Resolve input device (command line overrides saved config)
    if args.input:
        input_device_spec = args.input
        print(f"Input device (from command line): {args.input}")
    else:
        input_device_spec = config.get("input_device", "default")
        print(f"Input device (from saved config): {input_device_spec}")

    input_dev = manager.resolve_device(input_device_spec, 'input')
    if input_dev:
        input_device = str(input_dev.index)
        print(f"  -> Resolved to: {input_dev.name} (index {input_dev.index})")
    else:
        input_device = "default"
        print(f"  -> Could not resolve, using system default")

    # Build language configs
    if args.languages:
        # Command line override
        print(f"Languages (from command line): {', '.join(args.languages)}")
        pipeline_configs = [
            PipelineConfig(
                language_code=lang,
                language_name=language_names.get(lang, lang),
            )
            for lang in args.languages
        ]
    else:
        # Use saved config
        print(f"Languages (from saved config):")
        pipeline_configs = []
        for lang in config.get("languages", []):
            if lang.get("enabled", False):
                # Get output channel (0=left, 1=right, None=both)
                output_channel = lang.get("output_channel", None)
                channel_str = ""
                if output_channel is not None:
                    channel_str = f" [{'LEFT' if output_channel == 0 else 'RIGHT'} channel]"

                output_dev = manager.resolve_device(lang.get("output_device", "default"), 'output')
                if output_dev:
                    output_device = str(output_dev.index)
                    print(f"  {lang['name']}: {output_dev.name} (index {output_dev.index}){channel_str}")
                else:
                    output_device = "default"
                    print(f"  {lang['name']}: system default{channel_str}")

                pipeline_configs.append(PipelineConfig(
                    language_code=lang["code"],
                    language_name=lang["name"],
                    output_device=output_device,
                    output_channel=output_channel,
                    enabled=True,
                ))

    if not pipeline_configs:
        print("\nNo languages configured! Run with --setup to configure.")
        return 1

    # Resolve ASR model (command line overrides saved config)
    if args.model:
        asr_model = args.model
        print(f"ASR model (from command line): {asr_model}")
    else:
        asr_model = config.get("asr_model", "large-v3")
        print(f"ASR model (from saved config): {asr_model}")
    print("=" * 60)

    coordinator = TranslationCoordinator(
        input_device=input_device,
        languages=pipeline_configs,
        asr_model=asr_model,
    )

    coordinator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
