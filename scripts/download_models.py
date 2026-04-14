#!/usr/bin/env python3
"""
Model Download Script for Church Audio Translator

Downloads and caches all required AI models:
- Whisper ASR models (speech-to-text)
- Opus-MT translation models (en->es, en->ht, etc.)
- Piper TTS models (text-to-speech)

Usage:
    python download_models.py --all           # Download all models
    python download_models.py --asr           # Download ASR models only
    python download_models.py --translation   # Download translation models only
    python download_models.py --tts           # Download TTS models only
    python download_models.py --list          # List available models
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")


# Model definitions
ASR_MODELS = {
    "tiny.en": {
        "name": "Whisper Tiny English",
        "size_mb": 75,
        "description": "Fastest, lowest accuracy (English only)",
    },
    "base.en": {
        "name": "Whisper Base English",
        "size_mb": 140,
        "description": "Fast, good for clear speech (English only)",
    },
    "small.en": {
        "name": "Whisper Small English",
        "size_mb": 460,
        "description": "Good balance of speed and accuracy (English only)",
    },
    "medium.en": {
        "name": "Whisper Medium English",
        "size_mb": 1500,
        "description": "High accuracy (English only)",
    },
    "large-v3": {
        "name": "Whisper Large V3",
        "size_mb": 3000,
        "description": "Highest accuracy, multilingual",
        "default": True,
    },
}

TRANSLATION_MODELS = {
    "Helsinki-NLP/opus-mt-en-es": {
        "name": "English to Spanish",
        "size_mb": 300,
        "language": "es",
        "default": True,
    },
    "Helsinki-NLP/opus-mt-en-ht": {
        "name": "English to Haitian Creole",
        "size_mb": 300,
        "language": "ht",
        "default": True,
    },
    "Helsinki-NLP/opus-mt-en-fr": {
        "name": "English to French",
        "size_mb": 300,
        "language": "fr",
    },
    "Helsinki-NLP/opus-mt-en-pt": {
        "name": "English to Portuguese",
        "size_mb": 300,
        "language": "pt",
    },
    "Helsinki-NLP/opus-mt-en-de": {
        "name": "English to German",
        "size_mb": 300,
        "language": "de",
    },
}

TTS_MODELS = {
    "es_ES-davefx-medium": {
        "name": "Spanish (Spain) - Dave FX",
        "language": "es",
        "size_mb": 60,
        "default": True,
    },
    "es_MX-ald-medium": {
        "name": "Spanish (Mexico) - Ald",
        "language": "es",
        "size_mb": 60,
    },
    "fr_FR-upmc-medium": {
        "name": "French - UPMC",
        "language": "fr",
        "size_mb": 60,
        "note": "Also used for Haitian Creole",
        "default": True,
    },
    "pt_BR-faber-medium": {
        "name": "Portuguese (Brazil) - Faber",
        "language": "pt",
        "size_mb": 60,
    },
    "de_DE-thorsten-medium": {
        "name": "German - Thorsten",
        "language": "de",
        "size_mb": 60,
    },
}


def get_models_dir() -> Path:
    """Get the models directory."""
    # Check environment variable first
    if "MODELS_DIR" in os.environ:
        return Path(os.environ["MODELS_DIR"])

    # Use project default
    return PROJECT_ROOT / "models"


def format_size(size_mb: int) -> str:
    """Format size in MB or GB."""
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def list_models():
    """List all available models."""
    print_header("Available Models")

    print("ASR (Speech-to-Text) Models:")
    print("-" * 40)
    for model_id, info in ASR_MODELS.items():
        default = " [DEFAULT]" if info.get("default") else ""
        print(f"  {model_id}: {info['name']}{default}")
        print(f"    Size: {format_size(info['size_mb'])}")
        print(f"    {info['description']}")
        print()

    print("\nTranslation Models:")
    print("-" * 40)
    for model_id, info in TRANSLATION_MODELS.items():
        default = " [DEFAULT]" if info.get("default") else ""
        print(f"  {model_id}: {info['name']}{default}")
        print(f"    Size: {format_size(info['size_mb'])}")
        print()

    print("\nTTS (Text-to-Speech) Models:")
    print("-" * 40)
    for model_id, info in TTS_MODELS.items():
        default = " [DEFAULT]" if info.get("default") else ""
        note = f" ({info['note']})" if info.get("note") else ""
        print(f"  {model_id}: {info['name']}{default}{note}")
        print(f"    Size: {format_size(info['size_mb'])}")
        print()


def download_asr_model(model_id: str, models_dir: Path) -> bool:
    """Download a Whisper ASR model."""
    if model_id not in ASR_MODELS:
        print(f"Unknown ASR model: {model_id}")
        return False

    info = ASR_MODELS[model_id]
    print(f"Downloading ASR model: {info['name']} ({format_size(info['size_mb'])})")

    try:
        from faster_whisper import WhisperModel

        # Download by loading the model
        download_root = str(models_dir / "asr")
        os.makedirs(download_root, exist_ok=True)

        # This will download the model if not present
        model = WhisperModel(
            model_id,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
        )

        # Unload model
        del model

        print(f"  ✓ Downloaded {model_id}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {model_id}: {e}")
        return False


def download_translation_model(model_id: str, models_dir: Path) -> bool:
    """Download a translation model."""
    if model_id not in TRANSLATION_MODELS:
        print(f"Unknown translation model: {model_id}")
        return False

    info = TRANSLATION_MODELS[model_id]
    print(f"Downloading translation model: {info['name']} ({format_size(info['size_mb'])})")

    try:
        from transformers import MarianMTModel, MarianTokenizer

        cache_dir = str(models_dir / "translation")
        os.makedirs(cache_dir, exist_ok=True)

        # Download model and tokenizer
        print(f"  Downloading tokenizer...")
        MarianTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        print(f"  Downloading model weights...")
        MarianMTModel.from_pretrained(model_id, cache_dir=cache_dir)

        print(f"  ✓ Downloaded {model_id}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {model_id}: {e}")
        return False


def download_tts_model(model_id: str, models_dir: Path) -> bool:
    """Download a Piper TTS model."""
    if model_id not in TTS_MODELS:
        print(f"Unknown TTS model: {model_id}")
        return False

    info = TTS_MODELS[model_id]
    print(f"Downloading TTS model: {info['name']} ({format_size(info['size_mb'])})")

    try:
        import urllib.request

        tts_dir = models_dir / "tts"
        os.makedirs(tts_dir, exist_ok=True)

        # Piper model URL pattern
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

        # Parse model name to get path
        parts = model_id.split("-")
        lang_region = parts[0]  # e.g., "es_ES"
        voice_name = parts[1]  # e.g., "davefx"
        quality = parts[2] if len(parts) > 2 else "medium"  # e.g., "medium"

        # Construct URLs
        lang_code = lang_region.split("_")[0]  # e.g., "es"
        model_path = f"{lang_code}/{lang_region}/{voice_name}/{quality}"

        onnx_url = f"{base_url}/{model_path}/{model_id}.onnx"
        json_url = f"{base_url}/{model_path}/{model_id}.onnx.json"

        onnx_file = tts_dir / f"{model_id}.onnx"
        json_file = tts_dir / f"{model_id}.onnx.json"

        # Download files
        if not onnx_file.exists():
            print(f"  Downloading model file...")
            urllib.request.urlretrieve(onnx_url, onnx_file)

        if not json_file.exists():
            print(f"  Downloading config file...")
            urllib.request.urlretrieve(json_url, json_file)

        print(f"  ✓ Downloaded {model_id}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {model_id}: {e}")
        return False


def download_all_defaults(models_dir: Path):
    """Download all default models."""
    print_header("Downloading Default Models")

    total_size = 0
    for models in [ASR_MODELS, TRANSLATION_MODELS, TTS_MODELS]:
        for info in models.values():
            if info.get("default"):
                total_size += info["size_mb"]

    print(f"Total download size: approximately {format_size(total_size)}")
    print()

    # Download ASR models
    print("ASR Models:")
    for model_id, info in ASR_MODELS.items():
        if info.get("default"):
            download_asr_model(model_id, models_dir)

    # Download translation models
    print("\nTranslation Models:")
    for model_id, info in TRANSLATION_MODELS.items():
        if info.get("default"):
            download_translation_model(model_id, models_dir)

    # Download TTS models
    print("\nTTS Models:")
    for model_id, info in TTS_MODELS.items():
        if info.get("default"):
            download_tts_model(model_id, models_dir)


def download_asr_models(models_dir: Path, model_ids: Optional[List[str]] = None):
    """Download ASR models."""
    print_header("Downloading ASR Models")

    if model_ids is None:
        # Download default only
        model_ids = [m for m, info in ASR_MODELS.items() if info.get("default")]

    for model_id in model_ids:
        download_asr_model(model_id, models_dir)


def download_translation_models(models_dir: Path, model_ids: Optional[List[str]] = None):
    """Download translation models."""
    print_header("Downloading Translation Models")

    if model_ids is None:
        # Download defaults only
        model_ids = [m for m, info in TRANSLATION_MODELS.items() if info.get("default")]

    for model_id in model_ids:
        download_translation_model(model_id, models_dir)


def download_tts_models(models_dir: Path, model_ids: Optional[List[str]] = None):
    """Download TTS models."""
    print_header("Downloading TTS Models")

    if model_ids is None:
        # Download defaults only
        model_ids = [m for m, info in TTS_MODELS.items() if info.get("default")]

    for model_id in model_ids:
        download_tts_model(model_id, models_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download AI models for Church Audio Translator"
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Download all default models"
    )
    parser.add_argument(
        "--asr", action="store_true",
        help="Download ASR (speech-to-text) models"
    )
    parser.add_argument(
        "--translation", action="store_true",
        help="Download translation models"
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="Download TTS (text-to-speech) models"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--models-dir", type=str,
        help="Custom models directory"
    )
    parser.add_argument(
        "--model", type=str, action="append",
        help="Specific model to download (can be used multiple times)"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_models()
        return 0

    # Get models directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = get_models_dir()

    print(f"Models directory: {models_dir}")

    # Handle specific models
    if args.model:
        for model_id in args.model:
            if model_id in ASR_MODELS:
                download_asr_model(model_id, models_dir)
            elif model_id in TRANSLATION_MODELS:
                download_translation_model(model_id, models_dir)
            elif model_id in TTS_MODELS:
                download_tts_model(model_id, models_dir)
            else:
                print(f"Unknown model: {model_id}")
        return 0

    # Handle category downloads
    if args.all:
        download_all_defaults(models_dir)
    else:
        if args.asr:
            download_asr_models(models_dir)
        if args.translation:
            download_translation_models(models_dir)
        if args.tts:
            download_tts_models(models_dir)

        # If nothing specified, show help
        if not (args.asr or args.translation or args.tts):
            parser.print_help()
            print("\nUse --all to download all default models.")
            return 1

    print("\n" + "=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
