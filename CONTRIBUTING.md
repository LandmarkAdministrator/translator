# Contributing

Contributions are welcome. This project is used in production at a real church, so stability matters — but improvements, bug fixes, and new language support are all appreciated.

## Ways to Contribute

- **Bug reports** — Open an issue with your OS, GPU, and the error output
- **New language support** — Language config files and voice selections for additional languages
- **Hardware testing** — Reports on what works (or doesn't) on specific AMD/NVIDIA hardware
- **Documentation** — Corrections, clarity improvements, translations of the setup guide
- **Code** — Bug fixes, performance improvements, new features

## Development Setup

```bash
git clone https://github.com/LandmarkAdministrator/translator.git
cd translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run tests:
```bash
python -m pytest tests/
```

## Pull Requests

1. Fork the repo and create a branch from `master`
2. Make your changes
3. Test on real hardware if possible (audio pipelines are hard to test without devices)
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Please include:
- OS and kernel version (`uname -a`)
- GPU model and driver version (`rocminfo` or `nvidia-smi`)
- Full error output from the terminal or `journalctl --user -u church-translator -n 100`
- Steps to reproduce

## Adding a New Language

To add support for a new language:

1. Find a Helsinki-NLP Opus-MT model for English → your language on [HuggingFace](https://huggingface.co/Helsinki-NLP)
2. Find a [Piper TTS voice](https://github.com/rhasspy/piper/blob/master/VOICES.md) for your language
3. Add the Opus-MT model ID to `TranslationService.MODEL_MAP` in [src/pipeline/translation.py](src/pipeline/translation.py)
4. Add the Piper voice entry to `VOICE_MAP` in [src/pipeline/tts.py](src/pipeline/tts.py)
5. Add the language to the `all_languages` list in [scripts/setup.py](scripts/setup.py)
6. Download models: `python scripts/download_models.py --all`
7. Test with `python run.py --setup` (enable the new language), then run the pipeline
8. Submit a pull request

## Code Style

- Python: follow PEP 8, prefer clarity over cleverness
- No hard-coded paths or usernames
- Runtime-generated settings belong in `config/settings.yaml` (gitignored) — not committed files
