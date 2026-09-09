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
- Full error output from the terminal, `journalctl --user -u translate.service -n 100`,
  or the tail of `~/translate.log`; `scripts/gpu_doctor.sh` output for GPU problems
- Steps to reproduce

## Adding a New Language

Translation is NLLB-200 for every language, so a new language needs no
translation model — only its FLORES code, a voice, and configuration. The
recipe is in the README under "Adding a language": the code in
`NLLB_LANG_CODES` (`src/pipeline/translation.py`), a voice — Kokoro where it
has one, otherwise an MMS-TTS model id in `MMS_MODELS` (`src/pipeline/tts.py`)
with the matching `XX_TTS` export in `scripts/run_production.sh`, Piper as the
last resort — an entry in `config/settings.yaml` and `config/site.json`,
short phrases in `src/pipeline/translate_short_dict.py` reviewed by a
speaker, and `tests/test_pipeline_config.py` passing. Russian (2026-09-06)
is the worked example: about half a day. Budget one extra NLLB pass
(~0.5 s) of latency per added audio language.

## Code Style

- Python: follow PEP 8, prefer clarity over cleverness
- No hard-coded paths or usernames
- Runtime-generated settings belong in `config/settings.yaml` (gitignored) — not committed files
