# Church Audio Translator

Real-time English speech translation for church services. Audio in, translated audio out — simultaneously to multiple languages.

Built for congregations that serve speakers of multiple languages. Runs entirely offline on local hardware after setup, with no subscriptions or cloud services required.

## Features

- **Real-time** — English speech translated and spoken within 1–2 seconds
- **Simultaneous** — Multiple languages at once (Spanish + Haitian Creole out of the box)
- **Stereo channel split** — Two languages on one stereo output (Spanish left / Haitian Creole right)
- **GPU accelerated** — AMD ROCm or NVIDIA CUDA (2–3x faster than CPU)
- **Fully offline** — No internet required after initial setup and model download
- **Headless** — Runs as a systemd service, starts automatically at boot
- **Open source** — All components are FOSS

## How It Works

```
Microphone → Whisper ASR → English text
                              ├─→ Opus-MT (en→es) → Piper TTS → Spanish audio out
                              └─→ Opus-MT (en→ht) → Piper TTS → Haitian Creole audio out
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Modern x86_64 | AMD Ryzen / Intel Core (recent gen) |
| RAM | 8 GB | 16 GB+ |
| GPU | *(optional)* | AMD RDNA2+ or NVIDIA Maxwell+ |
| Storage | 15 GB free | 20 GB free |
| Audio in | Any microphone | USB audio interface |
| Audio out | Any speaker | Separate output per language |

**Tested on:**
- AMD Ryzen AI 9 HX 370 + Radeon 890M (ROCm)
- NVIDIA RTX 3060 (CUDA)

## Installation

Requires a fresh **Debian 13 (Trixie)** install. The installer handles everything else.

```bash
git clone https://github.com/LandmarkAdministrator/translator.git ~/translator
cd ~/translator
./install.sh
```

The installer auto-detects your GPU. To specify manually:

```bash
./install.sh --rocm    # AMD GPUs (RX 6000+, Radeon 680M / 780M / 890M)
./install.sh --cuda    # NVIDIA GPUs (GTX 900+, RTX series)
./install.sh --cpu     # CPU only
```

**What the installer does:**
1. Enables required Debian repositories
2. Installs GPU drivers (ROCm or NVIDIA kernel driver)
3. Creates Python virtual environment
4. Installs Python dependencies with the correct GPU backend
5. Downloads ML models (~4 GB)
6. Sets up the systemd service

See [docs/SETUP.md](docs/SETUP.md) for the full step-by-step guide and troubleshooting.

## First Run

```bash
source venv/bin/activate

# Interactive setup — select audio devices, test the pipeline, save config
python run.py --setup

# Run with saved configuration
python run.py

# Test GPU and all components
python run.py --test

# List available audio devices
python run.py --list-devices
```

## Running as a Service

```bash
# Install and enable the systemd user service
./scripts/install_service.sh install

# Start / stop / status
systemctl --user start church-translator
systemctl --user stop church-translator
systemctl --user status church-translator

# View live logs
journalctl --user -u church-translator -f
```

The service starts automatically at boot (no login required).

## Configuration

Audio devices and language settings are configured interactively with `python run.py --setup` and saved to `config/settings.yaml`. You can also create named presets in `config/presets/`.

Main configuration files:
- `config/config.yaml` — ASR, translation, TTS, and pipeline tuning
- `config/languages/spanish.yaml` — Spanish translation model and TTS voice
- `config/languages/haitian_creole.yaml` — Haitian Creole settings

## Adding Languages

The system supports any language pair that has:
1. A Helsinki-NLP Opus-MT model on HuggingFace (English → target)
2. A Piper TTS voice for the target language

To add a language, create a `config/languages/<code>.yaml` file following the existing examples and add it to your preset.

## Performance

| Stage | AMD 890M (ROCm) | NVIDIA RTX 3060 (CUDA) | CPU only |
|-------|-----------------|------------------------|----------|
| ASR (2s audio) | ~200ms | ~180ms | ~500ms |
| Translation | ~50ms | ~40ms | ~160ms |
| TTS | ~100ms | ~80ms | ~100ms |
| **End-to-end latency** | **~0.8s** | **~0.7s** | **~2–3s** |

## Project Structure

```
translator/
├── install.sh                  # Automated installer
├── run.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── config/
│   ├── config.yaml             # Core settings
│   ├── languages/              # Per-language model and voice config
│   └── presets/                # Named audio device configurations
├── src/
│   ├── audio/                  # Audio input/output with PipeWire/ALSA
│   ├── config/                 # Configuration loader
│   ├── pipeline/               # ASR, translation, TTS pipeline
│   └── utils/                  # GPU setup, logging
├── scripts/
│   ├── download_models.py      # Download ML models
│   ├── install_service.sh      # Systemd service installer
│   ├── test_gpu.py             # GPU verification
│   └── env.sh                  # Environment setup (AMD iGPU)
├── systemd/
│   └── translator.service      # Service file reference (generated by install_service.sh)
└── docs/
    ├── SETUP.md                # Full installation guide
    └── DEPLOYMENT.md           # Deployment and operations guide
```

## Dependency Licenses

This project uses the following open-source components:

| Component | License | Notes |
|-----------|---------|-------|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Speech recognition |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | Inference engine |
| [Helsinki-NLP Opus-MT](https://huggingface.co/Helsinki-NLP) | CC-BY 4.0 | Translation models |
| [Piper TTS](https://github.com/rhasspy/piper) | MIT | Text-to-speech |
| [piper-tts](https://github.com/rhasspy/piper) | MIT | Python TTS package |
| [PyTorch](https://pytorch.org/) | BSD 3-Clause | ML framework |
| [Transformers](https://github.com/huggingface/transformers) | Apache 2.0 | Model loading |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | MIT | Audio I/O |
| [NumPy](https://numpy.org/) | BSD 3-Clause | Array processing |
| [loguru](https://github.com/Delgan/loguru) | MIT | Logging |

**Translation model note:** The Helsinki-NLP Opus-MT models are released under CC-BY 4.0, which requires attribution for redistribution. This project does not redistribute the models — they are downloaded directly from HuggingFace during setup.

**GPU drivers:** AMD ROCm is open source (MIT/Apache 2.0). NVIDIA kernel drivers are proprietary but freely available. PyTorch bundles its own CUDA runtime — no separate CUDA toolkit installation is needed.

## License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it for any purpose, including commercial use.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

Built for multilingual church congregations. Inspired by the need to include everyone in worship regardless of language.
