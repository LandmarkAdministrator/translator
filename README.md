# Church Audio Translator

Real-time English speech translation for church services. Audio in, translated audio out — simultaneously to multiple languages.

Built for congregations that serve speakers of multiple languages. Runs entirely offline on local hardware after setup, with no subscriptions or cloud services required.

## Features

- **Real-time** — Low-latency English speech translation (speed depends on hardware)
- **Two ASR backends** — Batched Whisper (default, GPU) or Parakeet TDT (`--parakeet`, CPU) for sub-second streaming updates
- **Simultaneous** — Multiple languages at once (Spanish + Haitian Creole out of the box)
- **Stereo channel split** — Two languages on one stereo output (Spanish left / Haitian Creole right)
- **GPU accelerated** — AMD ROCm 7.2.2 or NVIDIA CUDA **(required for Whisper backends — CPU-only Whisper is not supported)**
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
| GPU | **Required** — AMD RDNA2+ (RX 6000+, Radeon 680M/780M/890M) or NVIDIA Maxwell+ (GTX 900+, RTX) | ≥8 GB VRAM for `large-v3` |
| Storage | 15 GB free | 20 GB free |
| Audio in | Any microphone | USB audio interface |
| Audio out | Any speaker | Separate output per language |

> **GPU is mandatory.**  The translator runs `large-v3` Whisper by default,
> which is ~3× slower than real time on CPU — unusable for live services.
> `run.py` refuses to start if no GPU is detected.

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
./install.sh --rocm                # AMD GPUs (RX 6000+, Radeon 680M / 780M / 890M)
./install.sh --cuda                # NVIDIA GPUs (GTX 900+, RTX series)
./install.sh --rocm --parakeet     # Also install the Parakeet streaming backend
```

**What the installer does:**
1. Enables required Debian repositories
2. Installs GPU drivers (ROCm 7.2.2 or NVIDIA kernel driver)
3. Creates Python virtual environment
4. Installs Python dependencies with the correct GPU backend
5. Downloads ML models (~4 GB)
6. (Optional, with `--parakeet`) Installs `onnxruntime-rocm` + Parakeet TDT 0.6b v3 ONNX model
7. Sets up the systemd service

The Parakeet backend can also be added later by running
`./scripts/install_parakeet.sh` from inside the activated venv.

See [docs/SETUP.md](docs/SETUP.md) for the full step-by-step guide and troubleshooting.

## First Run

```bash
source venv/bin/activate

# Interactive setup — select audio devices, test the pipeline, save config
python run.py --setup

# Run with saved configuration (batched Whisper on GPU — default)
python run.py

# Parakeet TDT 0.6b v3 streaming ASR (CPU, ~sub-second commits)
python run.py --parakeet

# Test GPU and all components
python run.py --test

# List available audio devices
python run.py --list-devices
```

The default batched Whisper path waits for an utterance to finish before
translating. `--parakeet` emits partial transcripts in real time with a
token-level LocalAgreement-2 commit policy.

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

Audio devices and enabled languages are configured interactively with `python run.py --setup` and saved to `config/settings.yaml`. Edit that file directly (or re-run `--setup`) to change settings — there is no separate per-language YAML to edit. The ASR model is fixed at `large-v3` and is not configurable.

## Adding Languages

The system supports any language pair that has:
1. A Helsinki-NLP Opus-MT model on HuggingFace (English → target)
2. A Piper TTS voice for the target language

Adding a new language currently requires a small code change:
1. Add the Opus-MT model ID to `TranslationService.MODEL_MAP` in [src/pipeline/translation.py](src/pipeline/translation.py).
2. Add the Piper voice entry to `VOICE_MAP` in [src/pipeline/tts.py](src/pipeline/tts.py).
3. Add the language to the `all_languages` list in [scripts/setup.py](scripts/setup.py).
4. Download the models with `python scripts/download_models.py --all`.
5. Re-run `python run.py --setup` to enable the new language.

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
├── install.sh                  # Automated installer (--rocm | --cuda | --parakeet)
├── run.py                      # Main entry point (default batch Whisper | --parakeet)
├── requirements/
│   ├── base.txt                # Runtime deps (audio, yaml, librosa, hf hub…)
│   └── ml.txt                  # ML deps (torch, transformers, faster-whisper…)
├── config/
│   └── settings.yaml           # Generated by --setup (devices, languages)
├── src/
│   ├── audio/                  # Audio input/output with PipeWire/ALSA
│   ├── config/                 # Settings loader (config/settings.yaml)
│   ├── pipeline/               # ASR (batched Whisper / Parakeet), translation, TTS
│   └── utils/                  # GPU setup, logging
├── scripts/
│   ├── download_models.py      # Download ML models
│   ├── install_parakeet.sh     # Parakeet backend installer (onnxruntime-rocm + ONNX model)
│   ├── install_rocm.sh         # ROCm 7.2.2 installer
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
