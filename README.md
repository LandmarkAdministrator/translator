# Church Audio Translator

Real-time English speech translation for church services. The sound desk's
feed goes in; Spanish, Haitian Creole and Russian come out — as audio to the
room and as live text and audio on congregants' phones.

Built for congregations that serve speakers of several languages. Runs
entirely on local hardware after the models are downloaded; no subscriptions,
no cloud services.

## Features

- **Streaming ASR with punctuation** — NVIDIA `parakeet-unified-en-0.6b`
  (NeMo), decoding as the speaker talks. Sentences are sent to translation on
  the ASR's own sentence boundaries, not on a timer, so translators see whole
  thoughts (`docs/CHANGES-2026-09-06.md` §2 has the measurements).
- **One translation model for every language** — Meta NLLB-200 (1.3B
  distilled, fp16); the weights are loaded once and shared.
- **A voice per language** — Kokoro-82M for Spanish, Meta MMS-TTS for Haitian
  Creole and Russian, Piper as a fallback.
- **Simultaneous outputs** — two languages on one stereo interface (left /
  right), further languages on any other output the machine has.
- **A page for phones and screens** — live English and translated text plus
  per-language audio over WebSocket, installable as a PWA, always reachable
  (it shows a standby notice between services). English has two views —
  *Sentences* (what is translated) and *Live* (the recognizer's words, a few
  seconds earlier) — text scrolls up smoothly as new lines arrive, and the
  type size is adjustable; a TV in the auditorium is set up once by URL,
  e.g. `/?view=live&size=42&theme=dark`. `/?demo=1` plays a canned passage
  for judging the page without a service. An authenticated admin panel
  starts and stops translation and edits the schedule and audio routing.
- **Runs itself** — service windows from `config/schedule.conf`, a scheduler
  that starts and stops the service and restarts it on a real hang, and a GPU
  thermal guard.
- **Context biasing** — a phrase list (`config/bias_phrases.txt`) of Bible
  books, KJV forms and local names boosts them inside the decoder.
- **Fully offline** after setup; **open source** throughout.

## How it works

```
mixer feed ─► Parakeet ASR (NeMo subprocess) ─► fragments ─► SentenceBuffer ─► sentences
                                                                              │
              ┌───────────────────────────────────────────────────────────────┘
              ├─► NLLB en→es ─► Kokoro  ─► Behringer left   ┐
              ├─► NLLB en→ht ─► MMS-TTS ─► Behringer right  ├─► relay ─► web page (text + audio)
              └─► NLLB en→ru ─► MMS-TTS ─► onboard jack     ┘
```

Two processes: `translate.service` runs the pipeline during service windows;
`translate-web.service` serves the page and admin panel all the time and
receives events from the pipeline over a Unix socket.

## Hardware

| Component | Minimum | Production |
|-----------|---------|------------|
| GPU | CUDA or ROCm, ≥ 6 GB VRAM (NLLB fp16 ≈ 2.6 GB, Parakeet ≈ 1.5 GB) | NVIDIA RTX 3060 |
| CPU / RAM | Modern x86_64, 16 GB | Ryzen mini PC |
| Audio in | Any input; a feed from the mixer is far cleaner than a microphone | USB (Behringer UCA202) |
| Audio out | One stereo output per two languages | Behringer + onboard jack |
| Storage | 20 GB for models and both virtual environments | |

> **A GPU is required.** `run.py` refuses to start without one.

Tested on an NVIDIA RTX 3060 (CUDA 12.8; production) and an AMD Ryzen AI 9 HX
370 with Radeon 890M (ROCm; development).

## Installation

Debian 13 (Trixie). The installer sets up the main virtual environment,
drivers and models:

```bash
git clone https://github.com/LandmarkAdministrator/translator.git ~/translator
cd ~/translator
./install.sh                       # auto-detects the GPU; or --rocm / --cuda
```

The streaming ASR runs in a **second** virtual environment because NeMo needs
Python 3.11 and its own PyTorch build:

```bash
python3.11 -m venv ~/nemo-venv
~/nemo-venv/bin/pip install -r requirements-nemo.txt     # pinned from production
```

`scripts/install_site.sh` does exactly that, plus the configuration files,
the admin password, the scheduler and every systemd unit (and TLS with
`--tls`); `install.sh` calls it at the end, and it is safe to rerun after any
`git pull` (`--check` reports without changing anything). `./install.sh
--parakeet` additionally installs the onnx-asr Parakeet
TDT model, which the pipeline uses when `PARAKEET_MODEL` is not
`unified-remote` — a fallback for a machine without the NeMo venv, with lower
accuracy and no punctuation-driven sentence boundaries.

See [docs/SETUP.md](docs/SETUP.md) for the step-by-step guide.

## Running

```bash
./scripts/run_production.sh                       # live: sets every env var the stack needs
./scripts/run_production.sh --input-file x.wav    # the same pipeline over a file (reproducible tests)
./venv/bin/python run.py --setup                  # pick audio devices and languages
./venv/bin/python run.py --list-devices
```

`run_production.sh` is what the service runs; its exports (translation model
and device, one `<CODE>_TTS` backend per language, sentence-buffer policy) are
the production configuration. In production the launcher
`scripts/ops/start-translate-unified` wraps it with `PARAKEET_MODEL=unified-remote`,
the NeMo interpreter, and the biasing phrase list.

Before a service, `./venv/bin/python tests/test_pipeline_config.py` confirms
every configured language resolves a translation model and a voice, without
loading anything — the failure it exists for took the whole service down once.

## Running as a service

The production units are in `systemd/` and the scripts they run in
`scripts/ops/`:

| Unit | Role |
|------|------|
| `translate.service` (user) | the pipeline; `ExecStart` is the launcher above |
| `translate-web.service` (user) | page + admin panel, always on |
| `translate-window.timer` (user) | every 5 min: `translate-window-check.sh` starts/stops translation by `config/schedule.conf`, drains the archive backlog first, restarts a hung service, and keeps `ExecStart` pointed at the launcher |
| `gpu-thermal-guard.service` (user) | stops the backlog at 85 °C, and the service too if it is running |
| `translate-cert-renew.timer` (system) | daily at 03:20: renews the page's TLS certificate from the internal CA |

Logs: `~/translate.log` (the service's stdout, what the scheduler watches),
`logs/` (loguru), `journalctl --user -u translate-web`.

## Configuration

| File | Holds | Edited by |
|------|-------|-----------|
| `config/settings.yaml` | input device; per language the output device, channel and enabled flag | admin panel, `run.py --setup`, or by hand — comments survive a save |
| `config/schedule.conf` | service windows and the backlog drain lead time | admin panel or by hand |
| `config/site.json` | church name, service times, the languages the page offers and their strings | by hand; served to the page as `/config.json` |
| `config/bias_phrases.txt` | phrases boosted in the decoder | by hand |
| `scripts/run_production.sh` | models, devices, TTS backend per language, sentence-buffer policy | by hand |

Device names are matched by substring so ALSA card renumbering does not break
routing. The admin credentials live outside the repo in
`~/.config/translator/admin.json` (scrypt).

## Adding a language

1. NLLB needs a FLORES code in `NLLB_LANG_CODES` (`src/pipeline/translation.py`);
   the common ones are there.
2. Pick a voice: an MMS-TTS model id in `MMS_MODELS` (`src/pipeline/tts.py`),
   then `export XX_TTS="mms"` in `scripts/run_production.sh`. **Without the
   export the language falls through to Piper, which raises during load and
   crash-loops the whole service.**
3. Add the language to `config/settings.yaml` with an output device and
   channel (or use the admin panel), and to `config/site.json` so the page
   offers it.
4. Short phrases ("Amen.", "Let us pray.") bypass NLLB through
   `src/pipeline/translate_short_dict.py`; add entries reviewed by a speaker.
5. Run `tests/test_pipeline_config.py`.

## Performance

Measured on the RTX 3060 (2026-09-03 head-to-head against the retired
Whisper program, same sermon recording): word error rate **2.69 %** vs 3.34 %,
and about **8 s** from words spoken to translated audio vs 46 s. ASR decode is
~165 ms per 1.5 s chunk; NLLB ~0.5–0.9 s per sentence; TTS ~0.7 s.

## Project structure

```
translator/
├── run.py                        # entry point (streaming pipeline)
├── install.sh                    # installer: drivers, venv, models
├── requirements.txt              # main venv
├── requirements-nemo.txt         # NeMo venv (Python 3.11), frozen from production
├── config/                       # settings.yaml, schedule.conf, site.json, bias_phrases.txt
├── src/
│   ├── audio/                    # capture, resampling, stereo/channel output
│   ├── pipeline/                 # coordinator, ASR client + NeMo server, sentence buffer,
│   │                             #   translation, TTS, short-phrase dictionary
│   ├── web/                      # HTTP/WebSocket server, bus, relay, auth, admin, config API
│   └── config/, utils/
├── scripts/
│   ├── run_production.sh         # the production environment
│   └── ops/                      # launcher, scheduler, thermal guard, cert renewal
├── systemd/                      # the production units
├── tests/                        # unit + smoke tests, capture/replay and evaluation tools
└── docs/                         # SETUP, DEPLOYMENT, dated change records
```

## Dependency licences

| Component | Licence | Notes |
|-----------|---------|-------|
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Apache 2.0 | streaming ASR runtime |
| [parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | CC-BY-4.0 | ASR model |
| [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | CC-BY-NC-4.0 | translation model |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache 2.0 | Spanish voice |
| [MMS-TTS](https://huggingface.co/facebook/mms-tts-hat) | CC-BY-NC-4.0 | Creole and Russian voices |
| [Piper](https://github.com/rhasspy/piper) | MIT | fallback voices |
| [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers) | BSD-3 / Apache 2.0 | frameworks |
| [sounddevice](https://python-sounddevice.readthedocs.io/), [NumPy](https://numpy.org/), [loguru](https://github.com/Delgan/loguru) | MIT / BSD-3 / MIT | |

The models are downloaded from Hugging Face at setup and are not
redistributed here. NLLB-200 and MMS-TTS are licensed for non-commercial use.

## License

MIT — see [LICENSE](LICENSE). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

Built for multilingual church congregations, so that no one is left out of
worship for want of a language.
