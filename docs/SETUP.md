# Church Audio Translator - Setup Guide

Complete setup guide for installing the Church Audio Translator on a fresh Debian 13 system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install](#quick-install)
- [Manual Installation](#manual-installation)
- [GPU-Specific Setup](#gpu-specific-setup)
- [Audio Device Configuration](#audio-device-configuration)
- [First Run](#first-run)
- [Running as a Service](#running-as-a-service)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **CPU:** Modern x86_64 processor (AMD or Intel)
- **RAM:** 8GB minimum, 16GB recommended
- **GPU (required):** CPU-only operation is not supported — the translator
  refuses to start without a ROCm or CUDA GPU.  NLLB-200 1.3B (fp16, ≈2.6 GB)
  and the Parakeet streaming ASR (≈1.5 GB) both run on it.
  - AMD: RDNA2 or newer (RX 6000+, Radeon 680M/780M/890M)
  - NVIDIA: Maxwell or newer (GTX 900+, RTX series); production is an RTX 3060
  - ≥6 GB VRAM
- **Audio:** One input device, multiple output devices for different languages
- **Storage:** 10GB free space for models and dependencies

### Software Requirements

- **OS:** Debian 13 (Trixie) - fresh install recommended
- **Kernel:** 6.10+ (from backports for best GPU support)
- **Python:** 3.11+ (included in Debian 13)

---

## Quick Install

The install is two scripts. `./install.sh` handles the operating system: GPU
drivers, the main virtual environment, base models. `scripts/install_site.sh`
handles the site: the NeMo virtual environment for the streaming ASR, the
configuration files, the admin password, the scheduler and every systemd
unit, optionally TLS from an internal CA and a model prefetch. `install.sh`
calls it at the end; run it again by itself after any `git pull` — it is
idempotent and reports each step as ok / changed / skipped / FAILED, and
`--check` reports without changing anything.

```bash
./install.sh --cuda                                   # or --rocm; then, or later:
./scripts/install_site.sh --check                     # what is missing
./scripts/install_site.sh --web-host 0.0.0.0          # LAN page without a proxy
./scripts/install_site.sh --tls https://ca.example/acme/directory \
    --ca-cert root.crt --domain translate.example.org --email admin@example.org
./scripts/install_site.sh --prefetch                  # download all models now
```

Afterwards: sign in to `/admin`, set the audio devices and service windows,
and edit `config/site.json` (church name, service times, languages).


For a fully automated installation on Debian 13:

```bash
# Download the translator (or clone from git)
cd /home/$USER
git clone https://github.com/LandmarkAdministrator/translator.git translator
cd translator

# Run the installer (auto-detects GPU)
./install.sh

# Or specify GPU type explicitly:
./install.sh --rocm               # For AMD GPUs
./install.sh --cuda               # For NVIDIA GPUs

# Add --parakeet to also set up the onnx-asr Parakeet TDT model — the ASR
# fallback for a machine without the NeMo virtual environment (below):
./install.sh --rocm --parakeet
```

The production ASR (`nvidia/parakeet-unified-en-0.6b`) runs in a second
virtual environment, because NeMo needs Python 3.11 and its own PyTorch.
`scripts/install_site.sh` creates it (`~/nemo-venv`, Python 3.11 via uv,
from `requirements-nemo.txt`, which is frozen from production) on CUDA hosts;
on ROCm hosts make it by hand:

```bash
uv venv --python 3.11 ~/nemo-venv
~/nemo-venv/bin/pip install -r requirements-nemo.txt   # swap the cu128 index line for the ROCm one
```

The service launcher (`scripts/ops/start-translate-unified`) points the
pipeline at it with `PARAKEET_MODEL=unified-remote` and `UNIFIED_PYTHON`.

> CPU-only installation is not supported — translation and TTS need a GPU.

`install.sh` will:
1. Enable required Debian repositories (backports, contrib, non-free)
2. Install system dependencies
3. Install GPU drivers (ROCm 7.2.x, or NVIDIA's driver as described under
   [NVIDIA CUDA Setup](#nvidia-cuda-setup)) if applicable
4. Create the main Python virtual environment
5. Install Python dependencies with the matching GPU backend
   (PyTorch 2.11.0 +rocm7.2 or +cu128, transformers 5.x, huggingface_hub 1.x)
6. Download the base models (NLLB-200, Kokoro, MMS-TTS, Piper)
7. (If `--parakeet`) install onnxruntime-rocm + onnx-asr and pre-download
   the Parakeet TDT 0.6b v3 ONNX model (the no-NeMo fallback)
8. Run `scripts/install_site.sh` for everything else — the NeMo venv, the
   configuration, the admin password, the scheduler and the systemd units —
   and `scripts/gpu_doctor.sh` as the final check

---

## Manual Installation

If you prefer step-by-step installation:

### Step 1: Enable Required Repositories

```bash
# Enable contrib, non-free, and non-free-firmware
sudo sed -i 's/main$/main contrib non-free non-free-firmware/' /etc/apt/sources.list

# Enable backports for newer kernel
echo "deb http://deb.debian.org/debian trixie-backports main contrib non-free non-free-firmware" | \
  sudo tee /etc/apt/sources.list.d/backports.list

sudo apt update
```

### Step 2: Install Latest Kernel

**Required for AMD iGPUs (Radeon 680M/780M/890M). Recommended for all AMD GPUs.**

Kernel 6.10+ is required for ROCm to detect integrated AMD GPUs. Without it,
`rocminfo` will not find the GPU and the translator will fall back to CPU mode.

```bash
sudo apt install -t trixie-backports linux-image-amd64 linux-headers-amd64
sudo reboot
```

> **Important:** Always reboot into the new kernel *before* installing ROCm.
> If using `install.sh`, it detects when the kernel is too old, installs it,
> and prompts you to reboot. Re-run `./install.sh --rocm` after rebooting to
> continue — it picks up where it left off.

### Step 3: Install System Dependencies

```bash
sudo apt install -y \
    python3 python3-venv python3-pip python3-dev \
    git curl wget \
    libsndfile1 libsoundio-dev portaudio19-dev libasound2-dev \
    ffmpeg \
    pipewire pipewire-alsa pipewire-pulse wireplumber \
    build-essential
```

### Step 4: Install GPU Drivers

See [GPU-Specific Setup](#gpu-specific-setup) below.

### Step 5: Create Project Directory

```bash
cd /home/$USER
git clone https://github.com/LandmarkAdministrator/translator.git translator
cd translator
```

### Step 6: Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### Step 7: Install Python Dependencies

For AMD ROCm (tested: ROCm 7.2.2 + PyTorch 2.11.0+rocm7.2):
```bash
# Use the newest rocmX.Y wheel published by PyTorch; browse
# https://download.pytorch.org/whl/ to confirm the latest, or just let
# ./install.sh --rocm auto-detect it for you.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements/base.txt -r requirements/ml.txt
```

For NVIDIA CUDA (driver 570 or newer; production runs 610):
```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/base.txt -r requirements/ml.txt
```

> CPU-only installation is not supported; see note in [Quick Install](#quick-install).

### Step 8: Download Models

```bash
python scripts/download_models.py --all
```

### Step 9 (optional): Install the onnx-asr Parakeet fallback

The pipeline uses onnx-asr's Parakeet TDT whenever `PARAKEET_MODEL` is not
`unified-remote` (that is, without the NeMo venv). To install onnxruntime-rocm,
onnx-asr, and pre-download the model:

```bash
source venv/bin/activate
./scripts/install_parakeet.sh
```

See [Parakeet Streaming Backend](#parakeet-streaming-backend-optional) below
for details on what this does and its known limitations.

---

## GPU-Specific Setup

### AMD ROCm Setup

**Use ROCm 7.2.2** (the tested baseline — see
[DEPLOYMENT.md](DEPLOYMENT.md) for the pinned versions used in production).
ROCm 7.x is **required** for integrated GPUs (680M/780M/890M); older
versions will not detect these.

Repo layout note: ROCm 7.x ships Ubuntu Noble and Jammy packages only.
Debian 13 (Trixie) is library-compatible with Noble, so we add the Noble
repo.  ROCm 6.x has Debian packages, but the project now requires 7.2+.

```bash
# Add ROCm repository
sudo mkdir -p /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Pin to the tested version (check https://repo.radeon.com/rocm/apt/ for
# later 7.2.x point releases).  Debian Trixie uses the Noble packages.
ROCM_VERSION="7.2.2"
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# Priority-pin the ROCm repo so apt prefers its versions over Debian's own
# ROCm packages (Debian 13 ships partial older ROCm).
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | \
  sudo tee /etc/apt/preferences.d/rocm-pin-600

sudo apt update
sudo apt install -y rocm-hip-sdk rocm-libs rocm-dev rocminfo rocm-smi-lib

# Add user to required groups
sudo usermod -aG render,video $USER
```

> **Note:** The install script (`./install.sh`) automates the above,
> detects the latest ROCm 7.x and matches the Debian/Ubuntu codename
> automatically — the manual steps above are for reference only.

**Important:** Log out and back in after adding groups.

#### Verify ROCm Installation

```bash
# Check GPU is detected
/opt/rocm/bin/rocminfo | grep "Name:"

# Test PyTorch GPU access
source venv/bin/activate
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

#### ROCm Environment Variables

For some AMD GPUs (especially APUs), you may need:

```bash
# Add to ~/.bashrc or create .env.rocm file
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For gfx1150 (890M)
export PATH=/opt/rocm/bin:$PATH
```

GPU Architecture Reference:
| GPU | Architecture | HSA_OVERRIDE Value |
|-----|-------------|-------------------|
| Radeon 890M | gfx1150 | 11.0.0 |
| Radeon 780M | gfx1103 | 11.0.0 |
| Radeon 680M | gfx1035 | 10.3.0 |
| RX 7900 XT | gfx1100 | (not needed) |
| RX 6800 XT | gfx1030 | (not needed) |

**Note:** Always use the latest ROCm version for best performance. ROCm 7.x+ is **required** for integrated GPUs (iGPUs) like the 680M, 780M, and 890M - older versions will not detect these GPUs.

### NVIDIA CUDA Setup

`./install.sh --cuda` does the following. It is the procedure that brought up
the production host (Debian 13, RTX 3060) on 2026-09-01, after Debian's own
`nvidia-driver` from non-free failed to build against a backports kernel.

1. **Secure Boot must be off.** DKMS builds an unsigned kernel module and a
   Secure Boot kernel refuses to load it. `mokutil --sb-state` tells you;
   disable it in the firmware setup. (AMD is unaffected: amdgpu is in-tree.)
2. **Headers for the running kernel** (`linux-headers-$(uname -r)`), or the
   backports kernel and headers together, followed by a reboot.
3. **NVIDIA's own Debian repository** via `cuda-keyring`, then
   `nvidia-kernel-open-dkms nvidia-driver nvidia-driver-cuda` — the open
   module for Turing (RTX 20xx / GTX 16xx) and newer, `--nvidia-proprietary`
   for older cards. The package blacklists nouveau itself.
4. **Reboot** and re-run `./install.sh --cuda`: it sees `nvidia-smi` working
   and continues with the Python environment (torch 2.11 cu128, which needs
   driver 570 or newer; the repository provides 610).

No CUDA toolkit is installed — the PyTorch wheels carry their own runtime.
`scripts/gpu_doctor.sh` checks each of these and names the fix for whatever
is wrong; `install.sh` runs it at the end and `install_site.sh` at the start.

By hand, the same thing is:

```bash
mokutil --sb-state                       # must say "SecureBoot disabled"
sudo apt install -y linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y nvidia-kernel-open-dkms nvidia-driver nvidia-driver-cuda
sudo reboot
nvidia-smi                               # driver 610.x and the card listed
```

---

## ASR backends

Both are streaming Parakeet models; `PARAKEET_MODEL` selects one. (The
Whisper batch backend was retired on 2026-09-06.)

| `PARAKEET_MODEL`                    | Backend                                                  | Where it runs |
|-------------------------------------|----------------------------------------------------------|---------------|
| `unified-remote` (production)       | `nvidia/parakeet-unified-en-0.6b` via NeMo, in its own venv | GPU           |
| *(unset)* `nemo-parakeet-tdt-0.6b-v3` | Parakeet TDT 0.6b v3 via onnx-asr                       | CPU (see note) |

The onnx-asr fallback is installed separately — either during install
(`./install.sh --rocm --parakeet`) or later:

```bash
source venv/bin/activate
./scripts/install_parakeet.sh
```

The script:
1. Uninstalls any stock `onnxruntime` wheel (it conflicts with
   `onnxruntime-rocm`, which replaces it as a superset).
2. Installs `onnxruntime-rocm` from
   `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`.
3. Patches the AMD wheel's ELF `GNU_STACK` flag from `RWE` to `RW`, so
   Linux 6.x loaders will accept it (they reject executable stacks).
4. Installs `onnx-asr[hub]`.
5. Pre-downloads the Parakeet TDT 0.6b v3 model into
   `models/asr/parakeet/` via Hugging Face (it honors `HF_HOME`).
6. Runs a silent-buffer smoke test to confirm the model loads.

### Known limitation: Parakeet runs on CPU, not GPU

The current `onnxruntime-rocm-1.22.2.post1` wheel was built against the
ROCm 6.x ABI — it links `libhipblas.so.2` and `libamdhip64.so.6`.  On
ROCm 7.2 the system ships `libhipblas.so.3` and `libamdhip64.so.7`, so at
runtime `ROCMExecutionProvider` and `MIGraphXExecutionProvider` fail to
load and `onnxruntime` silently falls back to CPU.  You'll see lines like:

```
Failed to load library libonnxruntime_providers_rocm.so with error:
libhipblas.so.2: cannot open shared object file: No such file or directory
```

This is expected and benign.  On a Ryzen AI 9 HX 370 (Radeon 890M iGPU
hardware), Parakeet TDT 0.6b v3 hits RTF ≈ 0.06 on CPU — about 16× faster
than real time — which leaves the GPU to translation and TTS.

**Do not** create a compatibility symlink `libhipblas.so.3 → libhipblas.so.2`
— it's a major-version ABI bump and will crash or silently produce wrong
results.  When AMD publishes a 1.24+ wheel built for ROCm 7.x, the ROCm
provider will start working with no code changes (`parakeet_asr.py`
already requests ROCm first, CPU second, via
`ort.get_available_providers()`).

### Reinstalling / reverting

- `install_parakeet.sh` is idempotent — re-running it uninstalls the stock
  `onnxruntime` again (harmless no-op if already removed), reinstalls
  `onnxruntime-rocm`, re-applies the ELF patch (no-op if already clean),
  and re-verifies the model.
- To revert to a Parakeet-free install:
  ```bash
  pip uninstall -y onnxruntime-rocm onnx-asr
  pip install 'onnxruntime>=1.24.0'
  rm -rf models/asr/parakeet
  ```

---

## Audio Device Configuration

### List Available Devices

```bash
source venv/bin/activate
python run.py --list-devices
```

### Device Selection

Audio devices and per-language outputs are saved in `config/settings.yaml`, which is generated by `python run.py --setup`. The file looks like:

```yaml
input_device: "ThinkPad USB-C Dock"
languages:
  - code: es
    name: Spanish
    enabled: true
    output_device: "5"     # Output device index or name
    output_channel: null   # null = mono/both; 0 = left only; 1 = right only
  - code: ht
    name: Haitian Creole
    enabled: true
    output_device: "6"
    output_channel: null
```

Re-run `python run.py --setup` any time you want to change devices or toggle languages — there is no separate preset file.

### Stereo Channel Separation

To split two languages onto one stereo output, point both languages at the same `output_device` and set `output_channel: 0` (left) for one and `output_channel: 1` (right) for the other:

```yaml
languages:
  - code: es
    output_device: "5"
    output_channel: 0   # Spanish on left
  - code: ht
    output_device: "5"
    output_channel: 1   # Haitian Creole on right
```

`--setup` offers this choice interactively.

### Testing Audio

```bash
# Test input (record 5 seconds)
python -c "
import sounddevice as sd
import numpy as np
print('Recording 5 seconds...')
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
sd.wait()
print(f'Recorded {len(audio)} samples, max amplitude: {np.abs(audio).max():.3f}')
"

# Test output on specific device
python -c "
import sounddevice as sd
import numpy as np
# Generate 1 second beep
t = np.linspace(0, 1, 16000)
audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
sd.play(audio, samplerate=16000, device=5)  # Change device index
sd.wait()
"
```

---

## First Run

### Interactive Setup

```bash
source venv/bin/activate
python run.py --setup
```

The setup wizard guides you through:
1. Selecting the audio input device
2. Configuring the output device and channel for each language
3. Saving the result to `config/settings.yaml`

The admin panel (`/admin`, once `translate-web.service` is running) does the
same from a browser and is the usual way on a serving machine.

### Command Line Usage

```bash
# Run with defaults
python run.py

# Specify languages
python run.py -l es ht fr

# Specify input device
python run.py -i "USB Audio Device"
```

### Test Mode

```bash
# Test GPU and components
python run.py --test

# Verbose mode for debugging
python run.py --verbose
```

---

## Running as a Service

`scripts/install_site.sh` installs the user units (sources in `systemd/`)
and enables lingering so they run without a login:

| Unit | Role |
|---|---|
| `translate.service` | the pipeline — started and stopped by the scheduler inside the windows in `config/schedule.conf`; never enabled at boot |
| `translate-web.service` | the page, the WebSocket stream and `/admin`; always on |
| `translate-window.timer` | the scheduler, every five minutes |
| `translate-tally.timer` | the nightly service tally at 23:30 |
| `gpu-thermal-guard.service` | optional (`install_site.sh --thermal-guard`): stops translation at 85 °C, for a card with improvised cooling |

```bash
systemctl --user status translate.service translate-web.service
journalctl --user -u translate.service -f          # the pipeline's own log is ~/translate.log
systemctl --user list-timers                       # next window check and tally
```

Start and stop by hand from `/admin` (which pauses the schedule until
"Resume automatic schedule"), or `touch ~/translate-manual.flag` to keep the
scheduler's hands off everything while it exists. `docs/DEPLOYMENT.md` covers
updating, rollback and diagnosis.

---

## Troubleshooting

### GPU Not Detected

**AMD iGPU (Radeon 680M / 780M / 890M) — check kernel version first:**
```bash
uname -r   # Must be 6.10 or higher
```
If the kernel is older than 6.10, ROCm cannot detect the iGPU regardless of
other settings. Install the backports kernel and reboot before proceeding:
```bash
sudo apt install -t trixie-backports linux-image-amd64 linux-headers-amd64
sudo reboot
# Then re-run: ./install.sh --rocm
```

**AMD — after confirming kernel is 6.10+:**
```bash
# Check ROCm sees the GPU
/opt/rocm/bin/rocminfo | grep "Name:"

# Check user groups (log out and back in after adding)
groups | grep -E "render|video"

# iGPU HSA override (required for 890M/780M/680M)
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # 890M / 780M
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # 680M
python -c "import torch; print(torch.cuda.is_available())"
```

**NVIDIA:**
```bash
nvidia-smi
# If not found, reinstall driver
sudo apt install --reinstall nvidia-driver
```

### Audio Issues

```bash
# Check PipeWire is running
systemctl --user status pipewire wireplumber

# List all audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Check for audio device permissions
ls -la /dev/snd/
```

### Model Download Failures

```bash
# Download all models
python scripts/download_models.py --all

# Or download specific models
python scripts/download_models.py --asr
python scripts/download_models.py --translation
python scripts/download_models.py --tts
```

### High Latency

- Confirm the GPU is actually in use: watch `rocm-smi` (AMD) or `nvidia-smi`
  (NVIDIA) during a session and check for activity.
- Close other GPU-intensive applications (browsers, video players, etc.);
  on the production host the sermon-archive worker is drained before every
  window for this reason (`docs/BACKLOG-CONTRACT.md`).
- Check the nightly tally in `/admin`: it reports the delay from first word
  to translated audio per service, so a slow day shows up with numbers.
- The streaming ASR's context preset (`UNIFIED_LEFT/CHUNK/RIGHT_SECS`) is
  already the fastest setting that does not cost accuracy; do not enlarge it.

### Service Won't Start

```bash
# Check logs
journalctl --user -u translate.service -n 50
tail -50 ~/translate.log

# Verify venv activation works
/home/$USER/translator/venv/bin/python --version

# Check saved settings
python -c "from src.config.settings import SettingsManager; print(SettingsManager().load())"
```

---

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying to production systems
- See [README.md](../README.md) for usage reference
- Re-run `python run.py --setup` to reconfigure devices, languages, or the ASR model; the result is saved to `config/settings.yaml`
