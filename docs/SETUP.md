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
  refuses to start without a ROCm or CUDA GPU.  Real-time transcription with
  the default `large-v3` Whisper model needs GPU acceleration to keep up.
  - AMD: RDNA2 or newer (RX 6000+, Radeon 680M/780M/890M)
  - NVIDIA: Maxwell or newer (GTX 900+, RTX series)
  - Recommended ≥8 GB VRAM for `large-v3`; 4–6 GB VRAM works if you step down
    to `medium.en` or `small.en` via `--setup`.
- **Audio:** One input device, multiple output devices for different languages
- **Storage:** 10GB free space for models and dependencies

### Software Requirements

- **OS:** Debian 13 (Trixie) - fresh install recommended
- **Kernel:** 6.10+ (from backports for best GPU support)
- **Python:** 3.11+ (included in Debian 13)

---

## Quick Install

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

# Add --parakeet to also set up the Parakeet TDT 0.6b streaming backend
# (onnx-asr + onnxruntime-rocm + Parakeet ONNX model).  This is optional —
# the default Whisper backend is unchanged and still selected at runtime:
./install.sh --rocm --parakeet    # ROCm + Parakeet streaming backend
```

> CPU-only installation is not supported — the batch Whisper path requires
> a GPU.  The Parakeet streaming backend runs on CPU, but it
> shares the venv with Whisper/translation/TTS, all of which still need GPU.

The installer will:
1. Enable required Debian repositories (backports, contrib, non-free)
2. Install system dependencies
3. Install GPU drivers (ROCm 7.2.x or CUDA) if applicable
4. Create Python virtual environment
5. Install Python dependencies with correct GPU backend
   (PyTorch 2.11+rocm7.2 or +cu124, transformers 5.x, huggingface_hub 1.x)
6. Download ML models (Whisper, Opus-MT, Piper TTS)
7. (If `--parakeet`) install onnxruntime-rocm + onnx-asr and pre-download
   the Parakeet TDT 0.6b v3 ONNX model
8. Set up the systemd service

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

For NVIDIA CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/base.txt -r requirements/ml.txt
```

> CPU-only installation is not supported; see note in [Quick Install](#quick-install).

### Step 8: Download Models

```bash
python scripts/download_models.py --all
```

### Step 9 (optional): Install Parakeet Streaming Backend

To enable `python run.py --parakeet`, install onnxruntime-rocm, onnx-asr, and
pre-download the Parakeet TDT 0.6b v3 ONNX model:

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

```bash
# Enable non-free repos (required for nvidia-driver on Debian)
sudo sed -i 's/main$/main contrib non-free non-free-firmware/' /etc/apt/sources.list
sudo apt update

# Install the kernel driver only — PyTorch bundles its own CUDA runtime
sudo apt install -y nvidia-driver firmware-misc-nonfree

# Reboot to load driver
sudo reboot
```

> **Note:** Do not install `nvidia-cuda-toolkit` from apt. PyTorch ships its own CUDA
> runtime inside the pip wheel, so only the kernel driver is needed.

#### Verify CUDA Installation

```bash
nvidia-smi
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Parakeet Streaming Backend (optional)

The project ships two ASR backends and you pick one at runtime:

| Flag                       | Backend                                              | Where it runs      |
|----------------------------|------------------------------------------------------|--------------------|
| *(default, no flag)*       | Whisper batch (faster-whisper via CTranslate2)       | GPU                |
| `--parakeet`               | NVIDIA Parakeet TDT 0.6b v3 via onnx-asr             | CPU (see note)     |

Parakeet is enabled by installing its runtime dependencies separately —
either during install (`./install.sh --rocm --parakeet`) or later:

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
than real time — so the batch Whisper path can keep the GPU free for
MarianMT translation while Parakeet handles ASR on CPU.

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
1. Selecting audio input device
2. Configuring output devices for each language
3. Testing the pipeline
4. Saving configuration as a preset

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

### Install the Systemd Service

```bash
# User service (recommended, no sudo for operation)
./scripts/install_service.sh install

# Or system-wide service
sudo ./scripts/install_service.sh --system install
```

### Service Commands

For user service:
```bash
systemctl --user start church-translator
systemctl --user stop church-translator
systemctl --user status church-translator
journalctl --user -u church-translator -f
```

For system service:
```bash
sudo systemctl start church-translator
sudo systemctl stop church-translator
sudo systemctl status church-translator
sudo journalctl -u church-translator -f
```

### Autostart at Boot

The user service requires "lingering" to start without login:

```bash
sudo loginctl enable-linger $USER
```

This is done automatically by the install script.

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
- Close other GPU-intensive applications (browsers, video players, etc.).
- Reduce chunk duration in config.
- **Fallback:** on low-VRAM GPUs (≤6 GB) large-v3 may not fit.  Re-run
  `./translator --setup` → "Configure ASR model" and pick `medium.en` or
  `small.en`.  Smaller models cost some accuracy but greatly reduce VRAM
  and latency.

### Service Won't Start

```bash
# Check logs
journalctl --user -u church-translator -n 50

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
