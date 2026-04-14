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
./install.sh --rocm    # For AMD GPUs
./install.sh --cuda    # For NVIDIA GPUs
./install.sh --cpu     # CPU only (no GPU acceleration)
```

The installer will:
1. Enable required Debian repositories (backports, contrib, non-free)
2. Install system dependencies
3. Install GPU drivers (ROCm or CUDA) if applicable
4. Create Python virtual environment
5. Install Python dependencies with correct GPU backend
6. Download ML models (Whisper, Opus-MT, Piper TTS)
7. Set up the systemd service

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

For AMD ROCm:
```bash
# Use the newest rocmX.Y wheel published by PyTorch; browse
# https://download.pytorch.org/whl/ to confirm the latest, or just let
# ./install.sh --rocm auto-detect it for you.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

For NVIDIA CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 8: Download Models

```bash
python scripts/download_models.py --all
```

---

## GPU-Specific Setup

### AMD ROCm Setup

**Use the latest ROCm version** for best performance. ROCm 7.x+ is **required** for integrated GPUs (680M/780M/890M).

```bash
# Add ROCm repository
sudo mkdir -p /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Check latest version at https://repo.radeon.com/rocm/apt/
# Replace VERSION with the latest (e.g., 7.0, 7.1, etc.)
ROCM_VERSION="7.0"  # Update to latest available
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} trixie main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install -y rocm-hip-runtime rocm-hip-sdk rocm-libs

# Add user to required groups
sudo usermod -aG render,video $USER
```

> **Note:** The install script (`./install.sh`) automatically detects the latest available
> ROCm version and correct Debian codename — the manual steps above are for reference only.

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
asr_model: large-v3
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
