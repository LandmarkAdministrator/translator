# Church Audio Translator - Deployment Guide

Guide for deploying the Church Audio Translator to production systems.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Creating a Deployment Package](#creating-a-deployment-package)
- [Fresh System Deployment](#fresh-system-deployment)
- [Cloning an Existing Installation](#cloning-an-existing-installation)
- [Multi-System Deployment](#multi-system-deployment)
- [Post-Deployment Configuration](#post-deployment-configuration)
- [Updating Deployments](#updating-deployments)
- [Backup and Recovery](#backup-and-recovery)

---

## Deployment Overview

### Deployment Methods

| Method | Best For | Includes Models | Disk Usage |
|--------|----------|-----------------|------------|
| Fresh Install | New systems | Downloads during install | ~10GB |
| Full Package | Air-gapped systems | Yes | ~8GB package |
| Code Only | Systems with internet | Downloads during install | ~50MB package |
| Clone | Identical hardware | Yes | ~10GB |

### System Requirements

Target systems must have:
- Debian 13 (Trixie) or compatible
- 8GB+ RAM
- 10GB free disk space
- Audio input/output devices
- (Optional) AMD or NVIDIA GPU

---

## Creating a Deployment Package

### Full Package (with Models)

Creates a complete package including all models (~8GB):

```bash
cd /home/$USER/translator

# Create deployment package
./install.sh --create-package

# Or manually:
tar -czf translator-full.tar.gz \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*' \
    --exclude='.git' \
    src/ scripts/ config/ docs/ models/ \
    install.sh run.py requirements.txt README.md
```

### Code-Only Package (without Models)

Creates a smaller package (~50MB), models download during install:

```bash
tar -czf translator-code.tar.gz \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*' \
    --exclude='.git' \
    --exclude='models/*' \
    src/ scripts/ config/ docs/ \
    install.sh run.py requirements.txt README.md
```

### Models-Only Package

For updating models on existing installations:

```bash
tar -czf translator-models.tar.gz models/
```

---

## Fresh System Deployment

### Step 1: Prepare Target System

On the target Debian 13 system:

```bash
# Enable required repos (if not already)
sudo sed -i 's/main$/main contrib non-free non-free-firmware/' /etc/apt/sources.list

# Install minimal requirements
sudo apt update
sudo apt install -y git curl wget
```

### Step 2: Transfer Package

```bash
# Option A: SCP from source system
scp translator-full.tar.gz user@target:/home/user/

# Option B: Download from shared location
wget https://your-server/translator-full.tar.gz

# Option C: Clone from git (internet required)
git clone <repository-url> translator
```

### Step 3: Extract and Install

```bash
cd /home/$USER
tar -xzf translator-full.tar.gz -C translator/
cd translator

# Run installer with GPU detection
./install.sh

# Or specify GPU type
./install.sh --rocm    # AMD GPU
./install.sh --cuda    # NVIDIA GPU
./install.sh --cpu     # No GPU
```

### Step 4: Configure Audio

```bash
source venv/bin/activate

# List available devices
python run.py --list-devices

# Run interactive setup
python run.py --setup
```

### Step 5: Test

```bash
# Test all components
python run.py --test

# Test translation
python run.py --verbose
# Speak into microphone, verify output
```

### Step 6: Install Service

```bash
./scripts/install_service.sh install
systemctl --user start church-translator
systemctl --user status church-translator
```

---

## Cloning an Existing Installation

For deploying to identical hardware (same GPU, similar audio):

### On Source System

```bash
cd /home/$USER

# Create complete archive including venv
tar -czf translator-clone.tar.gz \
    --exclude='logs/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    translator/
```

### On Target System

```bash
cd /home/$USER
tar -xzf translator-clone.tar.gz

cd translator

# Install system dependencies only (skip Python packages)
./install.sh --system-deps-only

# If same GPU type, just verify
source venv/bin/activate
python run.py --test

# Reconfigure audio devices (usually different)
python run.py --setup

# Install service
./scripts/install_service.sh install
```

**Note:** Cloning only works between systems with the same:
- CPU architecture (x86_64)
- GPU type (AMD ROCm / NVIDIA CUDA / CPU)
- Python version

---

## Multi-System Deployment

For deploying to multiple church locations or venues:

### Create Base Configuration

```bash
# On master system, create a base preset
mkdir -p config/deployments/

# Create location-specific presets
cat > config/deployments/main_sanctuary.yaml << 'EOF'
preset:
  name: Main Sanctuary
  description: Primary worship space

audio_input:
  device: "USB Audio Device"

languages:
  spanish:
    enabled: true
    output_device: "5"
  haitian_creole:
    enabled: true
    output_device: "6"
EOF
```

### Deployment Script

Create a deployment script for consistency:

```bash
#!/bin/bash
# deploy.sh - Deploy translator to remote system

TARGET_HOST="$1"
TARGET_USER="${2:-administrator}"
GPU_TYPE="${3:-auto}"

if [[ -z "$TARGET_HOST" ]]; then
    echo "Usage: $0 <hostname> [username] [gpu_type]"
    exit 1
fi

# Create package
echo "Creating deployment package..."
tar -czf /tmp/translator-deploy.tar.gz \
    --exclude='venv' --exclude='.venv' \
    --exclude='__pycache__' --exclude='logs/*' \
    -C /home/$USER translator/

# Transfer
echo "Transferring to $TARGET_HOST..."
scp /tmp/translator-deploy.tar.gz "$TARGET_USER@$TARGET_HOST:/home/$TARGET_USER/"

# Install remotely
echo "Installing on $TARGET_HOST..."
ssh "$TARGET_USER@$TARGET_HOST" << EOF
cd /home/$TARGET_USER
tar -xzf translator-deploy.tar.gz
cd translator
./install.sh --$GPU_TYPE
./scripts/install_service.sh install
EOF

echo "Deployment complete. Configure audio devices on target system."
```

---

## Post-Deployment Configuration

### Audio Device Mapping

Audio device indices vary between systems. After deployment:

```bash
# Discover devices on new system
python run.py --list-devices

# Update preset with correct device indices
nano config/presets/default.yaml
```

### Create Location Preset

```bash
python run.py --setup
# Configure devices interactively
# Save as new preset
```

### Environment Variables

For AMD GPUs, you may need to set HSA override:

```bash
# Check GPU architecture
/opt/rocm/bin/rocminfo | grep "Name:"

# Create environment file if needed
echo 'HSA_OVERRIDE_GFX_VERSION=11.0.0' > .env.rocm
```

### Test Configuration

```bash
# Verify GPU is working
python run.py --test

# Test with actual audio
python run.py --verbose
# Verify transcription and translation output
```

---

## Updating Deployments

### Code Updates

```bash
# On target system
cd /home/$USER/translator

# Pull latest code (if using git)
git pull

# Or extract new code package
tar -xzf translator-code-new.tar.gz --strip-components=1

# Reinstall dependencies (if requirements changed)
source venv/bin/activate
pip install -r requirements.txt

# Restart service
systemctl --user restart church-translator
```

### Model Updates

```bash
# Download new models
source venv/bin/activate
python scripts/download_models.py --all

# Or extract model package
tar -xzf translator-models-new.tar.gz

# Restart service
systemctl --user restart church-translator
```

### Full Update

```bash
# Stop service
systemctl --user stop church-translator

# Backup config
cp -r config config.backup

# Extract new version
cd /home/$USER
rm -rf translator.old
mv translator translator.old
tar -xzf translator-full-new.tar.gz -C translator/

# Restore config
cp -r translator.old/config/presets/* translator/config/presets/

# Reinstall
cd translator
./install.sh

# Restart
systemctl --user start church-translator
```

---

## Backup and Recovery

### What to Backup

Priority files for backup:

```bash
# Essential (small)
config/                 # All configuration
config/presets/         # Custom presets

# Large but replaceable
models/                 # Can re-download
venv/                   # Can reinstall
```

### Backup Script

```bash
#!/bin/bash
# backup_translator.sh

BACKUP_DIR="/home/$USER/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup config (essential)
tar -czf "$BACKUP_DIR/translator-config-$DATE.tar.gz" \
    -C /home/$USER/translator config/

# Full backup (optional)
tar -czf "$BACKUP_DIR/translator-full-$DATE.tar.gz" \
    --exclude='logs/*' \
    --exclude='__pycache__' \
    -C /home/$USER translator/

echo "Backup created: $BACKUP_DIR/translator-config-$DATE.tar.gz"
```

### Recovery

```bash
# Stop service
systemctl --user stop church-translator

# Restore config
cd /home/$USER/translator
tar -xzf /home/$USER/backups/translator-config-YYYYMMDD.tar.gz

# Or full restore
cd /home/$USER
rm -rf translator
tar -xzf /home/$USER/backups/translator-full-YYYYMMDD.tar.gz

# Reinstall if needed
cd translator
./install.sh

# Start service
systemctl --user start church-translator
```

---

## Air-Gapped Deployment

For systems without internet access:

### On Internet-Connected System

```bash
# Download all dependencies
mkdir -p offline_packages

# Download Python packages
pip download -d offline_packages/ -r requirements.txt
pip download -d offline_packages/ torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2  # or cu124/cpu

# Download models
python scripts/download_models.py --all

# Create complete offline package
tar -czf translator-offline.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    src/ scripts/ config/ docs/ models/ \
    offline_packages/ \
    install.sh run.py requirements.txt README.md
```

### On Air-Gapped System

```bash
# Transfer via USB drive
mount /dev/sdb1 /mnt/usb
cp /mnt/usb/translator-offline.tar.gz /home/$USER/

# Extract
cd /home/$USER
tar -xzf translator-offline.tar.gz -C translator/
cd translator

# Install system deps (requires local apt mirror or DVD)
./install.sh --system-deps-only

# Install Python packages from local cache
python3 -m venv venv
source venv/bin/activate
pip install --no-index --find-links=offline_packages/ -r requirements.txt
pip install --no-index --find-links=offline_packages/ torch torchvision torchaudio

# Models are already included
python run.py --test
```

---

## Troubleshooting Deployments

### Installation Fails

```bash
# Check install log
cat /tmp/translator_install.log

# Verify system requirements
uname -r                    # Kernel version
python3 --version           # Python version
lspci | grep -i vga         # GPU detection
```

### Service Won't Start

```bash
# Check service logs
journalctl --user -u church-translator -n 100

# Verify paths
ls -la /home/$USER/translator/venv/bin/python
ls -la /home/$USER/translator/run.py

# Test manually
cd /home/$USER/translator
source venv/bin/activate
python run.py --test
```

### GPU Not Working After Deployment

```bash
# Check ROCm (AMD) - should be 7.x+ for best performance
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocminfo 2>&1 | head -5  # Check version

# Check CUDA (NVIDIA)
nvidia-smi

# Reinstall GPU packages
source venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Note:** Always use the latest ROCm version for best performance. ROCm 7.x+ is **required** for integrated GPUs (680M, 780M, 890M).

### Audio Devices Different

```bash
# Re-run device discovery
python run.py --list-devices

# Update preset
python run.py --setup

# Or edit config directly
nano config/presets/default.yaml
```
