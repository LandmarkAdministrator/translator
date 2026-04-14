#!/bin/bash
#
# Church Audio Translator - Installation Script
#
# Supports: Debian 13 (Trixie), Ubuntu 22.04+
# GPU: AMD ROCm 7.x+ (required for iGPUs), NVIDIA CUDA 12.x+, or CPU-only
#
# Usage:
#   ./install.sh              # Interactive install (requires a GPU)
#   ./install.sh --rocm       # Force AMD ROCm installation
#   ./install.sh --cuda       # Force NVIDIA CUDA installation
#   ./install.sh --help       # Show help
#
# Note: CPU-only installation is not supported — the translator requires
# a ROCm or CUDA GPU to run in real time.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default installation directory is the repo itself (where this script lives)
# Override with --dir or INSTALL_DIR env var if needed
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"

# Preserve the real user when run via sudo (usermod, .bashrc, systemd service dir)
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"

# Non-interactive mode: skip all confirm() prompts and use defaults
YES=false

# Log file
LOG_FILE="/tmp/church-translator-install.log"

#=============================================================================
# Helper Functions
#=============================================================================

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
}

header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

confirm() {
    local prompt="$1"
    local default="${2:-y}"

    # In non-interactive mode, use the default without prompting
    if [[ "$YES" == "true" ]]; then
        [[ "$default" =~ ^[Yy]$ ]]
        return
    fi

    if [[ "$default" == "y" ]]; then
        prompt="$prompt [Y/n] "
    else
        prompt="$prompt [y/N] "
    fi

    read -p "$prompt" response
    response=${response:-$default}
    [[ "$response" =~ ^[Yy]$ ]]
}

command_exists() {
    command -v "$1" &> /dev/null
}

#=============================================================================
# System Detection
#=============================================================================

detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        OS_CODENAME="$VERSION_CODENAME"
    else
        error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi

    log "Detected OS: $OS_ID $OS_VERSION ($OS_CODENAME)"
}

detect_gpu() {
    GPU_TYPE="cpu"
    GPU_NAME="None detected"

    # Check for AMD GPU
    if lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -qi "amd\|radeon"; then
        GPU_TYPE="amd"
        GPU_NAME=$(lspci | grep -i "vga\|3d\|display" | grep -i "amd\|radeon" | head -1 | sed 's/.*: //')
    fi

    # Check for NVIDIA GPU
    if lspci 2>/dev/null | grep -i "vga\|3d\|display" | grep -qi "nvidia"; then
        GPU_TYPE="nvidia"
        GPU_NAME=$(lspci | grep -i "vga\|3d\|display" | grep -i "nvidia" | head -1 | sed 's/.*: //')
    fi

    log "Detected GPU: $GPU_NAME ($GPU_TYPE)"
}

detect_amd_gpu_arch() {
    # Try to detect AMD GPU architecture for ROCm compatibility
    AMD_GPU_ARCH=""
    AMD_IS_IGPU=false

    if [[ "$GPU_TYPE" == "amd" ]]; then
        # Check for iGPUs (integrated GPUs) - these require ROCm 7.x+
        if echo "$GPU_NAME" | grep -qi "680m\|780m\|890m\|radeon.*graphics"; then
            AMD_IS_IGPU=true
            log "Detected AMD integrated GPU (iGPU) - ROCm 7.x required"
        fi

        # Check for common AMD GPU families
        if echo "$GPU_NAME" | grep -qi "890m\|gfx1150"; then
            AMD_GPU_ARCH="gfx1150"  # RDNA3.5 (Strix Point)
        elif echo "$GPU_NAME" | grep -qi "780m\|gfx1103"; then
            AMD_GPU_ARCH="gfx1103"  # RDNA3 (Phoenix)
        elif echo "$GPU_NAME" | grep -qi "680m\|gfx1035"; then
            AMD_GPU_ARCH="gfx1035"  # RDNA2 (Rembrandt)
        elif echo "$GPU_NAME" | grep -qi "radeon.*7\|rx.*7\|gfx11"; then
            AMD_GPU_ARCH="gfx1100"  # RDNA3 discrete
        elif echo "$GPU_NAME" | grep -qi "radeon.*6\|rx.*6\|gfx10"; then
            AMD_GPU_ARCH="gfx1030"  # RDNA2 discrete
        elif echo "$GPU_NAME" | grep -qi "vega\|gfx9"; then
            AMD_GPU_ARCH="gfx900"   # Vega
        fi

        if [[ -n "$AMD_GPU_ARCH" ]]; then
            log "Detected AMD GPU architecture: $AMD_GPU_ARCH"
        else
            warn "Could not detect AMD GPU architecture. May need manual configuration."
        fi
    fi
}

#=============================================================================
# Dependency Installation
#=============================================================================

install_system_deps() {
    header "Installing System Dependencies"

    log "Updating package lists..."
    sudo apt update

    log "Installing build tools and libraries..."
    sudo apt install -y \
        build-essential \
        cmake \
        pkg-config \
        git \
        curl \
        wget \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        portaudio19-dev \
        libasound2-dev \
        pipewire \
        pipewire-alsa \
        pipewire-pulse \
        wireplumber \
        alsa-utils \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        sox \
        libsox-fmt-all \
        libopenblas-dev \
        libffi-dev \
        libssl-dev

    log "System dependencies installed successfully."
}

check_kernel_version() {
    # Returns 0 if kernel is new enough for ROCm iGPU support, 1 if not
    local kernel_version
    kernel_version=$(uname -r | cut -d. -f1-2)
    local major minor
    major=$(echo "$kernel_version" | cut -d. -f1)
    minor=$(echo "$kernel_version" | cut -d. -f2)

    # Require 6.10+ for AMD iGPU ROCm support
    if [[ "$major" -gt 6 ]] || [[ "$major" -eq 6 && "$minor" -ge 10 ]]; then
        return 0
    fi
    return 1
}

setup_backports() {
    header "Setting Up Debian Backports"

    if [[ "$OS_ID" != "debian" ]]; then
        log "Not Debian, skipping backports setup."
        return 0
    fi

    local backports_file="/etc/apt/sources.list.d/${OS_CODENAME}-backports.list"
    local kernel_ok=true

    # Check if current kernel is sufficient
    if ! check_kernel_version; then
        kernel_ok=false
    fi

    # If kernel is already sufficient and running non-interactively, skip entirely.
    # Avoids adding a backports repo that may not yet exist (e.g. trixie-backports on new installs).
    if [[ "$kernel_ok" == "true" && "$YES" == "true" ]]; then
        log "Kernel already sufficient ($(uname -r)), skipping backports setup."
        return 0
    fi

    if [[ "$kernel_ok" == "false" ]]; then
        local current_kernel
        current_kernel=$(uname -r)
        if [[ "$AMD_IS_IGPU" == "true" ]]; then
            echo ""
            error "Kernel ${current_kernel} is too old for AMD iGPU ROCm support."
            error "Kernel 6.10+ is REQUIRED for the Radeon 680M/780M/890M."
            echo ""
        else
            warn "Current kernel: ${current_kernel}. Kernel 6.10+ is recommended for AMD ROCm."
        fi
    fi

    # Set up backports repo if not already done
    if [[ ! -f "$backports_file" ]]; then
        if [[ "$AMD_IS_IGPU" == "true" && "$kernel_ok" == "false" ]]; then
            log "Adding backports repository (required for iGPU support)..."
            echo "deb http://deb.debian.org/debian/ ${OS_CODENAME}-backports main contrib non-free non-free-firmware" | \
                sudo tee "$backports_file"
            sudo apt update
        elif confirm "Enable Debian backports for latest kernel (recommended for AMD GPUs)?"; then
            log "Adding backports repository..."
            echo "deb http://deb.debian.org/debian/ ${OS_CODENAME}-backports main contrib non-free non-free-firmware" | \
                sudo tee "$backports_file"
            sudo apt update
        else
            return 0
        fi
    else
        log "Backports already configured."
    fi

    # Install kernel if needed
    if [[ "$kernel_ok" == "false" ]]; then
        if [[ "$AMD_IS_IGPU" == "true" ]]; then
            log "Installing kernel 6.10+ from backports (required for ${AMD_GPU_ARCH} iGPU)..."
            sudo apt install -t "${OS_CODENAME}-backports" linux-image-amd64 linux-headers-amd64 -y
            echo ""
            echo -e "${RED}============================================================${NC}"
            echo -e "${RED}  REBOOT REQUIRED BEFORE CONTINUING                        ${NC}"
            echo -e "${RED}============================================================${NC}"
            echo ""
            warn "A newer kernel was installed. ROCm CANNOT be installed until"
            warn "you boot into the new kernel."
            echo ""
            log "After rebooting, run this script again to continue:"
            echo "  ./install.sh --rocm"
            echo ""
            log "The script will detect that the kernel is now up to date"
            log "and continue with ROCm installation automatically."
            echo ""
            if confirm "Reboot now?"; then
                sudo reboot
            else
                echo "Please reboot manually, then re-run: ./install.sh --rocm"
                exit 0
            fi
        elif confirm "Install latest kernel from backports?"; then
            log "Installing latest kernel..."
            sudo apt install -t "${OS_CODENAME}-backports" linux-image-amd64 linux-headers-amd64 -y
            warn "A reboot is required after installation to use the new kernel."
            NEEDS_REBOOT=true
        fi
    fi
}

#=============================================================================
# ROCm Installation (AMD)
#=============================================================================

install_rocm() {
    header "Installing AMD ROCm"

    # Check if ROCm is already installed
    if command_exists rocminfo; then
        local rocm_version=$(rocminfo 2>/dev/null | grep -i "version" | head -1 || echo "unknown")
        log "ROCm already installed: $rocm_version"
        if ! confirm "Reinstall ROCm?"; then
            return 0
        fi
    fi

    log "Adding ROCm repository..."

    # Create keyring directory
    sudo mkdir -p /etc/apt/keyrings

    # Download and install ROCm GPG key
    wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

    # Detect latest ROCm version from repository
    local rocm_version=""
    log "Detecting latest ROCm version..."

    # Try to get the latest version from the repo (requires curl)
    if command -v curl &> /dev/null; then
        # Fetch directory listing and find highest version
        rocm_version=$(curl -s https://repo.radeon.com/rocm/apt/ 2>/dev/null | \
            grep -oP 'href="\K[0-9]+\.[0-9]+(\.[0-9]+)?' | \
            sort -V | tail -1)
    fi

    # Fallback to known latest if detection fails
    if [[ -z "$rocm_version" ]]; then
        rocm_version="6.3.2"  # Fallback - update this periodically
        warn "Could not detect latest ROCm version, using ${rocm_version}"
    else
        log "Detected latest ROCm version: ${rocm_version}"
    fi

    # Determine the apt distribution name.
    # ROCm 7.x dropped Debian packages — only ships Ubuntu Noble and Jammy.
    # Debian 13 (Trixie) is library-compatible with Ubuntu Noble and can use those packages.
    # ROCm 6.x had bookworm packages that worked directly on Debian 12/13.
    local rocm_dist
    local rocm_major
    rocm_major=$(echo "$rocm_version" | cut -d. -f1)

    if [[ "$OS_ID" == "ubuntu" ]]; then
        if [[ "$OS_CODENAME" == "noble" || "$OS_CODENAME" == "jammy" ]]; then
            rocm_dist="$OS_CODENAME"
        else
            rocm_dist="noble"  # Default to noble for newer Ubuntu releases
        fi
    elif [[ "$rocm_major" -ge 7 ]]; then
        # ROCm 7.x on Debian — noble packages are compatible with Debian Trixie
        rocm_dist="noble"
        log "ROCm ${rocm_version} on Debian: using Ubuntu Noble packages (compatible with Trixie)"
    else
        # ROCm 6.x on Debian — use bookworm packages
        rocm_dist="bookworm"
    fi

    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${rocm_version} ${rocm_dist} main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list

    # Set ROCm package priority
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | \
        sudo tee /etc/apt/preferences.d/rocm-pin-600

    sudo apt update

    log "Installing ROCm packages..."
    sudo apt install -y \
        rocm-hip-sdk \
        rocm-libs \
        rocm-dev \
        rocminfo \
        rocm-smi-lib

    # Add user to required groups
    log "Adding user to render and video groups..."
    sudo usermod -a -G render,video "$REAL_USER"

    # Set up environment
    setup_rocm_environment

    log "ROCm installation complete."
    warn "Please log out and back in for group changes to take effect."
    NEEDS_RELOGIN=true
}

setup_rocm_environment() {
    log "Setting up ROCm environment..."

    # Determine HSA_OVERRIDE_GFX_VERSION based on GPU architecture
    local hsa_override="11.0.0"  # Default for RDNA3+
    case "$AMD_GPU_ARCH" in
        gfx1150)  hsa_override="11.0.0" ;;  # Radeon 890M
        gfx1103)  hsa_override="11.0.0" ;;  # Radeon 780M
        gfx1035)  hsa_override="10.3.0" ;;  # Radeon 680M
        gfx1100)  hsa_override="" ;;        # RX 7000 discrete (no override needed)
        gfx1030)  hsa_override="" ;;        # RX 6000 discrete (no override needed)
    esac

    # Create ROCm environment file
    local env_file="$INSTALL_DIR/.env.rocm"
    cat > "$env_file" << EOF
# ROCm Environment Variables
export PATH=/opt/rocm/bin:\$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH
EOF

    # Only set HSA_OVERRIDE if needed (for iGPUs)
    if [[ -n "$hsa_override" ]]; then
        echo "export HSA_OVERRIDE_GFX_VERSION=$hsa_override" >> "$env_file"
        log "Set HSA_OVERRIDE_GFX_VERSION=$hsa_override for $AMD_GPU_ARCH"
    fi

    # Add to user's profile if not already there
    local profile_file="$REAL_HOME/.bashrc"
    if ! grep -q "church-translator.*rocm" "$profile_file" 2>/dev/null; then
        echo "" >> "$profile_file"
        echo "# Church Translator ROCm environment" >> "$profile_file"
        echo "[ -f \"$env_file\" ] && source \"$env_file\"" >> "$profile_file"
    fi

    log "ROCm environment configured."
}

#=============================================================================
# CUDA Installation (NVIDIA)
#=============================================================================

enable_nonfree_repos() {
    # Ensure Debian non-free/non-free-firmware repos are enabled (required for nvidia-driver)
    local sources_file="/etc/apt/sources.list"
    local needs_update=false

    if ! grep -q "non-free" "$sources_file" 2>/dev/null; then
        log "Enabling non-free and non-free-firmware repositories (required for NVIDIA driver)..."
        sudo sed -i 's/^\(deb.*main\)$/\1 contrib non-free non-free-firmware/' "$sources_file"
        needs_update=true
    fi

    # Also check sources.list.d
    if ! grep -rq "non-free" /etc/apt/sources.list.d/ 2>/dev/null && [[ "$needs_update" == "false" ]]; then
        log "non-free repos already enabled."
    fi

    if [[ "$needs_update" == "true" ]]; then
        sudo apt update
        log "non-free repositories enabled."
    fi
}

install_cuda() {
    header "Installing NVIDIA Driver"

    # Check if driver is already installed
    if command_exists nvidia-smi; then
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        log "NVIDIA driver already installed: $driver_version"
        if ! confirm "Reinstall NVIDIA driver?"; then
            return 0
        fi
    fi

    if [[ "$OS_ID" == "debian" ]]; then
        # Enable non-free repos first — nvidia-driver won't be found without them
        enable_nonfree_repos

        log "Installing NVIDIA driver..."
        # Note: We install the driver only — NOT nvidia-cuda-toolkit from apt.
        # The apt CUDA toolkit is often outdated. PyTorch and faster-whisper
        # bundle their own CUDA runtime, so only the driver is needed.
        sudo apt install -y nvidia-driver firmware-misc-nonfree

    elif [[ "$OS_ID" == "ubuntu" ]]; then
        # Ubuntu: Use official NVIDIA repo for latest driver
        local ubuntu_version
        ubuntu_version=$(echo "$OS_VERSION" | tr -d '.')
        # Default to 2204 packages which work on 22.04 and 24.04
        local cuda_repo_ubuntu="ubuntu2204"
        if [[ "$ubuntu_version" -ge 2404 ]]; then
            cuda_repo_ubuntu="ubuntu2404"
        fi
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo_ubuntu}/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
        sudo apt update
        sudo apt install -y nvidia-driver-550
    fi

    log "NVIDIA driver installation complete."
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  REBOOT REQUIRED BEFORE CONTINUING                        ${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    warn "The NVIDIA driver cannot load until you reboot."
    warn "GPU will NOT be detected by PyTorch until after reboot."
    echo ""
    log "After rebooting, run this script again to continue:"
    echo "  ./install.sh --cuda"
    echo ""
    log "The script will detect the driver is installed and skip"
    log "straight to Python environment and model setup."
    echo ""
    if confirm "Reboot now?"; then
        sudo reboot
    else
        echo "Please reboot manually, then re-run: ./install.sh --cuda"
        exit 0
    fi
}

#=============================================================================
# PyTorch ROCm Wheel Detection
#=============================================================================

detect_pytorch_rocm_version() {
    # Find the latest PyTorch ROCm wheel available at download.pytorch.org.
    # This ensures the PyTorch wheel matches the installed ROCm version as
    # closely as possible rather than using a hardcoded (potentially outdated) URL.
    local pytorch_rocm=""

    if command -v curl &> /dev/null; then
        pytorch_rocm=$(curl -s "https://download.pytorch.org/whl/" 2>/dev/null | \
            grep -oP '(?<=href=")rocm[0-9]+\.[0-9]+(?=/)' | \
            sort -V | tail -1)
    fi

    if [[ -n "$pytorch_rocm" ]]; then
        log "Detected latest PyTorch ROCm wheel: $pytorch_rocm"
        echo "$pytorch_rocm"
    else
        warn "Could not detect latest PyTorch ROCm wheel version, using rocm7.2 fallback"
        echo "rocm7.2"
    fi
}

#=============================================================================
# Python Environment Setup
#=============================================================================

venv_python() {
    # Run a command using the venv's Python/pip, as the real user when running as root.
    # Usage: venv_python python -c "..."  or  venv_python pip install ...
    local cmd="$1"; shift
    local bin="$INSTALL_DIR/venv/bin/$cmd"
    if [[ $EUID -eq 0 && "$REAL_USER" != "root" ]]; then
        sudo -u "$REAL_USER" "$bin" "$@"
    else
        "$bin" "$@"
    fi
}

setup_python_env() {
    header "Setting Up Python Environment"

    local venv_dir="$INSTALL_DIR/venv"

    if [[ -d "$venv_dir" ]]; then
        if confirm "Virtual environment exists. Recreate it?"; then
            rm -rf "$venv_dir"
        else
            log "Using existing virtual environment."
            return 0
        fi
    fi

    log "Creating virtual environment..."
    if [[ $EUID -eq 0 && "$REAL_USER" != "root" ]]; then
        sudo -u "$REAL_USER" python3 -m venv "$venv_dir"
    else
        python3 -m venv "$venv_dir"
    fi

    log "Upgrading pip..."
    venv_python pip install --upgrade pip setuptools wheel

    log "Python environment ready."
}

install_python_deps() {
    header "Installing Python Dependencies"

    # Install base dependencies first
    log "Installing base dependencies..."
    venv_python pip install -r "$INSTALL_DIR/requirements/base.txt"

    # Install GPU-specific dependencies
    case "$GPU_BACKEND" in
        rocm)
            log "Installing PyTorch with ROCm support..."
            local pytorch_rocm_ver
            pytorch_rocm_ver=$(detect_pytorch_rocm_version)
            venv_python pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${pytorch_rocm_ver}"
            ;;
        cuda)
            log "Installing PyTorch with CUDA support..."
            venv_python pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
            ;;
        cpu)
            log "Installing PyTorch (CPU only)..."
            venv_python pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac

    # Install ML dependencies
    log "Installing ML dependencies..."
    venv_python pip install -r "$INSTALL_DIR/requirements/ml.txt"

    log "Python dependencies installed."
}

#=============================================================================
# Application Setup
#=============================================================================

setup_directories() {
    header "Setting Up Installation Directory"

    # Check if installing to /opt (needs sudo)
    if [[ "$INSTALL_DIR" == /opt/* ]]; then
        if [[ ! -d "$INSTALL_DIR" ]]; then
            sudo mkdir -p "$INSTALL_DIR"
            sudo chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
        fi
    else
        mkdir -p "$INSTALL_DIR"
    fi

    # Create subdirectories
    mkdir -p "$INSTALL_DIR"/{config,models/{asr,translation,tts},logs}

    # Copy application files if installing from source directory
    if [[ "$SCRIPT_DIR" != "$INSTALL_DIR" && -d "$SCRIPT_DIR/src" ]]; then
        log "Copying application files..."
        cp -r "$SCRIPT_DIR/src" "$INSTALL_DIR/"
        cp -r "$SCRIPT_DIR/scripts" "$INSTALL_DIR/"
        cp -r "$SCRIPT_DIR/requirements" "$INSTALL_DIR/" 2>/dev/null || true
        cp "$SCRIPT_DIR/run.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/" 2>/dev/null || true
    fi

    log "Directory structure created at $INSTALL_DIR"
}

create_requirements_files() {
    header "Creating Requirements Files"

    mkdir -p "$INSTALL_DIR/requirements"

    # Base requirements (no GPU dependencies)
    cat > "$INSTALL_DIR/requirements/base.txt" << 'EOF'
# Core dependencies
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
pyyaml>=6.0.1
loguru>=0.7.2
tqdm>=4.66.1

# Audio processing
sounddevice>=0.4.6
soundfile>=0.12.1

# IPC and concurrency
psutil>=5.9.0

# Additional utilities
python-dotenv>=1.0.0
EOF

    # ML/AI requirements
    cat > "$INSTALL_DIR/requirements/ml.txt" << 'EOF'
# ASR (Speech-to-Text)
# Note: faster-whisper bundles its own CUDA/ROCm runtime via CTranslate2.
# openai-whisper is NOT needed — faster-whisper replaces it entirely.
faster-whisper>=1.1.0

# Translation
transformers>=4.35.0
ctranslate2>=4.0.0
sentencepiece>=0.1.99
sacremoses>=0.1.1
protobuf>=3.20.0

# TTS (Text-to-Speech)
piper-tts>=1.2.0
onnxruntime>=1.16.0
EOF

    # Development/testing requirements
    cat > "$INSTALL_DIR/requirements/dev.txt" << 'EOF'
# Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0

# Linting
flake8>=6.1.0
black>=23.0.0
EOF

    log "Requirements files created."
}

#=============================================================================
# Model Download
#=============================================================================

download_models() {
    header "Downloading AI Models"

    source "$INSTALL_DIR/venv/bin/activate"

    log "This will download approximately 2-3 GB of model files."
    if ! confirm "Download models now?"; then
        warn "Skipping model download. Run '$INSTALL_DIR/scripts/download_models.py' later."
        return 0
    fi

    # Run model download script
    if [[ -f "$INSTALL_DIR/scripts/download_models.py" ]]; then
        venv_python python "$INSTALL_DIR/scripts/download_models.py" --all
    else
        # Fallback: download models manually
        log "Downloading Whisper model..."
        venv_python python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', download_root='$INSTALL_DIR/models/asr')"

        log "Downloading translation models..."
        venv_python python -c "from transformers import MarianMTModel, MarianTokenizer; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='$INSTALL_DIR/models/translation'); MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='$INSTALL_DIR/models/translation')"
        venv_python python -c "from transformers import MarianMTModel, MarianTokenizer; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ht', cache_dir='$INSTALL_DIR/models/translation'); MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ht', cache_dir='$INSTALL_DIR/models/translation')"
    fi

    log "Models downloaded successfully."
}

#=============================================================================
# Systemd Service
#=============================================================================

install_systemd_service() {
    header "Installing Systemd Service"

    if ! confirm "Install systemd user service for autostart?"; then
        return 0
    fi

    local service_dir="$REAL_HOME/.config/systemd/user"
    mkdir -p "$service_dir"

    cat > "$service_dir/church-translator.service" << EOF
[Unit]
Description=Church Audio Translator
After=pipewire.service pulseaudio.service
Wants=pipewire.service

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/run.py
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"
EOF

    # Add GPU-specific environment
    if [[ "$GPU_BACKEND" == "rocm" ]]; then
        # Read HSA_OVERRIDE from the .env.rocm file written during ROCm setup
        # so the correct value is used for this specific GPU (not hardcoded)
        local hsa_override=""
        if [[ -f "$INSTALL_DIR/.env.rocm" ]]; then
            hsa_override=$(grep "HSA_OVERRIDE_GFX_VERSION" "$INSTALL_DIR/.env.rocm" | cut -d= -f2 | tr -d '"' | tr -d "'" | head -1)
        fi
        # Fall back to 11.0.0 (correct for 890M/780M) if not found
        hsa_override="${hsa_override:-11.0.0}"

        cat >> "$service_dir/church-translator.service" << EOF
Environment="HSA_OVERRIDE_GFX_VERSION=${hsa_override}"
Environment="PATH=/opt/rocm/bin:\$PATH"
EOF
    fi

    cat >> "$service_dir/church-translator.service" << EOF

[Install]
WantedBy=default.target
EOF

    # Reload systemd
    systemctl --user daemon-reload

    log "Systemd service installed."
    log "Enable with: systemctl --user enable church-translator"
    log "Start with:  systemctl --user start church-translator"
}

#=============================================================================
# Create Launcher Scripts
#=============================================================================

create_launcher_scripts() {
    header "Creating Launcher Scripts"

    # Main run script
    cat > "$INSTALL_DIR/translator" << EOF
#!/bin/bash
# Church Audio Translator Launcher

INSTALL_DIR="$INSTALL_DIR"
source "\$INSTALL_DIR/venv/bin/activate"

# Set GPU environment if needed
if [[ -f "\$INSTALL_DIR/.env.rocm" ]]; then
    source "\$INSTALL_DIR/.env.rocm"
fi

cd "\$INSTALL_DIR"
exec python run.py "\$@"
EOF
    chmod +x "$INSTALL_DIR/translator"

    # Setup script
    cat > "$INSTALL_DIR/translator-setup" << EOF
#!/bin/bash
# Church Audio Translator Setup Wizard

INSTALL_DIR="$INSTALL_DIR"
source "\$INSTALL_DIR/venv/bin/activate"

cd "\$INSTALL_DIR"
exec python run.py --setup "\$@"
EOF
    chmod +x "$INSTALL_DIR/translator-setup"

    # Create symlinks in /usr/local/bin if installing to /opt
    if [[ "$INSTALL_DIR" == /opt/* ]]; then
        if confirm "Create symlinks in /usr/local/bin for easy access?"; then
            sudo ln -sf "$INSTALL_DIR/translator" /usr/local/bin/church-translator
            sudo ln -sf "$INSTALL_DIR/translator-setup" /usr/local/bin/church-translator-setup
            log "Symlinks created. Run 'church-translator' from anywhere."
        fi
    fi

    log "Launcher scripts created."
}

#=============================================================================
# Verification
#=============================================================================

verify_installation() {
    header "Verifying Installation"

    source "$INSTALL_DIR/venv/bin/activate"

    local errors=0

    # Check Python packages
    log "Checking Python packages..."

    venv_python python -c "import torch; print(f'PyTorch: {torch.__version__}')" || ((errors++))
    venv_python python -c "import faster_whisper; print('faster-whisper: OK')" || ((errors++))
    venv_python python -c "import transformers; print(f'transformers: {transformers.__version__}')" || ((errors++))
    venv_python python -c "import sounddevice; print('sounddevice: OK')" || ((errors++))

    # Check GPU availability (skip if reboot is pending — driver won't be loaded yet)
    if [[ "$NEEDS_REBOOT" == "true" ]]; then
        warn "Skipping GPU check — reboot required before driver is active."
    else
        log "Checking GPU availability..."
        if [[ "$GPU_BACKEND" == "rocm" ]]; then
            venv_python python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')" || warn "ROCm not detected by PyTorch"
            rocminfo 2>/dev/null | head -20 || warn "rocminfo not available"
        elif [[ "$GPU_BACKEND" == "cuda" ]]; then
            venv_python python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || warn "CUDA not detected by PyTorch"
            nvidia-smi 2>/dev/null | head -10 || warn "nvidia-smi not available"
        fi
    fi

    # Check audio devices
    log "Checking audio devices..."
    venv_python python -c "import sounddevice as sd; print(f'Audio devices: {len(sd.query_devices())}')" || ((errors++))

    if [[ $errors -gt 0 ]]; then
        warn "Installation completed with $errors warnings."
    else
        log "All checks passed!"
    fi
}

#=============================================================================
# Main Installation Flow
#=============================================================================

show_help() {
    cat << EOF
Church Audio Translator - Installation Script

Usage: $0 [OPTIONS]

Options:
  --rocm          Force AMD ROCm GPU backend
  --cuda          Force NVIDIA CUDA GPU backend
  --dir PATH      Install to specified directory (default: repo directory)
  --skip-models   Skip downloading AI models
  --skip-service  Skip systemd service installation
  --yes           Non-interactive: accept all defaults (use with sudo)
  --help          Show this help message

Examples:
  $0                         # Interactive installation
  $0 --rocm                  # Install with AMD ROCm support
  $0 --cuda                  # Install with NVIDIA CUDA support
  $0 --dir ~/translator      # Install to home directory
  sudo $0 --yes              # Non-interactive install (no TTY required)

EOF
}

main() {
    # Parse arguments
    FORCE_GPU=""
    SKIP_MODELS=false
    SKIP_SERVICE=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rocm)
                FORCE_GPU="rocm"
                shift
                ;;
            --cuda)
                FORCE_GPU="cuda"
                shift
                ;;
            --dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-service)
                SKIP_SERVICE=true
                shift
                ;;
            --yes|-y)
                YES=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Start installation
    header "Church Audio Translator Installation"

    echo "Installation log: $LOG_FILE"
    echo ""

    # Detect system
    detect_os
    detect_gpu
    detect_amd_gpu_arch

    # Determine GPU backend — GPU is required; abort if none found.
    if [[ -n "$FORCE_GPU" ]]; then
        GPU_BACKEND="$FORCE_GPU"
    else
        case "$GPU_TYPE" in
            amd)
                GPU_BACKEND="rocm"
                ;;
            nvidia)
                GPU_BACKEND="cuda"
                ;;
            *)
                error "No GPU detected.  This project requires a ROCm (AMD) or CUDA (NVIDIA) GPU."
                error "If you know you have a GPU but auto-detect failed, re-run with --rocm or --cuda."
                exit 1
                ;;
        esac
    fi

    log "Selected GPU backend: $GPU_BACKEND"
    echo ""

    # Confirm installation
    echo "Installation Summary:"
    echo "  - Install directory: $INSTALL_DIR"
    echo "  - GPU backend: $GPU_BACKEND"
    echo "  - OS: $OS_ID $OS_VERSION"
    echo ""

    if ! confirm "Proceed with installation?"; then
        echo "Installation cancelled."
        exit 0
    fi

    # Run installation steps
    NEEDS_REBOOT=false
    NEEDS_RELOGIN=false

    install_system_deps

    if [[ "$OS_ID" == "debian" && "$GPU_BACKEND" == "rocm" ]]; then
        setup_backports
    fi

    case "$GPU_BACKEND" in
        rocm)
            install_rocm
            ;;
        cuda)
            install_cuda
            ;;
    esac

    setup_directories
    create_requirements_files
    setup_python_env
    install_python_deps

    if [[ "$SKIP_MODELS" != "true" ]]; then
        download_models
    fi

    create_launcher_scripts

    if [[ "$SKIP_SERVICE" != "true" ]]; then
        install_systemd_service
    fi

    verify_installation

    # Final messages
    header "Installation Complete!"

    echo "To run the translator:"
    echo "  $INSTALL_DIR/translator"
    echo ""
    echo "To configure audio devices:"
    echo "  $INSTALL_DIR/translator-setup"
    echo ""

    if [[ "$NEEDS_REBOOT" == "true" ]]; then
        warn "A system reboot is required for some changes to take effect."
    elif [[ "$NEEDS_RELOGIN" == "true" ]]; then
        warn "Please log out and back in for group changes to take effect."
    fi

    echo ""
    log "Installation completed successfully!"
}

# Run main function
main "$@"
