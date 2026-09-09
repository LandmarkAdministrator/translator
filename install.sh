#!/bin/bash
#
# Church Audio Translator - Installation Script
#
# Supports: Debian 13 (Trixie), Ubuntu 22.04+
# GPU: AMD ROCm 7.2+ (required for iGPUs), NVIDIA CUDA 12.x+
#
# Usage:
#   ./install.sh              # Interactive install (requires a GPU)
#   ./install.sh --rocm       # Force AMD ROCm installation
#   ./install.sh --cuda       # Force NVIDIA CUDA installation
#   ./install.sh --parakeet   # Also install onnx-asr + Parakeet ONNX model
#                             # (the ASR fallback when the NeMo venv is absent;
#                             # production ASR: requirements-nemo.txt)
#   ./install.sh --help       # Show help
#
# Note: CPU-only installation is not supported — the translator requires
# a ROCm or CUDA GPU to run in real time.  The Parakeet streaming backend
# runs on CPU (onnxruntime-rocm's ROCM/MIGraphX providers don't load against
# ROCm 7.2's hipblas.so.3), but CPU is comfortably >RTF 1 for Parakeet on
# modern hardware.
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

# PyTorch build for the main venv. The NeMo venv pins the same version
# (requirements-nemo.txt); keep the two together.
TORCH_VERSION="2.11.0"

# NVIDIA kernel module flavour: the open module (Turing / RTX 20xx / GTX 16xx
# and newer) unless --nvidia-proprietary.
NVIDIA_PROPRIETARY=false

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

secure_boot_enabled() {
    # DKMS builds an unsigned NVIDIA module; a Secure Boot kernel refuses it.
    # (AMD is unaffected: amdgpu ships inside the kernel.)
    if command_exists mokutil; then
        mokutil --sb-state 2>/dev/null | grep -qi "SecureBoot enabled"
        return
    fi
    local f
    f=$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1)
    [[ -n "$f" ]] && [[ "$(od -An -tu1 -j4 -N1 "$f" 2>/dev/null | tr -d ' ')" == "1" ]]
}

reboot_then_rerun() {
    # $1 = why, $2 = the flags to re-run with. The script is resumable: on
    # the next run every completed stage detects itself and is skipped.
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  REBOOT REQUIRED BEFORE CONTINUING                        ${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    warn "$1."
    log "After rebooting, run this script again to continue:"
    echo "  ./install.sh $2"
    echo ""
    if confirm "Reboot now?"; then
        sudo reboot
    else
        echo "Please reboot manually, then re-run: ./install.sh $2"
        exit 0
    fi
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
        libssl-dev \
        mokutil \
        pciutils

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

    # Fallback to known latest if detection fails.
    # 7.2.2 is what we currently test against on gfx1150 (Radeon 890M).
    if [[ -z "$rocm_version" ]]; then
        rocm_version="7.2.2"  # Fallback - update this periodically
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

    # Already loaded? Nothing to do — this is what makes the script resumable
    # after the reboot below.
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        log "NVIDIA driver already loaded: $driver_version"
        if [[ "${driver_version%%.*}" -lt 570 ]]; then
            warn "Driver $driver_version is older than 570; the torch cu128 wheels need 570 or newer."
            warn "Upgrade from NVIDIA's repository: remove the driver and re-run ./install.sh --cuda."
        fi
        return 0
    fi

    if [[ "$OS_ID" == "ubuntu" ]]; then
        # Untested here. NVIDIA's repository and its cuda-drivers meta-package
        # (the newest driver), same family as the Debian path below.
        local ubuntu_version cuda_repo_ubuntu="ubuntu2204"
        ubuntu_version=$(echo "$OS_VERSION" | tr -d '.')
        [[ "$ubuntu_version" -ge 2404 ]] && cuda_repo_ubuntu="ubuntu2404"
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo_ubuntu}/x86_64/cuda-keyring_1.1-1_all.deb"
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
        sudo apt update
        sudo apt install -y cuda-drivers
        reboot_then_rerun "The NVIDIA module loads at boot" "--cuda"
        return 0
    fi

    # Debian. This is the procedure that brought up the production host
    # (Debian 13, RTX 3060, 2026-09-01) after Debian's own nvidia-driver from
    # non-free failed to build against a backports kernel: NVIDIA's Debian
    # repository, the open kernel module through DKMS, headers matching the
    # running kernel, and one reboot.

    # 1. Secure Boot: the module DKMS builds is unsigned; the kernel refuses it.
    if secure_boot_enabled; then
        error "Secure Boot is enabled. The NVIDIA kernel module that DKMS builds is unsigned"
        error "and will not load. Disable Secure Boot in the firmware (BIOS/UEFI) setup,"
        error "boot again, and re-run: ./install.sh --cuda"
        exit 1
    fi
    log "Secure Boot is off."

    # 2. Headers for the running kernel, or DKMS has nothing to build against.
    if dpkg -s "linux-headers-$(uname -r)" >/dev/null 2>&1; then
        log "Kernel headers for $(uname -r) present."
    else
        log "Installing kernel headers for $(uname -r)..."
        if ! sudo apt install -y "linux-headers-$(uname -r)"; then
            # The running kernel's headers are gone from the archive: move to
            # the current backports kernel and headers together, then return.
            warn "No headers for the running kernel; installing the backports kernel and headers instead."
            local backports_file="/etc/apt/sources.list.d/${OS_CODENAME}-backports.list"
            [[ -f "$backports_file" ]] || echo "deb http://deb.debian.org/debian/ ${OS_CODENAME}-backports main contrib non-free non-free-firmware" | sudo tee "$backports_file"
            sudo apt update
            sudo apt install -y -t "${OS_CODENAME}-backports" linux-image-amd64 linux-headers-amd64
            reboot_then_rerun "The new kernel must be running before the driver module can be built for it" "--cuda"
        fi
    fi

    # 3. NVIDIA's repository for this Debian release.
    local keyring=/usr/share/keyrings/cuda-archive-keyring.gpg
    if [[ ! -f "$keyring" ]]; then
        local deb_major="${OS_VERSION%%.*}" tmp
        tmp=$(mktemp -d)
        log "Adding NVIDIA's Debian ${deb_major} repository..."
        if ! wget -qO "$tmp/cuda-keyring.deb" \
             "https://developer.download.nvidia.com/compute/cuda/repos/debian${deb_major}/x86_64/cuda-keyring_1.1-1_all.deb"; then
            error "Could not download cuda-keyring for debian${deb_major}."
            exit 1
        fi
        sudo dpkg -i "$tmp/cuda-keyring.deb"
        rm -rf "$tmp"
    fi
    sudo apt update

    # 4. The driver. Open kernel module by default; the package blacklists
    # nouveau itself (/etc/modprobe.d/nvidia.conf). No CUDA toolkit: the
    # PyTorch wheels carry their own runtime, only the driver is needed.
    local kmod="nvidia-kernel-open-dkms"
    [[ "$NVIDIA_PROPRIETARY" == "true" ]] && kmod="nvidia-kernel-dkms"
    log "Installing $kmod nvidia-driver nvidia-driver-cuda (DKMS builds the module; a few minutes)..."
    sudo apt install -y "$kmod" nvidia-driver nvidia-driver-cuda

    # 5. Did DKMS build it for THIS kernel?
    if /usr/sbin/dkms status 2>/dev/null | grep -q "nvidia.*$(uname -r).*installed"; then
        log "DKMS built the nvidia module for $(uname -r)."
    else
        error "DKMS did not report the nvidia module as installed for $(uname -r):"
        /usr/sbin/dkms status 2>/dev/null | sed 's/^/    /' || true
        error "Run: sudo dkms autoinstall   and read /var/lib/dkms/nvidia/*/build/make.log"
        exit 1
    fi

    reboot_then_rerun "The NVIDIA module loads at boot; PyTorch cannot see the GPU until then" "--cuda"
}

#=============================================================================
# PyTorch ROCm Wheel Detection
#=============================================================================

detect_pytorch_rocm_version() {
    # The wheel has to match the ROCm that is installed, not the newest one
    # PyTorch publishes: 7.2.2 in /opt/rocm wants the rocm7.2 wheel. Messages
    # go to stderr because the caller captures stdout as the answer.
    if [[ -f /opt/rocm/.info/version ]]; then
        local installed
        installed="rocm$(cut -d. -f1-2 /opt/rocm/.info/version)"
        log "PyTorch ROCm wheel matching the installed ROCm: $installed" >&2
        echo "$installed"
        return
    fi
    local pytorch_rocm=""
    if command -v curl &> /dev/null; then
        pytorch_rocm=$(curl -s "https://download.pytorch.org/whl/" 2>/dev/null | \
            grep -oP '(?<=href=")rocm[0-9]+\.[0-9]+(?=/)' | \
            sort -V | tail -1)
    fi
    if [[ -n "$pytorch_rocm" ]]; then
        log "Newest PyTorch ROCm wheel: $pytorch_rocm" >&2
        echo "$pytorch_rocm"
    else
        warn "Could not detect a PyTorch ROCm wheel version; using rocm7.2" >&2
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
    # torch only: nothing here uses torchvision or torchaudio, and the NeMo
    # venv brings its own torch of the same version.
    case "$GPU_BACKEND" in
        rocm)
            local pytorch_rocm_ver
            pytorch_rocm_ver=$(detect_pytorch_rocm_version)
            log "Installing PyTorch ${TORCH_VERSION} for ${pytorch_rocm_ver}..."
            venv_python pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${pytorch_rocm_ver}"
            ;;
        cuda)
            log "Installing PyTorch ${TORCH_VERSION} for CUDA 12.8 (driver 570 or newer)..."
            venv_python pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
            ;;
        cpu)
            log "Installing PyTorch ${TORCH_VERSION} (CPU only)..."
            venv_python pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
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
    # numpy is unpinned: PyTorch 2.11+ (ROCm 7.2 wheel) requires numpy 2.x,
    # and torch's own constraint picks a compatible version.
    cat > "$INSTALL_DIR/requirements/base.txt" << 'EOF'
# Core dependencies
numpy>=2.0.0
scipy>=1.13.0
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
    # Minimums reflect the versions known to run on ROCm 7.2 / PyTorch 2.11.
    cat > "$INSTALL_DIR/requirements/ml.txt" << 'EOF'
# ASR (Speech-to-Text): the streaming model runs in its own venv
# (requirements-nemo.txt); the onnx-asr fallback is installed by --parakeet.

# Translation
transformers>=5.0.0
sentencepiece>=0.1.99
sacremoses>=0.1.1
protobuf>=4.21.0
huggingface_hub>=1.0.0

# TTS (Text-to-Speech)
piper-tts>=1.4.0
onnxruntime>=1.24.0   # Replaced by onnxruntime-rocm when --parakeet is used
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
        log "Downloading translation models..."
        venv_python python -c "from transformers import MarianMTModel, MarianTokenizer; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='$INSTALL_DIR/models/translation'); MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir='$INSTALL_DIR/models/translation')"
        venv_python python -c "from transformers import MarianMTModel, MarianTokenizer; MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ht', cache_dir='$INSTALL_DIR/models/translation'); MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ht', cache_dir='$INSTALL_DIR/models/translation')"
    fi

    log "Models downloaded successfully."
}

#=============================================================================
# Parakeet Streaming Backend (optional)
#=============================================================================

install_parakeet() {
    header "Installing Parakeet Streaming Backend"

    local script="$INSTALL_DIR/scripts/install_parakeet.sh"
    if [[ ! -f "$script" ]]; then
        warn "scripts/install_parakeet.sh not found — skipping Parakeet install."
        return 0
    fi

    # install_parakeet.sh requires the venv to be activated.
    # When run under sudo we need to stay as REAL_USER so writes to
    # ~/translator/models and venv/ use the right ownership.
    if [[ $EUID -eq 0 && "$REAL_USER" != "root" ]]; then
        sudo -u "$REAL_USER" bash -c "source '$INSTALL_DIR/venv/bin/activate' && cd '$INSTALL_DIR' && bash '$script'"
    else
        (source "$INSTALL_DIR/venv/bin/activate" && cd "$INSTALL_DIR" && bash "$script")
    fi

    log "Parakeet backend installed.  Run with: $INSTALL_DIR/translator --parakeet"
}

#=============================================================================
# Create Launcher Scripts
#=============================================================================
# (The systemd units are the site layer's job — scripts/install_site.sh
# installs translate.service, translate-web.service and the timers from
# systemd/. The church-translator.service writer that used to live here was
# never called since the site layer arrived and was removed on 2026-09-09.)

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
    venv_python python -c "import loguru, yaml, numpy; print('base deps: OK')" || ((errors++))
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

    # The GPU doctor names the fix for anything wrong in the driver stack.
    if [[ "$NEEDS_REBOOT" != "true" ]]; then
        "$INSTALL_DIR/scripts/gpu_doctor.sh" || warn "GPU problems reported above — fix them before the site setup."
    fi

    # Check audio devices
    log "Checking audio devices..."
    venv_python python -c "import sounddevice as sd; print(f'Audio devices: {len(sd.query_devices())}')" || ((errors++))

    # Parakeet-specific checks (only if installed)
    if [[ "$INSTALL_PARAKEET" == "true" ]]; then
        log "Checking Parakeet stack..."
        venv_python python -c "import onnx_asr; import onnxruntime as ort; print(f'onnx-asr OK; providers={ort.get_available_providers()}')" || ((errors++))
    fi

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
  --parakeet      Also install onnx-asr + Parakeet ONNX model (the no-NeMo ASR fallback)
  --nvidia-proprietary  NVIDIA's proprietary kernel module (cards older than Turing / RTX 20xx)
  --dir PATH      Install to specified directory (default: repo directory)
  --skip-models   Skip downloading AI models
  --skip-service  Skip the site layer (scripts/install_site.sh: NeMo venv, units, config)
  --yes           Non-interactive: accept all defaults (use with sudo)
  --help          Show this help message

Examples:
  $0                         # Interactive installation
  $0 --rocm                  # Install with AMD ROCm support
  $0 --rocm --parakeet       # ROCm + Parakeet streaming backend
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
    INSTALL_PARAKEET=false

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
            --parakeet)
                INSTALL_PARAKEET=true
                shift
                ;;
            --nvidia-proprietary)
                NVIDIA_PROPRIETARY=true
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
    echo "  - Parakeet backend: $([[ "$INSTALL_PARAKEET" == "true" ]] && echo "yes" || echo "no (use --parakeet to enable)")"
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

    if [[ "$INSTALL_PARAKEET" == "true" ]]; then
        install_parakeet
    fi

    create_launcher_scripts

    verify_installation

    # The site layer — NeMo venv, config, admin password, scheduler, units —
    # is its own idempotent script so it can be rerun after every git pull.
    if [[ "$SKIP_SERVICE" != "true" ]]; then
        header "Site setup"
        if [[ "$YES" == "true" ]]; then
            "$INSTALL_DIR/scripts/install_site.sh" --yes || warn "site setup reported problems — rerun scripts/install_site.sh"
        else
            "$INSTALL_DIR/scripts/install_site.sh" || warn "site setup reported problems — rerun scripts/install_site.sh"
        fi
    fi

    # Final messages
    header "Installation Complete!"

    echo "The service starts itself inside the windows in config/schedule.conf."
    echo "Configure audio devices and windows from the admin panel (see the"
    echo "site setup summary above), or run the pipeline by hand:"
    echo "  $INSTALL_DIR/scripts/run_production.sh"
    echo ""
    echo "Rerun scripts/install_site.sh after any git pull; it changes only what differs."
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
