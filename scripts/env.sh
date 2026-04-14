#!/bin/bash
# Environment configuration for Church Audio Translator
# Source this file before running any Python scripts manually
#
# Usage:
#   source scripts/env.sh
#   python scripts/test_gpu.py
#
# The install script writes .env.rocm with the correct HSA_OVERRIDE value
# for your specific GPU. If that file exists, it is loaded automatically.
# For manual use, see the GPU reference table below.

# AMD iGPU HSA override — required for integrated Radeon GPUs
# Load from .env.rocm if available (written by install.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env.rocm" ]]; then
    # shellcheck source=/dev/null
    source "$PROJECT_DIR/.env.rocm"
else
    # GPU Architecture Reference:
    #   Radeon 890M / 780M  (gfx1150 / gfx1103) → HSA_OVERRIDE_GFX_VERSION=11.0.0
    #   Radeon 680M         (gfx1035)             → HSA_OVERRIDE_GFX_VERSION=10.3.0
    #   RX 7900 / RX 6800   (gfx1100 / gfx1030)  → (no override needed)
    #   NVIDIA GPUs         → (no override needed, uses CUDA)
    #
    # Uncomment and set the correct value for your GPU if not using install.sh:
    # export HSA_OVERRIDE_GFX_VERSION=11.0.0
    :
fi

# Suppress hipBLASLt informational messages on AMD iGPUs
if [[ -n "$HSA_OVERRIDE_GFX_VERSION" ]]; then
    export AMD_LOG_LEVEL=0
fi

# Add ROCm to PATH if installed
if [[ -d /opt/rocm/bin ]]; then
    export PATH=/opt/rocm/bin:$PATH
fi

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    fi
fi
