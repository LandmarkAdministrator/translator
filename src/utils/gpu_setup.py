"""
GPU Setup Module for AMD Radeon 890M

This module configures the environment for AMD Radeon 890M (gfx1150) GPU
acceleration. It MUST be imported before any PyTorch imports.

Usage:
    import src.utils.gpu_setup  # Import first!
    import torch  # Now PyTorch will work with the GPU
"""

import os

# Required for AMD Radeon 890M (gfx1150)
# The gfx1150 architecture (RDNA 3.5) is not yet included in pre-built
# PyTorch ROCm wheels. This override tells ROCm to use gfx1100 (RDNA 3)
# kernels which are compatible with gfx1150.
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

# Reduce hipBLASLt warnings (expected on gfx1150, falls back to hipBLAS)
os.environ.setdefault('AMD_LOG_LEVEL', '1')


def get_device():
    """
    Get the best available device for PyTorch operations.

    Returns:
        torch.device: 'cuda' if GPU is available, otherwise 'cpu'
    """
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def is_gpu_available():
    """
    Check if GPU acceleration is available.

    Returns:
        bool: True if GPU is available and working
    """
    import torch
    if not torch.cuda.is_available():
        return False

    try:
        # Test actual GPU operation
        x = torch.tensor([1.0], device='cuda')
        _ = x + x
        return True
    except Exception:
        return False


def get_gpu_info():
    """
    Get information about the available GPU.

    Returns:
        dict: GPU information or None if no GPU available
    """
    import torch
    if not torch.cuda.is_available():
        return None

    return {
        'name': torch.cuda.get_device_name(0),
        'capability': torch.cuda.get_device_capability(0),
        'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'architecture_override': os.environ.get('HSA_OVERRIDE_GFX_VERSION'),
    }
