#!/usr/bin/env python3
"""
GPU Testing Script for AMD Radeon 890M with ROCm

This script verifies that:
1. ROCm is installed and accessible
2. AMD Radeon 890M (gfx1150) is detected
3. PyTorch can access the GPU
4. Basic GPU operations work

Note: The Radeon 890M (gfx1150) requires HSA_OVERRIDE_GFX_VERSION=11.0.0
to use gfx1100 (RDNA 3) kernels which are compatible with gfx1150 (RDNA 3.5).
"""

import os
import sys
import subprocess

# CRITICAL: Set this BEFORE importing PyTorch
# Required for AMD Radeon 890M (gfx1150) - use gfx1100 kernels
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'


def check_rocm_installation():
    """Check if ROCm is installed and accessible."""
    print("=" * 60)
    print("1. Checking ROCm Installation")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocminfo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("✓ ROCm is installed")

            # Check for gfx1150 (Radeon 890M) or gfx1100 (when using override)
            if "gfx1150" in result.stdout:
                print("✓ AMD Radeon 890M (gfx1150) detected")
                return True
            elif "gfx1100" in result.stdout:
                print("✓ AMD GPU detected (gfx1100 compatible)")
                return True
            else:
                print("✗ Compatible AMD GPU NOT detected")
                return False
        else:
            print("✗ ROCm command failed")
            return False

    except FileNotFoundError:
        print("✗ ROCm is NOT installed (/opt/rocm/bin/rocminfo not found)")
        return False
    except Exception as e:
        print(f"✗ Error checking ROCm: {e}")
        return False


def check_pytorch_gpu():
    """Check if PyTorch can detect and use the GPU."""
    print("\n" + "=" * 60)
    print("2. Checking PyTorch GPU Support")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            print(f"✓ GPU is accessible via PyTorch")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ GPU is NOT accessible via PyTorch")
            return False

    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False


def check_user_groups():
    """Check if user is in required groups."""
    print("\n" + "=" * 60)
    print("3. Checking User Groups")
    print("=" * 60)

    try:
        import grp

        username = os.getenv('USER', 'unknown')
        groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]

        # Add primary group
        primary_gid = os.getgid()
        primary_group = grp.getgrgid(primary_gid).gr_name
        if primary_group not in groups:
            groups.append(primary_group)

        print(f"User '{username}' is in groups: {', '.join(sorted(groups))}")

        required_groups = ['video', 'render']
        missing_groups = [g for g in required_groups if g not in groups]

        if not missing_groups:
            print(f"✓ User is in all required groups: {', '.join(required_groups)}")
            return True
        else:
            print(f"✗ User is missing groups: {', '.join(missing_groups)}")
            return False

    except Exception as e:
        print(f"✗ Error checking user groups: {e}")
        return False


def test_gpu_operations():
    """Test basic GPU operations."""
    print("\n" + "=" * 60)
    print("4. Testing GPU Operations")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⊘ Skipping GPU operations (GPU not available)")
            return False

        # Test tensor creation on GPU
        print("Testing tensor creation on GPU...")
        x = torch.randn(1000, 1000, device='cuda')
        print(f"✓ Created tensor on GPU: {x.shape}")

        # Test matrix multiplication
        print("Testing matrix multiplication on GPU...")
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)
        print(f"✓ Matrix multiplication successful: {z.shape}")

        # Test data transfer
        print("Testing GPU ↔ CPU data transfer...")
        cpu_tensor = z.cpu()
        gpu_tensor = cpu_tensor.cuda()
        print("✓ Data transfer successful")

        # Measure performance
        print("\nPerformance test (1000x1000 matrix multiplication):")
        import time

        # GPU test
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU time: {gpu_time:.4f} seconds")

        # CPU test
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(100):
            _ = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        print(f"  CPU time: {cpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")

        return True

    except Exception as e:
        print(f"✗ GPU operations failed: {e}")
        return False


def main():
    """Run all GPU tests."""
    print("\n" + "=" * 60)
    print("AMD Radeon 890M ROCm GPU Test Suite")
    print("=" * 60)
    print(f"HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
    print("(Required for gfx1150 compatibility with gfx1100 kernels)")

    results = {
        'ROCm installed': check_rocm_installation(),
        'PyTorch GPU': check_pytorch_gpu(),
        'User groups': check_user_groups(),
        'GPU operations': test_gpu_operations()
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All tests passed! GPU acceleration is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
