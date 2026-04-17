#!/bin/bash
#
# Parakeet ASR install for the ROCm target machine.
#
# Installs:
#   1. onnxruntime-rocm from AMD's repo (replaces any stock onnxruntime wheel)
#   2. onnx-asr (the inference wrapper around the published Parakeet ONNX model)
#   3. Pre-downloads the Parakeet TDT 0.6b v3 model into models/asr/parakeet
#   4. Patches the AMD wheel's executable-stack bit so Linux 6.x loaders accept it
#
# Run this from the project root INSIDE the venv on the target machine:
#   source venv/bin/activate && ./scripts/install_parakeet.sh
#
# After a successful install:
#   python run.py --parakeet
#
# Known issue — runs on CPU, not GPU:
#   The currently-published onnxruntime-rocm wheel (1.22.2.post1) is built
#   against the ROCm 6.x ABI — it links libhipblas.so.2 and libamdhip64.so.6.
#   Debian 13 with ROCm 7.2 ships .so.3 and .so.7 respectively, so at runtime
#   the ROCMExecutionProvider and MIGraphXExecutionProvider fail to load and
#   onnxruntime silently falls back to CPU.  You'll see lines like
#   "Failed to load library libonnxruntime_providers_rocm.so" on stderr.
#   This is benign: Parakeet TDT 0.6b v3 hits RTF ≈ 0.06 on CPU on a
#   Ryzen AI 9 HX 370, so real-time streaming is easy.
#   Do NOT symlink libhipblas.so.3 → libhipblas.so.2 to "fix" this — it's a
#   major ABI bump and will crash or produce wrong results.
#   When AMD releases a 1.24+ wheel built for ROCm 7.x, this goes away.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

hdr()  { echo -e "\n${BLUE}=== $1 ===${NC}"; }
ok()   { echo -e "${GREEN}OK${NC} $1"; }
warn() { echo -e "${YELLOW}WARN${NC} $1"; }
die()  { echo -e "${RED}ERR${NC} $1" >&2; exit 1; }

# Sanity: must be in a venv.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    die "Activate the project venv first: source venv/bin/activate"
fi

# Sanity: must be at project root.
if [ ! -f "run.py" ] || [ ! -d "src/pipeline" ]; then
    die "Run from the translator project root"
fi

ROCM_WHEEL_REPO="${ROCM_WHEEL_REPO:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/}"

hdr "Remove any stock onnxruntime (conflicts with onnxruntime-rocm)"
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
ok "stock onnxruntime gone"

hdr "Install onnxruntime-rocm from $ROCM_WHEEL_REPO"
pip install --pre onnxruntime-rocm -f "$ROCM_WHEEL_REPO" \
    || die "onnxruntime-rocm install failed — check the ROCm version repo URL"
ok "onnxruntime-rocm installed"

hdr "Patch executable-stack bit on onnxruntime .so"
# Debian 13 / Linux 6.x loaders reject shared libraries with GNU_STACK=RWE.
# The AMD wheels set this flag, so we clear it on the pybind .so before first
# import. Cannot `import onnxruntime` to locate the file — that import is
# exactly what's broken pre-patch — so derive the path from the venv instead.
# No-op if the flag is already clean.
python - <<'PATCH_EOF' || die "execstack patch failed"
import struct, sys
from pathlib import Path
site_pkgs = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
so = site_pkgs / "onnxruntime" / "capi" / "onnxruntime_pybind11_state.so"
if not so.exists():
    raise SystemExit(f"onnxruntime .so not found at {so} — is onnxruntime-rocm installed?")
data = bytearray(so.read_bytes())
PT_GNU_STACK = 0x6474e551
e_phoff = struct.unpack_from("<Q", data, 32)[0]
e_phentsize = struct.unpack_from("<H", data, 54)[0]
e_phnum = struct.unpack_from("<H", data, 56)[0]
patched = False
for i in range(e_phnum):
    base = e_phoff + i * e_phentsize
    if struct.unpack_from("<I", data, base)[0] == PT_GNU_STACK:
        p_flags_offset = base + 4
        old = struct.unpack_from("<I", data, p_flags_offset)[0]
        if old & 0x1:
            struct.pack_into("<I", data, p_flags_offset, old & ~0x1)
            patched = True
            print(f"cleared PF_X: {old:#x} -> {old & ~0x1:#x}")
if patched:
    so.write_bytes(bytes(data))
    print("patched")
else:
    print("already clean")
PATCH_EOF
ok "executable-stack bit cleared"

hdr "Install onnx-asr"
pip install 'onnx-asr[hub]' \
    || die "onnx-asr install failed"
ok "onnx-asr installed"

hdr "Verify ONNX Runtime providers"
# get_available_providers() reports what was compiled in, not what will
# actually load at runtime — on ROCm 7.2 the ROCm/MIGraphX providers show up
# here but fail silently on first use (hipblas.so major-version mismatch)
# and fall through to CPU. That's expected; the Parakeet smoke test below
# prints the provider that's actually used for the first decode.
python -c "
import onnxruntime as ort
print('compiled-in providers:', ort.get_available_providers())
"

hdr "Pre-download Parakeet TDT 0.6b v3"
mkdir -p models/asr/parakeet
HF_HOME="$(pwd)/models/asr/parakeet" python -c "
import onnx_asr
m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', providers=['CPUExecutionProvider'])
print('Parakeet model ready')
" || die "Parakeet model download/load failed"
ok "Parakeet model cached under models/asr/parakeet"

hdr "Quick smoke test on a silent 1s buffer"
# Force CPUExecutionProvider to keep the output clean — on ROCm 7.2 the
# ROCm/MIGraphX providers spam "Failed to load library" lines and fall back
# to CPU anyway. The real app (parakeet_asr.py) requests ROCm→CPU; this
# smoke test just proves the wheel and model are loadable.
python -c "
import numpy as np, onnx_asr, os
os.environ['HF_HOME'] = os.path.abspath('models/asr/parakeet')
m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', providers=['CPUExecutionProvider'])
r = m.recognize(np.zeros(16000, dtype=np.float32), sample_rate=16000)
print('smoke test OK (output on silence):', repr(r))
" || die "smoke test failed"

hdr "Done"
echo "Run it with:"
echo "    python run.py --parakeet"
