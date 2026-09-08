#!/usr/bin/env bash
# scripts/gpu_doctor.sh — why does the GPU work, or not? Read-only.
#
#   scripts/gpu_doctor.sh           # full report with a fix for each problem
#   scripts/gpu_doctor.sh --brief   # one line and an exit status (install_site.sh uses it)
#
# Getting the GPU working is the hard part of an install, and the failure
# modes are known: Secure Boot refusing the unsigned NVIDIA DKMS module, no
# headers for the running kernel so DKMS built nothing, the driver installed
# but not loaded because nobody rebooted, nouveau still bound to the card, a
# driver too old for the CUDA the PyTorch wheel was built with, the user not
# in the render/video groups for ROCm, an iGPU without its HSA override. Each
# check below names the fix. Two reference machines pass this: the production
# host (RTX 3060, NVIDIA's Debian 13 repo, driver 610, torch cu128) and the
# development laptop (Radeon 890M, ROCm 7.2.2, Secure Boot on — harmless for
# the in-tree amdgpu driver).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BRIEF=0; [ "${1:-}" = "--brief" ] && BRIEF=1
FAILS=0; HINTS=(); SUMMARY=""

ok()   { [ "$BRIEF" = 1 ] || printf '  ok    %s\n' "$1"; }
note() { [ "$BRIEF" = 1 ] || printf '  --    %s\n' "$1"; }
bad()  { [ "$BRIEF" = 1 ] || printf '  BAD   %s\n' "$1"; FAILS=$((FAILS + 1)); HINTS+=("$2"); }
head_() { [ "$BRIEF" = 1 ] || printf '\n== %s\n' "$1"; }

secure_boot() {  # prints enabled / disabled / unknown
  if command -v mokutil >/dev/null 2>&1; then
    mokutil --sb-state 2>/dev/null | grep -qi "SecureBoot enabled" && echo enabled || echo disabled
  else
    local f; f=$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1)
    if [ -n "$f" ]; then [ "$(od -An -tu1 -j4 -N1 "$f" 2>/dev/null | tr -d ' ')" = "1" ] && echo enabled || echo disabled
    else echo unknown; fi
  fi
}

torch_check() {  # torch_check PYTHON LABEL  -> sets TORCH_LINE, returns 0 if the GPU computes
  local py="$1" label="$2"
  TORCH_LINE=$(timeout 120 "$py" - <<'PY' 2>&1 | tail -1
import torch
if not torch.cuda.is_available():
    print(f"torch {torch.__version__}: no GPU visible"); raise SystemExit(1)
a = torch.ones(256, 256, device="cuda"); s = float((a @ a).sum())
api = f"CUDA {torch.version.cuda}" if torch.version.cuda else f"HIP {torch.version.hip}"
print(f"torch {torch.__version__} ({api}) computes on {torch.cuda.get_device_name(0)}" if s == 256**3 else "GPU result WRONG")
raise SystemExit(0 if s == 256**3 else 1)
PY
  )
  return $?
}

# ---- which GPUs are in the box ----------------------------------------
head_ "GPUs on the PCI bus"
PCI=$(lspci 2>/dev/null | grep -Ei "vga|3d|display")
NV=$(printf '%s\n' "$PCI" | grep -i nvidia | head -1 | sed 's/.*: //')
AMD=$(printf '%s\n' "$PCI" | grep -E "\[AMD/ATI\]|Advanced Micro Devices|Radeon" | head -1 | sed 's/.*: //')
[ -n "$NV" ] && ok "NVIDIA: $NV"
[ -n "$AMD" ] && ok "AMD: $AMD"
[ -z "$NV$AMD" ] && bad "no NVIDIA or AMD display device on the PCI bus" "the pipeline needs a GPU; check the card is seated/connected and visible in the firmware"
SB=$(secure_boot)

# ---- NVIDIA ---------------------------------------------------------------
if [ -n "$NV" ]; then
  head_ "NVIDIA driver"
  case "$SB" in
    enabled) bad "Secure Boot is ENABLED" "the NVIDIA module DKMS builds is unsigned and will not load: disable Secure Boot in the firmware setup, then reboot" ;;
    disabled) ok "Secure Boot disabled (the unsigned DKMS module can load)" ;;
    *) note "Secure Boot state unknown (no mokutil, no efivars)" ;;
  esac
  if dpkg -s "linux-headers-$(uname -r)" >/dev/null 2>&1; then ok "headers for the running kernel $(uname -r)"
  else bad "no headers for the running kernel $(uname -r)" "sudo apt install linux-headers-$(uname -r)   (or the backports kernel+headers together, then reboot); DKMS cannot build without them"; fi
  if grep -rqs "developer.download.nvidia.com" /etc/apt/sources.list /etc/apt/sources.list.d/; then ok "NVIDIA's apt repository configured"
  else note "NVIDIA's apt repository not configured — Debian's own nvidia-driver may not build against a backports kernel (install.sh --cuda adds the repository)"; fi
  DRV=$(dpkg-query -W -f='${Version}' nvidia-driver 2>/dev/null); KM=$(dpkg -l 2>/dev/null | awk '/^ii  nvidia-kernel-(open-)?dkms /{print $2" "$3}')
  if [ -n "$DRV" ]; then ok "packages: nvidia-driver $DRV; ${KM:-no DKMS kernel module package}"
  else bad "nvidia-driver package not installed" "./install.sh --cuda"; fi
  DK=$(/usr/sbin/dkms status 2>/dev/null || dkms status 2>/dev/null)
  if printf '%s\n' "$DK" | grep -q "nvidia.*$(uname -r).*installed"; then ok "DKMS module built for $(uname -r)"
  elif [ -n "$KM" ]; then bad "DKMS has no nvidia module for $(uname -r) ($(printf '%s' "$DK" | tr '\n' ';' | cut -c1-80))" "sudo dkms autoinstall; if it fails read /var/lib/dkms/nvidia/*/build/make.log (usually headers or Secure Boot)"; fi
  if lsmod | grep -q "^nvidia "; then ok "nvidia kernel module loaded"
  elif lsmod | grep -q "^nouveau "; then bad "nouveau is bound to the card, nvidia is not loaded" "blacklist nouveau (/etc/modprobe.d/nvidia.conf), sudo update-initramfs -u, reboot"
  elif [ -n "$KM" ]; then bad "nvidia module built but not loaded" "reboot (the module loads at boot); or sudo modprobe nvidia"; fi
  if command -v nvidia-smi >/dev/null 2>&1 && SMI=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null); then
    VER=${SMI%%,*}; MAJ=${VER%%.*}
    if [ "${MAJ:-0}" -ge 570 ]; then ok "nvidia-smi: driver $SMI (CUDA 12.8 wheels need 570+)"
    else bad "driver $VER is older than 570" "the torch cu128 wheels need driver 570 or newer: upgrade from NVIDIA's repository (install.sh --cuda)"; fi
  else bad "nvidia-smi does not work" "the driver is not loaded — see the checks above"; fi
fi

# ---- AMD / ROCm -------------------------------------------------------------
if [ -n "$AMD" ] && [ -z "$NV" ]; then
  head_ "AMD ROCm"
  [ "$SB" = enabled ] && note "Secure Boot enabled — fine for AMD: amdgpu is part of the kernel, nothing unsigned is built"
  if [ -f /opt/rocm/.info/version ]; then ok "ROCm $(cat /opt/rocm/.info/version) in /opt/rocm"
  else bad "ROCm not installed (/opt/rocm missing)" "./install.sh --rocm"; fi
  for g in render video; do id -nG | tr ' ' '\n' | grep -qx "$g" && ok "user in group $g" || bad "user not in group $g" "sudo usermod -aG render,video $USER, then log out and back in"; done
  [ -e /dev/kfd ] && ok "/dev/kfd present ($(stat -c '%G %A' /dev/kfd))" || bad "/dev/kfd missing" "the amdgpu driver did not initialise the compute interface: check dmesg for amdgpu, and the kernel version (iGPUs need 6.10+)"
  K=$(uname -r); KMAJ=${K%%.*}; KMIN=$(echo "$K" | cut -d. -f2)
  if [ "$KMAJ" -gt 6 ] || { [ "$KMAJ" -eq 6 ] && [ "$KMIN" -ge 10 ]; }; then ok "kernel $K"
  else bad "kernel $K is older than 6.10" "Radeon 680M/780M/890M need 6.10+: install the backports kernel (install.sh does this)"; fi
  if printf '%s' "$AMD" | grep -qiE "680m|780m|890m|graphics"; then
    if [ -f "$REPO/.env.rocm" ] && grep -q HSA_OVERRIDE_GFX_VERSION "$REPO/.env.rocm"; then ok "iGPU HSA override set in .env.rocm ($(grep -o 'HSA_OVERRIDE_GFX_VERSION=.*' "$REPO/.env.rocm"))"
    else bad "iGPU without HSA_OVERRIDE_GFX_VERSION in $REPO/.env.rocm" "ROCm does not list these iGPUs officially; install.sh writes the override (11.0.0 for 780M/890M, 10.3.0 for 680M)"; fi
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    GFX=$(rocminfo 2>/dev/null | grep -m1 -o "gfx[0-9a-z]*")
    [ -n "$GFX" ] && ok "rocminfo sees $GFX" || bad "rocminfo lists no GPU agent" "usually the render group (log out/in after adding it) or /dev/kfd"
  fi
fi

# ---- PyTorch in the virtual environments ---------------------------------
head_ "PyTorch"
# run.py loads .env.rocm (the iGPU override) on ROCm boxes; mirror that here,
# and only there — a CUDA host may carry a stale copy whose $LD_LIBRARY_PATH
# reference would abort this script under set -u.
if [ -n "$AMD" ] && [ -z "$NV" ] && [ -f "$REPO/.env.rocm" ]; then
  set +u; . "$REPO/.env.rocm" 2>/dev/null || true; set -u
fi
if [ -x "$REPO/venv/bin/python" ]; then
  if torch_check "$REPO/venv/bin/python" main; then ok "main venv: $TORCH_LINE"
  else bad "main venv: $TORCH_LINE" "the venv's torch build must match the driver (cu128 needs driver 570+; ROCm wheels must match /opt/rocm's major.minor): ./install.sh reinstalls it"; fi
else note "no main venv at $REPO/venv yet"; fi
if [ -x "$HOME/nemo-venv/bin/python" ]; then
  if torch_check "$HOME/nemo-venv/bin/python" nemo; then ok "NeMo venv: $TORCH_LINE"
  else bad "NeMo venv: $TORCH_LINE" "recreate ~/nemo-venv from requirements-nemo.txt (scripts/install_site.sh)"; fi
else note "no NeMo venv yet (scripts/install_site.sh creates it)"; fi

# ---- verdict ---------------------------------------------------------------
if [ "$BRIEF" = 1 ]; then
  if [ "$FAILS" = 0 ]; then echo "GPU ok: ${NV:-$AMD}; $TORCH_LINE"; exit 0
  else echo "GPU: $FAILS problem(s) — ${HINTS[0]}"; exit 1; fi
fi
printf '\n'
if [ "$FAILS" = 0 ]; then echo "GPU stack is healthy."; exit 0; fi
echo "$FAILS problem(s). Fixes, in order:"
i=1; for h in "${HINTS[@]}"; do echo "  $i. $h"; i=$((i + 1)); done
exit 1
