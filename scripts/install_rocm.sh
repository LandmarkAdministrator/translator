#!/bin/bash
#
# ROCm 7.2 Installation Script for AMD Radeon 890M (gfx1150)
# Debian 13 (Bookworm)
#
# This script installs ROCm and configures the system for GPU acceleration
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run as root (use sudo)"
    echo "Usage: sudo bash $0"
    exit 1
fi

# Get the actual username (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
if [ "$ACTUAL_USER" = "root" ]; then
    print_error "Please run this script with sudo, not as root directly"
    echo "Usage: sudo bash $0"
    exit 1
fi

print_header "ROCm 7.2 Installation for AMD Radeon 890M"

print_info "Installing for user: $ACTUAL_USER"
print_info "System: $(lsb_release -ds)"
print_info "Kernel: $(uname -r)"

# Step 1: Check prerequisites
print_header "Step 1: Checking Prerequisites"

# Check Debian version
if ! grep -q "bookworm\|13" /etc/debian_version 2>/dev/null && ! grep -q "bookworm" /etc/os-release 2>/dev/null; then
    print_warning "This script is designed for Debian 13 (Bookworm)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for AMD GPU
if lspci | grep -i "VGA.*AMD\|Display.*AMD" > /dev/null; then
    GPU_INFO=$(lspci | grep -i "VGA.*AMD\|Display.*AMD" | head -n1)
    print_success "AMD GPU detected: $GPU_INFO"
else
    print_error "No AMD GPU detected!"
    print_warning "Continuing anyway, but ROCm may not work..."
fi

# Step 2: Create keyrings directory
print_header "Step 2: Setting Up ROCm Repository"

mkdir -p /etc/apt/keyrings
print_success "Created /etc/apt/keyrings directory"

# Step 3: Download and add ROCm GPG key
print_info "Downloading ROCm GPG key..."
if wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
   gpg --dearmor -o /etc/apt/keyrings/rocm.gpg; then
    print_success "ROCm GPG key added"
else
    print_error "Failed to download ROCm GPG key"
    exit 1
fi

# Step 4: Add ROCm repository (Official AMD Instructions)
print_info "Adding ROCm 7.2 and AMDGPU 7.0.3 repositories..."
# ROCm 7.2 is available, but AMDGPU only goes up to 7.0.3
# This combination is compatible according to AMD documentation
cat > /etc/apt/sources.list.d/rocm.list << 'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/7.0.3/ubuntu noble main
EOF

if [ -f /etc/apt/sources.list.d/rocm.list ]; then
    print_success "ROCm repositories added to /etc/apt/sources.list.d/rocm.list"
else
    print_error "Failed to add ROCm repository"
    exit 1
fi

# Set repository priority to prefer ROCm packages
print_info "Setting repository priority..."
cat > /etc/apt/preferences.d/rocm-pin-600 << 'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF
print_success "Repository priority configured"

# Step 5: Update package lists
print_header "Step 3: Updating Package Lists"
print_info "Running apt update (this may take a moment)..."

if apt update > /tmp/rocm_apt_update.log 2>&1; then
    print_success "Package lists updated"
else
    print_error "Failed to update package lists"
    echo "Check /tmp/rocm_apt_update.log for details"
    exit 1
fi

# Step 6: Install ROCm packages
print_header "Step 4: Installing ROCm"
print_warning "This will download ~2-3 GB of packages. It may take 10-20 minutes."
read -p "Continue with installation? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_info "Installation cancelled by user"
    exit 0
fi

print_info "Installing ROCm (official AMD package)..."
print_info "Progress will be shown below:"
echo ""

if apt install -y rocm; then
    print_success "ROCm installed successfully"
else
    print_error "Failed to install ROCm"
    print_info "You can try installing manually with:"
    echo "  sudo apt install rocm"
    exit 1
fi

# Step 7: Add user to required groups
print_header "Step 5: Configuring User Groups"

# Check current groups
CURRENT_GROUPS=$(groups $ACTUAL_USER)
print_info "Current groups for $ACTUAL_USER: $CURRENT_GROUPS"

# Add to video and render groups (required for GPU access)
GROUPS_TO_ADD=""
if ! groups $ACTUAL_USER | grep -q "\bvideo\b"; then
    GROUPS_TO_ADD="video"
fi

if ! groups $ACTUAL_USER | grep -q "\brender\b"; then
    if [ -n "$GROUPS_TO_ADD" ]; then
        GROUPS_TO_ADD="$GROUPS_TO_ADD,render"
    else
        GROUPS_TO_ADD="render"
    fi
fi

if [ -n "$GROUPS_TO_ADD" ]; then
    usermod -a -G "$GROUPS_TO_ADD" $ACTUAL_USER
    print_success "Added $ACTUAL_USER to groups: $GROUPS_TO_ADD"
else
    print_info "User already in 'video' and 'render' groups"
fi

# Step 8: Verify ROCm installation
print_header "Step 6: Verifying ROCm Installation"

if [ -f /opt/rocm/bin/rocminfo ]; then
    print_success "ROCm tools installed at /opt/rocm"

    print_info "Checking for AMD Radeon 890M (gfx1150)..."
    if /opt/rocm/bin/rocminfo | grep -q "gfx1150"; then
        print_success "AMD Radeon 890M (gfx1150) detected by ROCm!"
    else
        print_warning "gfx1150 not detected yet (may require reboot)"
        print_info "Available GPU architectures:"
        /opt/rocm/bin/rocminfo | grep "gfx" | head -n 5 || echo "  (none detected yet)"
    fi
else
    print_error "ROCm tools not found at /opt/rocm/bin/rocminfo"
fi

# Step 9: Create environment setup
print_header "Step 7: Creating Environment Configuration"

# Add ROCm to PATH in user's bashrc if not already there
USER_HOME=$(eval echo ~$ACTUAL_USER)
BASHRC="$USER_HOME/.bashrc"

if ! grep -q "/opt/rocm/bin" "$BASHRC" 2>/dev/null; then
    print_info "Adding ROCm to PATH in $BASHRC..."
    cat >> "$BASHRC" << 'EOF'

# ROCm environment
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
EOF
    chown $ACTUAL_USER:$ACTUAL_USER "$BASHRC"
    print_success "ROCm paths added to .bashrc"
else
    print_info "ROCm already in PATH"
fi

# Step 10: Installation summary
print_header "Installation Complete!"

echo -e "${GREEN}✓ ROCm 7.2 has been installed successfully${NC}\n"

print_warning "IMPORTANT: You must REBOOT or LOG OUT and LOG IN for changes to take effect!"
echo ""
echo "Group changes require a new login session."
echo ""

print_header "Next Steps"

echo "1. Reboot your system:"
echo -e "   ${YELLOW}sudo reboot${NC}"
echo ""
echo "2. After reboot, verify GPU detection:"
echo -e "   ${YELLOW}cd /home/administrator/translator${NC}"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo -e "   ${YELLOW}python scripts/test_gpu.py${NC}"
echo ""
echo "3. If all tests pass, continue with Phase 2 of the implementation"
echo ""

print_info "Installation log saved to /tmp/rocm_install.log"

# Save installation info
{
    echo "ROCm Installation Summary"
    echo "========================="
    echo "Date: $(date)"
    echo "User: $ACTUAL_USER"
    echo "System: $(lsb_release -ds)"
    echo "Kernel: $(uname -r)"
    echo ""
    echo "Installed packages:"
    dpkg -l | grep rocm | awk '{print $2, $3}'
    echo ""
    echo "GPU Info:"
    lspci | grep -i "VGA.*AMD\|Display.*AMD" || echo "No AMD GPU detected"
    echo ""
    echo "User groups:"
    groups $ACTUAL_USER
} > /tmp/rocm_install.log

print_success "Installation details saved to /tmp/rocm_install.log"

echo ""
print_header "Installation Script Finished"
