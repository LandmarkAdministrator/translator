#!/bin/bash
#
# Install/manage the Church Audio Translator systemd service
#
# Supports both user-level and system-level services.
# User services are recommended for most installations.
#
# Usage:
#   ./install_service.sh install    - Install user service
#   ./install_service.sh --system   - Install system-wide service (requires sudo)
#   ./install_service.sh remove     - Remove the service
#   ./install_service.sh start      - Start the service
#   ./install_service.sh stop       - Stop the service
#   ./install_service.sh status     - Check service status
#   ./install_service.sh logs       - View service logs
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
SERVICE_NAME="church-translator"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(dirname "$SCRIPT_DIR")"

# Verify installation
if [[ ! -f "$INSTALL_DIR/run.py" ]]; then
    error "Cannot find run.py in $INSTALL_DIR"
    exit 1
fi

# Default to user service
USE_SYSTEM_SERVICE=false

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --system)
                USE_SYSTEM_SERVICE=true
                shift
                ;;
            --user)
                USE_SYSTEM_SERVICE=false
                shift
                ;;
            *)
                COMMAND="$1"
                shift
                ;;
        esac
    done
}

# Get service file path based on mode
get_service_file() {
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        echo "/etc/systemd/system/${SERVICE_NAME}.service"
    else
        echo "$HOME/.config/systemd/user/${SERVICE_NAME}.service"
    fi
}

# Generate service file content
generate_service_file() {
    local gpu_env=""

    # Detect ROCm environment and read HSA_OVERRIDE from .env.rocm if available
    if [[ -f "$INSTALL_DIR/.env.rocm" ]] || [[ -d /opt/rocm ]]; then
        local hsa_override=""
        if [[ -f "$INSTALL_DIR/.env.rocm" ]]; then
            hsa_override=$(grep "HSA_OVERRIDE_GFX_VERSION" "$INSTALL_DIR/.env.rocm" 2>/dev/null | cut -d= -f2 || echo "")
        fi
        # Default to 11.0.0 for RDNA3+ if not specified
        hsa_override="${hsa_override:-11.0.0}"

        gpu_env="Environment=\"HSA_OVERRIDE_GFX_VERSION=$hsa_override\"
Environment=\"PATH=/opt/rocm/bin:\$PATH\""
    fi

    # Determine user for system service
    local user_section=""
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        user_section="User=$USER
Group=$USER"
    fi

    cat << EOF
[Unit]
Description=Church Audio Translator - Real-time speech translation
Documentation=https://github.com/your-org/church-translator
After=pipewire.service pulseaudio.service sound.target
Wants=pipewire.service

[Service]
Type=simple
$user_section
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/run.py
ExecStop=/bin/kill -SIGINT \$MAINPID
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s
StandardOutput=journal
StandardError=journal
Environment="PYTHONUNBUFFERED=1"
$gpu_env

# Resource limits
MemoryMax=8G
CPUQuota=80%

[Install]
WantedBy=default.target
EOF
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
  install   - Install and enable the service
  remove    - Stop and remove the service
  start     - Start the service
  stop      - Stop the service
  restart   - Restart the service
  status    - Check service status
  logs      - View service logs (follow mode)

Options:
  --user    - Use user-level service (default, no sudo required)
  --system  - Use system-level service (requires sudo)

Examples:
  $0 install              # Install user service
  $0 --system install     # Install system service (needs sudo)
  $0 logs                 # View logs

EOF
}

check_root() {
    if [[ "$USE_SYSTEM_SERVICE" == "true" && "$EUID" -ne 0 ]]; then
        error "System service requires sudo. Run: sudo $0 --system $COMMAND"
        exit 1
    fi
}

systemctl_cmd() {
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        sudo systemctl "$@"
    else
        systemctl --user "$@"
    fi
}

journalctl_cmd() {
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        sudo journalctl -u "$SERVICE_NAME" "$@"
    else
        journalctl --user -u "$SERVICE_NAME" "$@"
    fi
}

install_service() {
    check_root

    local service_file=$(get_service_file)
    local service_dir=$(dirname "$service_file")

    log "Installing ${SERVICE_NAME} service..."
    log "Service file: $service_file"

    # Create directory if needed
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        sudo mkdir -p "$service_dir"
    else
        mkdir -p "$service_dir"
    fi

    # Generate and write service file
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        generate_service_file | sudo tee "$service_file" > /dev/null
    else
        generate_service_file > "$service_file"
    fi

    # Reload systemd
    systemctl_cmd daemon-reload

    # Enable service
    systemctl_cmd enable "$SERVICE_NAME"

    # Enable lingering for user services (start at boot without login)
    if [[ "$USE_SYSTEM_SERVICE" == "false" ]]; then
        if command -v loginctl &> /dev/null; then
            log "Enabling lingering for $USER (service starts at boot)..."
            sudo loginctl enable-linger "$USER" 2>/dev/null || \
                warn "Could not enable lingering. Service may only start after login."
        fi
    fi

    log "Service installed and enabled."
    echo ""
    echo "Commands:"
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        echo "  sudo systemctl start $SERVICE_NAME    # Start"
        echo "  sudo systemctl stop $SERVICE_NAME     # Stop"
        echo "  sudo systemctl status $SERVICE_NAME   # Status"
        echo "  sudo journalctl -u $SERVICE_NAME -f   # Logs"
    else
        echo "  systemctl --user start $SERVICE_NAME    # Start"
        echo "  systemctl --user stop $SERVICE_NAME     # Stop"
        echo "  systemctl --user status $SERVICE_NAME   # Status"
        echo "  journalctl --user -u $SERVICE_NAME -f   # Logs"
    fi
}

remove_service() {
    check_root

    local service_file=$(get_service_file)

    log "Removing ${SERVICE_NAME} service..."

    # Stop if running
    systemctl_cmd stop "$SERVICE_NAME" 2>/dev/null || true

    # Disable
    systemctl_cmd disable "$SERVICE_NAME" 2>/dev/null || true

    # Remove service file
    if [[ "$USE_SYSTEM_SERVICE" == "true" ]]; then
        sudo rm -f "$service_file"
    else
        rm -f "$service_file"
    fi

    # Reload systemd
    systemctl_cmd daemon-reload

    log "Service removed."
}

start_service() {
    check_root
    systemctl_cmd start "$SERVICE_NAME"
    log "Service started."
    systemctl_cmd status "$SERVICE_NAME" --no-pager
}

stop_service() {
    check_root
    systemctl_cmd stop "$SERVICE_NAME"
    log "Service stopped."
}

restart_service() {
    check_root
    systemctl_cmd restart "$SERVICE_NAME"
    log "Service restarted."
    systemctl_cmd status "$SERVICE_NAME" --no-pager
}

show_status() {
    systemctl_cmd status "$SERVICE_NAME" --no-pager 2>/dev/null || \
        warn "Service not found or not running."
}

show_logs() {
    journalctl_cmd -f
}

# Main
parse_args "$@"

case "${COMMAND:-}" in
    install)
        install_service
        ;;
    remove|uninstall)
        remove_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
