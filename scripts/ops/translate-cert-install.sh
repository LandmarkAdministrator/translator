#!/bin/bash
# lego renew-hook: copy a freshly issued certificate where translate-web reads
# it and restart that unit so it is served. Installed to /usr/local/sbin.
#
# Site-specific values: the service account (administrator, uid 1000) and the
# certificate name, which is the host's address because the internal CA issues
# for IPs. A new deployment edits these three lines.
set -euo pipefail
trap 'logger -t translate-cert "ERROR: cert install hook failed at line $LINENO"' ERR
SVC_USER=administrator
SVC_UID=1000
CERT_NAME=10.1.170.184

SRC=/etc/lego/certificates
DST=/etc/translate-tls
install -d -m 750 -o "$SVC_USER" -g "$SVC_USER" "$DST"
install -m 644 -o "$SVC_USER" -g "$SVC_USER" "$SRC/$CERT_NAME.crt" "$DST/tls.crt"
install -m 600 -o "$SVC_USER" -g "$SVC_USER" "$SRC/$CERT_NAME.key" "$DST/tls.key"
logger -t translate-cert "installed cert, restarting app"
# Root restarting a --user unit needs that user's session bus.
su - "$SVC_USER" -c "XDG_RUNTIME_DIR=/run/user/$SVC_UID systemctl --user restart translate-web.service" \
  || logger -t translate-cert "WARNING: could not restart translate-web.service"
