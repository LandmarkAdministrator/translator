#!/bin/bash
# Renew the page's TLS certificate from the internal CA (lego, HTTP-01).
#
# Installed to /usr/local/sbin/translate-cert-renew.sh and run by the system
# unit translate-cert-renew.timer (systemd/). Reads /etc/lego/translate-cert.env:
#   ACME_HOST / ACME_PORT / ACME_PATH   the CA's directory endpoint
#   EMAIL                               registration contact
#   DOMAIN                              the name on the certificate
# and trusts the CA's root at /etc/lego/lbc-root-ca.crt.
#
# Certificates are six-day. `renew --days 3` is a no-op until three days
# remain, so the timer can run daily without churning; when it does renew,
# the hook (translate-cert-install.sh) copies the files and restarts the web
# unit — which is why the timer runs at 03:20 and not in a service window.
set -euo pipefail
. /etc/lego/translate-cert.env
export LEGO_CA_CERTIFICATES=/etc/lego/lbc-root-ca.crt
SERVER="https://${ACME_HOST}:${ACME_PORT}${ACME_PATH}"
LEGO="/usr/local/bin/lego --server $SERVER --email $EMAIL --domains $DOMAIN --http --accept-tos --path /etc/lego"
if [ -f "/etc/lego/certificates/${DOMAIN}.crt" ]; then
  $LEGO renew --days 3 --renew-hook /usr/local/sbin/translate-cert-install.sh
else
  # No cert yet — the CA was unreachable at setup time. Keep trying; the first
  # success installs it and the branch above takes over from then on.
  logger -t translate-cert "no certificate yet, attempting initial issuance via $SERVER"
  $LEGO run && /usr/local/sbin/translate-cert-install.sh
fi
