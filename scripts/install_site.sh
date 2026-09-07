#!/usr/bin/env bash
# scripts/install_site.sh — the site layer: everything a serving machine needs
# that ./install.sh (OS packages, GPU drivers, the main venv, base models)
# does not. Idempotent: run it again after a `git pull` and it changes only
# what differs, reporting each step as ok / changed / skipped / FAILED.
#
#   ./scripts/install_site.sh                 # interactive (asks for the admin password if unset)
#   ./scripts/install_site.sh --check         # report only; change nothing
#   ./scripts/install_site.sh --yes           # non-interactive; skips the password prompt
#   ./scripts/install_site.sh --prefetch      # also download every model now (~8 GB)
#   ./scripts/install_site.sh --web-host 0.0.0.0 --trusted-proxies 10.0.0.5
#   ./scripts/install_site.sh --tls https://ca.example/acme/directory --ca-cert root.crt \
#                             --domain translate.example.org --email admin@example.org
#   ./scripts/install_site.sh --no-thermal-guard
#
# What it sets up, in order:
#   1  preflight   repo at ~/translator, main venv, GPU, user systemd, linger
#   2  NeMo venv   ~/nemo-venv (Python 3.11 via uv) from requirements-nemo.txt
#   3  config      settings.yaml default if missing; reminders for site.json/schedule.conf
#   4  admin       ~/.config/translator/admin.json (scrypt), prompted unless --yes
#   5  scripts     scheduler, launcher, thermal guard into ~/bin; log directories
#   6  units       translate, translate-web, window timer, tally timer, thermal guard
#   7  TLS         optional: lego + internal CA, system timer at 03:20 (sudo)
#   8  prefetch    optional: NLLB, the voices, the streaming ASR model
#   9  verify      tests/test_pipeline_config.py, unit states, next steps
#
# Site-specific values live in config/site.json (church name, service times,
# languages), config/schedule.conf (windows) and config/settings.yaml (audio
# devices) — all editable afterwards from the admin panel or by hand.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOME_REPO="$HOME/translator"
UNITS="$HOME/.config/systemd/user"
CHECK=0; YES=0; PREFETCH=0; THERMAL=1
WEB_HOST=""; PROXIES=""
TLS_URL=""; CA_CERT=""; DOMAIN=""; EMAIL=""
FAILS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --yes|-y) YES=1 ;;
    --prefetch) PREFETCH=1 ;;
    --no-thermal-guard) THERMAL=0 ;;
    --web-host) WEB_HOST="$2"; shift ;;
    --trusted-proxies) PROXIES="$2"; shift ;;
    --tls) TLS_URL="$2"; shift ;;
    --ca-cert) CA_CERT="$2"; shift ;;
    --domain) DOMAIN="$2"; shift ;;
    --email) EMAIL="$2"; shift ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# ---- reporting ------------------------------------------------------------
say()  { printf '  %-9s %s\n' "$1" "$2"; }
ok()   { say "ok" "$1"; }
chg()  { say "changed" "$1"; }
skip() { say "skipped" "$1"; }
fail() { say "FAILED" "$1"; FAILS=$((FAILS + 1)); }
step() { printf '\n== %s\n' "$1"; }

# ---- 1. preflight ---------------------------------------------------------
step "1. preflight"
if [ "$REPO" != "$HOME_REPO" ]; then
  if [ -e "$HOME_REPO" ]; then
    if [ "$(readlink -f "$HOME_REPO")" = "$REPO" ]; then ok "$HOME_REPO -> $REPO"
    else fail "$HOME_REPO exists and is not this checkout ($REPO); the units expect the repo there"; fi
  elif [ "$CHECK" = 1 ]; then say "would" "ln -s $REPO $HOME_REPO (the units use %h/translator)"
  else ln -s "$REPO" "$HOME_REPO" && chg "$HOME_REPO -> $REPO (symlink; the units use %h/translator)"; fi
else ok "repo at $HOME_REPO"; fi

if [ -x "$REPO/venv/bin/python" ]; then ok "main venv ($("$REPO/venv/bin/python" --version 2>&1))"
else fail "no main venv at $REPO/venv — run ./install.sh first (drivers, venv, base models)"; fi

if command -v nvidia-smi >/dev/null 2>&1; then ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
elif command -v rocminfo >/dev/null 2>&1; then ok "GPU: ROCm ($(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*:\s*//'))"; THERMAL=0
else fail "no GPU tooling found (nvidia-smi / rocminfo); the pipeline refuses to start without a GPU"; fi

if systemctl --user show-environment >/dev/null 2>&1; then ok "user systemd reachable"
else fail "systemctl --user does not work here — log in as the service user over ssh or a console (needs a session bus)"; fi

if command -v loginctl >/dev/null 2>&1; then
  if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then ok "linger enabled (user units start at boot)"
  elif [ "$CHECK" = 1 ]; then say "would" "sudo loginctl enable-linger $USER"
  elif sudo -n true 2>/dev/null || [ -t 0 ]; then sudo loginctl enable-linger "$USER" && chg "linger enabled for $USER" || fail "could not enable linger (sudo loginctl enable-linger $USER)"
  else fail "linger is off and sudo is unavailable: run  sudo loginctl enable-linger $USER"; fi
fi

# ---- 2. NeMo venv ---------------------------------------------------------
step "2. NeMo venv (streaming ASR, Python 3.11)"
NEMO_PY="$HOME/nemo-venv/bin/python"
UV="$REPO/venv/bin/uv"
if command -v rocminfo >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
  skip "ROCm host: requirements-nemo.txt pins the CUDA torch; create ~/nemo-venv by hand with the rocm7.2 index (see the file header)"
elif [ -x "$NEMO_PY" ] && "$NEMO_PY" -c "import nemo" >/dev/null 2>&1; then
  ok "~/nemo-venv has nemo_toolkit ($("$NEMO_PY" -c 'import nemo; print(nemo.__version__)' 2>/dev/null))"
elif [ "$CHECK" = 1 ]; then
  say "would" "install uv into the main venv, uv python install 3.11, create ~/nemo-venv, pip install -r requirements-nemo.txt (~4 GB)"
elif [ ! -x "$REPO/venv/bin/python" ]; then
  skip "needs the main venv first"
else
  [ -x "$UV" ] || "$REPO/venv/bin/pip" install -q uv || fail "could not install uv into the main venv"
  if [ -x "$UV" ]; then
    "$UV" python install 3.11 >/dev/null && \
    { [ -x "$NEMO_PY" ] || "$UV" venv --python 3.11 "$HOME/nemo-venv" >/dev/null; } && \
    "$UV" pip install --python "$NEMO_PY" -r "$REPO/requirements-nemo.txt" \
      && chg "~/nemo-venv created and populated" \
      || fail "NeMo venv install failed — rerun; the log above names the package"
  fi
fi

# ---- 3. config -------------------------------------------------------------
step "3. configuration"
if [ -f "$REPO/config/settings.yaml" ]; then ok "config/settings.yaml present"
else
  if [ "$CHECK" = 1 ]; then say "would" "write a default config/settings.yaml (Spanish left, Creole right on the default output)"
  else
    cat > "$REPO/config/settings.yaml" <<'YAML'
# Audio routing. Edit from the admin panel (/admin) or by hand; device names
# match by substring so a card renumbering after a reboot does not matter.
input_device: default
languages:
- code: es
  name: Spanish
  output_device: default
  output_channel: 0
  enabled: true
- code: ht
  name: Haitian Creole
  output_device: default
  output_channel: 1
  enabled: true
YAML
    chg "config/settings.yaml written with defaults — set the real devices from /admin"
  fi
fi
for f in site.json schedule.conf bias_phrases.txt; do
  [ -f "$REPO/config/$f" ] && ok "config/$f present (edit for this congregation)" || fail "config/$f missing from the checkout"
done

# ---- 4. admin credentials -------------------------------------------------
step "4. admin credentials"
CRED="$HOME/.config/translator/admin.json"
if [ -f "$CRED" ]; then ok "$CRED"
elif [ "$CHECK" = 1 ] || [ "$YES" = 1 ]; then skip "no admin password yet — run  ./venv/bin/python scripts/set_admin_password.py"
elif [ -t 0 ]; then "$REPO/venv/bin/python" "$REPO/scripts/set_admin_password.py" && chg "admin password set" || fail "admin password not set"
else skip "no terminal to prompt on — run  ./venv/bin/python scripts/set_admin_password.py"; fi

# ---- 5. scripts and directories ------------------------------------------
step "5. scripts into ~/bin, log directories"
mkdir -p "$HOME/bin" "$HOME/sermons/logs/service-tally"
put() {  # put SRC DST MODE
  if cmp -s "$1" "$2" 2>/dev/null; then ok "$(basename "$2") unchanged"
  elif [ "$CHECK" = 1 ]; then say "would" "install $(basename "$2")"
  else install -m "$3" "$1" "$2.new" && mv -f "$2.new" "$2" && chg "$(basename "$2")"; fi
}
put "$REPO/scripts/ops/translate-window-check.sh" "$HOME/bin/translate-window-check.sh" 755
put "$REPO/scripts/ops/start-translate-unified"   "$HOME/bin/start-translate-unified"   755
[ "$THERMAL" = 1 ] && put "$REPO/scripts/ops/gpu-thermal-guard.sh" "$HOME/bin/gpu-thermal-guard.sh" 755

# ---- 6. user units --------------------------------------------------------
step "6. systemd user units"
mkdir -p "$UNITS"
CHANGED_UNITS=0; WEB_CHANGED=0
unit() {  # unit NAME  (copies verbatim when it differs)
  if cmp -s "$REPO/systemd/$1" "$UNITS/$1" 2>/dev/null; then ok "$1 unchanged"
  elif [ "$CHECK" = 1 ]; then say "would" "install $1"
  else cp "$REPO/systemd/$1" "$UNITS/$1" && chg "$1" && CHANGED_UNITS=1; fi
}
for u in translate.service translate-window.timer translate-window.service translate-tally.timer translate-tally.service; do unit "$u"; done
[ "$THERMAL" = 1 ] && unit gpu-thermal-guard.service
# translate-web.service carries two site values; keep what is installed
# unless told otherwise, default to a proxy-less LAN-safe setting on a new box.
WEB_SRC="$REPO/systemd/translate-web.service"; WEB_DST="$UNITS/translate-web.service"
cur_host=$(grep -o '^Environment=WEB_HOST=.*' "$WEB_DST" 2>/dev/null | cut -d= -f3)
cur_prox=$(grep -o '^Environment=TRANSLATOR_TRUSTED_PROXIES=.*' "$WEB_DST" 2>/dev/null | cut -d= -f3-)
host="${WEB_HOST:-${cur_host:-127.0.0.1}}"
prox="${PROXIES:-${cur_prox:-127.0.0.1,::1}}"
tmp=$(mktemp)
sed -e "s|^Environment=WEB_HOST=.*|Environment=WEB_HOST=$host|" \
    -e "s|^Environment=TRANSLATOR_TRUSTED_PROXIES=.*|Environment=TRANSLATOR_TRUSTED_PROXIES=$prox|" "$WEB_SRC" > "$tmp"
if cmp -s "$tmp" "$WEB_DST" 2>/dev/null; then ok "translate-web.service unchanged (host $host, proxies $prox)"
elif [ "$CHECK" = 1 ]; then say "would" "install translate-web.service (WEB_HOST=$host, trusted proxies $prox)"
else cp "$tmp" "$WEB_DST" && chg "translate-web.service (WEB_HOST=$host, trusted proxies $prox)" && WEB_CHANGED=1; fi
rm -f "$tmp"
if [ "$CHECK" = 0 ]; then
  systemctl --user daemon-reload
  systemctl --user enable --now translate-web.service translate-window.timer translate-tally.timer >/dev/null 2>&1 \
    && ok "enabled: translate-web.service, translate-window.timer, translate-tally.timer" \
    || fail "could not enable the user units"
  if [ "$THERMAL" = 1 ]; then systemctl --user enable --now gpu-thermal-guard.service >/dev/null 2>&1 && ok "enabled: gpu-thermal-guard.service"; fi
  # Only the web unit's own change warrants the two-second page blip of a
  # restart; the timers and the pipeline unit are re-read at their next use.
  [ "$WEB_CHANGED" = 1 ] && systemctl --user restart translate-web.service && chg "translate-web.service restarted"
  systemctl --user disable translate.service >/dev/null 2>&1; ok "translate.service left to the window timer (not enabled at boot)"
fi

# ---- 7. TLS (optional) ----------------------------------------------------
step "7. TLS certificate from an internal CA (optional)"
if [ -z "$TLS_URL" ]; then
  if [ -f /etc/translate-tls/tls.crt ]; then ok "certificate present in /etc/translate-tls (renewal timer: $(systemctl is-enabled translate-cert-renew.timer 2>/dev/null || echo not installed))"
  else skip "no --tls given; the page serves plain HTTP on WEB_HOST:8080 (put a TLS proxy in front, or rerun with --tls)"; fi
elif [ -z "$CA_CERT" ] || [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then fail "--tls needs --ca-cert, --domain and --email"
elif ! command -v lego >/dev/null 2>&1 && [ ! -x /usr/local/bin/lego ]; then
  fail "lego not found — install it to /usr/local/bin (https://github.com/go-acme/lego/releases) and rerun"
elif [ "$CHECK" = 1 ]; then say "would" "write /etc/lego/translate-cert.env, install the CA cert, the renew scripts and the 03:20 timer, issue the first certificate"
else
  proto_host=$(printf '%s' "$TLS_URL" | sed -E 's#^https?://##; s#/.*##'); acme_host=${proto_host%%:*}; acme_port=${proto_host##*:}
  [ "$acme_port" = "$proto_host" ] && acme_port=443
  acme_path=$(printf '%s' "$TLS_URL" | sed -E 's#^https?://[^/]+##')
  sudo install -d -m 755 /etc/lego && \
  printf 'ACME_HOST=%s\nACME_PORT=%s\nACME_PATH=%s\nEMAIL=%s\nDOMAIN=%s\n' "$acme_host" "$acme_port" "$acme_path" "$EMAIL" "$DOMAIN" | sudo tee /etc/lego/translate-cert.env >/dev/null && \
  sudo install -m 644 "$CA_CERT" /etc/lego/lbc-root-ca.crt && \
  sudo install -m 755 "$REPO/scripts/ops/translate-cert-renew.sh" /usr/local/sbin/translate-cert-renew.sh && \
  sed -e "s/^SVC_USER=.*/SVC_USER=$USER/" -e "s/^SVC_UID=.*/SVC_UID=$(id -u)/" -e "s/^CERT_NAME=.*/CERT_NAME=$DOMAIN/" \
      "$REPO/scripts/ops/translate-cert-install.sh" | sudo tee /usr/local/sbin/translate-cert-install.sh >/dev/null && \
  sudo chmod 755 /usr/local/sbin/translate-cert-install.sh && \
  sudo cp "$REPO/systemd/translate-cert-renew.timer" "$REPO/systemd/translate-cert-renew.service" /etc/systemd/system/ && \
  sudo systemctl daemon-reload && sudo systemctl enable --now translate-cert-renew.timer >/dev/null && \
  chg "TLS configured; first issuance:" && sudo /usr/local/sbin/translate-cert-renew.sh && ok "certificate issued and installed" \
    || fail "TLS setup did not complete — see the messages above"
fi

# ---- 8. prefetch models (optional) ---------------------------------------
step "8. models"
if [ "$PREFETCH" = 1 ]; then
  if [ "$CHECK" = 1 ]; then say "would" "download NLLB-200 1.3B, the configured voices and parakeet-unified-en-0.6b now"
  else "$REPO/venv/bin/python" "$REPO/scripts/prefetch_models.py" && ok "models present" || fail "prefetch reported a problem"; fi
else skip "not prefetched (--prefetch): the first start downloads ~8 GB and takes several minutes"; fi

# ---- 9. verify ------------------------------------------------------------
step "9. verify"
if [ -x "$REPO/venv/bin/python" ]; then
  if "$REPO/venv/bin/python" "$REPO/tests/test_pipeline_config.py" >/tmp/install_site_check.txt 2>&1; then ok "every configured language resolves a translation model and a voice"
  else fail "tests/test_pipeline_config.py failed — see /tmp/install_site_check.txt"; fi
fi
for u in translate-web.service translate-window.timer translate-tally.timer; do
  state=$(systemctl --user is-active "$u" 2>/dev/null); say "${state:-unknown}" "$u"
done

printf '\n'
if [ "$FAILS" -gt 0 ]; then echo "$FAILS step(s) FAILED — fix and rerun (it is safe to rerun)."; exit 1; fi
[ "$CHECK" = 1 ] && { echo "check only; nothing changed."; exit 0; }
cat <<EOF
Done. Next:
  1. Sign in to the admin panel and set the audio devices and service windows:
       http://${WEB_HOST:-127.0.0.1}:8080/admin   (or https://<this host>:8443/admin with TLS)
  2. Edit config/site.json — church name, service times, the languages the page offers.
  3. Plug the mixer feed into the input and the room amplifier into the outputs.
  4. Watch a window start: tail -f ~/translate.log  (protocol 2, HEARTBEAT, [EN] lines).
EOF
