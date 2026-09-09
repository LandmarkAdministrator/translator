#!/usr/bin/env bash
# Copy artwork/ (gitignored collateral: logo, QR code, cards) to the Synology
# "Repo" share. Additive — nothing is deleted on the NAS if it disappears here.
#   scripts/backup_artwork.sh            # laptop: /mnt/synology/repo/translator/artwork/
#   ARTWORK_BACKUP=/path scripts/backup_artwork.sh
set -eu
SRC="$(cd "$(dirname "$0")/.." && pwd)/artwork/"
DST="${ARTWORK_BACKUP:-/mnt/synology/repo/translator/artwork/}"
[ -d "$SRC" ] || { echo "no artwork/ folder at $SRC"; exit 0; }
ls /mnt/synology/repo >/dev/null 2>&1 || true      # trigger the automount
mountpoint -q /mnt/synology/repo || { echo "Synology Repo share is not mounted at /mnt/synology/repo — nothing copied"; exit 1; }
mkdir -p "$DST"
# -rt: recurse, keep times; CIFS cannot take owners/permissions (-a would warn).
rsync -rt --exclude '.~lock*' "$SRC" "$DST"
echo "artwork mirrored to $DST ($(du -sh "$DST" | cut -f1), $(find "$DST" -type f | wc -l) files)"
