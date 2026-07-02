#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${KIOSK_URL:-http://localhost:3000}"
CHROMIUM="${CHROMIUM:-chromium-browser}"
PROFILE_DIR="${KIOSK_PROFILE_DIR:-${HOME}/.config/voice-agent-kiosk-chromium}"
MAX_WAIT=180

if pgrep -f "chromium.*--kiosk.*${URL}" >/dev/null 2>&1; then
  exit 0
fi

for _ in $(seq 1 "$MAX_WAIT"); do
  if curl -sf --max-time 2 "$URL" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if pgrep -f "chromium.*--kiosk.*${URL}" >/dev/null 2>&1; then
  exit 0
fi

mkdir -p "$PROFILE_DIR"

"${SCRIPT_DIR}/hide-cursor.sh" &

exec "$CHROMIUM" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-translate \
  --no-first-run \
  --check-for-update-interval=31536000 \
  --user-data-dir="$PROFILE_DIR" \
  "$URL"
