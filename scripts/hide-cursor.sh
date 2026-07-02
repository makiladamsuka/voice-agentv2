#!/usr/bin/env bash
# Hide the mouse pointer on labwc (Wayland). Re-hides after brief idle.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABWC_RC="${HOME}/.config/labwc/rc.xml"

ensure_labwc_keybind() {
  mkdir -p "${HOME}/.config/labwc"

  if [ ! -s "$LABWC_RC" ] || ! grep -q '</keyboard>' "$LABWC_RC" 2>/dev/null; then
    cp /etc/xdg/labwc/rc.xml "$LABWC_RC"
  fi

  if grep -q 'HideCursor' "$LABWC_RC"; then
    return 0
  fi

  python3 - "$LABWC_RC" <<'PY'
import sys
from pathlib import Path

rc_path = Path(sys.argv[1])
text = rc_path.read_text()
if "HideCursor" in text:
    sys.exit(0)

keybind = """
    <!-- voice-agent HideCursor keybind -->
    <keybind key="A-W-h">
      <action name="HideCursor" />
      <action name="WarpCursor" x="-1" y="-1" />
    </keybind>
"""

if "</keyboard>" not in text:
    raise SystemExit("labwc rc.xml has no </keyboard> section")

text = text.replace("</keyboard>", f"{keybind}  </keyboard>", 1)
rc_path.write_text(text)
PY
}

hide_cursor() {
  if command -v wtype >/dev/null 2>&1; then
    wtype -M alt -M logo -P h -m alt -m logo 2>/dev/null || true
  fi
}

start_idle_hider() {
  if ! command -v swayidle >/dev/null 2>&1 || ! command -v wtype >/dev/null 2>&1; then
    return 0
  fi

  pkill -f 'swayidle.*voice-agent-kiosk-cursor' 2>/dev/null || true
  swayidle -w \
    timeout 1 'wtype -M alt -M logo -P h -m alt -m logo' \
    resume 'true' &
  echo $! > /tmp/voice-agent-kiosk-cursor-swayidle.pid
}

ensure_labwc_keybind

# Give the compositor time to finish starting.
sleep 2
hide_cursor
start_idle_hider
