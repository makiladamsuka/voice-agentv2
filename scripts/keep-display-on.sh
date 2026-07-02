#!/usr/bin/env bash
# Keep the display awake for kiosk use (X11 and Wayland).

if command -v xset >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  xset s off 2>/dev/null || true
  xset -dpms 2>/dev/null || true
  xset s noblank 2>/dev/null || true
fi

if [ -w /sys/module/kernel/parameters/consoleblank ]; then
  echo 0 > /sys/module/kernel/parameters/consoleblank 2>/dev/null || true
fi
