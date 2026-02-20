#!/usr/bin/env python3
"""
Eye Debug Tool - Press keys to cycle emotions in real time.

Keys:
  i = idle (normal)
  l = idle2 (listening - wide eyes)
  h = happy (squint-smile)
  t = thinking (squint-think)
  s = sad
  a = angry
  u = surprised
  n = suspicious
  z = sleepy
  q = quit

Run with: python debug_eyes.py
"""

import sys
import time
import threading
import termios
import tty

# ---- Start the display ----
import oled_display
print("🖥️  Starting display...")
oled_display.setup_and_start_display()
time.sleep(0.5)

KEYMAP = {
    'i': 'idle',
    'l': 'idle2',
    'h': 'happy',
    't': 'thinking',
    's': 'sad',
    'a': 'angry',
    'u': 'surprised',
    'n': 'suspicious',
    'z': 'sleepy',
}

def print_menu(current):
    print("\n" + "─" * 42)
    print(f"  Current emotion: >>> {current.upper()} <<<")
    print("─" * 42)
    for key, name in KEYMAP.items():
        marker = " ◀" if name == current else ""
        print(f"  [{key}] {name}{marker}")
    print("  [q] quit")
    print("─" * 42)
    print("  Press a key: ", end='', flush=True)

def get_key():
    """Read a single keypress without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# Optionally simulate speech amplitude on 'happy'
def pulse_speech_demo(stop_event):
    """Gives a speech amplitude pulse when in happy mode for demo."""
    import math
    t = 0
    while not stop_event.is_set():
        if oled_display.current_emotion == 'happy':
            amp = max(0, math.sin(t * 5.0)) * 0.8
            oled_display.set_speech_amplitude(amp)
            t += 0.05
        else:
            oled_display.set_speech_amplitude(0.0)
        time.sleep(0.05)

stop_event = threading.Event()
pulse_thread = threading.Thread(target=pulse_speech_demo, args=(stop_event,), daemon=True)
pulse_thread.start()

current = 'idle'
oled_display.start_emotion(current)
print_menu(current)

try:
    while True:
        key = get_key().lower()

        if key == 'q':
            print("\n\n👋 Quitting...")
            break

        if key in KEYMAP:
            current = KEYMAP[key]
            oled_display.start_emotion(current)
            print_menu(current)
        else:
            print(f"\n  (Unknown key: {repr(key)})", end='', flush=True)

finally:
    stop_event.set()
    oled_display.stop_display()
    print("✅ Display stopped.")
