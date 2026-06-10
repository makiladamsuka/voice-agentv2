"""
test_servos.py
Quick interactive test for the two head servos (pan + tilt) via PCA9685.

Run:
    python3 test_servos.py

Controls:
    a / d   → pan  left / right   (channel 0)
    w / s   → tilt up   / down    (channel 1)
    c       → centre both servos  (90°)
    n       → nod gesture
    k       → shake gesture
    q       → quit
"""

import time
import sys

# ── Hardware import ────────────────────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)
    print("✅ PCA9685 / ServoKit initialised.")
except Exception as e:
    print(f"❌ Could not initialise ServoKit: {e}")
    print("   Make sure I2C is enabled and adafruit-circuitpython-servokit is installed.")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────
PAN_CH   = 0
TILT_CH  = 1

PAN_MIN,  PAN_MAX  = 30,  150
TILT_MIN, TILT_MAX = 60,  120
STEP = 5   # degrees per key-press

# ── State ──────────────────────────────────────────────────────────────────
pan  = 90.0
tilt = 90.0

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def set_pan(angle):
    global pan
    pan = clamp(angle, PAN_MIN, PAN_MAX)
    kit.servo[PAN_CH].angle = pan
    print(f"  Pan={pan:.0f}°   Tilt={tilt:.0f}°", end="\r")

def set_tilt(angle):
    global tilt
    tilt = clamp(angle, TILT_MIN, TILT_MAX)
    kit.servo[TILT_CH].angle = tilt
    print(f"  Pan={pan:.0f}°   Tilt={tilt:.0f}°", end="\r")

def centre():
    set_pan(90)
    set_tilt(90)
    print("\n🎯 Centred both servos.")

def sweep(channel, lo, hi, steps=30, delay=0.03):
    """Slow sweep from lo to hi and back."""
    for angle in list(range(lo, hi+1, 2)) + list(range(hi, lo-1, -2)):
        kit.servo[channel].angle = angle
        time.sleep(delay)

def nod():
    """Simple nod: tilt down then back."""
    print("\n🤖 Nodding...")
    for _ in range(2):
        set_tilt(tilt + 10); time.sleep(0.08)
        set_tilt(tilt - 10); time.sleep(0.08)
    set_tilt(90)

def shake():
    """Quick head shake."""
    print("\n🤖 Shaking...")
    orig = pan
    set_pan(orig - 10); time.sleep(0.08)
    set_pan(orig + 10); time.sleep(0.08)
    set_pan(orig - 10); time.sleep(0.08)
    set_pan(orig);      time.sleep(0.08)

# ── Startup sweep ──────────────────────────────────────────────────────────
print("\n🔧 Servo Test Script")
print("   Press  a/d = pan,  w/s = tilt,  c = centre,  n = nod,  k = shake,  q = quit\n")

print("   Running startup sweep on PAN servo...")
sweep(PAN_CH, PAN_MIN, PAN_MAX)
centre()
time.sleep(0.3)

print("   Running startup sweep on TILT servo...")
sweep(TILT_CH, TILT_MIN, TILT_MAX)
centre()
time.sleep(0.3)

print("✅ Startup sweep done. Ready for key input.\n")

# ── Key input loop ─────────────────────────────────────────────────────────
try:
    import tty, termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)

    while True:
        ch = sys.stdin.read(1)

        if ch == 'q':
            break
        elif ch == 'a':
            set_pan(pan - STEP)
        elif ch == 'd':
            set_pan(pan + STEP)
        elif ch == 'w':
            set_tilt(tilt - STEP)    # smaller angle = up (adjust sign if needed)
        elif ch == 's':
            set_tilt(tilt + STEP)
        elif ch == 'c':
            centre()
        elif ch == 'n':
            nod()
        elif ch == 'k':
            shake()

    termios.tcsetattr(fd, termios.TCSADRAIN, old)

except ImportError:
    # Fallback for Windows or restricted envs
    termios = None
    while True:
        cmd = input("cmd (a/d/w/s/c/n/k/q): ").strip().lower()
        if cmd == 'q':    break
        elif cmd == 'a':  set_pan(pan - STEP)
        elif cmd == 'd':  set_pan(pan + STEP)
        elif cmd == 'w':  set_tilt(tilt - STEP)
        elif cmd == 's':  set_tilt(tilt + STEP)
        elif cmd == 'c':  centre()
        elif cmd == 'n':  nod()
        elif cmd == 'k':  shake()

finally:
    centre()
    print("\n👋 Servos centred. Goodbye.")
