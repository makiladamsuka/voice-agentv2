"""
OLED/TFT Emotion Display - Dual SPI Eyes (BlockyEye)
Drives two ST7735 displays via SPI0 and SPI1.
"""

import sys
import time
import threading
import atexit
import board
import busio
import digitalio
from PIL import Image

# Import procedural eyes engine
from procedural_eyes import ProceduralEyeDisplay, EMOTION_PRESETS

# --- Try to import display drivers ---
try:
    from adafruit_rgb_display import st7735
    DISPLAY_AVAILABLE = True
    print("✅ adafruit-circuitpython-rgb-display available")
except ImportError:
    DISPLAY_AVAILABLE = False
    print("⚠️ adafruit-circuitpython-rgb-display not found (Headless Mode)")

# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS

# --- Global State ---
_eye_display = None
_display_thread = None
disp_l = None
disp_r = None
DISPLAY_RUNNING = False
_stop_event = threading.Event()
current_emotion = "idle"

def setup_and_start_display():
    global _eye_display, _display_thread, disp_l, disp_r, DISPLAY_RUNNING

    if _display_thread and _display_thread.is_alive():
        print("⚠️ Display thread already running")
        return

    print("🖥️ Initializing Dual SPI Displays...")

    if DISPLAY_AVAILABLE:
        try:
            # SPI 0 (Left Screen)
            spi0 = board.SPI()
            disp_l = st7735.ST7735R(
                spi0, 
                rotation=0, 
                baudrate=24000000, 
                bgr=True,
                cs=digitalio.DigitalInOut(board.CE1),   
                dc=digitalio.DigitalInOut(board.D24),   
                rst=digitalio.DigitalInOut(board.D25)
            )
            print("✅ Left Display (SPI0) Initialized")

            # SPI 1 (Right Screen)
            spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
            disp_r = st7735.ST7735R(
                spi1, 
                rotation=0, 
                baudrate=24000000, 
                bgr=True,
                cs=digitalio.DigitalInOut(board.D18),   
                dc=digitalio.DigitalInOut(board.D23),   
                rst=digitalio.DigitalInOut(board.D27)
            )
            print("✅ Right Display (SPI1) Initialized")
            
            # Clear screens
            black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
            disp_l.image(black)
            disp_r.image(black)
            
        except Exception as e:
            print(f"❌ Error initializing SPI displays: {e}")
            disp_l = None
            disp_r = None
    else:
        print("⚠️ Running in Headless Mode (No hardware)")

    # Initialize Engine
    _eye_display = ProceduralEyeDisplay()
    _eye_display.set_emotion("idle")

    # Start Thread
    DISPLAY_RUNNING = True
    _stop_event.clear()
    _display_thread = threading.Thread(target=_display_loop, daemon=True)
    _display_thread.start()
    print("👀 Display thread started")

def stop_display():
    global DISPLAY_RUNNING
    print("🛑 Stopping display thread...")
    DISPLAY_RUNNING = False
    _stop_event.set()
    if _display_thread:
        _display_thread.join(timeout=2.0)
    
    # Clear screens on exit
    if disp_l and disp_r:
        try:
            black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
            disp_l.image(black)
            disp_r.image(black)
        except:
            pass
    print("✅ Display stopped")

def start_emotion(emotion_name):
    global current_emotion
    current_emotion = emotion_name
    if _eye_display:
        _eye_display.set_emotion(emotion_name)

def stop_emotion():
    start_emotion("idle")

def display_emotion(emotion_name):
    # Alias for start_emotion (legacy compatibility)
    start_emotion(emotion_name)

def update_face_target(x, y):
    """
    Update target for face tracking.
    x, y: normalized coordinates (-1.0 to 1.0)
    """
    if _eye_display:
        _eye_display.set_face_target(x, y)

def _display_loop():
    last_frame = time.time()
    
    while DISPLAY_RUNNING and not _stop_event.is_set():
        now = time.time()
        dt = now - last_frame
        last_frame = now

        if _eye_display:
            # Render returns tuple (left_img, right_img)
            frames = _eye_display.render_frame(dt)
            
            # Check if we got a tuple (Dual Display) or single image (Fallback/Mono)
            if isinstance(frames, tuple):
                img_l, img_r = frames
            else:
                img_l = frames
                img_r = frames 

            # Send to hardware
            if disp_l:
                disp_l.image(img_l)
            if disp_r:
                disp_r.image(img_r)

        # FPS Lock
        elapsed = time.time() - now
        rest = FRAME_DELAY - elapsed
        if rest > 0:
            time.sleep(rest)

# Cleanup on exit
atexit.register(stop_display)
