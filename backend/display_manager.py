"""
Display Manager - Dual SPI Eyes (BlockyEye)
Drives two ST7735 displays via SPI0 and SPI1.
"""

import sys
import time
import threading
import atexit
try:
    import board
    import busio
    import digitalio
    HARDWARE_LIBS_AVAILABLE = True
except ImportError:
    HARDWARE_LIBS_AVAILABLE = False
    print("⚠️ Hardware libraries (board, busio, digitalio) not found. Switching to Mock mode.")

from PIL import Image

# Preview support for local testing
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Import eye engine
from eye_engine import ProceduralEyeDisplay, EMOTION_PRESETS

# --- Try to import display drivers ---
try:
    from adafruit_rgb_display import st7735
    DISPLAY_AVAILABLE = True
    print("✅ adafruit-circuitpython-rgb-display available")
except ImportError:
    DISPLAY_AVAILABLE = False
    print("⚠️ adafruit-circuitpython-rgb-display not found (Headless Mode)")

# --- Configuration ---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 128
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
speech_amplitude = 0.0  # 0.0 to 1.0, driven by audio output

# Preview surface for Pygame
_preview_screen = None

def setup_and_start_display():
    global _eye_display, _display_thread, disp_l, disp_r, DISPLAY_RUNNING

    if _display_thread and _display_thread.is_alive():
        print("⚠️ Display thread already running")
        return

    print("🖥️ Initializing Dual SPI Displays...")

    if DISPLAY_AVAILABLE and HARDWARE_LIBS_AVAILABLE:
        try:
            # SPI 0 (Left Screen)
            spi0 = board.SPI()
            disp_l = st7735.ST7735R(
                spi0, 
                rotation=90, # Landscape
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
                rotation=90, # Landscape
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
        # Pygame preview initialization moved to _display_loop thread for stability

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

def start_emotion(emotion_name, duration=None, chain=None, blink_shift=False):
    global current_emotion
    current_emotion = emotion_name
    if _eye_display:
        _eye_display.set_emotion(emotion_name, duration=duration, chain=chain, blink_shift=blink_shift)

def stop_emotion():
    start_emotion("idle")

def display_emotion(emotion_name, duration=None, chain=None, blink_shift=False):
    # Alias for start_emotion (legacy compatibility)
    start_emotion(emotion_name, duration=duration, chain=chain, blink_shift=blink_shift)

def update_face_target(x, y, rotation=0.0):
    """
    Update target for face tracking.
    x, y: normalized coordinates (-1.0 to 1.0)
    rotation: face roll in degrees
    """
    if _eye_display:
        _eye_display.set_face_target(x, y, rotation)

def set_speech_amplitude(amplitude: float):
    """
    Set the current speech amplitude (0.0 to 1.0).
    Called from audio output stream to drive eye reactivity.
    """
    global speech_amplitude
    speech_amplitude = max(0.0, min(1.0, amplitude))
    if _eye_display:
        _eye_display.left_eye.speech_amplitude = speech_amplitude
        _eye_display.right_eye.speech_amplitude = speech_amplitude

def _display_loop():
    global _preview_screen
    last_frame = time.time()
    
    # Initialize Pygame Preview if needed (must be in same thread as flip/events)
    if not DISPLAY_AVAILABLE and PYGAME_AVAILABLE:
        print("📺 Initializing Pygame Preview Window (Background Thread)...")
        try:
            pygame.init()
            # Wider window for landscape: Margin(20) + Eye(160) + Gap(100) + Eye(160) + Margin(20) = 460
            _preview_screen = pygame.display.set_mode((460, SCREEN_HEIGHT + 60))
            pygame.display.set_caption("Voice Agent Eyes - Preview (Landscape)")
            print("✅ Pygame Preview Ready")
        except Exception as e:
            print(f"⚠️ Could not init pygame preview: {e}")

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

            # Update Pygame preview
            if _preview_screen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                surf_l = pygame.image.frombuffer(img_l.tobytes(), img_l.size, img_l.mode)
                surf_r = pygame.image.frombuffer(img_r.tobytes(), img_r.size, img_r.mode)
                
                _preview_screen.fill((20, 20, 20)) # Dark gray bg
                
                # Layout: [20px margin] [Eye L] [100px gap] [Eye R] [20px margin]
                _preview_screen.blit(surf_l, (20, 30))
                _preview_screen.blit(surf_r, (SCREEN_WIDTH + 120, 30))
                pygame.display.flip()

        # FPS Lock
        elapsed = time.time() - now
        rest = FRAME_DELAY - elapsed
        if rest > 0:
            time.sleep(rest)

# Cleanup on exit
atexit.register(stop_display)
