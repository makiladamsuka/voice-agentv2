"""
OLED/TFT Emotion Display - Procedural Robot Eyes

Drives dual displays with real-time procedural eye animations.
Supports:
- ST7735 color TFT (128x160) via luma.lcd + SPI
- SSD1306 monochrome OLED (128x64) via luma.oled + I2C (fallback)
- Headless mode for development/testing on laptop

Enhanced with Vector-style life behaviors:
- Saccades, micro-drift, breathing rhythm
- Natural asymmetric blinks
- Smooth emotion transitions with color blending
"""

import sys
import time
import threading
import atexit
from enum import Enum
from PIL import Image

# Import procedural eyes engine
from procedural_eyes import ProceduralEyeDisplay, EMOTION_PRESETS

# --- Try to import display drivers ---
DISPLAY_TYPE = None  # "st7735", "ssd1306", or None (headless)

# Try ST7735 (color TFT) first
try:
    from luma.core.interface.serial import spi
    from luma.lcd.device import st7735
    DISPLAY_TYPE = "st7735"
    print("✅ luma.lcd available — ST7735 color TFT mode")
except ImportError:
    pass

# Fallback to SSD1306 (monochrome OLED)
if not DISPLAY_TYPE:
    try:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        DISPLAY_TYPE = "ssd1306"
        print("✅ luma.oled available — SSD1306 monochrome OLED mode")
    except ImportError:
        print("⚠️ No display driver available — headless mode (no OLED/TFT)")


# --- Configuration ---

# ST7735 SPI config (two displays on different CE pins)
SPI_PORT = 0
LEFT_TFT_CE = 0       # CE0 = GPIO8
RIGHT_TFT_CE = 1      # CE1 = GPIO7
TFT_DC_PIN = 24       # Data/Command GPIO
TFT_RST_PIN = 25      # Reset GPIO
TFT_WIDTH = 128
TFT_HEIGHT = 160

# SSD1306 I2C config (fallback)
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d
RIGHT_OLED_ADDRESS = 0x3c

# Animation settings
DEFAULT_EMOTION = "idle1"
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS

# Available emotions (procedural — no video files needed!)
EMOTIONS = list(EMOTION_PRESETS.keys())


class EmotionMode(Enum):
    """Emotion playback modes"""
    ONE_SHOT = "one_shot"     # Play once, return to idle
    LOOPING = "looping"       # Keep looping until stopped
    SUSTAINED = "sustained"   # Keep last frame until changed


# --- Global State ---
_eye_display: ProceduralEyeDisplay = None
_display_thread = None
DEVICES = None
DISPLAY_RUNNING = False
current_emotion = DEFAULT_EMOTION
current_mode = EmotionMode.ONE_SHOT
_one_shot_duration = 2.0  # How long a one-shot emotion plays before returning to idle
_one_shot_start = 0.0
_stop_current_emotion = threading.Event()


class DummyDevice:
    """Fallback device when no display hardware is available."""
    def clear(self): pass
    def display(self, image): pass
    def hide(self): pass
    width = 128
    height = 160


# ============================================================================
# DEVICE SETUP
# ============================================================================

def _setup_st7735_device(ce_pin, name):
    """Initialize a single ST7735 TFT device."""
    try:
        serial = spi(port=SPI_PORT, device=ce_pin, gpio_DC=TFT_DC_PIN,
                     gpio_RST=TFT_RST_PIN, bus_speed_hz=32000000)
        device = st7735(serial, width=TFT_WIDTH, height=TFT_HEIGHT,
                        rotate=0, bgr=True)
        print(f"✅ {name} TFT (CE{ce_pin}) initialized — {TFT_WIDTH}x{TFT_HEIGHT} color")
        return device
    except Exception as e:
        print(f"⚠️ Could not connect to {name} TFT (CE{ce_pin}): {e}")
        return DummyDevice()


def _setup_ssd1306_device(address, name):
    """Initialize a single SSD1306 OLED device (fallback)."""
    try:
        serial = i2c(port=I2C_PORT, address=address)
        device = ssd1306(serial)
        print(f"✅ {name} OLED ({hex(address)}) initialized — 128x64 mono")
        return device
    except Exception as e:
        print(f"⚠️ Could not connect to {name} OLED ({hex(address)}): {e}")
        return DummyDevice()


# ============================================================================
# FRAME OUTPUT — Sends procedural frame to displays
# ============================================================================

def _send_frame_st7735(frame_img):
    """
    Send a 128x128 RGB frame to dual ST7735 TFT displays.
    Each eye gets its half, resized to 128x160.
    """
    global DEVICES
    if not DEVICES:
        return

    left_device, right_device = DEVICES

    try:
        # Split the 128x128 frame into left and right halves
        left_half = frame_img.crop((0, 0, 64, 128))    # Left eye
        right_half = frame_img.crop((64, 0, 128, 128))  # Right eye

        # Resize to fit TFT dimensions (64x128 -> 128x160)
        # Rotate 90 degrees so each half fills the TFT properly
        left_tft = left_half.rotate(-90, expand=True).resize(
            (TFT_WIDTH, TFT_HEIGHT), Image.BILINEAR
        )
        right_tft = right_half.rotate(90, expand=True).resize(
            (TFT_WIDTH, TFT_HEIGHT), Image.BILINEAR
        )

        # Ensure RGB mode for color displays
        if left_tft.mode != 'RGB':
            left_tft = left_tft.convert('RGB')
        if right_tft.mode != 'RGB':
            right_tft = right_tft.convert('RGB')

        left_device.display(left_tft)
        right_device.display(right_tft)

    except Exception as e:
        pass  # Silently skip frame errors


def _send_frame_ssd1306(frame_img):
    """
    Send a 128x128 frame to dual SSD1306 OLED displays.
    Converts to monochrome, splits, rotates for the OLEDs.
    """
    global DEVICES
    if not DEVICES:
        return

    left_device, right_device = DEVICES

    try:
        # Convert to monochrome
        mono = frame_img.convert('1')

        # Split
        left_half = mono.crop((0, 0, 64, 128))
        right_half = mono.crop((64, 0, 128, 128))

        # Rotate for OLED orientation
        left_oled = left_half.rotate(-90, expand=True)
        right_oled = right_half.rotate(90, expand=True)

        left_device.display(left_oled)
        right_device.display(right_oled)

    except Exception:
        pass


def _send_frame(frame_img):
    """Route frame to the correct display type."""
    if DISPLAY_TYPE == "st7735":
        _send_frame_st7735(frame_img)
    elif DISPLAY_TYPE == "ssd1306":
        _send_frame_ssd1306(frame_img)
    # Headless: do nothing


# ============================================================================
# DISPLAY THREAD
# ============================================================================

def _display_thread_function():
    """Main rendering loop — generates and displays procedural eye frames."""
    global current_emotion, current_mode, DISPLAY_RUNNING, _eye_display
    global _one_shot_start, _stop_current_emotion

    last_time = time.time()

    while DISPLAY_RUNNING:
        now = time.time()
        dt = now - last_time
        last_time = now

        # Check for one-shot timeout (return to idle after duration)
        if current_mode == EmotionMode.ONE_SHOT and current_emotion != DEFAULT_EMOTION:
            if now - _one_shot_start > _one_shot_duration:
                current_emotion = DEFAULT_EMOTION
                _eye_display.set_emotion(DEFAULT_EMOTION)
                print(f"👀 One-shot done, returning to idle")

        # Check for stop signal
        if _stop_current_emotion.is_set():
            _stop_current_emotion.clear()
            current_emotion = DEFAULT_EMOTION
            _eye_display.set_emotion(DEFAULT_EMOTION)

        # Render frame
        frame = _eye_display.render_frame(dt)

        # Send to displays
        _send_frame(frame)

        # Frame timing
        elapsed = time.time() - now
        sleep_time = FRAME_DELAY - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================================
# PUBLIC API (same interface as before — agent.py needs no changes)
# ============================================================================

def setup_and_start_display():
    """
    Initializes devices and starts the procedural display loop.
    Call this at agent startup.
    """
    global DEVICES, DISPLAY_RUNNING, _eye_display, _display_thread

    # Initialize procedural eye display
    _eye_display = ProceduralEyeDisplay()

    # Setup display devices
    if DISPLAY_TYPE == "st7735":
        left_device = _setup_st7735_device(LEFT_TFT_CE, "LEFT")
        right_device = _setup_st7735_device(RIGHT_TFT_CE, "RIGHT")
        DEVICES = (left_device, right_device)
    elif DISPLAY_TYPE == "ssd1306":
        left_device = _setup_ssd1306_device(LEFT_OLED_ADDRESS, "LEFT")
        right_device = _setup_ssd1306_device(RIGHT_OLED_ADDRESS, "RIGHT")
        DEVICES = (left_device, right_device)
    else:
        print("⚠️ No display hardware — running headless (procedural engine still active)")
        DEVICES = (DummyDevice(), DummyDevice())

    # Start display thread
    DISPLAY_RUNNING = True
    _display_thread = threading.Thread(target=_display_thread_function, daemon=True)
    _display_thread.start()
    print(f"👀 Procedural eye display started ({DESIRED_FPS} FPS, {DISPLAY_TYPE or 'headless'})")

    atexit.register(stop_display)

    return _display_thread


def display_emotion(emotion_name: str, mode: EmotionMode = EmotionMode.ONE_SHOT) -> bool:
    """
    Display an emotion. Smoothly transitions from current state.

    Args:
        emotion_name: The emotion to show
        mode: EmotionMode.ONE_SHOT or EmotionMode.LOOPING

    Returns:
        True if emotion was set successfully
    """
    global current_emotion, current_mode, _one_shot_start, _stop_current_emotion

    if not DISPLAY_RUNNING or not _eye_display:
        return False

    requested = emotion_name.strip().lower()

    # Validate emotion
    if requested not in EMOTIONS:
        print(f"⚠️ Unknown emotion '{requested}', available: {EMOTIONS}")
        return False

    # Dedup: don't restart if already playing this emotion in looping mode
    if current_emotion == requested and mode == EmotionMode.LOOPING:
        return True

    # Set the new emotion
    _stop_current_emotion.clear()
    current_emotion = requested
    current_mode = mode
    _eye_display.set_emotion(requested)

    if mode == EmotionMode.ONE_SHOT:
        _one_shot_start = time.time()

    print(f"👀 Emotion: {requested.upper()} ({mode.value})")
    return True


def start_emotion(emotion_name: str) -> bool:
    """
    Start playing an emotion in looping mode (typical for speech).
    """
    return display_emotion(emotion_name, EmotionMode.LOOPING)


def stop_emotion() -> bool:
    """
    Stop the current emotion and return to idle.
    Use this when done speaking.
    """
    global _stop_current_emotion
    print("🎬 stop_emotion() called — returning to idle")
    _stop_current_emotion.set()
    return True


def stop_display():
    """Stops the display thread and clears the screens."""
    global DISPLAY_RUNNING, DEVICES
    print("👀 Stopping display...")
    DISPLAY_RUNNING = False
    time.sleep(0.3)

    if DEVICES:
        try:
            DEVICES[0].clear()
            DEVICES[1].clear()
        except Exception:
            pass


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    display_thread = setup_and_start_display()

    if display_thread:
        print("\n--- Testing Procedural Eyes ---")
        time.sleep(2)

        # Test looping mode
        print("▶ Starting 'happy' (looping)...")
        start_emotion("happy")
        time.sleep(3)

        print("⏹ Stopping emotion...")
        stop_emotion()
        time.sleep(2)

        # Test one-shot
        print("▶ Playing 'sad' (one-shot)...")
        display_emotion("sad", EmotionMode.ONE_SHOT)
        time.sleep(3)

        # Test angry
        print("▶ Playing 'angry' (looping)...")
        start_emotion("angry")
        time.sleep(3)

        print("▶ Transitioning to 'loving'...")
        start_emotion("loving")
        time.sleep(3)

        print("⏹ Stopping...")
        stop_emotion()
        time.sleep(2)

        stop_display()
    else:
        print("Display not available")
