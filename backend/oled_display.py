"""
OLED Emotion Display - Dual SSD1306 robot eyes
Displays emotions on dual OLED screens connected via I2C.

Enhanced version with:
- Looping mode: Keep emotion playing during speech
- Living idle: Random blinks and subtle movements
"""

import sys 
from pathlib import Path
from PIL import Image
import time
import os
import glob 
import threading 
import collections 
import random
from enum import Enum
import atexit

# Try to import luma.oled - gracefully fail if not on Pi
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    LUMA_AVAILABLE = True
except ImportError:
    LUMA_AVAILABLE = False
    print("⚠️ luma.oled not available - OLED display disabled")


# --- Configuration ---
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d
RIGHT_OLED_ADDRESS = 0x3c 

# Video frames directory - relative to this file's location
BASE_DIRECTORY = str(Path(__file__).parent / "videos")
DEFAULT_EMOTION = "idle1"  # idle1 = no one talking, idle2 = user talking

ZOOM_FACTOR = 0.8
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS if DESIRED_FPS > 0 else 0.0

# Available emotions
EMOTIONS = ["idle1", "idle2", "looking", "happy", "sad", "angry", "boring", "smile", "loving"]

# Living idle settings
BLINK_INTERVAL_MIN = 3.0  # Minimum seconds between blinks
BLINK_INTERVAL_MAX = 7.0  # Maximum seconds between blinks
LOOK_INTERVAL_MIN = 8.0   # Minimum seconds between look movements
LOOK_INTERVAL_MAX = 15.0  # Maximum seconds between look movements
 
# Emotion fallbacks (if directory is missing)
EMOTION_FALLBACKS = {
    "happy": "smile",
    "loving": "smile",
    "boring": "idle2"
}


class EmotionMode(Enum):
    """Emotion playback modes"""
    ONE_SHOT = "one_shot"     # Play once, return to idle
    LOOPING = "looping"       # Keep looping until stopped
    SUSTAINED = "sustained"   # Keep last frame until changed


# --- Global State ---
current_emotion = DEFAULT_EMOTION 
current_mode = EmotionMode.ONE_SHOT
_stop_current_emotion = threading.Event()  # Signal to stop current emotion
video_queue = collections.deque() 
FRAME_CACHE = {} 
DEVICES = None
DISPLAY_RUNNING = False

# Living idle state
_last_blink_time = 0
_next_blink_interval = 5.0
_last_look_time = 0
_next_look_interval = 10.0


class DummyDevice:
    """Fallback device when OLED not available"""
    def clear(self): pass
    def display(self, image): pass
    width = 128
    height = 64


def _setup_device(address, name):
    """Initializes a single SSD1306 device or returns a dummy device on failure."""
    if not LUMA_AVAILABLE:
        return DummyDevice()
        
    try:
        serial = i2c(port=I2C_PORT, address=address)
        device = ssd1306(serial)
        print(f"✅ {name} OLED (Address: {hex(address)}) initialized")
        return device
    except Exception as e:
        print(f"⚠️ Could not connect to {name} OLED ({hex(address)}): {e}")
        return DummyDevice()


def _load_emotion_frames(emotion):
    """Loads all image file paths for a specific emotion into the cache."""
    frame_path = os.path.join(BASE_DIRECTORY, emotion)
    search_path = os.path.join(frame_path, "*.png")
    frame_files = sorted(glob.glob(search_path)) 
    
    if not frame_files:
        return []
        
    print(f"📹 Loaded {len(frame_files)} frames for '{emotion}'")
    return frame_files


def _process_frame(img, frame_number):
    """Processes a single image frame (zoom, split, rotate) and displays it."""
    global DEVICES
    if not DEVICES: return
    left_device, right_device = DEVICES
    
    try:
        if img.size != (128, 128): return 
        target_crop_size = int(128 * ZOOM_FACTOR)
        offset = (128 - target_crop_size) // 2
        crop_box = (offset, offset, 128 - offset, 128 - offset) 
        img_zoomed = img.crop(crop_box).resize((128, 128), Image.BILINEAR)

        left_half_pil = img_zoomed.crop((0, 0, 64, 128))
        right_half_pil = img_zoomed.crop((64, 0, 128, 128))

        left_oled_image = left_half_pil.rotate(-90, expand=True).convert('1')
        right_oled_image = right_half_pil.rotate(90, expand=True).convert('1')

        left_device.display(left_oled_image)
        right_device.display(right_oled_image)
            
    except Exception:
        pass


def _play_emotion_once(emotion_name):
    """Plays a single full cycle of the specified emotion video."""
    emotion_frames = FRAME_CACHE.get(emotion_name)
    
    if not emotion_frames:
        print(f"⚠️ Cannot play '{emotion_name}' - frames missing")
        return False
        
    for frame_index in range(len(emotion_frames)):
        if not DISPLAY_RUNNING or _stop_current_emotion.is_set():
            return False

        file_path = emotion_frames[frame_index]
        start_time = time.time()
        
        try:
            current_frame_img = Image.open(file_path)
            _process_frame(current_frame_img, frame_index) 
        except Exception:
            pass

        time_spent = time.time() - start_time
        sleep_time = FRAME_DELAY - time_spent
        if sleep_time > 0:
            time.sleep(sleep_time)

    return True


def _play_emotion_looping(emotion_name):
    """Plays the emotion in a loop until stopped."""
    emotion_frames = FRAME_CACHE.get(emotion_name)
    
    if not emotion_frames:
        print(f"⚠️ Cannot play '{emotion_name}' - frames missing")
        return
    
    print(f"🔄 Looping: {emotion_name.upper()}")
    
    while DISPLAY_RUNNING and not _stop_current_emotion.is_set():
        for frame_index in range(len(emotion_frames)):
            if not DISPLAY_RUNNING or _stop_current_emotion.is_set():
                return

            file_path = emotion_frames[frame_index]
            start_time = time.time()
            
            try:
                current_frame_img = Image.open(file_path)
                _process_frame(current_frame_img, frame_index) 
            except Exception:
                pass

            time_spent = time.time() - start_time
            sleep_time = FRAME_DELAY - time_spent
            if sleep_time > 0:
                time.sleep(sleep_time)


def _play_living_idle():
    """Plays idle animation with occasional blinks and look movements."""
    global _last_blink_time, _next_blink_interval, _last_look_time, _next_look_interval
    
    emotion_frames = FRAME_CACHE.get(DEFAULT_EMOTION)
    if not emotion_frames:
        time.sleep(FRAME_DELAY)
        return
    
    current_time = time.time()
    
    # Check for blink
    if current_time - _last_blink_time > _next_blink_interval:
        # Try to play blink animation if available
        if "blink" in FRAME_CACHE or _load_and_cache("blink"):
            _play_emotion_once("blink")
        _last_blink_time = current_time
        _next_blink_interval = random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        return
    
    # Check for look movement
    if current_time - _last_look_time > _next_look_interval:
        # Try to play looking animation if available
        if "looking" in FRAME_CACHE or _load_and_cache("looking"):
            _play_emotion_once("looking")
        _last_look_time = current_time
        _next_look_interval = random.uniform(LOOK_INTERVAL_MIN, LOOK_INTERVAL_MAX)
        return
    
    # Normal idle frame
    frame_index = int((current_time * DESIRED_FPS) % len(emotion_frames))
    file_path = emotion_frames[frame_index]
    
    try:
        current_frame_img = Image.open(file_path)
        _process_frame(current_frame_img, frame_index)
    except Exception:
        pass
    
    time.sleep(FRAME_DELAY)


def _load_and_cache(emotion):
    """Try to load and cache an emotion, returns True if successful."""
    global FRAME_CACHE
    if emotion not in FRAME_CACHE:
        frames = _load_emotion_frames(emotion)
        if frames:
            FRAME_CACHE[emotion] = frames
            return True
    return emotion in FRAME_CACHE and len(FRAME_CACHE[emotion]) > 0


def _display_thread_function():
    """The continuous video playback loop."""
    global current_emotion, current_mode, video_queue, DISPLAY_RUNNING, _stop_current_emotion
    
    current_emotion = DEFAULT_EMOTION 
    _stop_current_emotion.clear()

    while DISPLAY_RUNNING:
        # Check for new video request
        if video_queue:
            request = video_queue.popleft()
            emotion_name = request["emotion"]
            mode = request["mode"]
            
            current_emotion = emotion_name
            current_mode = mode
            _stop_current_emotion.clear()
            
            print(f"👀 Playing: {emotion_name.upper()} ({mode.value})")
            
            if mode == EmotionMode.LOOPING:
                _play_emotion_looping(emotion_name)
            else:  # ONE_SHOT
                _play_emotion_once(emotion_name)
                
            # After playing, return to idle
            print(f"👀 OLED: Finished {emotion_name.upper()}, returning to idle")
            current_emotion = DEFAULT_EMOTION
            _stop_current_emotion.clear()
            continue

        # Play living idle when nothing in queue
        _play_living_idle()


def setup_and_start_display():
    """
    Initializes devices, pre-loads the default emotion, and starts the display loop.
    Call this at agent startup.
    """
    global DEVICES, FRAME_CACHE, DISPLAY_RUNNING, _last_blink_time, _last_look_time
    
    if not LUMA_AVAILABLE:
        print("⚠️ OLED display disabled (luma.oled not installed)")
        return None
    
    # Check if videos directory exists
    if not os.path.exists(BASE_DIRECTORY):
        print(f"⚠️ OLED videos directory not found: {BASE_DIRECTORY}")
        return None
    
    # Setup devices
    left_device = _setup_device(LEFT_OLED_ADDRESS, "LEFT")
    right_device = _setup_device(RIGHT_OLED_ADDRESS, "RIGHT")
    DEVICES = (left_device, right_device)
        
    # Pre-load idle emotion
    FRAME_CACHE[DEFAULT_EMOTION] = _load_emotion_frames(DEFAULT_EMOTION)
    if not FRAME_CACHE.get(DEFAULT_EMOTION):
        print(f"⚠️ Default 'idle' frames missing in {BASE_DIRECTORY}/idle")
        return None

    # Initialize living idle timers
    _last_blink_time = time.time()
    _last_look_time = time.time()

    # Start display thread
    DISPLAY_RUNNING = True
    display_thread = threading.Thread(target=_display_thread_function, daemon=True)
    display_thread.start()
    print(f"👀 OLED display started ({DESIRED_FPS} FPS) with living idle")
    
    # Register cleanup for process exit
    atexit.register(stop_display)
    
    return display_thread


def display_emotion(emotion_name: str, mode: EmotionMode = EmotionMode.ONE_SHOT) -> bool:
    """
    Queue an emotion to display. Clears queue and interrupts current play.
    
    Args:
        emotion_name: The emotion to play
        mode: EmotionMode.ONE_SHOT or EmotionMode.LOOPING
    
    Returns:
        True if emotion was queued successfully
    """
    global video_queue, current_emotion, FRAME_CACHE, DISPLAY_RUNNING, _stop_current_emotion
    
    if not DISPLAY_RUNNING:
        return False
        
    requested_emotion = emotion_name.strip().lower()
    
    # Apply fallback if folder missing
    if not os.path.exists(os.path.join(BASE_DIRECTORY, requested_emotion)) and requested_emotion in EMOTION_FALLBACKS:
        fallback = EMOTION_FALLBACKS[requested_emotion]
        print(f"🎬 OLED: Fallback '{requested_emotion}' -> '{fallback}'")
        requested_emotion = fallback
    
    # Validate emotion
    if requested_emotion not in EMOTIONS:
        print(f"⚠️ Emotion '{requested_emotion}' not found in {EMOTIONS}")
        return False
        
    # DEDUPLICATION: If already playing this emotion in LOOPING mode, don't restart it
    if current_emotion == requested_emotion and mode == EmotionMode.LOOPING:
        return True

    # Load frames if not cached
    if requested_emotion not in FRAME_CACHE:
        FRAME_CACHE[requested_emotion] = _load_emotion_frames(requested_emotion)
        
    # If frames are still not available after loading attempt, return False
    if not FRAME_CACHE.get(requested_emotion):
        print(f"⚠️ Cannot play '{requested_emotion}' - frames missing")
        return False

    # CLEAR QUEUE and Interrupt current animation to start new one immediately
    video_queue.clear()
    _stop_current_emotion.set()
    
    video_queue.append({
        "emotion": requested_emotion,
        "mode": mode
    })
    
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
    print("🎬 OLED: stop_emotion() called")
    _stop_current_emotion.set()
    return True


def stop_display():
    """Stops the display thread and clears the screens."""
    global DISPLAY_RUNNING, DEVICES, _stop_current_emotion
    print("👀 Stopping OLED display...")
    _stop_current_emotion.set()
    DISPLAY_RUNNING = False
    time.sleep(0.5) 

    if DEVICES:
        DEVICES[0].clear() 
        DEVICES[1].clear()


# Test when run directly
if __name__ == "__main__":
    display_thread = setup_and_start_display()
    
    if display_thread:
        print("\nTesting emotions with looping mode...")
        time.sleep(2)
        
        # Test looping mode
        print("Starting 'happy' in looping mode...")
        start_emotion("happy")
        time.sleep(4)
        
        print("Stopping emotion...")
        stop_emotion()
        time.sleep(2)
        
        # Test one-shot
        print("Playing 'sad' one-shot...")
        display_emotion("sad", EmotionMode.ONE_SHOT)
        time.sleep(4)
        
        stop_display()
    else:
        print("OLED display not available")
