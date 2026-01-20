"""
OLED Emotion Display - Dual SSD1306 robot eyes
Displays emotions on dual OLED screens connected via I2C.
"""

import sys 
from pathlib import Path
from PIL import Image
import time
import os
import glob 
import threading 
import collections 

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

# Video frames directory - change this to your path
BASE_DIRECTORY = "/home/nema/Documents/NEma/oled/videos" 
DEFAULT_EMOTION = "idle" 

ZOOM_FACTOR = 0.8
DESIRED_FPS = 30
FRAME_DELAY = 1.0 / DESIRED_FPS if DESIRED_FPS > 0 else 0.0

# Available emotions
EMOTIONS = ["idle", "looking", "happy", "sad", "angry", "boring", "smile"]


# --- Global State ---
current_emotion = DEFAULT_EMOTION 
video_queue = collections.deque() 
FRAME_CACHE = {} 
DEVICES = None
DISPLAY_RUNNING = False


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


def _play_emotion_one_shot(emotion_name):
    """Plays a single full cycle of the specified emotion video."""
    emotion_frames = FRAME_CACHE.get(emotion_name)
    
    if not emotion_frames:
        print(f"⚠️ Cannot play '{emotion_name}' - frames missing")
        return False
        
    print(f"👀 Playing: {emotion_name.upper()}")
    
    for frame_index in range(len(emotion_frames)):
        if not DISPLAY_RUNNING:
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

    return True


def _display_thread_function():
    """The continuous video playback loop."""
    global current_emotion, video_queue, DISPLAY_RUNNING
    
    current_emotion = DEFAULT_EMOTION 
    idle_frame_index = 0 

    while DISPLAY_RUNNING:
        # Check for new video request
        if video_queue:
            requested_emotion = video_queue.popleft()
            _play_emotion_one_shot(requested_emotion)
            current_emotion = DEFAULT_EMOTION
            idle_frame_index = 0 

        # Play idle frame
        emotion_frames = FRAME_CACHE.get(DEFAULT_EMOTION)
        
        if not emotion_frames:
            time.sleep(FRAME_DELAY)
            continue

        file_path = emotion_frames[idle_frame_index]
        start_time = time.time()
        
        try:
            current_frame_img = Image.open(file_path)
            _process_frame(current_frame_img, idle_frame_index)
        except Exception:
            pass 

        idle_frame_index = (idle_frame_index + 1) % len(emotion_frames)
        
        time_spent = time.time() - start_time
        sleep_time = FRAME_DELAY - time_spent
        if sleep_time > 0:
            time.sleep(sleep_time)


def setup_and_start_display():
    """
    Initializes devices, pre-loads the default emotion, and starts the display loop.
    Call this at agent startup.
    """
    global DEVICES, FRAME_CACHE, DISPLAY_RUNNING
    
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

    # Start display thread
    DISPLAY_RUNNING = True
    display_thread = threading.Thread(target=_display_thread_function, daemon=True)
    display_thread.start()
    print(f"👀 OLED display started ({DESIRED_FPS} FPS)")
    
    return display_thread


def display_emotion(emotion_name: str) -> bool:
    """
    Queue an emotion to display.
    Call this when LLM generates a response with emotion tag.
    
    Args:
        emotion_name: One of: idle, looking, happy, sad, angry, boring, smile
    """
    global video_queue, current_emotion, FRAME_CACHE, DISPLAY_RUNNING
    
    if not DISPLAY_RUNNING:
        return False
        
    requested_emotion = emotion_name.strip().lower()
    
    # Validate emotion
    if requested_emotion not in EMOTIONS:
        print(f"⚠️ Unknown emotion: {requested_emotion}")
        return False

    # Load frames if not cached
    if requested_emotion not in FRAME_CACHE:
        FRAME_CACHE[requested_emotion] = _load_emotion_frames(requested_emotion)
        
    # Add to queue
    if FRAME_CACHE.get(requested_emotion):
        if requested_emotion not in video_queue and requested_emotion != current_emotion:
            video_queue.append(requested_emotion)
            current_emotion = requested_emotion 
            return True
    
    return False


def stop_display():
    """Stops the display thread and clears the screens."""
    global DISPLAY_RUNNING, DEVICES
    print("👀 Stopping OLED display...")
    DISPLAY_RUNNING = False
    time.sleep(0.5) 

    if DEVICES:
        DEVICES[0].clear() 
        DEVICES[1].clear()


# Test when run directly
if __name__ == "__main__":
    display_thread = setup_and_start_display()
    
    if display_thread:
        print("\nTesting emotions...")
        time.sleep(2)
        
        for emotion in ["happy", "sad", "smile"]:
            print(f"Testing: {emotion}")
            display_emotion(emotion)
            time.sleep(3)
        
        stop_display()
    else:
        print("OLED display not available")
