"""
Procedural Eyes - Vector-style animated robot eyes

Real-time eye rendering with smooth parameter interpolation.
Replaces video-based emotions with procedural graphics.
"""

import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from PIL import Image, ImageDraw

# Try to import luma.oled - gracefully fail if not on Pi
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    LUMA_AVAILABLE = True
except ImportError:
    LUMA_AVAILABLE = False
    print("⚠️ luma.oled not available - OLED display disabled")


# === Configuration ===
I2C_PORT = 1
LEFT_OLED_ADDRESS = 0x3d
RIGHT_OLED_ADDRESS = 0x3c

# Eye dimensions for 128x64 OLED (we use 64x64 per eye)
EYE_SIZE = 64
CANVAS_SIZE = 128  # Full canvas (both eyes)

# Animation settings
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
TRANSITION_DURATION = 0.3  # Default emotion transition time
BLINK_DURATION = 0.15


# === Easing Functions ===
def ease_in_out_cubic(t: float) -> float:
    """Smooth start and end easing."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_quad(t: float) -> float:
    """Quick start, slow end."""
    return 1 - (1 - t) * (1 - t)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


# === Eye Parameters ===
@dataclass
class EyeParams:
    """Parameters that define eye appearance."""
    eyelid_top: float = 0.2      # 0=open, 1=fully closed from top
    eyelid_bottom: float = 0.0   # 0=normal, 1=raised (squint)
    pupil_size: float = 0.5      # 0=tiny, 1=large
    gaze_x: float = 0.5          # 0=left, 0.5=center, 1=right
    gaze_y: float = 0.5          # 0=up, 0.5=center, 1=down
    eye_height: float = 1.0      # Overall eye height multiplier
    
    def copy(self) -> 'EyeParams':
        return EyeParams(
            eyelid_top=self.eyelid_top,
            eyelid_bottom=self.eyelid_bottom,
            pupil_size=self.pupil_size,
            gaze_x=self.gaze_x,
            gaze_y=self.gaze_y,
            eye_height=self.eye_height
        )
    
    def lerp_to(self, target: 'EyeParams', t: float) -> 'EyeParams':
        """Interpolate towards target parameters."""
        return EyeParams(
            eyelid_top=lerp(self.eyelid_top, target.eyelid_top, t),
            eyelid_bottom=lerp(self.eyelid_bottom, target.eyelid_bottom, t),
            pupil_size=lerp(self.pupil_size, target.pupil_size, t),
            gaze_x=lerp(self.gaze_x, target.gaze_x, t),
            gaze_y=lerp(self.gaze_y, target.gaze_y, t),
            eye_height=lerp(self.eye_height, target.eye_height, t)
        )


# === Emotion Presets ===
EMOTION_PRESETS: Dict[str, EyeParams] = {
    # Neutral states
    "idle1": EyeParams(eyelid_top=0.2, eyelid_bottom=0.0, pupil_size=0.5),
    "idle2": EyeParams(eyelid_top=0.1, eyelid_bottom=0.0, pupil_size=0.6),  # Alert/listening
    
    # Positive emotions
    "happy": EyeParams(eyelid_top=0.45, eyelid_bottom=0.35, pupil_size=0.55),
    "smile": EyeParams(eyelid_top=0.35, eyelid_bottom=0.25, pupil_size=0.5),
    "loving": EyeParams(eyelid_top=0.3, eyelid_bottom=0.2, pupil_size=0.75),
    
    # Negative emotions
    "sad": EyeParams(eyelid_top=0.5, eyelid_bottom=0.0, pupil_size=0.4, gaze_y=0.6),
    "angry": EyeParams(eyelid_top=0.45, eyelid_bottom=0.4, pupil_size=0.35),
    "boring": EyeParams(eyelid_top=0.6, eyelid_bottom=0.0, pupil_size=0.4),
    
    # Other states
    "looking": EyeParams(eyelid_top=0.05, eyelid_bottom=0.0, pupil_size=0.7),
    "surprised": EyeParams(eyelid_top=0.0, eyelid_bottom=0.0, pupil_size=0.8, eye_height=1.1),
    
    # Blink (used internally)
    "blink": EyeParams(eyelid_top=1.0, eyelid_bottom=0.5, pupil_size=0.5),
}


# === Eye Renderer ===
class EyeRenderer:
    """Renders a single eye based on parameters."""
    
    def __init__(self, size: int = EYE_SIZE):
        self.size = size
        self.center_x = size // 2
        self.center_y = size // 2
        
        # Eye shape constants
        self.eye_width = int(size * 0.8)
        self.eye_height_base = int(size * 0.6)
        self.pupil_max_radius = int(size * 0.2)
        
    def render(self, params: EyeParams) -> Image.Image:
        """Render eye to PIL Image."""
        img = Image.new('1', (self.size, self.size), 0)  # Black background
        draw = ImageDraw.Draw(img)
        
        # Calculate dimensions
        eye_h = int(self.eye_height_base * params.eye_height)
        top_y = self.center_y - eye_h // 2
        bottom_y = self.center_y + eye_h // 2
        
        # Draw eye white (ellipse)
        left_x = self.center_x - self.eye_width // 2
        right_x = self.center_x + self.eye_width // 2
        draw.ellipse([left_x, top_y, right_x, bottom_y], fill=1, outline=1)
        
        # Apply eyelids (cover parts of the ellipse)
        lid_top_y = top_y + int(eye_h * params.eyelid_top)
        lid_bottom_y = bottom_y - int(eye_h * params.eyelid_bottom)
        
        # Top eyelid (black rectangle from top)
        if params.eyelid_top > 0:
            draw.rectangle([0, 0, self.size, lid_top_y], fill=0)
        
        # Bottom eyelid (black rectangle from bottom)  
        if params.eyelid_bottom > 0:
            draw.rectangle([0, lid_bottom_y, self.size, self.size], fill=0)
        
        # Draw pupil
        visible_h = lid_bottom_y - lid_top_y
        if visible_h > 5:  # Only draw if eye is open enough
            pupil_radius = int(self.pupil_max_radius * params.pupil_size)
            
            # Gaze offset (limited to eye bounds)
            max_offset_x = (self.eye_width // 2) - pupil_radius - 5
            max_offset_y = (visible_h // 2) - pupil_radius - 3
            
            gaze_offset_x = int((params.gaze_x - 0.5) * 2 * max_offset_x)
            gaze_offset_y = int((params.gaze_y - 0.5) * 2 * max_offset_y)
            
            pupil_x = self.center_x + gaze_offset_x
            pupil_y = (lid_top_y + lid_bottom_y) // 2 + gaze_offset_y
            
            # Draw pupil as black circle
            draw.ellipse([
                pupil_x - pupil_radius,
                pupil_y - pupil_radius,
                pupil_x + pupil_radius,
                pupil_y + pupil_radius
            ], fill=0)
        
        return img


# === Main Procedural Eyes Class ===
class ProceduralEyes:
    """
    Main class for procedural eye animation.
    Manages both eyes, animation state, and OLED output.
    """
    
    def __init__(self):
        self.renderer = EyeRenderer()
        
        # Current and target parameters
        self.current = EyeParams()
        self.target = EyeParams()
        self.base_emotion = "idle1"  # For returning after blink
        
        # Animation state
        self.transition_start = 0.0
        self.transition_duration = TRANSITION_DURATION
        self.start_params: Optional[EyeParams] = None
        
        # Blink state
        self.is_blinking = False
        self.blink_start = 0.0
        self.next_blink_time = time.time() + 3.0
        
        # Micro-movement state
        self.micro_offset_x = 0.0
        self.micro_offset_y = 0.0
        self.last_micro_update = 0.0
        
        # OLED devices
        self.left_device = None
        self.right_device = None
        
        # Thread control
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def _setup_devices(self):
        """Initialize OLED displays."""
        if not LUMA_AVAILABLE:
            print("⚠️ OLED not available")
            return False
        
        try:
            serial_left = i2c(port=I2C_PORT, address=LEFT_OLED_ADDRESS)
            self.left_device = ssd1306(serial_left)
            print(f"✅ LEFT OLED initialized")
            
            serial_right = i2c(port=I2C_PORT, address=RIGHT_OLED_ADDRESS)
            self.right_device = ssd1306(serial_right)
            print(f"✅ RIGHT OLED initialized")
            
            return True
        except Exception as e:
            print(f"⚠️ OLED init failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the animation thread."""
        if self.running:
            return True
        
        if not self._setup_devices():
            print("⚠️ Running in headless mode (no OLED)")
        
        self.running = True
        self.thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.thread.start()
        print(f"👀 Procedural eyes started ({TARGET_FPS} FPS)")
        return True
    
    def stop(self):
        """Stop animation and clear displays."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.left_device:
            self.left_device.clear()
        if self.right_device:
            self.right_device.clear()
        
        print("👀 Procedural eyes stopped")
    
    def set_emotion(self, emotion: str, duration: float = TRANSITION_DURATION):
        """
        Transition to an emotion.
        
        Args:
            emotion: Name of emotion preset
            duration: Transition time in seconds
        """
        if emotion not in EMOTION_PRESETS:
            print(f"⚠️ Unknown emotion: {emotion}")
            return
        
        print(f"🎭 Setting emotion: {emotion}")
        self.base_emotion = emotion
        self.target = EMOTION_PRESETS[emotion].copy()
        self.start_params = self.current.copy()
        self.transition_start = time.time()
        self.transition_duration = duration
    
    def blink(self):
        """Trigger a blink animation."""
        if self.is_blinking:
            return
        
        self.is_blinking = True
        self.blink_start = time.time()
        self.start_params = self.current.copy()
    
    def look_at(self, x: float, y: float, duration: float = 0.2):
        """
        Move gaze to position.
        
        Args:
            x: 0=left, 0.5=center, 1=right
            y: 0=up, 0.5=center, 1=down
        """
        new_target = self.target.copy()
        new_target.gaze_x = max(0, min(1, x))
        new_target.gaze_y = max(0, min(1, y))
        
        self.target = new_target
        self.start_params = self.current.copy()
        self.transition_start = time.time()
        self.transition_duration = duration
    
    def _update_animation(self):
        """Update current parameters based on animation state."""
        now = time.time()
        
        # Handle blink
        if self.is_blinking:
            blink_progress = (now - self.blink_start) / BLINK_DURATION
            
            if blink_progress >= 1.0:
                # Blink complete
                self.is_blinking = False
                self.current = EMOTION_PRESETS[self.base_emotion].copy()
            elif blink_progress < 0.5:
                # Closing
                t = ease_out_quad(blink_progress * 2)
                self.current = self.start_params.lerp_to(EMOTION_PRESETS["blink"], t)
            else:
                # Opening
                t = ease_out_quad((blink_progress - 0.5) * 2)
                self.current = EMOTION_PRESETS["blink"].lerp_to(EMOTION_PRESETS[self.base_emotion], t)
            return
        
        # Handle emotion transition
        if self.start_params and self.transition_duration > 0:
            elapsed = now - self.transition_start
            progress = min(1.0, elapsed / self.transition_duration)
            t = ease_in_out_cubic(progress)
            self.current = self.start_params.lerp_to(self.target, t)
            
            if progress >= 1.0:
                self.start_params = None
        
        # Add micro-movements for idle
        if now - self.last_micro_update > 2.0:
            self.micro_offset_x = (math.sin(now * 0.5) * 0.03)
            self.micro_offset_y = (math.cos(now * 0.7) * 0.02)
            self.last_micro_update = now
        
        # Apply micro-movements
        self.current.gaze_x = self.target.gaze_x + self.micro_offset_x
        self.current.gaze_y = self.target.gaze_y + self.micro_offset_y
        
        # Auto-blink
        if now > self.next_blink_time:
            self.blink()
            self.next_blink_time = now + 3.0 + (math.sin(now) + 1) * 2  # 3-7 seconds
    
    def _render_frame(self) -> Tuple[Image.Image, Image.Image]:
        """Render both eyes."""
        # Left eye (mirror gaze_x for natural looking)
        left_params = self.current.copy()
        left_params.gaze_x = 1.0 - self.current.gaze_x  # Mirror
        left_img = self.renderer.render(left_params)
        
        # Right eye
        right_img = self.renderer.render(self.current)
        
        return left_img, right_img
    
    def _display_frame(self, left_img: Image.Image, right_img: Image.Image):
        """Send frame to OLED displays."""
        if self.left_device:
            # Rotate for OLED orientation
            left_rotated = left_img.rotate(-90, expand=True)
            self.left_device.display(left_rotated)
        
        if self.right_device:
            right_rotated = right_img.rotate(90, expand=True)
            self.right_device.display(right_rotated)
    
    def _animation_loop(self):
        """Main animation loop running in thread."""
        while self.running:
            start = time.time()
            
            # Update animation state
            self._update_animation()
            
            # Render and display
            left_img, right_img = self._render_frame()
            self._display_frame(left_img, right_img)
            
            # Maintain frame rate
            elapsed = time.time() - start
            sleep_time = FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# === Module-level interface (matches oled_display.py) ===
_eyes: Optional[ProceduralEyes] = None
DISPLAY_RUNNING = False


def setup_and_start_display():
    """Start procedural eyes. Call at agent startup."""
    global _eyes, DISPLAY_RUNNING
    
    _eyes = ProceduralEyes()
    if _eyes.start():
        DISPLAY_RUNNING = True
        return _eyes.thread
    return None


def display_emotion(emotion: str) -> bool:
    """Display an emotion (one-shot, returns to idle)."""
    global _eyes
    if _eyes and DISPLAY_RUNNING:
        _eyes.set_emotion(emotion)
        return True
    return False


def start_emotion(emotion: str) -> bool:
    """Start playing an emotion (looping mode)."""
    return display_emotion(emotion)


def stop_emotion() -> bool:
    """Return to idle state."""
    global _eyes
    if _eyes and DISPLAY_RUNNING:
        _eyes.set_emotion("idle1")
        return True
    return False


def stop_display():
    """Stop the display."""
    global _eyes, DISPLAY_RUNNING
    if _eyes:
        _eyes.stop()
    DISPLAY_RUNNING = False


# === Test ===
if __name__ == "__main__":
    print("Testing Procedural Eyes\n" + "=" * 40)
    
    eyes = ProceduralEyes()
    eyes.start()
    
    # Test emotions
    emotions = ["idle1", "happy", "sad", "angry", "looking", "loving"]
    
    try:
        for emotion in emotions:
            print(f"\n>>> {emotion}")
            eyes.set_emotion(emotion)
            time.sleep(2)
        
        print("\n>>> Testing blink")
        eyes.blink()
        time.sleep(1)
        
        print("\n>>> Testing look_at")
        eyes.look_at(0.2, 0.3)
        time.sleep(1)
        eyes.look_at(0.8, 0.7)
        time.sleep(1)
        
        print("\nPress Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    eyes.stop()
