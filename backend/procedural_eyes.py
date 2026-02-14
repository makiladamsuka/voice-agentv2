"""
Procedural Eyes Engine - Vector-Style Living Robot Eyes

Generates real-time animated eye frames using parametric rendering.
Every frame is unique — eyes breathe, blink, saccade, and drift
just like a living creature.

Supports:
- ST7735 color TFT displays (128x160, RGB) via luma.lcd or SPI
- SSD1306 monochrome OLEDs (128x64) via luma.oled (fallback)
- Headless mode (no hardware — for development/testing on laptop)

Output: 128x128 RGB PIL Image per frame
(Display adapter handles split/rotate/send to hardware)
"""

import math
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from PIL import Image, ImageDraw


# ============================================================================
# EYE PARAMETERS (all animatable)
# ============================================================================

@dataclass
class EyeParams:
    """Parameters defining the appearance of a single eye."""
    # Shape
    eye_width: float = 28.0       # Width of eye opening
    eye_height: float = 40.0      # Height of eye opening
    corner_radius: float = 12.0   # Rounded corners

    # Eyelids (0.0 = fully open, 1.0 = fully closed)
    top_lid: float = 0.05         # Top eyelid closure (slight natural droop)
    bottom_lid: float = 0.0       # Bottom eyelid closure
    top_lid_angle: float = 0.0    # Tilt angle (-1 = angry inward, +1 = sad droop)

    # Pupil / gaze
    pupil_scale: float = 0.55     # Pupil size relative to eye
    gaze_x: float = 0.0           # Horizontal gaze (-1 left, +1 right)
    gaze_y: float = 0.0           # Vertical gaze (-1 up, +1 down)

    # Global offsets
    y_offset: float = 0.0         # Vertical offset (breathing)
    x_offset: float = 0.0         # Horizontal offset (for asymmetry)
    openness: float = 1.0         # Overall eye openness multiplier (0=closed, 1=open)

    # Color (RGB tuples — only used with color displays)
    eye_color: Tuple[int, int, int] = (255, 255, 255)       # Eye white/sclera color
    bg_color: Tuple[int, int, int] = (0, 0, 0)              # Background color


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Interpolate between two RGB colors."""
    return (
        int(lerp(a[0], b[0], t)),
        int(lerp(a[1], b[1], t)),
        int(lerp(a[2], b[2], t)),
    )


def ease_in_out_cubic(t: float) -> float:
    """Smooth ease-in-out cubic curve."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        return 1.0 - pow(-2.0 * t + 2.0, 3) / 2.0


def ease_out_quad(t: float) -> float:
    """Quick start, slow finish — used for blink open."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_quad(t: float) -> float:
    """Slow start, quick finish — used for blink close."""
    return t * t


def blend_params(current: EyeParams, target: EyeParams, t: float) -> EyeParams:
    """Blend between two EyeParams with interpolation factor t (0-1)."""
    result = EyeParams()
    for attr in ['eye_width', 'eye_height', 'corner_radius',
                 'top_lid', 'bottom_lid', 'top_lid_angle',
                 'pupil_scale', 'gaze_x', 'gaze_y',
                 'y_offset', 'x_offset', 'openness']:
        setattr(result, attr, lerp(getattr(current, attr), getattr(target, attr), t))
    # Blend colors
    result.eye_color = lerp_color(current.eye_color, target.eye_color, t)
    result.bg_color = lerp_color(current.bg_color, target.bg_color, t)
    return result


# ============================================================================
# EMOTION PRESETS
# ============================================================================

# Default eye color: bright cyan-white for that Vector/robot feel
DEFAULT_EYE_COLOR = (180, 230, 255)    # Cool blue-white
DEFAULT_BG_COLOR = (0, 0, 0)           # Black background

EMOTION_PRESETS: Dict[str, EyeParams] = {
    "idle1": EyeParams(
        eye_width=28, eye_height=40, corner_radius=12,
        top_lid=0.05, bottom_lid=0.0, top_lid_angle=0.0,
        pupil_scale=0.55, gaze_x=0.0, gaze_y=0.0,
        openness=1.0,
        eye_color=DEFAULT_EYE_COLOR, bg_color=DEFAULT_BG_COLOR,
    ),
    "idle2": EyeParams(  # Listening — alert, slightly wider
        eye_width=30, eye_height=44, corner_radius=12,
        top_lid=0.0, bottom_lid=0.0, top_lid_angle=0.0,
        pupil_scale=0.50, gaze_x=0.0, gaze_y=-0.05,
        openness=1.0,
        eye_color=(200, 240, 255), bg_color=DEFAULT_BG_COLOR,
    ),
    "happy": EyeParams(
        eye_width=30, eye_height=36, corner_radius=14,
        top_lid=0.0, bottom_lid=0.35, top_lid_angle=0.0,
        pupil_scale=0.50, gaze_x=0.0, gaze_y=-0.05,
        openness=1.0,
        eye_color=(180, 255, 200), bg_color=DEFAULT_BG_COLOR,  # Warm green tint
    ),
    "smile": EyeParams(
        eye_width=28, eye_height=34, corner_radius=14,
        top_lid=0.0, bottom_lid=0.25, top_lid_angle=0.0,
        pupil_scale=0.52, gaze_x=0.0, gaze_y=0.0,
        openness=1.0,
        eye_color=(200, 245, 220), bg_color=DEFAULT_BG_COLOR,
    ),
    "sad": EyeParams(
        eye_width=26, eye_height=36, corner_radius=10,
        top_lid=0.2, bottom_lid=0.0, top_lid_angle=0.5,
        pupil_scale=0.60, gaze_x=0.0, gaze_y=0.15,
        openness=0.85,
        eye_color=(150, 180, 230), bg_color=DEFAULT_BG_COLOR,  # Cool blue tint
    ),
    "angry": EyeParams(
        eye_width=30, eye_height=30, corner_radius=8,
        top_lid=0.3, bottom_lid=0.05, top_lid_angle=-0.7,
        pupil_scale=0.45, gaze_x=0.0, gaze_y=0.0,
        openness=0.9,
        eye_color=(255, 160, 140), bg_color=DEFAULT_BG_COLOR,  # Red tint
    ),
    "looking": EyeParams(  # Curious / searching
        eye_width=30, eye_height=44, corner_radius=12,
        top_lid=0.0, bottom_lid=0.0, top_lid_angle=0.0,
        pupil_scale=0.48, gaze_x=0.3, gaze_y=-0.1,
        openness=1.0,
        eye_color=(200, 220, 255), bg_color=DEFAULT_BG_COLOR,
    ),
    "boring": EyeParams(  # Sleepy / bored
        eye_width=28, eye_height=28, corner_radius=14,
        top_lid=0.45, bottom_lid=0.1, top_lid_angle=0.0,
        pupil_scale=0.55, gaze_x=0.0, gaze_y=0.1,
        openness=0.6,
        eye_color=(160, 170, 180), bg_color=DEFAULT_BG_COLOR,  # Dim/grey
    ),
    "loving": EyeParams(  # Soft, warm
        eye_width=28, eye_height=36, corner_radius=14,
        top_lid=0.05, bottom_lid=0.2, top_lid_angle=0.15,
        pupil_scale=0.65, gaze_x=0.0, gaze_y=0.0,
        openness=0.95,
        eye_color=(255, 200, 220), bg_color=DEFAULT_BG_COLOR,  # Pink tint
    ),
}


# ============================================================================
# RENDERER — Draws a single frame
# ============================================================================

class EyeRenderer:
    """Renders a pair of eyes into a 128x128 RGB image."""

    CANVAS_SIZE = 128
    # Eye centers (on the 128x128 canvas)
    LEFT_EYE_CENTER = (32, 64)
    RIGHT_EYE_CENTER = (96, 64)

    def render(self, params: EyeParams, right_params: Optional[EyeParams] = None,
               mono: bool = False) -> Image.Image:
        """
        Render a full frame with both eyes.

        Args:
            params: Parameters for left eye (also used for right if right_params is None)
            right_params: Optional separate params for right eye (for asymmetry)
            mono: If True, output monochrome '1' mode (for SSD1306 fallback)

        Returns:
            128x128 PIL Image (RGB or '1' mode)
        """
        img = Image.new('RGB', (self.CANVAS_SIZE, self.CANVAS_SIZE), params.bg_color)
        draw = ImageDraw.Draw(img)

        rp = right_params if right_params else params

        self._draw_eye(draw, self.LEFT_EYE_CENTER, params, is_left=True)
        self._draw_eye(draw, self.RIGHT_EYE_CENTER, rp, is_left=False)

        if mono:
            return img.convert('1')
        return img

    def _draw_eye(self, draw: ImageDraw.Draw, center: Tuple[int, int],
                  params: EyeParams, is_left: bool):
        """Draw a single eye with eyelid masks."""
        cx, cy = center
        cy += params.y_offset
        cx += params.x_offset

        # Effective dimensions
        w = params.eye_width * params.openness
        h = params.eye_height * params.openness
        r = min(params.corner_radius, w / 2, h / 2)

        if w < 2 or h < 2:
            return  # Eye is closed

        # --- Draw eye (rounded rectangle) ---
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2

        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=int(r),
            fill=params.eye_color
        )

        # --- Draw eyelids (black masks over the eye) ---
        self._draw_eyelids(draw, cx, cy, w, h, params, is_left)

    def _draw_eyelids(self, draw: ImageDraw.Draw, cx: float, cy: float,
                      w: float, h: float, params: EyeParams, is_left: bool):
        """Draw top and bottom eyelid masks."""
        half_w = w / 2 + 2  # Slight overflow to ensure clean mask
        half_h = h / 2
        bg = params.bg_color

        # --- Top eyelid ---
        if params.top_lid > 0.01:
            lid_drop = half_h * params.top_lid * 2  # How far the lid drops

            # Angle: create asymmetric droop
            angle = params.top_lid_angle
            if not is_left:
                angle = -angle

            left_drop = lid_drop + angle * half_h * 0.4
            right_drop = lid_drop - angle * half_h * 0.4

            # Draw as polygon (trapezoid mask)
            points = [
                (cx - half_w - 2, cy - half_h - 2),
                (cx + half_w + 2, cy - half_h - 2),
                (cx + half_w + 2, cy - half_h + right_drop),
                (cx - half_w - 2, cy - half_h + left_drop),
            ]
            draw.polygon(points, fill=bg)

        # --- Bottom eyelid ---
        if params.bottom_lid > 0.01:
            lid_rise = half_h * params.bottom_lid * 2

            points = [
                (cx - half_w - 2, cy + half_h + 2),
                (cx + half_w + 2, cy + half_h + 2),
                (cx + half_w + 2, cy + half_h - lid_rise),
                (cx - half_w - 2, cy + half_h - lid_rise),
            ]
            draw.polygon(points, fill=bg)


# ============================================================================
# LIFE ENGINE — Adds organic micro-behaviors
# ============================================================================

class LifeEngine:
    """
    Manages all the subtle autonomous behaviors that make eyes feel alive.
    Runs independently from emotion state — these happen on top of everything.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Saccade state
        self._saccade_x = 0.0
        self._saccade_y = 0.0
        self._next_saccade_time = 0.0

        # Micro-drift state (slow wandering)
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._drift_target_x = 0.0
        self._drift_target_y = 0.0
        self._drift_speed = 0.3

        # Breathing state
        self._breath_phase = random.uniform(0, math.pi * 2)

        # Blink state
        self._blink_progress = 0.0
        self._is_blinking = False
        self._blink_phase = "idle"  # idle, closing, closed, opening
        self._blink_timer = 0.0
        self._next_blink_time = time.time() + random.uniform(2.0, 5.0)
        self._double_blink = False
        self._double_blink_done = False

        # Squint variation
        self._squint_offset = 0.0
        self._squint_target = 0.0
        self._next_squint_time = time.time() + random.uniform(5.0, 15.0)

        # Timing
        self._last_update = time.time()

    def update(self, dt: float) -> dict:
        """
        Update all life behaviors and return offsets to apply.

        Returns:
            Dict with offset values to add to current eye params
        """
        now = time.time()

        with self._lock:
            offsets = {
                'gaze_x': 0.0,
                'gaze_y': 0.0,
                'y_offset': 0.0,
                'top_lid': 0.0,
                'bottom_lid': 0.0,
                'openness_mult': 1.0,
            }

            self._update_saccades(now, offsets)
            self._update_drift(dt, offsets)
            self._update_breathing(dt, offsets)
            self._update_blink(now, dt, offsets)
            self._update_squint(now, dt, offsets)

            return offsets

    def _update_saccades(self, now: float, offsets: dict):
        """Quick, tiny pupil jumps."""
        if now >= self._next_saccade_time:
            magnitude = random.gauss(0, 0.04)
            angle = random.uniform(0, math.pi * 2)
            self._saccade_x = magnitude * math.cos(angle)
            self._saccade_y = magnitude * math.sin(angle) * 0.5
            self._next_saccade_time = now + random.uniform(0.3, 1.2)

        offsets['gaze_x'] += self._saccade_x
        offsets['gaze_y'] += self._saccade_y

    def _update_drift(self, dt: float, offsets: dict):
        """Slow, continuous pupil drift."""
        if random.random() < dt * 0.15:
            self._drift_target_x = random.gauss(0, 0.06)
            self._drift_target_y = random.gauss(0, 0.04)

        self._drift_x += (self._drift_target_x - self._drift_x) * self._drift_speed * dt
        self._drift_y += (self._drift_target_y - self._drift_y) * self._drift_speed * dt

        offsets['gaze_x'] += self._drift_x
        offsets['gaze_y'] += self._drift_y

    def _update_breathing(self, dt: float, offsets: dict):
        """Subtle sinusoidal vertical oscillation ~0.2 Hz."""
        self._breath_phase += dt * 0.2 * math.pi * 2
        if self._breath_phase > math.pi * 2:
            self._breath_phase -= math.pi * 2

        offsets['y_offset'] += math.sin(self._breath_phase) * 1.0

    def _update_blink(self, now: float, dt: float, offsets: dict):
        """
        Natural blink with asymmetric timing:
        - Close: ~60ms (fast)
        - Hold: ~30ms
        - Open: ~160ms (slow)
        """
        CLOSE_DURATION = 0.06
        HOLD_DURATION = 0.03
        OPEN_DURATION = 0.16

        if self._blink_phase == "idle":
            if now >= self._next_blink_time:
                self._blink_phase = "closing"
                self._blink_timer = 0.0
                self._double_blink = random.random() < 0.15
                self._double_blink_done = False

        elif self._blink_phase == "closing":
            self._blink_timer += dt
            t = min(1.0, self._blink_timer / CLOSE_DURATION)
            self._blink_progress = ease_in_quad(t)
            if t >= 1.0:
                self._blink_phase = "closed"
                self._blink_timer = 0.0

        elif self._blink_phase == "closed":
            self._blink_timer += dt
            self._blink_progress = 1.0
            if self._blink_timer >= HOLD_DURATION:
                self._blink_phase = "opening"
                self._blink_timer = 0.0

        elif self._blink_phase == "opening":
            self._blink_timer += dt
            t = min(1.0, self._blink_timer / OPEN_DURATION)
            self._blink_progress = 1.0 - ease_out_quad(t)
            if t >= 1.0:
                self._blink_progress = 0.0
                if self._double_blink and not self._double_blink_done:
                    self._blink_phase = "closing"
                    self._blink_timer = 0.0
                    self._double_blink_done = True
                else:
                    self._blink_phase = "idle"
                    self._next_blink_time = now + random.uniform(2.5, 6.0)

        if self._blink_progress > 0.01:
            offsets['openness_mult'] = 1.0 - self._blink_progress * 0.95

    def _update_squint(self, now: float, dt: float, offsets: dict):
        """Occasional subtle eyelid micro-adjustments."""
        if now >= self._next_squint_time:
            self._squint_target = random.gauss(0, 0.03)
            self._next_squint_time = now + random.uniform(4.0, 12.0)

        self._squint_offset += (self._squint_target - self._squint_offset) * 0.5 * dt
        offsets['top_lid'] += max(0, self._squint_offset)
        offsets['bottom_lid'] += max(0, -self._squint_offset)

    def force_blink(self):
        """Trigger an immediate blink."""
        with self._lock:
            self._blink_phase = "closing"
            self._blink_timer = 0.0
            self._double_blink = False


# ============================================================================
# EYE DISPLAY CONTROLLER — Ties it all together
# ============================================================================

class ProceduralEyeDisplay:
    """
    Main controller for the procedural eye system.

    Usage:
        display = ProceduralEyeDisplay()
        display.set_emotion("happy")
        frame = display.render_frame(dt)  # Call at ~30fps
    """

    TRANSITION_SPEED = 3.5  # Higher = faster emotion transitions

    def __init__(self):
        self.renderer = EyeRenderer()
        self.life = LifeEngine()

        # Current interpolated parameters
        self._current_params = EyeParams(
            eye_color=DEFAULT_EYE_COLOR, bg_color=DEFAULT_BG_COLOR
        )
        # Target emotion parameters
        self._target_params = EyeParams(
            eye_color=DEFAULT_EYE_COLOR, bg_color=DEFAULT_BG_COLOR
        )
        self._target_emotion = "idle1"

        self._lock = threading.Lock()

    def set_emotion(self, emotion_name: str):
        """Set the target emotion. Eyes will smoothly transition to it."""
        emotion = emotion_name.strip().lower()
        if emotion not in EMOTION_PRESETS:
            print(f"⚠️ Unknown emotion '{emotion}', defaulting to idle1")
            emotion = "idle1"

        with self._lock:
            if emotion != self._target_emotion:
                self._target_emotion = emotion
                self._target_params = EMOTION_PRESETS[emotion]

    def get_emotion(self) -> str:
        """Get the current target emotion name."""
        return self._target_emotion

    def render_frame(self, dt: float, mono: bool = False) -> Image.Image:
        """
        Generate one frame of animation.

        Args:
            dt: Delta time since last frame in seconds
            mono: If True, output monochrome '1' mode (SSD1306 fallback)

        Returns:
            128x128 PIL Image (RGB or '1')
        """
        with self._lock:
            # 1. Blend current params toward target emotion
            blend_t = min(1.0, self.TRANSITION_SPEED * dt)
            blend_t = ease_in_out_cubic(blend_t)
            self._current_params = blend_params(
                self._current_params, self._target_params, blend_t
            )

        # 2. Get life behavior offsets
        life_offsets = self.life.update(dt)

        # 3. Create final render params with life overlays
        left_params = EyeParams(
            eye_width=self._current_params.eye_width,
            eye_height=self._current_params.eye_height,
            corner_radius=self._current_params.corner_radius,
            top_lid=max(0, self._current_params.top_lid + life_offsets['top_lid']),
            bottom_lid=max(0, self._current_params.bottom_lid + life_offsets['bottom_lid']),
            top_lid_angle=self._current_params.top_lid_angle,
            pupil_scale=self._current_params.pupil_scale,
            gaze_x=self._current_params.gaze_x + life_offsets['gaze_x'],
            gaze_y=self._current_params.gaze_y + life_offsets['gaze_y'],
            y_offset=self._current_params.y_offset + life_offsets['y_offset'],
            x_offset=self._current_params.x_offset,
            openness=self._current_params.openness * life_offsets['openness_mult'],
            eye_color=self._current_params.eye_color,
            bg_color=self._current_params.bg_color,
        )

        # Right eye: slight asymmetry
        right_params = EyeParams(
            eye_width=left_params.eye_width,
            eye_height=left_params.eye_height,
            corner_radius=left_params.corner_radius,
            top_lid=left_params.top_lid,
            bottom_lid=left_params.bottom_lid,
            top_lid_angle=left_params.top_lid_angle,
            pupil_scale=left_params.pupil_scale,
            gaze_x=left_params.gaze_x,
            gaze_y=left_params.gaze_y,
            y_offset=left_params.y_offset + 0.3,  # Tiny vertical offset
            x_offset=left_params.x_offset,
            openness=left_params.openness,
            eye_color=left_params.eye_color,
            bg_color=left_params.bg_color,
        )

        # 4. Render
        return self.renderer.render(left_params, right_params, mono=mono)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import os

    print("🔬 Procedural Eyes — Standalone Test (Color ST7735)")
    print("=" * 50)

    display = ProceduralEyeDisplay()

    output_dir = "/tmp/procedural_eyes_test"
    os.makedirs(output_dir, exist_ok=True)

    emotions = ["idle1", "happy", "sad", "angry", "looking", "smile", "boring", "loving", "idle2"]

    frame_count = 0
    dt = 1.0 / 30.0

    for emotion in emotions:
        print(f"\n  Testing: {emotion}")
        display.set_emotion(emotion)

        # Render 45 frames (~1.5s) per emotion
        for i in range(45):
            frame = display.render_frame(dt)

            # Save every 15th frame
            if i % 15 == 0:
                path = os.path.join(output_dir, f"frame_{frame_count:04d}_{emotion}_{i:03d}.png")
                frame_resized = frame.resize((256, 256), Image.NEAREST)
                frame_resized.save(path)

            frame_count += 1

    print(f"\n✅ Test complete! {frame_count} frames rendered")
    print(f"   Sample frames saved to: {output_dir}")
    print(f"   Total emotions tested: {len(emotions)}")
    print(f"   Mode: RGB color (for ST7735 TFT)")
