"""
Procedural Eyes Engine - BlockyEye Version
Adapted from user script for Dual SPI Displays.
"""

import math
import time
import random
from PIL import Image, ImageDraw

# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
EYE_COLOR = (255, 255, 255) # White
BG_COLOR = (0, 0, 0)      # Black
EYE_SIZE = 120           # Base size
FLOOR_Y = SCREEN_HEIGHT - 5

# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 8.0
BLINK_SPEED_MAX = 12.0

# Thinking: Hover between top-left and top-right corners
THINKING_PHASES = [
    {"x": 0.22, "y": 0.25, "dur": (1.5, 2.5)},  # Top-Left
    {"x": 0.78, "y": 0.25, "dur": (1.5, 2.5)},  # Top-Right
]

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":  {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True, "pos": (0.0, 0.0)},
    "joy":   {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0,  "mirror_angle": True, "pos": (0.0, -15.0)},
    "excited": {"scale_w": 0.9, "scale_h": 1.3, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0), "behavior": "bounce"},
    "amused": {"scale_w": 1.0, "scale_h": 0.8, "top_lid": 0.0, "bottom_lid": 0.2, "lid_angle": 0.0, "mirror_angle": True, "pos": (15.0, -15.0), "asym": True},
    "friendly": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0)},
    "proud": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, -25.0)},
    "sad":   {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.4,  "bottom_lid": 0.0,  "lid_angle": -12.0,  "mirror_angle": True, "pos": (0.0, 20.0)},
    "lonely": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (-30.0, 30.0)},
    "bored": {"scale_w": 1.1, "scale_h": 0.8, "top_lid": 0.45, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0), "behavior": "drift"},
    "tired": {"scale_w": 1.1, "scale_h": 0.9, "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 15.0), "behavior": "sink"},
    "disappointed": {"scale_w": 1.0, "scale_h": 1.1, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 30.0), "anticipate": True},
    "thinking": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.1, "lid_angle": 0.0, "mirror_angle": True, "pos": (25.0, -25.0)},
    "confused": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "asym": True, "pos": (0.0, 0.0), "behavior": "jitter"},
    "curious": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0), "behavior": "lean"},
    "concentrating": {"scale_w": 1.2, "scale_h": 0.4, "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0)},
    "remembering": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (-25.0, -25.0), "behavior": "saccades"},
    "surprised": {"scale_w": 0.8, "scale_h": 1.25, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, 0.0)},
    "skeptical": {"scale_w": 1.05, "scale_h": 0.7, "top_lid": 0.0, "bottom_lid": 0.4, "lid_angle": 0.0, "mirror_angle": True, "pos": (30.0, 0.0)},
    "angry": {"scale_w": 1.0, "scale_h": 0.85, "top_lid": 0.45, "bottom_lid": 0.0,  "lid_angle": 15.0, "mirror_angle": True, "pos": (0.0, 15.0)},
    "shy": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.2, "bottom_lid": 0.0, "lid_angle": 4.0, "mirror_angle": True, "pos": (-25.0, 25.0)},
    "glitch": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "behavior": "glitch"},
    # --- Aliases for Agent.py Compatibility ---
    "happy": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0,  "mirror_angle": True, "pos": (0.0, -15.0)}, # Alias for joy
    "talking": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0,  "mirror_angle": True, "pos": (0.0, -15.0)}, # Alias for happy/joy
    "idle1": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True, "pos": (0.0, 0.0)},   # Alias for idle
    "idle2": {"scale_w": 1.02, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True, "pos": (0.0, -5.0)}, # Listening state
}

class BlockyEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]

        self.vel_x = 0.0
        self.vel_y = 0.0

        self.base_w = EYE_SIZE * scale
        self.base_h = EYE_SIZE * scale

        self.current_w = self.base_w
        self.current_h = self.base_h
        self.target_w = self.base_w
        self.target_h = self.base_h

        self.vel_w = 0.0
        self.vel_h = 0.0

        self.w = self.base_w
        self.h = self.base_h

        self.current_rotation = 0.0
        self.target_rotation = 0.0
        self.rot_sensitivity = random.uniform(0.3, 0.5)
        self.rot_speed = random.uniform(0.15, 0.25)

        self.is_left = is_left
        # Blink state
        self.blink_state = "IDLE"
        self.vy = 0
        self.blink_speed_mult = 1.0
        self.saccade_pending = False  # Will shift position mid-blink
        self.saccade_offset = [0.0, 0.0]  # Random shift during blink

        self.target_scale_w = 1.0
        self.target_scale_h = 1.0
        self.scale_w = 1.0
        self.scale_h = 1.0
        self.scale_w_vel = 0.0
        self.scale_h_vel = 0.0
        self.top_lid = 0.0
        self.bottom_lid = 0.0
        self.lid_angle = 0.0
        self.top_lid_vel = 0.0
        self.bottom_lid_vel = 0.0
        self.lid_angle_vel = 0.0
        self.target_top_lid = 0.0
        self.target_bottom_lid = 0.0
        self.target_lid_angle = 0.0
        self.current_emotion = "idle"
        self.happy_phase = random.uniform(0.0, math.pi * 2)
        self.happy_burst_until = 0.0

        self.noise_t = random.uniform(0, 100)
        
        # Thinking animation state
        self.thinking_phase = 0.0
        
        # Happy hop state: occasional left/right jump with vertical bounce
        self.happy_jump_x = 0.0       # Current hop X offset
        self.happy_jump_y = 0.0       # Current hop Y offset (upward pop)
        self.next_happy_jump = 0.0    # When to next hop
        
        # Speech reactivity
        self.speech_amplitude = 0.0  # 0.0 to 1.0
        
        # Advanced Animation State
        self.anticipation_timer = 0.0
        self.anticipation_dip = [0.0, 0.0]
        self.asym_delay = 0.0 if is_left else random.uniform(0.02, 0.06)
        
        self.jitter_x = 0.0
        self.jitter_y = 0.0
        self.noise_t = random.uniform(0, 1000)
        
        # --- Behavioral State ---
        self.emotion_queue = []
        self.decay_timer = 0.0
        self.blink_shift_pending = None
        self.shared_saccade_offset = [0.0, 0.0]

    def start_blink(self, speed_mult=None, saccade=False):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            if speed_mult is not None:
                self.blink_speed_mult = speed_mult
            else:
                self.blink_speed_mult = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            self.vy = 40 * self.blink_speed_mult
            self.saccade_pending = saccade
            if saccade:
                self.saccade_offset = [
                    random.uniform(-18, 18),
                    random.uniform(-12, 12)
                ]

    def set_emotion(self, emotion_name: str, duration=None, chain=None, blink_shift=False):
        if emotion_name not in EMOTION_PRESETS:
            return
            
        # Blink-Shift: wait for blink closure to swap emotion
        if blink_shift:
            self.blink_shift_pending = {"name": emotion_name, "duration": duration, "chain": chain}
            self.start_blink()
            return

        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        
        # Chaining
        if chain:
            self.emotion_queue = chain if isinstance(chain, list) else [chain]

        # Decay (Emotional Gravity)
        if duration:
            self.decay_timer = time.time() + duration
        elif emotion_name != "idle":
            self.decay_timer = time.time() + 3.0 # Default 3s for non-idle
        else:
            self.decay_timer = 0.0

        # Check for anticipation
        needs_anticipation = preset.get("anticipate") or abs(preset.get("pos", (0,0))[1]) > 20 or preset.get("scale_h", 1) > 1.3
        if needs_anticipation:
            self.anticipation_timer = time.time() + 0.12
            self.anticipation_dip = [0.0, 6.0]

        self.target_scale_w = preset["scale_w"]
        self.target_scale_h = preset["scale_h"]
        self.target_top_lid = preset["top_lid"]
        self.target_bottom_lid = preset["bottom_lid"]
        
        # Mirror angle logic
        angle = preset["lid_angle"]
        if not self.is_left and preset.get("mirror_angle", True):
            if abs(angle) > 0.1:
                angle = -angle
        self.target_lid_angle = angle

    def update(self):
        if self.blink_state == "IDLE":
            now = time.time()
            t = now + self.noise_t
            
            # --- Autonomous Behaviors ---
            
            # 1. Decay (Emotional Gravity)
            if self.decay_timer > 0 and now > self.decay_timer:
                if self.emotion_queue:
                    next_emo = self.emotion_queue.pop(0)
                    self.set_emotion(next_emo)
                else:
                    self.set_emotion("idle")

            # 2. Saccadic Scanning (Idle only)
            # Now handled externally by ProceduralEyeDisplay for symmetry
            
            # 3. Micro-Saccades (High frequency jitter)
            jitter_amp = 0.3
            if self.current_emotion == "remembering": jitter_amp = 0.8
            self.jitter_x = (math.sin(t * 30.0) * jitter_amp + math.sin(t * 15.0) * 0.2) + self.shared_saccade_offset[0]
            self.jitter_y = (math.cos(t * 22.0) * jitter_amp) + self.shared_saccade_offset[1]

            # 2. Continuous Behaviors
            behavior = EMOTION_PRESETS[self.current_emotion].get("behavior")
            b_off_x, b_off_y = 0.0, 0.0
            
            if behavior == "bounce":
                b_off_y = math.sin(now * 12.0) * 4.5
            elif behavior == "drift":
                b_off_x = math.sin(now * 1.5) * 15.0
            elif behavior == "sink":
                # Slow sink then snap back
                phase = (now * 0.6) % 1.0
                if phase < 0.85: b_off_y = phase * 12.0
                else: b_off_y = (1.0 - phase) * -6.0
            elif behavior == "jitter":
                b_off_x = random.uniform(-2, 2)
                b_off_y = random.uniform(-2, 2)
            elif behavior == "lean":
                lean_dir = 1.0 if self.is_left else -1.0
                b_off_x = lean_dir * 10.0
            elif behavior == "glitch":
                if random.random() < 0.08:
                    b_off_x = random.uniform(-50, 50)
                    b_off_y = random.uniform(-50, 50)
            elif behavior == "saccades":
                if random.random() < 0.15:
                    self.jitter_x = random.uniform(-4, 4)
                    self.jitter_y = random.uniform(-4, 4)

            # 3. Anticipation Dip
            anticipate_x, anticipate_y = 0.0, 0.0
            if now < self.anticipation_timer:
                anticipate_y = self.anticipation_dip[1]

            preset_pos = EMOTION_PRESETS[self.current_emotion].get("pos", (0, 0))
            
            # Gaze Offset from face tracking (target_pos) + Emotion Offset
            target_x_phys = self.target_pos[0] + preset_pos[0] + b_off_x + self.jitter_x + anticipate_x
            target_y_phys = self.target_pos[1] + preset_pos[1] + b_off_y + self.jitter_y + anticipate_y

            # Tighten Screen Clamping (account for rotation & padding)
            angle_rad = math.radians(abs(self.current_rotation))
            eff_half_w = (self.target_w * math.cos(angle_rad) + self.target_h * math.sin(angle_rad)) / 2
            eff_half_h = (self.target_w * math.sin(angle_rad) + self.target_h * math.cos(angle_rad)) / 2
            
            pad = 10 
            target_x_phys = max(eff_half_w + pad, min(SCREEN_WIDTH - eff_half_w - pad, target_x_phys))
            target_y_phys = max(eff_half_h + pad, min(SCREEN_HEIGHT - eff_half_h - pad, target_y_phys))

            # 4. Spring Physics (Higher Friction / Less Bounce)
            k_pos = 0.07 # Softer
            d_pos_x = 0.72 # Higher friction
            self.vel_x = (self.vel_x + (target_x_phys - self.current_pos[0]) * k_pos) * d_pos_x

            # Gravity Feel + Vertical Resistance
            d_pos_y = 0.85
            if self.vel_y < 0: d_pos_y = 0.72 # Higher friction moving up
            else: d_pos_y = 0.82 # More friction moving down

            # Subtle gravity pull bias
            self.vel_y = (self.vel_y + (target_y_phys + 1.5 - self.current_pos[1]) * k_pos) * d_pos_y

            self.current_pos[0] += self.vel_x
            self.current_pos[1] += self.vel_y

            # Hard constraint on current_pos to prevent any visual overflow
            self.current_pos[0] = max(eff_half_w + 2, min(SCREEN_WIDTH - eff_half_w - 2, self.current_pos[0]))
            self.current_pos[1] = max(eff_half_h + 2, min(SCREEN_HEIGHT - eff_half_h - 2, self.current_pos[1]))

            # Rotation
            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.4 + rel_y * 0.6) * self.rot_sensitivity
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            # Shape Springs (Lower Bounce)
            k_shape = 0.20
            d_shape = 0.65
            
            # Voice Pulse (Audio Reactivity)
            pulse_scale = self.speech_amplitude * 0.25
            pulse_squish = self.speech_amplitude * 0.4
            
            self.scale_w_vel = (self.scale_w_vel + (self.target_scale_w - self.scale_w) * k_shape) * d_shape
            self.scale_h_vel = (self.scale_h_vel + ((self.target_scale_h + pulse_scale) - self.scale_h) * k_shape) * d_shape
            self.scale_w += self.scale_w_vel
            self.scale_h += self.scale_h_vel

            # Update final dimensions for draw()
            self.w = self.base_w * self.scale_w
            self.h = self.base_h * self.scale_h

            self.top_lid_vel = (self.top_lid_vel + (self.target_top_lid - self.top_lid) * k_shape) * d_shape
            
            # Speech Reactivity: add extra squint to bottom lid
            active_bottom_lid = self.target_bottom_lid
            if self.speech_amplitude > 0.05:
                # Only apply to emotions where it makes sense (open-ish bottom lids)
                if self.target_bottom_lid < 0.4:
                    active_bottom_lid = min(0.45, self.target_bottom_lid + pulse_squish)

            self.bottom_lid_vel = (self.bottom_lid_vel + (active_bottom_lid - self.bottom_lid) * k_shape) * d_shape
            self.lid_angle_vel = (self.lid_angle_vel + (self.target_lid_angle - self.lid_angle) * k_shape) * d_shape

            self.top_lid += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle += self.lid_angle_vel

            self.target_w = (self.base_w * self.scale_w)
            self.target_h = (self.base_h * self.scale_h)

        elif self.blink_state == "DROPPING":
            self.vy += 10 * self.blink_speed_mult
            self.current_pos[1] += self.vy
            self.current_w = self.base_w - 10
            self.current_h = self.base_h + 20
            self.target_w = self.current_w
            self.target_h = self.current_h

            if self.current_pos[1] + self.current_h // 2 >= FLOOR_Y:
                self.current_pos[1] = FLOOR_Y - self.current_h // 2
                self.blink_state = "SQUASHING"
                self.velocity = [0.0, 0.0]

        elif self.blink_state == "SQUASHING":
            squeeze_speed = 65 * self.blink_speed_mult
            spread_speed = 40 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2

            if self.current_h <= 22:
                self.current_h = 22
                
                # Blink-Shift: swap emotion while eyes are closed
                if self.blink_shift_pending:
                    p = self.blink_shift_pending
                    self.blink_shift_pending = None
                    self.set_emotion(p["name"], duration=p["duration"], chain=p["chain"], blink_shift=False)

                if self.saccade_pending:
                    self.target_pos[0] = self.base_x + self.saccade_offset[0]
                    self.target_pos[1] = self.base_y + self.saccade_offset[1]
                    self.current_pos[0] = self.target_pos[0]
                    self.saccade_pending = False
                self.blink_state = "JUMPING"

        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.15, min(0.95, 0.85 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed

            self.vel_x = (self.vel_x + (self.target_pos[0] - self.current_pos[0]) * 0.1) * 0.8
            self.current_pos[0] += self.vel_x

            target_y = self.target_pos[1]
            self.current_pos[1] += (target_y - self.current_pos[1]) * 0.8

            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.blink_state = "IDLE"
                self.vy = 0
                self.vel_x = 0
                self.vel_y = 0

        self.w = self.current_w
        self.h = self.current_h

    def draw_radial_rect(self, draw, x, y, w, h, color, radius, pupil_offset=(0,0)):
        center_x = x + w/2
        center_y = y + h/2
        steps = 4
        for i in range(steps):
            size_factor = 1.0 - (i / steps)
            current_w = w * size_factor
            current_h = h * size_factor
            if current_w <= 0 or current_h <= 0:
                continue
            b_factor = 0.85 + 0.15 * (i / steps)
            cur_color = (int(color[0] * b_factor), int(color[1] * b_factor), int(color[2] * b_factor))
            shift_x = pupil_offset[0] * (1.0 - size_factor) * 15
            shift_y = pupil_offset[1] * (1.0 - size_factor) * 15
            cx = center_x + shift_x
            cy = center_y + shift_y
            x0 = cx - current_w / 2
            y0 = cy - current_h / 2
            x1 = cx + current_w / 2
            y1 = cy + current_h / 2
            base_radius = int(radius)
            cur_radius = min(base_radius, int(min(current_w, current_h) / 2))
            draw.rounded_rectangle([x0, y0, x1, y1], radius=cur_radius, fill=cur_color)

    def draw_eyelids(self, eye_img, rect):
        """Draw eyelids as clean filled rects that clip the eye shape."""
        if self.top_lid < 0.01 and self.bottom_lid < 0.01:
            return

        from PIL import ImageDraw
        x0, y0, x1, y1 = rect
        w = x1 - x0
        h = y1 - y0
        eye_draw = ImageDraw.Draw(eye_img)
        lid_color = (0, 0, 0, 255)  # Explicit RGBA — no alpha guessing

        # 1. Top Lid — solid rect from top of eye downward
        if self.top_lid > 0.01:
            lid_h = h * self.top_lid
            tilt = self.lid_angle * 0.3
            # Flat rect that covers lid_h pixels from top of the eye
            eye_draw.rectangle(
                [x0 - 3, y0 - 3, x1 + 3, y0 + lid_h],
                fill=lid_color
            )
            # For tilted emotions: add an extra wedge on one side to angle the inner edge
            if abs(tilt) > 0.5:
                if tilt > 0:  # Right side higher, so cut a triangle from bottom-right
                    eye_draw.polygon([
                        (x1 - w * 0.5, y0 + lid_h),
                        (x1 + 3, y0 + lid_h),
                        (x1 + 3, y0 + lid_h - tilt * 2),
                    ], fill=lid_color)
                else:  # Left side higher
                    eye_draw.polygon([
                        (x0 - 3, y0 + lid_h),
                        (x0 + w * 0.5, y0 + lid_h),
                        (x0 - 3, y0 + lid_h + tilt * 2),
                    ], fill=lid_color)

        # 2. Bottom Lid — solid rect from bottom of eye upward
        if self.bottom_lid > 0.01:
            lid_h = h * self.bottom_lid
            eye_draw.rectangle(
                [x0 - 3, y1 - lid_h, x1 + 3, y1 + 3],
                fill=lid_color
            )

    def draw(self):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))

        eye_img_size = int(max(self.base_w, self.base_h) * 2.0) # Reduced from 2.5
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        radius_factor = 0.36 if self.current_emotion == "thinking" else 0.25
        base_radius = int(min(self.base_w, self.base_h) * radius_factor)
        corner_radius = min(base_radius, int(min(draw_w, draw_h) / 2))
        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))

        cx, cy = eye_img_size / 2, eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2

        # Draw the eye shape only (NO lid on transparent canvas)
        self.draw_radial_rect(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))

        # Use BILINEAR for faster rotation than BICUBIC
        rotated = eye_img.rotate(self.current_rotation, resample=Image.BILINEAR, expand=False)

        # Create final frame as RGB directly to save a conversion later
        final_frame = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        
        # Masked paste is often faster than alpha_composite for full frames
        final_frame.paste(rotated, (paste_x, paste_y), rotated)

        # Draw eyelids DIRECTLY on final_frame in screen coordinates.
        if self.top_lid > 0.01 or self.bottom_lid > 0.01:
            fd_draw = ImageDraw.Draw(final_frame)
            sy = int(self.current_pos[1])
            hh = draw_h // 2
            LID_COLOR = (0, 0, 0) # RGB is fine now

            if self.top_lid > 0.01:
                lid_h = int(draw_h * self.top_lid)
                fd_draw.rectangle(
                    [0, 0, SCREEN_WIDTH, sy - hh + lid_h],
                    fill=LID_COLOR
                )

            if self.bottom_lid > 0.01:
                lid_h = int(draw_h * self.bottom_lid)
                fd_draw.rectangle(
                    [0, sy + hh - lid_h, SCREEN_WIDTH, SCREEN_HEIGHT],
                    fill=LID_COLOR
                )

        return final_frame

# --- Main Controller Class ---
class ProceduralEyeDisplay:
    def __init__(self):
        center_x = SCREEN_WIDTH / 2
        center_y = SCREEN_HEIGHT / 2
        self.left_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=True)
        self.right_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=False)
        
        self.next_blink_time = time.time() + random.uniform(3, 6)
        
        # Thinking phase state machine
        self.thinking_phase_idx = -1
        self.thinking_phase_end = 0.0
        
        self.target_x_off = 0.0
        self.target_y_off = 0.0
        self.smoothed_x_off = 0.0
        self.smoothed_y_off = 0.0
        self.target_face_rotation = 0.0
        self.smoothed_face_rotation = 0.0
        
        # Shared Saccade (Parallel scanning)
        self.saccade_timer = 0.0
        self.shared_saccade_offset = [0.0, 0.0]

    def set_emotion(self, emotion_name: str, duration=None, chain=None, blink_shift=False):
        if emotion_name not in EMOTION_PRESETS:
            return
        
        self.left_eye.set_emotion(emotion_name, duration=duration, chain=chain, blink_shift=blink_shift)
        self.right_eye.set_emotion(emotion_name, duration=duration, chain=chain, blink_shift=blink_shift)
        
        if emotion_name == "thinking":
            self.thinking_phase_idx = -1
            self.thinking_phase_end = time.time()

    def set_face_target(self, x, y, rotation=0.0):
        """
        Set target from face tracking.
        x, y should be normalized (-1.0 to 1.0)
        """
        MAX_X_OFFSET = 50
        MAX_Y_OFFSET = 35
        
        # Target offsets
        self.target_x_off = x * MAX_X_OFFSET
        self.target_y_off = y * MAX_Y_OFFSET
        self.target_face_rotation = rotation

    def render_frame(self, dt: float, mono: bool = False):
        now = time.time()
        
        # 1. Shared Saccadic Scanning (Idle only)
        # Calculate a single offset for both eyes so they move in parallel
        if self.left_eye.current_emotion in ("idle", "idle1"):
            if now > self.saccade_timer:
                self.shared_saccade_offset = [random.uniform(-12, 12), random.uniform(-8, 8)]
                self.saccade_timer = now + random.uniform(2.5, 5.0)
        else:
            self.shared_saccade_offset = [0.0, 0.0]

        # 2. Trigger Blinks ONLY during Idle state
        if now > self.next_blink_time:
            if self.left_eye.current_emotion in ("idle", "idle1"):
                blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
                do_saccade = (random.random() < 0.30)
                self.left_eye.start_blink(blink_speed, saccade=do_saccade)
                self.right_eye.start_blink(blink_speed, saccade=do_saccade)
            self.next_blink_time = now + random.uniform(3.5, 7.0)

        # Thinking: hover between corners
        if self.left_eye.current_emotion == "thinking":
            now = time.time()
            if now >= self.thinking_phase_end:
                self.thinking_phase_idx = (self.thinking_phase_idx + 1) % len(THINKING_PHASES)
                phase = THINKING_PHASES[self.thinking_phase_idx]
                self.thinking_phase_end = now + random.uniform(*phase["dur"])

            phase = THINKING_PHASES[self.thinking_phase_idx]
            target_x = SCREEN_WIDTH * phase["x"]
            target_y = SCREEN_HEIGHT * phase["y"]
            for eye in (self.left_eye, self.right_eye):
                eye.target_pos[0] = target_x
                eye.target_pos[1] = target_y

        smooth_alpha = 0.15
        self.smoothed_x_off += (self.target_x_off - self.smoothed_x_off) * smooth_alpha
        self.smoothed_y_off += (self.target_y_off - self.smoothed_y_off) * smooth_alpha
        self.smoothed_face_rotation += (self.target_face_rotation - self.smoothed_face_rotation) * 0.1
        
        # Update eyes
        for eye in (self.left_eye, self.right_eye):
            # Sync the shared saccade offset
            eye.shared_saccade_offset = self.shared_saccade_offset
            
            # Apply face tracking ONLY during idle states
            if eye.blink_state == "IDLE" and not eye.saccade_pending:
                if eye.current_emotion in ("idle", "idle1"):
                    eye.target_pos[0] += (eye.base_x + self.smoothed_x_off - eye.target_pos[0]) * 0.015
                    eye.target_pos[1] += (eye.base_y + self.smoothed_y_off - eye.target_pos[1]) * 0.015
                    eye.target_rotation = self.smoothed_face_rotation
                else:
                    # Snappy return to center for other emotions
                    eye.target_pos[0] += (eye.base_x - eye.target_pos[0]) * 0.15
                    eye.target_pos[1] += (eye.base_y - eye.target_pos[1]) * 0.15
                    eye.target_rotation = 0.0
            eye.update()
            
        # Render
        img_l = self.left_eye.draw()
        img_r = self.right_eye.draw()
        
        if mono:
            return img_l.convert('1') # Fallback for OLED if needed
            
        return (img_l, img_r)
