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
BLINK_SPEED_MIN = 8.0   # Was 3.5 — now much snappier
BLINK_SPEED_MAX = 12.0  # Was 5.0

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":  {"scale_w": 1.0, "scale_h": 1.0,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},
    "idle1": {"scale_w": 1.0, "scale_h": 1.0,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},  # Alias
    "idle2": {"scale_w": 1.0, "scale_h": 1.02, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},  # Listening: barely wider, very still
    "happy": {"scale_w": 1.0,  "scale_h": 0.82, "top_lid": 0.0, "bottom_lid": 0.42, "lid_angle": 0.0, "mirror_angle": True},  # Squint-smile, safe scale
    "sad":   {"scale_w": 1.1, "scale_h": 1.1,  "top_lid": 0.4,  "bottom_lid": 0.0,  "lid_angle": 18.0,  "mirror_angle": True},
    "angry": {"scale_w": 1.0, "scale_h": 0.85, "top_lid": 0.45, "bottom_lid": 0.0,  "lid_angle": -22.0, "mirror_angle": True},
    "surprised": {"scale_w": 0.9, "scale_h": 1.5, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.1, "scale_h": 0.55, "top_lid": 0.45, "bottom_lid": 0.45, "lid_angle": 0.0, "mirror_angle": True},
    "sleepy": {"scale_w": 1.1, "scale_h": 1.0,  "top_lid": 0.65, "bottom_lid": 0.0,  "lid_angle": 0.0,  "mirror_angle": True},
    "looking": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.28, "bottom_lid": 0.0,  "lid_angle": -8.0, "mirror_angle": False},
    "thinking": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.1, "lid_angle": 0.0,  "mirror_angle": True},  # Squint-think
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
        
        # Thinking animation state (shared value set by ProceduralEyeDisplay for sync)
        self.thinking_gaze_x = 18.0    # Set externally to keep both eyes in sync
        self.thinking_gaze_y = 0.0
        self.thinking_look_up = -8.0   # Slight upward gaze
        
        # Happy hop state: occasional left/right jump with vertical bounce
        self.happy_jump_x = 0.0       # Current hop X offset
        self.happy_jump_y = 0.0       # Current hop Y offset (upward pop)
        self.next_happy_jump = 0.0    # When to next hop
        
        # Speech reactivity
        self.speech_amplitude = 0.0  # 0.0 to 1.0

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

    def set_emotion(self, emotion_name: str, intensity: float = 1.0):
        if emotion_name not in EMOTION_PRESETS:
            return

        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = time.time() + 0.35

        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        idle = EMOTION_PRESETS["idle"]

        intensity = max(0.0, min(1.0, intensity))
        scale_w = idle["scale_w"] + (preset["scale_w"] - idle["scale_w"]) * intensity
        scale_h = idle["scale_h"] + (preset["scale_h"] - idle["scale_h"]) * intensity
        top_lid = idle["top_lid"] + (preset["top_lid"] - idle["top_lid"]) * intensity
        bottom_lid = idle["bottom_lid"] + (preset["bottom_lid"] - idle["bottom_lid"]) * intensity
        lid_angle = idle["lid_angle"] + (preset["lid_angle"] - idle["lid_angle"]) * intensity

        self.target_scale_w = scale_w
        self.target_scale_h = scale_h
        self.target_top_lid = top_lid
        self.target_bottom_lid = bottom_lid
        # No mirroring - both eyes tilt same direction
        self.target_lid_angle = lid_angle

    def update(self):
        if self.blink_state == "IDLE":
            t = time.time() + self.noise_t
            # Larger noise = more visible idle movement
            noise_x = (math.sin(t * 1.3) * 2.5 + math.sin(t * 0.7) * 1.5)
            noise_y = (math.cos(t * 1.1) * 2.0 + math.cos(t * 0.9) * 1.2)

            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y

            burst_active = time.time() < self.happy_burst_until
            if burst_active:
                # Gentle pop upward — no lid slamming shut
                target_y_phys -= 10.0

            # idle2 (listening): go very still, slight upward drift — attentive gaze
            if self.current_emotion == "idle2":
                noise_x *= 0.1  # Almost completely still
                noise_y *= 0.1
                target_y_phys -= 4.0  # Slight upward attentive look

            if self.current_emotion == "happy":
                # Occasional left/right hop with vertical bounce
                now = time.time()
                if now > self.next_happy_jump:
                    self.happy_jump_x = random.choice([-13.0, 13.0])
                    self.happy_jump_y = -10.0  # Pop upward on hop
                    self.next_happy_jump = now + random.uniform(2.0, 4.0)
                # Spring decay back toward 0
                self.happy_jump_x *= 0.88
                self.happy_jump_y *= 0.82  # Slightly faster decay for snappy bounce
                target_x_phys += self.happy_jump_x
                target_y_phys += self.happy_jump_y
                # Lower resting position slightly (push down)
                target_y_phys += 6.0
                
            # Thinking: gaze driven by ProceduralEyeDisplay target_pos directly — skip here

            # Clamp so eye never leaves screen
            half_w = self.base_w * self.scale_w * 0.5
            half_h = self.base_h * self.scale_h * 0.5
            target_x_phys = max(half_w + 4, min(SCREEN_WIDTH - half_w - 4, target_x_phys))
            target_y_phys = max(half_h + 4, min(SCREEN_HEIGHT - half_h - 4, target_y_phys))

            # Speech Reactivity: only squint bottom lid with amplitude (no scale change)
            if self.current_emotion == "happy" and self.speech_amplitude > 0.05:
                squint = self.speech_amplitude * 0.20
                preset_lid = EMOTION_PRESETS["happy"]["bottom_lid"]
                self.target_bottom_lid = min(preset_lid + squint, preset_lid + 0.15)

            dx = target_x_phys - self.current_pos[0]
            dy = target_y_phys - self.current_pos[1]

            speed_x = 0.20
            speed_y = 0.22
            if dy < -1.0:
                speed_y = 0.14
            elif dy > 1.0:
                speed_y = 0.38

            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y

            self.vel_x = dx * speed_x
            self.vel_y = dy * speed_y

            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            if self.current_emotion == "happy":
                look_rot += math.sin(time.time() * 8.0 + self.happy_phase) * 1.2
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)

            stretch_mult = 1.0 if self.current_emotion != "happy" else 0.4
            move_stretch_x = (dx * speed_x) * 2.5 * stretch_mult
            move_stretch_y = (dy * speed_y) * 2.5 * stretch_mult

            k = 0.45  # Higher = faster transitions (was 0.22)
            d = 0.65  # Damping (was 0.55)
            self.scale_w_vel = (self.scale_w_vel + (self.target_scale_w - self.scale_w) * k) * d
            self.scale_h_vel = (self.scale_h_vel + (self.target_scale_h - self.scale_h) * k) * d
            self.scale_w += self.scale_w_vel
            self.scale_h += self.scale_h_vel

            self.top_lid_vel = (self.top_lid_vel + (self.target_top_lid - self.top_lid) * k) * d
            self.bottom_lid_vel = (self.bottom_lid_vel + (self.target_bottom_lid - self.bottom_lid) * k) * d
            self.lid_angle_vel = (self.lid_angle_vel + (self.target_lid_angle - self.lid_angle) * k) * d

            self.top_lid += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle += self.lid_angle_vel

            self.target_w = (self.base_w * self.scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.scale_h) + breath_h - (move_stretch_y * 0.2)

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

        if self.blink_state == "IDLE":
            k = 0.08
            d = 0.90
            force_w = (self.target_w - self.current_w) * k
            self.vel_w = (self.vel_w + force_w) * d
            self.current_w += self.vel_w

            force_h = (self.target_h - self.current_h) * k
            self.vel_h = (self.vel_h + force_h) * d
            self.current_h += self.vel_h
        else:
            self.vel_w = 0
            self.vel_h = 0

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
        from PIL import ImageFilter
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR
        blur_r = 5  # Soften lid edge to match eye's rounded shape

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_src = Image.new("RGBA", (int(w + 40), int(lid_h + 20)), (*lid_color, 255))
            lid_src = lid_src.filter(ImageFilter.GaussianBlur(radius=blur_r))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 10)
            paste_x = max(0, lid_x)
            paste_y = max(0, lid_y)
            eye_img.alpha_composite(lid_src, (paste_x, paste_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = Image.new("RGBA", (int(w + 40), int(lid_h + 20)), (*lid_color, 255))
            lid_src = lid_src.filter(ImageFilter.GaussianBlur(radius=blur_r))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 10 - lid_src.height)
            paste_x = max(0, lid_x)
            paste_y = max(0, lid_y)
            eye_img.alpha_composite(lid_src, (paste_x, paste_y))

    def draw(self):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))

        # We create the full screen image here
        # Optimization: Create ONE image and return it
        eye_img_size = int(max(self.base_w, self.base_h) * 2.5)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        # Thinking gets slightly rounder corners; all others keep original shape
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

        self.draw_radial_rect(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, corner_radius, (off_x, off_y))
        self.draw_eyelids(eye_img, (x0, y0, x1, y1))

        rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC, expand=False)
        
        # Create final frame
        final_frame = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        final_frame.alpha_composite(rotated, (paste_x, paste_y))
        return final_frame

# --- Main Controller Class ---
class ProceduralEyeDisplay:
    def __init__(self):
        center_x = SCREEN_WIDTH / 2
        center_y = SCREEN_HEIGHT / 2
        self.left_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=True)
        self.right_eye = BlockyEye(center_x, center_y, scale=1.0, is_left=False)
        
        self.next_blink_time = time.time() + random.uniform(3, 6)
        
        # Shared thinking gaze state (drives both eyes in sync)
        self.thinking_gaze_x = 18.0
        self.thinking_gaze_y = 0.0
        self.next_thinking_shift = time.time() + random.uniform(2.0, 3.5)
        
        self.target_x_off = 0.0
        self.target_y_off = 0.0
        self.smoothed_x_off = 0.0
        self.smoothed_y_off = 0.0

    def set_emotion(self, emotion_name: str):
        # Map some common aliases to presets
        if emotion_name == "idle": emotion_name = "idle1"
        
        # Talking -> happy/active
        if emotion_name == "talking": emotion_name = "happy"
        
        self.left_eye.set_emotion(emotion_name)
        self.right_eye.set_emotion(emotion_name)
        
        # Reset thinking gaze timer so switching starts immediately
        if emotion_name == "thinking":
            self.thinking_gaze_x = 18.0  # Start right
            self.next_thinking_shift = time.time()  # Fire immediately on first frame

    def set_face_target(self, x, y):
        """
        Set target from face tracking.
        x, y should be normalized (-1.0 to 1.0)
        """
        MAX_X_OFFSET = 50
        MAX_Y_OFFSET = 35
        
        # Target offsets
        self.target_x_off = x * MAX_X_OFFSET
        self.target_y_off = y * MAX_Y_OFFSET

    def render_frame(self, dt: float, mono: bool = False):
        # Suppress blinks during happy and thinking — those states have their own animations
        is_animated = (self.left_eye.current_emotion in ("happy", "thinking"))
        if time.time() > self.next_blink_time:
            if not is_animated:
                blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
                do_saccade = (random.random() < 0.30)
                self.left_eye.start_blink(blink_speed, saccade=do_saccade)
                self.right_eye.start_blink(blink_speed, saccade=do_saccade)
            self.next_blink_time = time.time() + random.uniform(3.5, 7.0)

        # Thinking: pin eyes hard to one side — never return to center
        if self.left_eye.current_emotion == "thinking":
            now = time.time()
            if now > self.next_thinking_shift:
                self.thinking_gaze_x = -self.thinking_gaze_x  # Flip side
                self.next_thinking_shift = now + random.uniform(0.4, 1.0)
            # Calculate hard side position (near screen edge)
            side_x = SCREEN_WIDTH * (0.78 if self.thinking_gaze_x > 0 else 0.22)
            look_up_y = SCREEN_HEIGHT * 0.38  # Slightly above center
            for eye in (self.left_eye, self.right_eye):
                eye.target_pos[0] = side_x
                eye.target_pos[1] = look_up_y
            
        # Smooth tracking (from face monitor)
        smooth_alpha = 0.15
        self.smoothed_x_off += (self.target_x_off - self.smoothed_x_off) * smooth_alpha
        self.smoothed_y_off += (self.target_y_off - self.smoothed_y_off) * smooth_alpha
        
        # Update eyes
        for eye in (self.left_eye, self.right_eye):
            # During thinking, target_pos is driven directly above — skip normal drift
            if eye.blink_state == "IDLE" and not eye.saccade_pending and eye.current_emotion != "thinking":
                eye.target_pos[0] += (eye.base_x + self.smoothed_x_off - eye.target_pos[0]) * 0.015
                eye.target_pos[1] += (eye.base_y + self.smoothed_y_off - eye.target_pos[1]) * 0.015
            eye.update()
            
        # Render
        img_l = self.left_eye.draw().convert("RGB")
        img_r = self.right_eye.draw().convert("RGB")
        
        if mono:
            return img_l.convert('1') # Fallback for OLED if needed
            
        return (img_l, img_r)
