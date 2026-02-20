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

# Blink Speed (Higher = Faster close/open)
BLINK_CLOSE_SPEED = 0.35  # Fraction of height per frame to close
BLINK_OPEN_SPEED  = 0.28  # Fraction of height per frame to open

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":  {"scale_w": 1.0, "scale_h": 1.0,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},
    "idle1": {"scale_w": 1.0, "scale_h": 1.0,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},  # Alias
    "idle2": {"scale_w": 1.0, "scale_h": 1.3,  "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0,   "mirror_angle": True},  # Listening: wide eyes
    "happy": {"scale_w": 1.2, "scale_h": 0.65, "top_lid": 0.0,  "bottom_lid": 0.5,  "lid_angle": -8.0,  "mirror_angle": True},  # Squint-smile
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
        
        # Thinking animation state
        self.thinking_phase = 0.0
        
        # Speech reactivity
        self.speech_amplitude = 0.0  # 0.0 to 1.0

    def start_blink(self, saccade=False):
        if self.blink_state == "IDLE":
            self.blink_state = "CLOSING"
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
                target_y_phys -= 8.0
                self.target_top_lid = max(self.target_top_lid, 0.9)
                self.target_bottom_lid = max(self.target_bottom_lid, 0.9)
                self.target_lid_angle = 0.0

            if self.current_emotion == "happy":
                ht = time.time() * 6.0 + self.happy_phase
                target_y_phys -= 2.5 + math.sin(ht) * 2.0
                target_x_phys += math.sin(ht * 1.7) * 1.2
                
            # Thinking animation: Look up and slightly left/right
            if self.current_emotion == "thinking":
                self.thinking_phase += 0.1
                # Rapid eye movement or looking up
                target_y_phys -= 15.0 # Look up
                # wander x slightly
                target_x_phys += math.sin(self.thinking_phase) * 5.0

            # Speech Reactivity: Squint with speech amplitude
            # This only applies when 'happy' (talking) state is active.
            if self.current_emotion == "happy" and self.speech_amplitude > 0.05:
                # Squint bottom lid slightly (makes eyes look "active" when speaking)
                squint = self.speech_amplitude * 0.25
                self.target_bottom_lid = max(self.target_bottom_lid, squint)
                # Also slightly boost the vertical scale (energized eye)
                self.target_scale_h = max(self.target_scale_h, self.target_scale_h + self.speech_amplitude * 0.08)
                # Micro bounce upward
                target_y_phys -= self.speech_amplitude * 3.0

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

            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5

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

        elif self.blink_state == "CLOSING":
            # Fast eyelid-style close: reduce height in place
            close_amount = self.base_h * BLINK_CLOSE_SPEED
            self.current_h -= close_amount
            self.current_w = self.base_w  # Stay same width
            self.target_w = self.current_w
            self.target_h = self.current_h

            if self.current_h <= 4:
                self.current_h = 4
                # SACCADE: jump position while eye is fully closed
                if self.saccade_pending:
                    self.target_pos[0] = self.base_x + self.saccade_offset[0]
                    self.target_pos[1] = self.base_y + self.saccade_offset[1]
                    self.current_pos[0] = self.target_pos[0]
                    self.current_pos[1] = self.target_pos[1]
                    self.saccade_pending = False
                self.blink_state = "OPENING"

        elif self.blink_state == "OPENING":
            # Fast eyelid-style open: restore height in place
            open_amount = self.base_h * BLINK_OPEN_SPEED
            self.current_h += open_amount
            self.current_w = self.base_w
            self.target_w = self.current_w
            self.target_h = self.current_h

            if self.current_h >= self.base_h:
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
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_src = Image.new("RGBA", (int(w + 20), int(lid_h + 20)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 10)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = Image.new("RGBA", (int(w + 20), int(lid_h + 20)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 10 - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))

        # We create the full screen image here
        # Optimization: Create ONE image and return it
        eye_img_size = int(max(self.base_w, self.base_h) * 2.5)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        base_radius = int(min(self.base_w, self.base_h) * 0.25)
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
        # Update shared blink logic
        if time.time() > self.next_blink_time:
            # ~30% of blinks are saccade blinks (vanish + reappear in new spot)
            do_saccade = (random.random() < 0.30)
            self.left_eye.start_blink(saccade=do_saccade)
            self.right_eye.start_blink(saccade=do_saccade)
            self.next_blink_time = time.time() + random.uniform(3.5, 7.0)
            
        # Smooth tracking (from face monitor)
        smooth_alpha = 0.15
        self.smoothed_x_off += (self.target_x_off - self.smoothed_x_off) * smooth_alpha
        self.smoothed_y_off += (self.target_y_off - self.smoothed_y_off) * smooth_alpha
        
        # Update eyes
        for eye in (self.left_eye, self.right_eye):
            # The saccade_offset lives inside BlockyEye.target_pos directly.
            # When not in a saccade, track face + smoothed offset.
            # After a saccade, slowly let eyes drift back to base + tracking.
            if eye.blink_state == "IDLE" and not eye.saccade_pending:
                # Gently return saccade drift to tracking position
                eye.target_pos[0] += (eye.base_x + self.smoothed_x_off - eye.target_pos[0]) * 0.015
                eye.target_pos[1] += (eye.base_y + self.smoothed_y_off - eye.target_pos[1]) * 0.015
            # Physics update
            eye.update()
            
        # Render
        img_l = self.left_eye.draw().convert("RGB")
        img_r = self.right_eye.draw().convert("RGB")
        
        if mono:
            return img_l.convert('1') # Fallback for OLED if needed
            
        return (img_l, img_r)
