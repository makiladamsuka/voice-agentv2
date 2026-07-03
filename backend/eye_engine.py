"""
Procedural Eyes Engine - V5 Port
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
EYE_SIZE = 120           # Base size for 128 width
FLOOR_Y = SCREEN_HEIGHT - 5

# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 3.2
BLINK_SPEED_MAX = 4.2

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":  {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0, "pos_x": 0.0, "pos_y": 0.0},
    "joy":   {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0, "pos_y": -15.0},
    "excited": {"scale_w": 0.9, "scale_h": 1.3, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "amused": {"scale_w": 1.0, "scale_h": 0.8, "top_lid": 0.0, "bottom_lid": 0.2, "lid_angle": 0.0, "pos_x": 15.0, "pos_y": -15.0},
    "friendly": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "proud": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_y": -25.0},
    "sad":   {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.4,  "bottom_lid": 0.0,  "lid_angle": -12.0, "pos_y": 20.0},
    "lonely": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_x": -30.0, "pos_y": 30.0},
    "bored": {"scale_w": 1.1, "scale_h": 0.8, "top_lid": 0.45, "bottom_lid": 0.0, "lid_angle": 0.0},
    "tired": {"scale_w": 1.1, "scale_h": 0.9, "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_y": 15.0},
    "disappointed": {"scale_w": 1.0, "scale_h": 1.1, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_y": 30.0},
    "thinking": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.1, "lid_angle": 0.0, "pos_x": 25.0, "pos_y": -25.0},
    "confused": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "curious": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "concentrating": {"scale_w": 1.2, "scale_h": 0.4, "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0},
    "remembering": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_x": -25.0, "pos_y": -25.0},
    "surprised": {"scale_w": 0.8, "scale_h": 1.25, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "skeptical": {"scale_w": 1.05, "scale_h": 0.7, "top_lid": 0.0, "bottom_lid": 0.4, "lid_angle": 0.0, "pos_x": 30.0},
    "angry": {"scale_w": 1.0, "scale_h": 0.85, "top_lid": 0.45, "bottom_lid": 0.0,  "lid_angle": 15.0, "pos_y": 15.0},
    "shy": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.2, "bottom_lid": 0.0, "lid_angle": 6.0, "pos_x": -25.0, "pos_y": 25.0},
    "searching": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    "glitch": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0},
    # Aliases
    "happy": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0, "pos_y": -15.0},
    "talking": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0, "pos_y": -15.0},
    "idle1": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0},
    "idle2": {"scale_w": 1.02, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos_y": -5.0},
}

class BlockyEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos  = [float(x), float(y)]
        self.vel_x = self.vel_y = 0.0
        self.base_w = self.base_h = EYE_SIZE * scale
        self.current_w = self.target_w = self.base_w
        self.current_h = self.target_h = self.base_h
        self.vel_w = self.vel_h = 0.0
        self.w = self.base_w; self.h = self.base_h
        self.current_rotation = self.target_rotation = 0.0
        self.rot_sensitivity = random.uniform(0.3, 0.5)
        self.rot_speed = random.uniform(0.15, 0.25)
        self.is_left = is_left
        self.blink_state = "IDLE"; self.vy = 0
        self.blink_speed_mult = 1.0
        self.target_scale_w = self.scale_w = 1.0
        self.target_scale_h = self.scale_h = 1.0
        self.scale_w_vel = self.scale_h_vel = 0.0
        self.top_lid = self.bottom_lid = self.lid_angle = 0.0
        self.top_lid_vel = self.bottom_lid_vel = self.lid_angle_vel = 0.0
        self.target_top_lid = self.target_bottom_lid = self.target_lid_angle = 0.0
        self.current_emotion = "idle"
        self.emotion_pos_bias_x = self.emotion_pos_bias_y = 0.0
        self.shared_saccade_offset = [0.0, 0.0]
        self.speech_amplitude = 0.0

    def start_blink(self, speed_mult=None, blink_speed_min=BLINK_SPEED_MIN, blink_speed_max=BLINK_SPEED_MAX):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            self.blink_speed_mult = speed_mult if speed_mult is not None else random.uniform(blink_speed_min, blink_speed_max)
            self.vy = 40 * self.blink_speed_mult

    def set_emotion(self, name: str, duration=None, chain=None, blink_shift=False, intensity: float = 1.0):
        # Ignore extra v2 kwargs for now, just apply the v5 logic
        if name not in EMOTION_PRESETS:
            return
        
        self.current_emotion = name
        preset = EMOTION_PRESETS[name]
        idle = EMOTION_PRESETS["idle"]
        
        # In v5, pos bias was part of left_bias/right_bias. Here we just use the preset's pos_x/pos_y
        self.emotion_pos_bias_x = preset.get("pos_x", 0.0) * intensity
        self.emotion_pos_bias_y = preset.get("pos_y", 0.0) * intensity
        
        self.target_scale_w  = idle.get("scale_w", 1.0)  + (preset.get("scale_w", 1.0)  - idle.get("scale_w", 1.0))  * intensity
        self.target_scale_h  = idle.get("scale_h", 1.0)  + (preset.get("scale_h", 1.0)  - idle.get("scale_h", 1.0))  * intensity
        self.target_top_lid  = idle.get("top_lid", 0.0)  + (preset.get("top_lid", 0.0)  - idle.get("top_lid", 0.0))  * intensity
        self.target_bottom_lid = idle.get("bottom_lid", 0.0) + (preset.get("bottom_lid", 0.0) - idle.get("bottom_lid", 0.0)) * intensity
        
        lid = idle.get("lid_angle", 0.0) + (preset.get("lid_angle", 0.0) - idle.get("lid_angle", 0.0)) * intensity
        
        # Mirror angle for right eye
        if preset.get("mirror_angle", True) and not self.is_left and abs(lid) > 0:
            lid = -lid
        self.target_lid_angle = lid

    def update(self):
        if self.blink_state == "IDLE":
            tl = self.target_top_lid; bl = self.target_bottom_lid; la = self.target_lid_angle
            
            # Combine base, target offsets, emotion offsets, and saccades
            tx = self.target_pos[0] + self.emotion_pos_bias_x + self.shared_saccade_offset[0]
            ty = self.target_pos[1] + self.emotion_pos_bias_y + self.shared_saccade_offset[1]
            
            dx = tx - self.current_pos[0]; dy = ty - self.current_pos[1]
            self.current_pos[0] += dx * 0.20
            self.current_pos[1] += dy * 0.22
            
            # Simple rotation smoothing towards target
            self.current_rotation += (self.target_rotation - self.current_rotation) * self.rot_speed
            
            t2 = time.time()
            bw = math.sin(t2*1.5+self.base_x)*1.5 + math.sin(t2*0.5)*1.0
            bh = math.cos(t2*1.8+self.base_y)*1.5 + math.cos(t2*0.6)*1.0
            
            # Add speech amplitude to scale/lid
            k=0.12; d=0.7
            if self.current_emotion=="surprised": k=0.30; d=0.52
            
            active_scale_w = self.target_scale_w + self.speech_amplitude * 0.1
            active_scale_h = self.target_scale_h + self.speech_amplitude * 0.2
            
            self.scale_w_vel=(self.scale_w_vel+(active_scale_w-self.scale_w)*k)*d; self.scale_w+=self.scale_w_vel
            self.scale_h_vel=(self.scale_h_vel+(active_scale_h-self.scale_h)*k)*d; self.scale_h+=self.scale_h_vel
            self.top_lid_vel=(self.top_lid_vel+(tl-self.top_lid)*k)*d; self.top_lid+=self.top_lid_vel
            self.bottom_lid_vel=(self.bottom_lid_vel+(bl-self.bottom_lid)*k)*d; self.bottom_lid+=self.bottom_lid_vel
            self.lid_angle_vel=(self.lid_angle_vel+(la-self.lid_angle)*k)*d; self.lid_angle+=self.lid_angle_vel
            
            self.top_lid = max(0.0, min(0.90, self.top_lid))
            self.bottom_lid = max(0.0, min(0.82, self.bottom_lid))
            self.lid_angle = max(-22.0, min(22.0, self.lid_angle))
            
            self.target_w=self.base_w*self.scale_w+bw; self.target_h=self.base_h*self.scale_h+bh
        elif self.blink_state=="DROPPING":
            self.vy+=10*self.blink_speed_mult; self.current_pos[1]+=self.vy
            self.current_w=self.base_w-10; self.current_h=self.base_h+20
            self.target_w=self.current_w; self.target_h=self.current_h
            if self.current_pos[1]+self.current_h//2>=FLOOR_Y:
                self.current_pos[1]=FLOOR_Y-self.current_h//2; self.blink_state="SQUASHING"
        elif self.blink_state=="SQUASHING":
            self.current_h-=65*self.blink_speed_mult; self.current_w+=40*self.blink_speed_mult
            self.current_pos[1]=FLOOR_Y-self.current_h//2
            if self.current_h<=22: self.current_h=22; self.blink_state="JUMPING"
        elif self.blink_state=="JUMPING":
            r=max(0.15,min(0.95,0.85*self.blink_speed_mult))
            self.current_h+=(self.base_h-self.current_h)*r; self.current_w+=(self.base_w-self.current_w)*r
            self.current_pos[0]+=(self.target_pos[0]-self.current_pos[0])*0.8
            if abs(self.current_h-self.base_h)<5:
                self.current_h=self.base_h; self.current_w=self.base_w; self.blink_state="IDLE"; self.vy=0
        
        if self.blink_state=="IDLE":
            k=0.08; d=0.90
            self.vel_w=(self.vel_w+(self.target_w-self.current_w)*k)*d; self.current_w+=self.vel_w
            self.vel_h=(self.vel_h+(self.target_h-self.current_h)*k)*d; self.current_h+=self.vel_h
        else:
            self.vel_w=self.vel_h=0
            
        self.w=self.current_w; self.h=self.current_h
        hw=max(2.0,self.w*0.5); hh=max(2.0,self.h*0.5)
        self.current_pos[0]=max(hw,min(SCREEN_WIDTH-hw,self.current_pos[0]))
        self.current_pos[1]=max(hh,min(SCREEN_HEIGHT-hh,self.current_pos[1]))

    @staticmethod
    def _solid_lid_block(width: int, height: int, angle: float):
        from PIL import Image
        lid = Image.new("RGBA", (max(1, width), max(1, height)), (*BG_COLOR, 255))
        if abs(angle) <= 0.1:
            return lid
        rotated = lid.rotate(angle, resample=Image.BICUBIC, expand=True)
        px = rotated.load()
        w, h = rotated.size
        for y in range(h):
            for x in range(w):
                if px[x, y][3] > 32:
                    px[x, y] = (*BG_COLOR, 255)
                else:
                    px[x, y] = (0, 0, 0, 0)
        return rotated

    def draw_eyelids(self, eye_img, x0: float, y0: float, x1: float, y1: float) -> None:
        w = int(x1 - x0)
        h = int(y1 - y0)
        if w < 1 or h < 1:
            return

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_src = self._solid_lid_block(int(w * 2.1), lid_h + 64, self.lid_angle)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 32)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = self._solid_lid_block(int(w * 2.1), lid_h + 28, self.lid_angle)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 13 - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self):
        # Adapt for v2 API which expects `draw()` to return the final image
        bg_image = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), (*BG_COLOR, 255))
        
        draw_w = max(6, min(int(self.w), SCREEN_WIDTH - 4))
        draw_h = max(6, min(int(self.h), SCREEN_HEIGHT - 4))
        eye_img_size = int(max(self.base_w, self.base_h) * 2.6)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        cx = eye_img_size / 2
        cy = eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2
        
        # Round eyes from v5
        eye_draw.ellipse([x0, y0, x1, y1], fill=EYE_COLOR)
        self.draw_eyelids(eye_img, x0, y0, x1, y1)

        paste_x = int(self.current_pos[0] - eye_img_size / 2)
        paste_y = int(self.current_pos[1] - eye_img_size / 2)
        bg_image.alpha_composite(eye_img, (paste_x, paste_y))
        
        # Apply rotation to the final eye image composited
        if abs(self.current_rotation) > 0.1:
            bg_image = bg_image.rotate(self.current_rotation, resample=Image.BICUBIC, center=(self.current_pos[0], self.current_pos[1]))
            
        return bg_image.convert("RGB")

# --- Main Controller Class (Maintained API for v2) ---
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
        
        # Apply roll multiplier and clamp
        ROLL_MULT = -0.75
        MAX_ROLL = 10.0
        self.target_face_rotation = max(-MAX_ROLL, min(MAX_ROLL, rotation * ROLL_MULT))

    def render_frame(self, dt: float, mono: bool = False):
        now = time.time()
        
        # 1. Shared Saccadic Scanning (Idle only)
        if self.left_eye.current_emotion in ("idle", "idle1", "idle2"):
            if now > self.saccade_timer:
                self.shared_saccade_offset = [random.uniform(-12, 12), random.uniform(-8, 8)]
                self.saccade_timer = now + random.uniform(2.5, 5.0)
        else:
            self.shared_saccade_offset = [0.0, 0.0]

        # 2. Trigger Blinks ONLY during Idle state
        if now > self.next_blink_time:
            if self.left_eye.current_emotion in ("idle", "idle1", "idle2"):
                blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
                self.left_eye.start_blink(blink_speed)
                self.right_eye.start_blink(blink_speed)
            self.next_blink_time = now + random.uniform(3.5, 7.0)

        # 3. Smoothing for face tracking
        smooth_alpha = 0.15
        self.smoothed_x_off += (self.target_x_off - self.smoothed_x_off) * smooth_alpha
        self.smoothed_y_off += (self.target_y_off - self.smoothed_y_off) * smooth_alpha
        self.smoothed_face_rotation += (self.target_face_rotation - self.smoothed_face_rotation) * 0.1
        
        # 4. Update eyes
        for eye in (self.left_eye, self.right_eye):
            eye.shared_saccade_offset = self.shared_saccade_offset
            
            if eye.blink_state == "IDLE":
                if eye.current_emotion in ("idle", "idle1", "idle2"):
                    eye.target_pos[0] = eye.base_x + self.smoothed_x_off
                    eye.target_pos[1] = eye.base_y + self.smoothed_y_off
                    eye.target_rotation = self.smoothed_face_rotation
                else:
                    eye.target_pos[0] = eye.base_x
                    eye.target_pos[1] = eye.base_y
                    eye.target_rotation = 0.0
            
            eye.update()
            
        # 5. Render
        img_l = self.left_eye.draw()
        img_r = self.right_eye.draw()
        
        if mono:
            return img_l.convert('1') 
            
        return (img_l, img_r)
