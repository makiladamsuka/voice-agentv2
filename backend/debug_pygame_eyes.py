import pygame
import cv2
import time
import math
import random
import os

# --- Configuration ---
SCREEN_W, SCREEN_H = 128, 160
GAP = 50
WINDOW_WIDTH = SCREEN_W * 2 + GAP
WINDOW_HEIGHT = SCREEN_H + 20 
EYE_COLOR = (0, 255, 255)  
BG_COLOR = (10, 10, 10)
EYE_SIZE = 100 
FLOOR_Y = SCREEN_H - 10 
 
# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 3.0
BLINK_SPEED_MAX = 5.0


# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle":  {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 0.0, "pos": (0, 0)},
    "joy":   {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0,  "bottom_lid": 0.0,  "lid_angle": 10.0, "pos": (0, -15)},
    "excited": {"scale_w": 0.9, "scale_h": 1.3, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 0), "behavior": "bounce"},
    "amused": {"scale_w": 1.0, "scale_h": 0.8, "top_lid": 0.0, "bottom_lid": 0.2, "lid_angle": 0.0, "pos": (15, -15), "asym": True},
    "friendly": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 0)},
    "proud": {"scale_w": 1.0, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, -25)},
    "sad":   {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.4,  "bottom_lid": 0.0,  "lid_angle": -12.0, "pos": (0, 20)},
    "lonely": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (-30, 30)},
    "bored": {"scale_w": 1.1, "scale_h": 0.8, "top_lid": 0.45, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 0), "behavior": "drift"},
    "tired": {"scale_w": 1.1, "scale_h": 0.9, "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 15), "behavior": "sink"},
    "disappointed": {"scale_w": 1.0, "scale_h": 1.1, "top_lid": 0.3, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 30), "anticipate": True},
    "thinking": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.3, "bottom_lid": 0.1, "lid_angle": 0.0, "pos": (25, -25)},
    "confused": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "asym": True, "pos": (0, 0), "behavior": "jitter"},
    "curious": {"scale_w": 1.1, "scale_h": 1.1, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 0), "behavior": "lean"},
    "concentrating": {"scale_w": 1.2, "scale_h": 0.4, "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0, "pos": (0, 0)},
    "remembering": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (-25, -25), "behavior": "saccades"},
    "surprised": {"scale_w": 0.8, "scale_h": 1.25, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "pos": (0, 0)},
    "skeptical": {"scale_w": 1.05, "scale_h": 0.7, "top_lid": 0.0, "bottom_lid": 0.4, "lid_angle": 0.0, "pos": (30, 0)},
    "angry": {"scale_w": 1.0, "scale_h": 0.85, "top_lid": 0.45, "bottom_lid": 0.0,  "lid_angle": 15.0, "pos": (0, 15)},
    "shy": {"scale_w": 0.9, "scale_h": 0.9, "top_lid": 0.2, "bottom_lid": 0.0, "lid_angle": 6.0, "pos": (-25, 25)},
    "glitch": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "behavior": "glitch"}
}

class BlockyEye:
    def __init__(self, x, y, scale=1.0, rotation=0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]
        self.velocity = [0.0, 0.0]
        
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
        self.target_rotation = 0.0  # valid external rotation

        self.rot_sensitivity = random.uniform(0.3, 0.5) 
        self.rot_speed = random.uniform(0.15, 0.25) # Faster rotation (was 0.08-0.15)
        
        self.is_left = is_left 
        self.blink_state = "IDLE" 
        self.vy = 0 
        self.blink_speed_mult = 1.0 
        
        # New Emotion Attributes
        self.target_scale_w = 1.0
        self.target_scale_h = 1.0
        
        self.top_lid = 0.0
        self.bottom_lid = 0.0
        self.lid_angle = 0.0
        
        self.target_top_lid = 0.0
        self.target_bottom_lid = 0.0
        self.target_lid_angle = 0.0
        
        self.current_emotion = "idle"
        
        # Advanced Animation State
        self.anticipation_timer = 0.0
        self.anticipation_dip = [0.0, 0.0]
        self.asym_delay = 0.0 if is_left else random.uniform(0.02, 0.06)
        
        self.jitter_x = 0.0
        self.jitter_y = 0.0
        self.noise_t = random.uniform(0, 1000)
        self.speech_amplitude = 0.0
        
        # --- Behavioral State ---
        self.emotion_queue = []
        self.decay_timer = 0.0
        self.blink_shift_pending = None
        self.shared_saccade_offset = [0.0, 0.0]

    def set_emotion(self, emotion_name, duration=None, chain=None, blink_shift=False):
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

        # Check for anticipation (dip before big scale/pos change)
        needs_anticipation = preset.get("anticipate") or abs(preset.get("pos", (0,0))[1]) > 20 or preset.get("scale_h", 1) > 1.3
        if needs_anticipation:
            self.anticipation_timer = time.time() + 0.12
            self.anticipation_dip = [0.0, 6.0] # Dip down slightly

        self.target_scale_w = preset["scale_w"]
        self.target_scale_h = preset["scale_h"]
        self.target_top_lid = preset["top_lid"]
        self.target_bottom_lid = preset["bottom_lid"]
        
        # Mirror angle for right eye
        angle = preset["lid_angle"]
        if not self.is_left and abs(angle) > 0.1:
            angle = -angle
        self.target_lid_angle = angle
 

    def start_blink(self, speed_mult=None):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            if speed_mult:
                self.blink_speed_mult = speed_mult
            else:
                self.blink_speed_mult = random.uniform(2.0, 3.0) # Faster blink
            self.vy = 40 * self.blink_speed_mult



    def update(self):
        now = time.time()
        # --- Movement Physics (SPRING PHYSICS) ---
        if self.blink_state == "IDLE":
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
            # Now handled externally in the main loop for symmetry
            
            # 3. Micro-Saccades (High frequency jitter)
            jitter_amp = 0.3
            if self.current_emotion == "remembering": jitter_amp = 0.8
            self.jitter_x = (math.sin(t * 30.0) * jitter_amp + math.sin(t * 15.0) * 0.2) + self.shared_saccade_offset[0]
            self.jitter_y = (math.cos(t * 22.0) * jitter_amp) + self.shared_saccade_offset[1]

            # 4. Continuous Behaviors
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

            # 4. Final Position Calculation (with Tighter Clamping)
            preset_pos = EMOTION_PRESETS[self.current_emotion].get("pos", (0, 0))
            target_x_phys = self.target_pos[0] + preset_pos[0] + b_off_x + self.jitter_x + anticipate_x
            target_y_phys = self.target_pos[1] + preset_pos[1] + b_off_y + self.jitter_y + anticipate_y

            # Tighten Screen Clamping (account for rotation & padding)
            # A rotated rectangle needs more space: w_eff = w*|cos| + h*|sin|
            angle_rad = math.radians(abs(self.current_rotation))
            eff_half_w = (self.w * math.cos(angle_rad) + self.h * math.sin(angle_rad)) / 2
            eff_half_h = (self.w * math.sin(angle_rad) + self.h * math.cos(angle_rad)) / 2
            
            pad = 10 
            target_x_phys = max(eff_half_w + pad, min(SCREEN_W - eff_half_w - pad, target_x_phys))
            target_y_phys = max(eff_half_h + pad, min(SCREEN_H - eff_half_h - pad, target_y_phys))

            # Spring Physics for Position (Higher Friction / Less Bounce)
            k_pos = 0.08
            d_pos_x = 0.70 
            self.velocity[0] = (self.velocity[0] + (target_x_phys - self.current_pos[0]) * k_pos) * d_pos_x

            # Gravity Feel + Vertical Resistance
            d_pos_y = 0.82
            if self.velocity[1] < 0: d_pos_y = 0.68 # Higher friction moving up
            else: d_pos_y = 0.80 # Slightly more friction moving down

            # Subtle gravity pull bias
            self.velocity[1] = (self.velocity[1] + (target_y_phys + 1.5 - self.current_pos[1]) * k_pos) * d_pos_y
            
            self.current_pos[0] += self.velocity[0]
            self.current_pos[1] += self.velocity[1]

            # Hard constraint on current_pos to prevent any visual overflow
            self.current_pos[0] = max(eff_half_w + 2, min(SCREEN_W - eff_half_w - 2, self.current_pos[0]))
            self.current_pos[1] = max(eff_half_h + 2, min(SCREEN_H - eff_half_h - 2, self.current_pos[1]))
            
            # Rotation Logic
            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed
            
            # Shape Springs (Lower Bounce)
            k_shape = 0.20
            d_shape = 0.65  # Higher friction
            
            # Voice Pulse (Audio Reactivity)
            pulse_scale = self.speech_amplitude * 0.25
            pulse_squish = self.speech_amplitude * 0.4
            
            self.vel_w = (self.vel_w + (self.base_w * self.target_scale_w - self.current_w) * k_shape) * d_shape
            self.vel_h = (self.vel_h + (self.base_h * (self.target_scale_h + pulse_scale) - self.current_h) * k_shape) * d_shape
            
            self.current_w += self.vel_w
            self.current_h += self.vel_h
            
            # --- Emotion Eyelid Interpolation ---
            speed_lid = 0.25
            self.top_lid += (self.target_top_lid - self.top_lid) * speed_lid
            
            # Speech Reactivity: add extra squint to bottom lid
            active_bottom_lid = self.target_bottom_lid
            if self.speech_amplitude > 0.05:
                # Only apply to emotions where it makes sense (open-ish bottom lids)
                if self.target_bottom_lid < 0.4:
                    active_bottom_lid = min(0.45, self.target_bottom_lid + self.speech_amplitude * 0.4)
            
            self.bottom_lid += (active_bottom_lid - self.bottom_lid) * speed_lid
            self.lid_angle += (self.target_lid_angle - self.lid_angle) * speed_lid
            
        elif self.blink_state == "DROPPING":
            self.vy += 10 * self.blink_speed_mult
            self.current_pos[1] += self.vy
            
            # Shrink to bottom (User Request)
            self.current_w = max(10, self.base_w - 20)
            self.current_h = max(10, self.base_h - 20)
            
            self.target_w = self.current_w
            self.target_h = self.current_h
            
            if self.current_pos[1] + self.current_h // 2 >= FLOOR_Y:
                self.current_pos[1] = FLOOR_Y - self.current_h // 2
                self.blink_state = "SQUASHING"
                self.velocity = [0.0, 0.0]
                
        elif self.blink_state == "SQUASHING":
            squeeze_speed = 45 * self.blink_speed_mult 
            spread_speed = 30 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            
            # Constraint: Keep inside eye frame (max width 120 < 128)
            if self.current_w > 120:
                self.current_w = 120
            
            # Constraint: Prevent negative height (which causes out of bounds)
            if self.current_h < 10:
                self.current_h = 10
            
            self.current_pos[1] = FLOOR_Y - self.current_h // 2
            
            if self.current_h <= 25: 
                self.current_h = 25 
                self.blink_state = "JUMPING"
                
                # Blink-Shift: swap emotion while eyes are closed
                if self.blink_shift_pending:
                    p = self.blink_shift_pending
                    self.blink_shift_pending = None
                    self.set_emotion(p["name"], duration=p["duration"], chain=p["chain"], blink_shift=False)
                
        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.1, min(0.9, 0.7 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed
            self.current_pos[0] += (self.target_pos[0] - self.current_pos[0]) * 0.2
            
            target_y = self.target_pos[1]
            self.current_pos[1] += (target_y - self.current_pos[1]) * 0.8 
            
            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.blink_state = "IDLE"
                self.vy = 0



        # Spring Physics
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

    def draw_radial_rect(self, surface, rect, color, radius, pupil_offset=(0,0)):
        grad_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        steps = 15
        for i in range(steps):
            size_factor = 1.0 - (i / steps) 
            current_w = int(rect.width * size_factor)
            current_h = int(rect.height * size_factor)
            if current_w <= 0 or current_h <= 0: continue
            
            b_factor = 0.4 + 0.6 * (i / steps) 
            cur_color = (int(color[0] * b_factor), int(color[1] * b_factor), int(color[2] * b_factor))
            
            dest_rect = pygame.Rect(0, 0, current_w, current_h)
            shift_x = pupil_offset[0] * (1.0 - size_factor) * 20 
            shift_y = pupil_offset[1] * (1.0 - size_factor) * 20
            
            cx = rect.width // 2 + shift_x
            cy = rect.height // 2 + shift_y
            dest_rect.center = (cx, cy)
            
            cur_radius = max(4, int(radius * size_factor))
            pygame.draw.rect(grad_surf, cur_color, dest_rect, border_radius=cur_radius)
        surface.blit(grad_surf, rect)

    def draw_eyelids(self, surface, rect):
        """Draw eyelids as full-width black rects that fully cover the eye above/below."""
        if self.top_lid < 0.01 and self.bottom_lid < 0.01:
            return

        w, h = rect.width, rect.height
        cx, cy = rect.centerx, rect.centery

        # 1. Top Lid — full width from top of screen to inner lid edge
        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            inner_y = cy - h // 2 + lid_h
            pygame.draw.rect(surface, BG_COLOR, (0, 0, surface.get_width(), inner_y))

        # 2. Bottom Lid — full width from inner lid edge to bottom of screen
        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            inner_y = cy + h // 2 - lid_h
            pygame.draw.rect(surface, BG_COLOR, (0, inner_y, surface.get_width(), surface.get_height() - inner_y))

    def draw(self, surface, offset_x=0, offset_y=0):
        draw_w = max(4, int(self.w))
        draw_h = max(4, int(self.h))
        surf_w = int(self.base_w * 1.5)
        surf_h = int(self.base_h * 1.5) 
        
        eye_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        
        rect = pygame.Rect(0, 0, draw_w, draw_h)
        rect.center = (surf_w // 2, surf_h // 2)
        corner_radius = int(self.base_w * 0.25) 
        
        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))
        
        self.draw_radial_rect(eye_surf, rect, EYE_COLOR, corner_radius, pupil_offset=(off_x, off_y))
        
        # NOTE: eyelids drawn AFTER rotation, on main surface — see below
        
        rotated_surf = pygame.transform.rotozoom(eye_surf, self.current_rotation, 1.0)
        dest_rect = rotated_surf.get_rect()
        dest_rect.center = (self.current_pos[0] + offset_x, self.current_pos[1] + offset_y)
        surface.blit(rotated_surf, dest_rect)
        
        # Draw eyelids on the MAIN surface AFTER the eye is blitted.
        # This ensures the black rect is in screen space and covers everything.
        if self.top_lid > 0.01 or self.bottom_lid > 0.01:
            scr_cx = int(self.current_pos[0] + offset_x)
            scr_cy = int(self.current_pos[1] + offset_y)
            half_h = draw_h // 2

            if self.top_lid > 0.01:
                lid_h = int(draw_h * self.top_lid)
                inner_y = scr_cy - half_h + lid_h
                # Cover from y=0 to inner lid edge, full window width
                pygame.draw.rect(surface, BG_COLOR, (0, 0, surface.get_width(), inner_y))

            if self.bottom_lid > 0.01:
                lid_h = int(draw_h * self.bottom_lid)
                inner_y = scr_cy + half_h - lid_h
                pygame.draw.rect(surface, BG_COLOR, (0, inner_y, surface.get_width(), surface.get_height() - inner_y))

# --- Main Setup ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Face Tracking Eyes Debug")
clock = pygame.time.Clock()

# Initialize Camera
cap = None
for i in range(4):
    print(f"Testing camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Success! Using camera index {i}")
            break
        else:
            cap.release()
    cap = None

if cap is None or not cap.isOpened():
    print("Error: Could not open any camera.")
    # Proceed anyway

# --- YuNet Setup (Optional if file exists) ---
detector = None
yunet_model_path = 'face_detection_yunet_2023mar.onnx'
if os.path.exists(yunet_model_path):
    detector = cv2.FaceDetectorYN.create(
        model=yunet_model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
else:
    print(f"Warning: {yunet_model_path} not found. Face tracking disabled.")

left_eye = BlockyEye(SCREEN_W // 2, SCREEN_H // 2, scale=1.1, rotation=0, is_left=True)
right_eye = BlockyEye(SCREEN_W // 2, SCREEN_H // 2, scale=1.1, rotation=0, is_left=False)

running = True
next_blink_time = time.time() + random.uniform(1, 4)
saccade_timer = 0.0
shared_saccade_offset = [0.0, 0.0]

# Eye movement constraints
MAX_X_OFFSET = 15 
MAX_Y_OFFSET = 10 
frame_count = 0

# Natural Rotation Logic
mimic_rotation = False
next_rotation_decision_time = time.time() + 2.0
target_rotation_smooth = 0.0

while running:
    screen.fill(BG_COLOR)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            # Map keys to emotions
            key_map = {
                pygame.K_1: "joy", pygame.K_2: "excited", pygame.K_3: "amused", 
                pygame.K_4: "friendly", pygame.K_5: "proud", pygame.K_6: "sad",
                pygame.K_7: "lonely", pygame.K_8: "bored", pygame.K_9: "tired",
                pygame.K_0: "disappointed", pygame.K_q: "thinking", pygame.K_w: "confused",
                pygame.K_e: "curious", pygame.K_r: "concentrating", pygame.K_t: "remembering",
                pygame.K_y: "surprised", pygame.K_u: "skeptical", pygame.K_i: "angry",
                pygame.K_o: "shy", pygame.K_p: "glitch", pygame.K_SPACE: "idle"
            }
            if event.key in key_map:
                emotion = key_map[event.key]
                left_eye.set_emotion(emotion)
                right_eye.set_emotion(emotion)
                print(f"DEBUG EMOTION: {emotion.upper()}")
            
            # --- Advanced Behavior Testing ---
            elif event.key == pygame.K_j:
                # Joke Chain: Concentrating (1.2s) -> Surprised (0.4s) -> Joy (Decay)
                print("DEBUG: TRIGGERING JOKE CHAIN (concentrating -> surprised -> joy)")
                left_eye.set_emotion("concentrating", duration=1.2, chain=["surprised", "joy"])
                right_eye.set_emotion("concentrating", duration=1.2, chain=["surprised", "joy"])
            
            elif event.key == pygame.K_b:
                # Blink Shift Test: Move to Sad ONLY after closing eyes
                print("DEBUG: TRIGGERING BLINK-SHIFT (target: SAD)")
                left_eye.set_emotion("sad", blink_shift=True)
                right_eye.set_emotion("sad", blink_shift=True)

    # --- Speech Amplitude Simulation (Hold 'S') ---
    keys = pygame.key.get_pressed()
    target_amp = 0.0
    if keys[pygame.K_s]:
        # Oscilate for visual interest
        target_amp = 0.5 + math.sin(time.time() * 15.0) * 0.4
    
    # Smooth amplitude
    left_eye.speech_amplitude += (target_amp - left_eye.speech_amplitude) * 0.3
    right_eye.speech_amplitude = left_eye.speech_amplitude

    # --- Video Capture & Face Detection ---
    if cap and cap.isOpened() and detector:
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            small_h, small_w = int(h * 0.5), int(w * 0.5)
            small_frame = cv2.resize(frame, (small_w, small_h)) 
            
            if frame_count % 2 == 0:
                detector.setInputSize((small_w, small_h))
                faces = detector.detect(small_frame)
                
                target_offset_x = 0
                target_offset_y = 0
                current_roll = 0.0

                if faces[1] is not None:
                    detected_faces = faces[1]
                    largest_face = max(detected_faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest_face[0:4]
                    re_x, re_y = largest_face[4], largest_face[5]
                    le_x, le_y = largest_face[6], largest_face[7]
                    
                    center_x = (fx + fw/2) / small_w
                    center_y = (fy + fh/2) / small_h
                    norm_x = -(center_x - 0.5) * 2.0 
                    norm_y = (center_y - 0.5) * 2.0
                    
                    dx = re_x - le_x
                    dy = re_y - le_y
                    if dx != 0:
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        current_roll = max(-6, min(6, angle_deg))

                    target_offset_x = norm_x * MAX_X_OFFSET
                    target_offset_y = norm_y * MAX_Y_OFFSET
                    
                    # Clamp
                    target_offset_x = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, target_offset_x))
                    target_offset_y = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, target_offset_y))

                # Natural Rotation Logic: Follow head tilt more consistently in debug
                if time.time() > next_rotation_decision_time:
                    # Increase chance to follow rotation in debug mode so user can see it works
                    mimic_rotation = random.random() < 0.8 
                    next_rotation_decision_time = time.time() + random.uniform(2.0, 5.0)

                target_rotation_smooth = current_roll if mimic_rotation else (current_roll * 0.3)

                # Apply to eyes ONLY if in idle state
                if left_eye.current_emotion == "idle":
                    left_eye.target_pos[0] = left_eye.base_x + target_offset_x
                    left_eye.target_pos[1] = left_eye.base_y + target_offset_y
                    left_eye.target_rotation = target_rotation_smooth
                    
                    right_eye.target_pos[0] = right_eye.base_x + target_offset_x
                    right_eye.target_pos[1] = right_eye.base_y + target_offset_y
                    right_eye.target_rotation = target_rotation_smooth
                else:
                    # Return to center if not in idle
                    left_eye.target_pos[0] = left_eye.base_x
                    left_eye.target_pos[1] = left_eye.base_y
                    left_eye.target_rotation = 0.0
                    
                    right_eye.target_pos[0] = right_eye.base_x
                    right_eye.target_pos[1] = right_eye.base_y
                    right_eye.target_rotation = 0.0
            
                if faces[1] is not None and frame_count % 2 == 0:
                    # Draw Box and Landmarks on the preview frame
                    cv2.rectangle(small_frame, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0, 255, 0), 2)
                    cv2.circle(small_frame, (int(re_x), int(re_y)), 3, (255, 0, 0), -1) # Blue
                    cv2.circle(small_frame, (int(le_x), int(le_y)), 3, (0, 0, 255), -1) # Red
                    
                    # Info Text
                    cv2.putText(small_frame, f"Roll: {current_roll:.1f}deg", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Face Tracker (YuNet) - Debug View', small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
    # --- Shared Saccadic Scanning (Idle only) ---
    if left_eye.current_emotion == "idle":
        if time.time() > saccade_timer:
            shared_saccade_offset = [random.uniform(-12, 12), random.uniform(-8, 8)]
            saccade_timer = time.time() + random.uniform(2.5, 5.0)
    else:
        shared_saccade_offset = [0.0, 0.0]
    
    left_eye.shared_saccade_offset = shared_saccade_offset
    right_eye.shared_saccade_offset = shared_saccade_offset

    frame_count += 1
    # --- Blinking ---
    if time.time() > next_blink_time:
        # Only blink if in idle state
        if left_eye.current_emotion == "idle":
            blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            left_eye.start_blink(blink_speed)
            right_eye.start_blink(blink_speed)
        next_blink_time = time.time() + random.uniform(2, 5)

    left_eye.update()
    right_eye.update()
    
    # Draw Borders
    pygame.draw.rect(screen, (30, 30, 30), (0, 10, SCREEN_W, SCREEN_H), 1)
    pygame.draw.rect(screen, (30, 30, 30), (SCREEN_W + GAP, 10, SCREEN_W, SCREEN_H), 1)
    
    left_eye.draw(screen, offset_x=0, offset_y=10)
    right_eye.draw(screen, offset_x=SCREEN_W + GAP, offset_y=10)

    pygame.display.flip()
    clock.tick(60)

if cap: cap.release()
cv2.destroyAllWindows()
pygame.quit()
