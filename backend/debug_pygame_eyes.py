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

# --- Face Tracking Config ---
CAMERA_ID = 0 # Default to 0, script will scan below
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle": {
        "scale_w": 1.0, "scale_h": 1.0, 
        "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0
    },
    "happy": {
        "scale_w": 1.1, "scale_h": 0.7, 
        "top_lid": 0.0, "bottom_lid": 0.5, "lid_angle": 0.0
    },
    "sad": {
        "scale_w": 1.1, "scale_h": 1.1, 
        "top_lid": 0.35, "bottom_lid": 0.0, "lid_angle": 15.0
    },
    "angry": {
        "scale_w": 1.0, "scale_h": 0.9, 
        "top_lid": 0.35, "bottom_lid": 0.0, "lid_angle": -20.0
    },
    "surprised": {
        "scale_w": 0.9, "scale_h": 1.3, 
        "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0
    },
    "suspicious": {
        "scale_w": 1.1, "scale_h": 0.6, 
        "top_lid": 0.4, "bottom_lid": 0.4, "lid_angle": 0.0
    },
    "sleepy": {
        "scale_w": 1.1, "scale_h": 1.0,
        "top_lid": 0.6, "bottom_lid": 0.0, "lid_angle": 0.0
    }
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

    def set_emotion(self, emotion_name):
        if emotion_name not in EMOTION_PRESETS:
            return
        
        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        
        self.target_scale_w = preset["scale_w"]
        self.target_scale_h = preset["scale_h"]
        self.target_top_lid = preset["top_lid"]
        self.target_bottom_lid = preset["bottom_lid"]
        
        # Invert angle for right eye if it's mirrored behavior (like sad/angry)
        angle = preset["lid_angle"]
        if not self.is_left and abs(angle) > 0:
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
        # --- Movement Physics (EASING + GRAVITY) ---
        if self.blink_state == "IDLE":
            dx = self.target_pos[0] - self.current_pos[0]
            dy = self.target_pos[1] - self.current_pos[1]
            
            speed_x = 0.25 # Much faster (was 0.08)
            speed_y = 0.30 # Faster (was 0.15)
            if dy < -1.0: speed_y = 0.15 # UP - Faster (was 0.05)
            elif dy > 1.0: speed_y = 0.50 # DOWN - Very Fast (was 0.35) 
                
            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y
            
            # Rotation Logic - Mix of Look Direction + Head Tilt
            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            
            # Combine look rotation and external head rotation
            final_target_rot = look_rot + self.target_rotation
            
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed
            
            # Breathing
            t = time.time()
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0) 
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)
            
            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5
            
            self.target_w = (self.base_w * self.target_scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.target_scale_h) + breath_h - (move_stretch_y * 0.2)
            
            # --- Emotion Interpolation ---
            speed = 0.2
            self.top_lid += (self.target_top_lid - self.top_lid) * speed
            self.bottom_lid += (self.target_bottom_lid - self.bottom_lid) * speed
            self.lid_angle += (self.target_lid_angle - self.lid_angle) * speed
            
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
        """Draw eyelids on top of the eye surface"""
        w, h = rect.width, rect.height
        
        # Colors
        lid_color = BG_COLOR
        
        # 1. Top Lid
        if self.top_lid > 0.01:
            lid_src = pygame.Surface((w + 20, int(h * self.top_lid) + 20), pygame.SRCALPHA)
            lid_src.fill((*lid_color, 255))
            
            if abs(self.lid_angle) > 0.1:
                lid_src = pygame.transform.rotate(lid_src, self.lid_angle)
            
            lid_rect = lid_src.get_rect()
            lid_rect.midtop = (w // 2, -10) # -10 buffer
            surface.blit(lid_src, lid_rect)


        # 2. Bottom Lid
        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = pygame.Surface((w + 20, lid_h + 20), pygame.SRCALPHA)
            lid_src.fill((*lid_color, 255))
            
            if abs(self.lid_angle) > 0.1:
                 lid_src = pygame.transform.rotate(lid_src, self.lid_angle)

            lid_rect = lid_src.get_rect()
            lid_rect.midbottom = (w // 2, h + 10)
            surface.blit(lid_src, lid_rect)

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
        
        self.draw_eyelids(eye_surf, rect)
        
        rotated_surf = pygame.transform.rotozoom(eye_surf, self.current_rotation, 1.0)
        dest_rect = rotated_surf.get_rect()
        dest_rect.center = (self.current_pos[0] + offset_x, self.current_pos[1] + offset_y)
        surface.blit(rotated_surf, dest_rect)

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
            if event.key == pygame.K_1:
                left_eye.set_emotion("idle")
                right_eye.set_emotion("idle")
                print("Emotion: IDLE")
            elif event.key == pygame.K_2:
                left_eye.set_emotion("happy")
                right_eye.set_emotion("happy")
                print("Emotion: HAPPY")
            elif event.key == pygame.K_3:
                left_eye.set_emotion("sad")
                right_eye.set_emotion("sad")
                print("Emotion: SAD")
            elif event.key == pygame.K_4:
                left_eye.set_emotion("angry")
                right_eye.set_emotion("angry")
                print("Emotion: ANGRY")
            elif event.key == pygame.K_5:
                left_eye.set_emotion("surprised")
                right_eye.set_emotion("surprised")
                print("Emotion: SURPRISED")
            elif event.key == pygame.K_6:
                left_eye.set_emotion("suspicious")
                right_eye.set_emotion("suspicious")
                print("Emotion: SUSPICIOUS")
            elif event.key == pygame.K_7:
                left_eye.set_emotion("sleepy")
                right_eye.set_emotion("sleepy")
                print("Emotion: SLEEPY")

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

                # Natural Rotation Logic
                if time.time() > next_rotation_decision_time:
                    mimic_rotation = random.random() < 0.3
                    next_rotation_decision_time = time.time() + random.uniform(2.0, 5.0)

                target_rotation_smooth = current_roll if mimic_rotation else 0.0

                # Apply to eyes
                left_eye.target_pos[0] = left_eye.base_x + target_offset_x
                left_eye.target_pos[1] = left_eye.base_y + target_offset_y
                left_eye.target_rotation = target_rotation_smooth
                
                right_eye.target_pos[0] = right_eye.base_x + target_offset_x
                right_eye.target_pos[1] = right_eye.base_y + target_offset_y
                right_eye.target_rotation = target_rotation_smooth
            
            if frame_count % 2 == 0:
                 cv2.imshow('Face Tracker (YuNet)', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
    frame_count += 1
    # --- Blinking ---
    if time.time() > next_blink_time:
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
