import pygame
import cv2
import time
import os
import random
import math

from eye_engine import ProceduralEyeDisplay, SCREEN_WIDTH, SCREEN_HEIGHT

# --- Configuration ---
GAP = 50
WINDOW_WIDTH = SCREEN_WIDTH * 2 + GAP
WINDOW_HEIGHT = SCREEN_HEIGHT + 20 
BG_COLOR = (10, 10, 10)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Face Tracking Eyes Debug (V5 Engine)")
clock = pygame.time.Clock()

# Initialize Camera
cap = None
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            break
        else:
            cap.release()
    cap = None

# --- YuNet Setup ---
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

engine = ProceduralEyeDisplay()
running = True
frame_count = 0
mimic_rotation = False
next_rotation_decision_time = time.time() + 2.0
target_rotation_smooth = 0.0

while running:
    screen.fill(BG_COLOR)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
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
                engine.set_emotion(emotion)
                print(f"DEBUG EMOTION: {emotion.upper()}")

    # --- Speech Amplitude Simulation (Hold 'S') ---
    keys = pygame.key.get_pressed()
    target_amp = 0.0
    if keys[pygame.K_s]:
        target_amp = 0.5 + math.sin(time.time() * 15.0) * 0.4
    
    engine.left_eye.speech_amplitude += (target_amp - engine.left_eye.speech_amplitude) * 0.3
    engine.right_eye.speech_amplitude = engine.left_eye.speech_amplitude

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
                        
                    engine.set_face_target(norm_x, norm_y, current_roll)

                if faces[1] is not None and frame_count % 2 == 0:
                    cv2.rectangle(small_frame, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0, 255, 0), 2)
                    cv2.putText(small_frame, f"Roll: {current_roll:.1f}deg", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Face Tracker (YuNet) - Debug View', small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
    frame_count += 1
    
    # Render Frame from Engine
    img_l, img_r = engine.render_frame(1.0/60.0)
    
    # Convert PIL to Pygame Surface
    surf_l = pygame.image.frombuffer(img_l.tobytes(), img_l.size, img_l.mode)
    surf_r = pygame.image.frombuffer(img_r.tobytes(), img_r.size, img_r.mode)
    
    screen.blit(surf_l, (0, 10))
    screen.blit(surf_r, (SCREEN_WIDTH + GAP, 10))

    pygame.display.flip()
    clock.tick(60)

if cap: cap.release()
cv2.destroyAllWindows()
pygame.quit()
