import pygame
import cv2
import time
import math
import random
import os
from PIL import Image

# Import the actual project logic
from procedural_eyes import ProceduralEyeDisplay, SCREEN_WIDTH, SCREEN_HEIGHT

# --- Pygame Layout Configuration ---
GAP = 50
WINDOW_WIDTH = SCREEN_WIDTH * 2 + GAP
WINDOW_HEIGHT = SCREEN_HEIGHT + 20 
BG_COLOR = (10, 10, 10)

def pil_to_pygame(pil_img):
    """Convert PIL image to pygame surface"""
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()
    return pygame.image.fromstring(data, size, mode)

# --- Main Setup ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Eye Simulator (Project Logic)")
clock = pygame.time.Clock()

# Initialize the actual ProceduralEyeDisplay
eyes = ProceduralEyeDisplay()

# Initialize Camera for Face Tracking
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
else:
    print(f"Warning: {yunet_model_path} not found. Face tracking disabled.")

running = True
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Map keys to emotions
            key_map = {
                pygame.K_1: "idle",
                pygame.K_2: "happy",
                pygame.K_3: "sad",
                pygame.K_4: "angry",
                pygame.K_5: "surprised",
                pygame.K_6: "suspicious",
                pygame.K_7: "sleepy",
                pygame.K_t: "thinking"
            }
            if event.key in key_map:
                emotion = key_map[event.key]
                eyes.set_emotion(emotion)
                print(f"Emotion: {emotion.upper()}")

    # --- Face Tracking ---
    if cap and cap.isOpened() and detector:
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            # Detect every few frames for speed
            if frame_count % 2 == 0:
                detector.setInputSize((w, h))
                faces = detector.detect(frame)
                
                if faces[1] is not None:
                    # Target the largest face
                    largest_face = max(faces[1], key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest_face[0:4]
                    
                    # Normalize to -1.0 to 1.0 (keeping logic consistent with agent.py)
                    norm_x = -((fx + fw/2) / w - 0.5) * 2.0
                    norm_y = ((fy + fh/2) / h - 0.5) * 2.0
                    
                    eyes.set_face_target(norm_x, norm_y)
                    
                    # Optional: Show debug rectangle
                    cv2.rectangle(frame, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0, 255, 0), 2)
                else:
                    # Return to center if no face
                    eyes.set_face_target(0, 0)

            # Show camera debug view
            if frame_count % 4 == 0:
                cv2.imshow('Face Tracking Debug', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
    
    frame_count += 1

    # --- Render Eyes ---
    # render_frame() returns (PIL_L, PIL_R)
    img_l, img_r = eyes.render_frame(dt=1/60)
    
    # Convert to Pygame surfaces
    surf_l = pil_to_pygame(img_l)
    surf_r = pil_to_pygame(img_r)
    
    # Draw to screen
    screen.fill(BG_COLOR)
    
    # Border rectangles for visualization
    pygame.draw.rect(screen, (30, 30, 30), (0, 10, SCREEN_WIDTH, SCREEN_HEIGHT), 1)
    pygame.draw.rect(screen, (30, 30, 30), (SCREEN_WIDTH + GAP, 10, SCREEN_WIDTH, SCREEN_HEIGHT), 1)
    
    # Blit eyes
    screen.blit(surf_l, (0, 10))
    screen.blit(surf_r, (SCREEN_WIDTH + GAP, 10))

    pygame.display.flip()
    clock.tick(60)

if cap:
    cap.release()
cv2.destroyAllWindows()
pygame.quit()
