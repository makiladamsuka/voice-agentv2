#!/usr/bin/env python3
"""
Face tracking with Picamera2 + YuNet + ServoKit + MJPEG debug stream.

Run:
    python3 face_tracker_servo_mjpeg.py

Dependencies:
    pip install opencv-python pillow adafruit-circuitpython-servokit
    sudo apt install python3-picamera2
"""

import io
import random
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
from PIL import Image

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Install with: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    from adafruit_servokit import ServoKit
except ImportError:
    print("Error: adafruit-circuitpython-servokit not found.")
    print("Install with: pip install adafruit-circuitpython-servokit")
    sys.exit(1)


# ---------------- Configuration ----------------
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
CAMERA_MAIN_RES = (1920, 1080)
CAMERA_RES = (1280, 720)
CAMERA_ROTATE_180 = False

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

# Servo wiring and limits (same style as testservos2.py)
PAN_CH = 0
TILT_CH = 1
PAN_MIN = 40.0
PAN_MAX = 130.0
TILT_MIN = 80.0
TILT_MAX = 130.0
PULSE_MIN = 450
PULSE_MAX = 2600

# Keep testservos2-style smoothing, tuned for lower latency with stable damping.
SMOOTHING = 0.10
SERVO_LOOP_DELAY = 0.01
MAX_SERVO_STEP_DEG = 1.4
SERVO_DEADZONE_DEG = 0.22

# Face-to-servo mapping range around center
PAN_TRACK_RANGE = 26.0
TILT_TRACK_RANGE = 18.0
CENTER_DEADZONE_X = 0.05
CENTER_DEADZONE_Y = 0.06
TARGET_FILTER_ALPHA = 0.30
FACE_SMOOTH_ALPHA = 0.35

NO_FACE_RECENTER_SEC = 1.5

# Persistent gaze tracking feature
FACE_LOST_HOLD_SEC = 3.0        # Hold last gaze position for 3 seconds
EYE_HEAD_RATIO = 0.65           # Eyes move 65% of head angle
EYE_MICRO_DRIFT_MAX = 2.5       # Random micro-drift ±2.5 degrees

# MJPEG debug stream
STREAM_ENABLED = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080
STREAM_FPS = 10
STREAM_JPEG_QUALITY = 75
STREAM_RES = (640, 360)

VISION_FPS = 16


# ---------------- Shared State ----------------
state_lock = threading.Lock()
frame_lock = threading.Lock()
latest_frame = None
stream_server = None

# Servo state (target/current) shared for worker + overlay
servo_state = {
    "target_pan": (PAN_MIN + PAN_MAX) / 2.0,
    "target_tilt": (TILT_MIN + TILT_MAX) / 2.0,
    "current_pan": (PAN_MIN + PAN_MAX) / 2.0,
    "current_tilt": (TILT_MIN + TILT_MAX) / 2.0,
}

# Vision state used for overlays/debug
vision_state = {
    "has_face": False,
    "face_box": None,
    "face_center": None,
    "smoothed_cx": CAMERA_RES[0] / 2.0,
    "smoothed_cy": CAMERA_RES[1] / 2.0,
    "last_face_ts": 0.0,
}

# Persistent gaze state
gaze_state = {
    "last_face_pan": (PAN_MIN + PAN_MAX) / 2.0,   # Last detected face pan angle
    "last_face_tilt": (TILT_MIN + TILT_MAX) / 2.0, # Last detected face tilt angle
    "face_lost_time": 0.0,                         # When did face disappear?
    "gaze_mode": "tracking",                      # tracking, holding, resetting
    "eye_offset_pan": 0.0,                        # Eye adjustment relative to head
    "eye_offset_tilt": 0.0,                       # Eye adjustment relative to head
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_deadzone_norm(value: float, deadzone: float) -> float:
    """Remove tiny center jitter while preserving full-scale output outside deadzone."""
    v = clamp(value, -1.0, 1.0)
    if abs(v) <= deadzone:
        return 0.0
    sign = 1.0 if v >= 0.0 else -1.0
    scaled = (abs(v) - deadzone) / (1.0 - deadzone)
    return sign * scaled


def find_model_path() -> Path:
    local = Path(__file__).resolve().with_name(MODEL_NAME)
    if local.exists():
        return local

    cwd = Path.cwd() / MODEL_NAME
    if cwd.exists():
        return cwd

    raise FileNotFoundError(f"Face model not found: {MODEL_NAME}")


# ---------------- MJPEG Server ----------------
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()

                if frame is None:
                    time.sleep(0.05)
                    continue

                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=STREAM_JPEG_QUALITY)
                jpg = buf.getvalue()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, _fmt, *_args):
        return


def start_stream_server():
    global stream_server
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), MJPEGHandler)
    thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream started: http://{STREAM_HOST}:{STREAM_PORT}/stream")


# ---------------- Workers ----------------
def vision_worker(stop_event: threading.Event, picam2: Picamera2, detector):
    interval = 1.0 / max(1.0, float(VISION_FPS))
    next_tick = time.perf_counter()

    while not stop_event.is_set():
        try:
            large_frame = picam2.capture_array()
            frame = cv2.resize(large_frame, CAMERA_RES)
            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)

            has_face = False
            face_box = None
            face_center = None

            if faces is not None and len(faces) > 0:
                has_face = True
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest[0:4]
                face_box = (int(fx), int(fy), int(fw), int(fh))
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0

                with state_lock:
                    scx_prev = vision_state["smoothed_cx"]
                    scy_prev = vision_state["smoothed_cy"]
                    scx = scx_prev + (cx - scx_prev) * FACE_SMOOTH_ALPHA
                    scy = scy_prev + (cy - scy_prev) * FACE_SMOOTH_ALPHA
                    vision_state["smoothed_cx"] = scx
                    vision_state["smoothed_cy"] = scy

                face_center = (int(scx), int(scy))

                norm_x = -((scx / CAMERA_RES[0]) - 0.5) * 2.0
                norm_y = ((scy / CAMERA_RES[1]) - 0.5) * 2.0
                norm_x = apply_deadzone_norm(norm_x, CENTER_DEADZONE_X)
                norm_y = apply_deadzone_norm(norm_y, CENTER_DEADZONE_Y)

                mapped_pan = ((PAN_MIN + PAN_MAX) / 2.0) + (norm_x * PAN_TRACK_RANGE)
                mapped_tilt = ((TILT_MIN + TILT_MAX) / 2.0) + (norm_y * TILT_TRACK_RANGE)

                mapped_pan = clamp(mapped_pan, PAN_MIN, PAN_MAX)
                mapped_tilt = clamp(mapped_tilt, TILT_MIN, TILT_MAX)

                with state_lock:
                    # Filter vision target updates to avoid detector jitter spikes.
                    prev_pan = servo_state["target_pan"]
                    prev_tilt = servo_state["target_tilt"]
                    target_pan = prev_pan + (mapped_pan - prev_pan) * TARGET_FILTER_ALPHA
                    target_tilt = prev_tilt + (mapped_tilt - prev_tilt) * TARGET_FILTER_ALPHA
                    servo_state["target_pan"] = target_pan
                    servo_state["target_tilt"] = target_tilt
                    vision_state["has_face"] = True
                    vision_state["face_box"] = face_box
                    vision_state["face_center"] = face_center
                    vision_state["last_face_ts"] = time.time()
                    
                    # Save last face position for persistent gaze
                    gaze_state["last_face_pan"] = target_pan
                    gaze_state["last_face_tilt"] = target_tilt
                    gaze_state["gaze_mode"] = "tracking"
                    gaze_state["face_lost_time"] = 0.0  # Reset timer
            else:
                with state_lock:
                    vision_state["has_face"] = False
                    vision_state["face_box"] = None
                    vision_state["face_center"] = None
                    
                    # Handle persistent gaze: hold last position for FACE_LOST_HOLD_SEC, then reset
                    if gaze_state["face_lost_time"] == 0.0:
                        # First frame without face - mark the time
                        gaze_state["face_lost_time"] = time.time()
                        gaze_state["gaze_mode"] = "holding"
                    
                    time_since_lost = time.time() - gaze_state["face_lost_time"]
                    
                    if time_since_lost < FACE_LOST_HOLD_SEC:
                        # Hold gaze at last face position
                        servo_state["target_pan"] = gaze_state["last_face_pan"]
                        servo_state["target_tilt"] = gaze_state["last_face_tilt"]
                        gaze_state["gaze_mode"] = "holding"
                    else:
                        # After FACE_LOST_HOLD_SEC, smooth reset to center
                        center_pan = (PAN_MIN + PAN_MAX) / 2.0
                        center_tilt = (TILT_MIN + TILT_MAX) / 2.0
                        servo_state["target_pan"] = center_pan
                        servo_state["target_tilt"] = center_tilt
                        gaze_state["gaze_mode"] = "resetting"

            if STREAM_ENABLED:
                stream_bgr = cv2.resize(frame, STREAM_RES)
                sx = STREAM_RES[0] / CAMERA_RES[0]
                sy = STREAM_RES[1] / CAMERA_RES[1]

                # Center crosshair
                cx_s = STREAM_RES[0] // 2
                cy_s = STREAM_RES[1] // 2
                cv2.line(stream_bgr, (cx_s - 18, cy_s), (cx_s + 18, cy_s), (255, 0, 0), 2)
                cv2.line(stream_bgr, (cx_s, cy_s - 18), (cx_s, cy_s + 18), (255, 0, 0), 2)

                if has_face and face_box is not None and face_center is not None:
                    fx, fy, fw, fh = face_box
                    cv2.rectangle(
                        stream_bgr,
                        (int(fx * sx), int(fy * sy)),
                        (int((fx + fw) * sx), int((fy + fh) * sy)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(stream_bgr, (int(face_center[0] * sx), int(face_center[1] * sy)), 4, (0, 255, 0), -1)

                with state_lock:
                    target_pan = servo_state["target_pan"]
                    target_tilt = servo_state["target_tilt"]
                    current_pan = servo_state["current_pan"]
                    current_tilt = servo_state["current_tilt"]

                cv2.putText(
                    stream_bgr,
                    f"PAN cur:{current_pan:5.1f} tgt:{target_pan:5.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    stream_bgr,
                    f"TILT cur:{current_tilt:5.1f} tgt:{target_tilt:5.1f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    stream_bgr,
                    "FACE: YES" if has_face else "FACE: NO",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if has_face else (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                frame_rgb = cv2.cvtColor(stream_bgr, cv2.COLOR_BGR2RGB)
                with frame_lock:
                    global latest_frame
                    latest_frame = frame_rgb

        except Exception as exc:
            print(f"Capture/Detect Error: {exc}")

        next_tick += interval
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()


def servo_worker(stop_event: threading.Event, kit: ServoKit):
    # Prime hardware at center before entering smoothing loop.
    with state_lock:
        pan_current = servo_state["current_pan"]
        tilt_current = servo_state["current_tilt"]
    kit.servo[PAN_CH].angle = pan_current
    kit.servo[TILT_CH].angle = tilt_current

    while not stop_event.is_set():
        with state_lock:
            pan_target = servo_state["target_pan"]
            tilt_target = servo_state["target_tilt"]
            pan_current = servo_state["current_pan"]
            tilt_current = servo_state["current_tilt"]

        # Fixed smoothing (testservos2 style) + deadzone + step clamp for stable behavior.
        pan_error = pan_target - pan_current
        tilt_error = tilt_target - tilt_current
        if abs(pan_error) < SERVO_DEADZONE_DEG:
            pan_error = 0.0
        if abs(tilt_error) < SERVO_DEADZONE_DEG:
            tilt_error = 0.0

        pan_step = clamp(pan_error * SMOOTHING, -MAX_SERVO_STEP_DEG, MAX_SERVO_STEP_DEG)
        tilt_step = clamp(tilt_error * SMOOTHING, -MAX_SERVO_STEP_DEG, MAX_SERVO_STEP_DEG)
        pan_current += pan_step
        tilt_current += tilt_step

        pan_current = clamp(pan_current, PAN_MIN, PAN_MAX)
        tilt_current = clamp(tilt_current, TILT_MIN, TILT_MAX)

        try:
            kit.servo[PAN_CH].angle = pan_current
            kit.servo[TILT_CH].angle = tilt_current
        except Exception as exc:
            print(f"Servo write error: {exc}")
        
        # Calculate head-eye synchronization (head leads, eyes follow with micro-adjustments)
        # Head moves full angle, eyes move EYE_HEAD_RATIO of head movement + random micro-drift
        center_pan = (PAN_MIN + PAN_MAX) / 2.0
        center_tilt = (TILT_MIN + TILT_MAX) / 2.0
        
        head_offset_pan = pan_current - center_pan
        head_offset_tilt = tilt_current - center_tilt
        
        # Eyes get fraction of head movement
        eye_pan_offset = head_offset_pan * EYE_HEAD_RATIO
        eye_tilt_offset = head_offset_tilt * EYE_HEAD_RATIO
        
        # Add tiny random micro-drift for natural \"living\" gaze

        micro_pan = random.uniform(-EYE_MICRO_DRIFT_MAX, EYE_MICRO_DRIFT_MAX)
        micro_tilt = random.uniform(-EYE_MICRO_DRIFT_MAX, EYE_MICRO_DRIFT_MAX)
        
        eye_pan_offset += micro_pan
        eye_tilt_offset += micro_tilt
        
        with state_lock:
            gaze_state["eye_offset_pan"] = eye_pan_offset
            gaze_state["eye_offset_tilt"] = eye_tilt_offset
            servo_state["current_pan"] = pan_current
            servo_state["current_tilt"] = tilt_current

        time.sleep(SERVO_LOOP_DELAY)


# ---------------- Main ----------------
def main():
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": CAMERA_MAIN_RES},
        raw={"size": (3280, 2464)},
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
    picam2.start()
    print(
        "Camera started: "
        f"full sensor (3280x2464) -> main ({CAMERA_MAIN_RES[0]}x{CAMERA_MAIN_RES[1]})"
    )

    print("Initializing YuNet face detector...")
    model_path = find_model_path()
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=CAMERA_RES,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    print(f"YuNet ready: {model_path}")

    print("Initializing ServoKit...")
    kit = ServoKit(channels=16)
    kit.servo[PAN_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
    kit.servo[TILT_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
    print("ServoKit ready.")

    if STREAM_ENABLED:
        start_stream_server()

    stop_event = threading.Event()
    vision_thread = threading.Thread(target=vision_worker, args=(stop_event, picam2, detector), daemon=True)
    servo_thread = threading.Thread(target=servo_worker, args=(stop_event, kit), daemon=True)

    vision_thread.start()
    servo_thread.start()

    print("Face tracker running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        vision_thread.join(timeout=2.0)
        servo_thread.join(timeout=2.0)

        try:
            kit.servo[PAN_CH].angle = None
            kit.servo[TILT_CH].angle = None
        except Exception as exc:
            print(f"Servo relax error: {exc}")

        try:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
        except Exception as exc:
            print(f"Camera close error: {exc}")

        global stream_server
        if stream_server is not None:
            try:
                stream_server.shutdown()
                stream_server.server_close()
                print("MJPEG stream stopped.")
            except Exception as exc:
                print(f"MJPEG shutdown error: {exc}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)
