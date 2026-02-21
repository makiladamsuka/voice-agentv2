"""
Continuous Face Monitor
Background service that monitors Raspberry Pi camera and tracks who's present.
Supports multi-person tracking with stability cache.
Uses picamera2 for Raspberry Pi camera.
"""

import cv2
import face_recognition
import threading
import time
import os
import math
from typing import Dict, List, Optional, Set
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    print("⚠️ Picamera2 not found. Camera functionality will be disabled.")
    HAS_PICAMERA = False
    class Picamera2: pass # Dummy class to prevent type errors

# from object_detector import ObjectDetector

# --- DEBUG SETTINGS ---
SHOW_DEBUG_VIDEO = False  # Set to True only if a monitor is attached to the Pi/PC
DEBUG_LOG_INTERVAL = 5.0  # Seconds between status prints (0 = disable)
CAMERA_ROTATE_180 = True  # Rotate camera if mounted upside down
# -----------------------

# --- STABILITY SETTINGS ---
FACE_CACHE_DURATION = 2.0  # Seconds to keep face in memory (prevents flicker)
GREETING_COOLDOWN = 60.0   # Seconds before re-greeting same person (1 minute)
# --------------------------

class FaceMonitor:
    """Background face monitoring service with multi-person tracking"""
    
    def __init__(self, known_faces: Dict[str, List]):
        self.known_faces = known_faces
        
        # Multi-person tracking (stable, from cache)
        self.current_people: Set[str] = set()    # Who's currently visible (stable)
        self.previous_people: Set[str] = set()   # For detecting arrivals/departures
        
        # Legacy single-person API (for backwards compatibility)
        self.current_person: Optional[str] = None
        self.previous_person: Optional[str] = None
        
        self.people_count = 0
        self.current_frame = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.picam2 = None  # Picamera2 instance
        
        # Face cache for stability (prevents flickering)
        self.face_cache: List[tuple] = []  # List of (timestamp, Set[names])
        self.fresh_people: Set[str] = set()  # Most recent detection (not cached)
        self.last_greeted: Dict[str, float] = {}  # Track when we greeted each person
        
        # Face Tracking Coordinates (Normalized -1.0 to 1.0)
        self.last_face_center: Optional[tuple] = None
        self.last_face_roll: float = 0.0
        
        # Object detection cache (last 5 seconds)
        self.object_cache = []  # List of (timestamp, detections)
        self.cache_duration = 5.0  # Keep last 5 seconds
        
        # YOLO / YuNet Initialization
        self.yolo_active = False
        self.detector = None 
        
        # --- YuNet Setup ---
        self.yunet_model_path = 'face_detection_yunet_2023mar.onnx'
        if os.path.exists(self.yunet_model_path):
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU
                )
                print(f"✅ YuNet Face Detector loaded from {self.yunet_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load YuNet detector: {e}")
        else:
            print(f"⚠️ {self.yunet_model_path} not found. Falling back to default detector.")
    
    # ==================== MULTI-PERSON API ====================
    
    def get_current_people(self) -> Set[str]:
        """Get all people currently visible (stable, from cache)"""
        with self.lock:
            return self.current_people.copy()
    
    def get_fresh_people(self) -> Set[str]:
        """Get people from the MOST RECENT detection (not cached).
        Use this for real-time context injection to avoid stale names."""
        with self.lock:
            return self.fresh_people.copy()
    
    def get_new_arrivals(self) -> List[str]:
        """Get people who just appeared (in fresh detection, weren't in previous).
        Uses fresh detection to avoid stale cache issues.
        Also filters out people greeted in last 60 seconds."""
        with self.lock:
            current_time = time.time()
            
            # Use fresh (most recent) detection, not cached stable
            arrivals = list(self.fresh_people - self.previous_people)
            
            # Filter out people we greeted recently (60 seconds cooldown)
            arrivals = [p for p in arrivals 
                       if current_time - self.last_greeted.get(p, 0) > GREETING_COOLDOWN]
            
            # Update previous to fresh (consume the event)
            self.previous_people = self.fresh_people.copy()
            
            return arrivals
    
    def mark_greeted(self, name: str):
        """Mark a person as greeted (prevents re-greeting for GREETING_COOLDOWN seconds)"""
        with self.lock:
            self.last_greeted[name] = time.time()
    
    def get_departures(self) -> List[str]:
        """Get people who left (were in previous, not in current).
        Note: Only triggers after FACE_CACHE_DURATION (2s) of absence."""
        with self.lock:
            departures = list(self.previous_people - self.current_people)
            return departures
    
    def _update_face_cache(self, detected_names: Set[str]):
        """Add face detections to cache and compute stable current_people"""
        current_time = time.time()
        
        # Add new detection
        self.face_cache.append((current_time, detected_names))
        
        # Remove old entries (> FACE_CACHE_DURATION)
        cutoff_time = current_time - FACE_CACHE_DURATION
        self.face_cache = [(t, names) for t, names in self.face_cache if t > cutoff_time]
        
        # Fresh people = most recent detection only (for accurate arrivals)
        self.fresh_people = detected_names.copy()
        
        # Stable people = anyone seen in last 2 seconds (prevents flicker)
        stable_people = set()
        for _, names in self.face_cache:
            stable_people.update(names)
        
        self.current_people = stable_people
        
        # Update legacy single-person field for backwards compatibility
        if len(stable_people) == 0:
            self.current_person = None
        elif len(stable_people) == 1:
            self.current_person = list(stable_people)[0]
        else:
            # Multiple people - could be mix of known/unknown
            known = [p for p in stable_people if p != "Unknown"]
            if known:
                self.current_person = known[0]  # First known person
            else:
                self.current_person = "Multiple"
    
    # ==================== LEGACY API (backwards compatible) ====================
    
    def get_current_person(self) -> Optional[str]:
        with self.lock:
            return self.current_person

    def get_face_center(self) -> Optional[tuple]:
        """Get the normalized coordinates (x, y) of the largest face.
        Range: -1.0 to 1.0. Returns None if no face visible."""
        with self.lock:
            return self.last_face_center

    def get_face_rotation(self) -> float:
        """Get the roll (tilt) angle of the largest face in degrees."""
        with self.lock:
            return self.last_face_roll

    def get_current_frame(self):
        with self.lock:
            return self.current_frame
    
    def person_changed(self) -> bool:
        """Check if person in front changed since last check.
        Only updates state when returning True (consumes the change)."""
        with self.lock:
            changed = self.current_person != self.previous_person
            if changed:
                self.previous_person = self.current_person
            return changed
    
    def person_arrived(self) -> bool:
        """Check if someone just appeared (was None, now someone).
        Does NOT consume the change - use person_changed() for that."""
        with self.lock:
            was_nobody = self.previous_person is None
            is_somebody = self.current_person is not None
            return was_nobody and is_somebody
    
    # ==================== OBJECT CACHE ====================
    
    def _update_object_cache(self, detections: List[Dict]):
        """Add detections to cache with timestamp"""
        current_time = time.time()
        self.object_cache.append((current_time, detections))
        cutoff_time = current_time - self.cache_duration
        self.object_cache = [(t, d) for t, d in self.object_cache if t > cutoff_time]
    
    def get_recent_objects(self, seconds: float = 5.0) -> List[Dict]:
        """Get all unique objects detected in last N seconds"""
        return []
        # current_time = time.time()
        # cutoff_time = current_time - seconds
        
        # with self.lock:
        #     recent = []
        #     for timestamp, detections in self.object_cache:
        #         if timestamp > cutoff_time:
        #             recent.extend(detections)
        #     
        #     unique_objects = {}
        #     for det in recent:
        #         class_name = det['class']
        #         if class_name not in unique_objects:
        #             unique_objects[class_name] = det
        #         elif det['confidence'] > unique_objects[class_name]['confidence']:
        #             unique_objects[class_name] = det
        #     
        #     return list(unique_objects.values())
    
    # ==================== LIFECYCLE ====================
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        
        # Auto-enable debug video ONLY if specifically True and not on Pi (with caution)
        # On Pi we usually want this False unless a monitor is attached.
        if not HAS_PICAMERA:
            # If we are on PC but didn't set it, maybe enable it. 
            # But let's respect the initial setting more strictly.
            pass
            
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"🎥 Face monitor started (multi-person mode, Debug: {SHOW_DEBUG_VIDEO})")
    
    def stop(self):
        self.is_running = False
        if self.picam2:
            try:
                self.picam2.stop()
            except:
                pass
        if SHOW_DEBUG_VIDEO:
            cv2.destroyAllWindows()
        print("🛑 Face monitor stopped")
            
    def _monitor_loop(self):
        """Monitor loop using picamera2 with cv2 fallback"""
        if not HAS_PICAMERA:
            print("⚠️ Picamera2 missing - falling back to USB Webcam (cv2)...")
            self._monitor_loop_cv2()
            return
            
        print("🎥 Initializing picamera2...")
        
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"format": 'XRGB8888', "size": (1280, 720)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.5)  # Allow camera to warm up
            
            # Test capture
            test_frame = self.picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                print(f"✅ Picamera2 started (size: {test_frame.shape[1]}x{test_frame.shape[0]})")
            else:
                print("❌ Failed to capture test frame")
                self.is_running = False
                return
                
        except Exception as e:
            print(f"⚠️ Picamera2 initialization failed: {e}")
            print(f"🔄 Falling back to USB Webcam (cv2)...")
            self._monitor_loop_cv2()
            return
        
        frame_count = 0
        last_debug_time = time.time()
        
        while self.is_running:
            try:
                # Capture frame from picamera2
                frame = self.picam2.capture_array()
                # XRGB8888 is actually BGRX format in memory, drop the X channel
                frame = frame[:, :, :3]  # Keep only BGR channels
                
                if CAMERA_ROTATE_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                with self.lock:
                    self.current_frame = frame.copy()
                
                frame_count += 1
                
                # Process every 5th frame for face recognition
                if frame_count % 5 == 0:
                    self._process_frame(frame)
                
                # Render logic
                if SHOW_DEBUG_VIDEO:
                    self._render_debug_window(frame)
                
                # Small delay to regulate FPS
                time.sleep(0.01)
            
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                time.sleep(0.5)
                continue

    def _monitor_loop_cv2(self):
        """Fallback loop using standard cv2.VideoCapture (for PC)"""
        print("🎥 Initializing USB Webcam (cv2)...")
        cap = None
        for index in range(5):
            print(f"🔍 Testing camera index {index}...")
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                print(f"✅ USB Webcam initialized at index {index}.")
                break
            cap.release()
            cap = None
            
        if cap is None:
            print("❌ Could not open any USB webcam (tried indices 0-4).")
            self.is_running = False
            return
            
        frame_count = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            with self.lock:
                self.current_frame = frame.copy()
            
            frame_count += 1
            if frame_count % 5 == 0:
                self._process_frame(frame)
            
            if SHOW_DEBUG_VIDEO:
                self._render_debug_window(frame)
            else:
                time.sleep(0.01)
        
        cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Face processing using YuNet with face_recognition fallback"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        face_locations = []
        face_rolls = [] # Track rolls for each detected face
        detected_names: Set[str] = set()
        largest_face_center = None
        largest_face_roll = 0.0
        max_area = 0

        # -- DETECTION --
        if self.detector:
            # YuNet Detection
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(frame)
            
            if faces is not None:
                for face in faces:
                    # YuNet coordinates: [x, y, w, h, re_x, re_y, le_x, le_y, ...]
                    coords = list(map(int, face[:14]))
                    x, y, w, h = coords[:4]
                    re_x, re_y = coords[4], coords[5]
                    le_x, le_y = coords[6], coords[7]

                    # Convert to face_recognition CSS format: (top, right, bottom, left)
                    face_locations.append((y, x + w, y + h, x))
                    
                    # Calculate roll for this face
                    dx = re_x - le_x
                    dy = re_y - le_y
                    roll = 0.0
                    if dx != 0:
                        roll = math.degrees(math.atan2(dy, dx))
                    face_rolls.append(roll)
        else:
            # Fallback to face_recognition (HOG/CNN)
            face_locations = face_recognition.face_locations(rgb_frame)

        # -- RECOGNITION & TRACKING --
        if len(face_locations) > 0:
            # Get encodings for all detected faces
            encs = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Calculate tracking coordinates for eye engine
                area = (bottom - top) * (right - left)
                if area > max_area:
                    max_area = area
                    cx, cy = (left + right) / 2, (top + bottom) / 2
                    # Normalized -1.0 to 1.0 (X is flipped for mirroring)
                    largest_face_center = ((0.5 - cx / width) * 2.0, (cy / height - 0.5) * 2.0)
                    # Pick roll for the largest face
                    if i < len(face_rolls):
                        largest_face_roll = face_rolls[i]

                # Identify person
                if i < len(encs):
                    match_name = "Unknown"
                    for kname, kencs in self.known_faces.items():
                        matches = face_recognition.compare_faces(kencs, encs[i], tolerance=0.5)
                        if True in matches: 
                            match_name = kname
                            break
                    detected_names.add(match_name)
        
        with self.lock:
            self.last_face_center = largest_face_center
            self.last_face_roll = largest_face_roll
            self._update_face_cache(detected_names)
            self._last_face_locs = face_locations
            self._last_detected_names = list(detected_names)
            
            # If no face detected, clear the center immediately 
            # (voice_agent.py also handles this but this ensures consistency)
            if largest_face_center is None:
                self.last_face_center = None
                self.last_face_roll = 0.0

    def _render_debug_window(self, frame):
        """Unified debug window renderer"""
        display = frame.copy()
        face_locs = getattr(self, '_last_face_locs', [])
        names = getattr(self, '_last_detected_names', [])
        
        for i, (top, right, bottom, left) in enumerate(face_locs):
            name = names[i] if i < len(names) else "Unknown"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            cv2.putText(display, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        cv2.imshow("Voice Agent - Face Monitor Debug", display)
        cv2.waitKey(1)
