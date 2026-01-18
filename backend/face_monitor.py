"""
Continuous Face Monitor
Background service that monitors webcam and tracks who's present.
Supports multi-person tracking with stability cache.
Supports both USB webcams and Raspberry Pi cameras via OpenCV.
"""

import cv2
import face_recognition
import threading
import time
import fcntl
from typing import Dict, List, Optional, Set
from pathlib import Path
from object_detector import ObjectDetector

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️ picamera2 not available - will use OpenCV only")

# Global lock file for camera access (ensures only one process uses picamera2)
CAMERA_LOCK_FILE = Path("/tmp/picamera2.lock")

# Global lock file for camera access (ensures only one process uses picamera2)
CAMERA_LOCK_FILE = Path("/tmp/picamera2.lock")

# --- DEBUG SETTINGS ---
SHOW_DEBUG_VIDEO = False  # Toggle debug window (set False for SSH/headless)
# ----------------------

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
        self.video_capture = None
        self.picam2 = None  # Picamera2 instance for Raspberry Pi
        self._camera_lock_fd = None  # File descriptor for camera lock
        
        # Face cache for stability (prevents flickering)
        self.face_cache: List[tuple] = []  # List of (timestamp, Set[names])
        self.fresh_people: Set[str] = set()  # Most recent detection (not cached)
        self.last_greeted: Dict[str, float] = {}  # Track when we greeted each person
        
        # Object detection cache (last 5 seconds)
        self.object_cache = []  # List of (timestamp, detections)
        self.cache_duration = 5.0  # Keep last 5 seconds
        
        # Init YOLO for debug view AND object cache (disabled by default for performance)
        print("🔍 YOLO disabled for better performance on Raspberry Pi")
        self.yolo_active = False
        self.detector = ObjectDetector(load_yolo=False)
    # ==================== NEW MULTI-PERSON API ====================
    
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
        Also filters out people greeted in last 5 seconds."""
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
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        with self.lock:
            recent = []
            for timestamp, detections in self.object_cache:
                if timestamp > cutoff_time:
                    recent.extend(detections)
            
            unique_objects = {}
            for det in recent:
                class_name = det['class']
                if class_name not in unique_objects:
                    unique_objects[class_name] = det
                elif det['confidence'] > unique_objects[class_name]['confidence']:
                    unique_objects[class_name] = det
            
            return list(unique_objects.values())
    
    # ==================== LIFECYCLE ====================
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("🎥 Face monitor started (multi-person mode)")
    
    def stop(self):
        self.is_running = False
        if self.picam2:
            self.picam2.stop()
        if self.video_capture:
            self.video_capture.release()
        if SHOW_DEBUG_VIDEO:
            cv2.destroyAllWindows()
        # Release camera lock if we have it
        if hasattr(self, '_camera_lock_fd') and self._camera_lock_fd:
            try:
                fcntl.flock(self._camera_lock_fd.fileno(), fcntl.LOCK_UN)
                self._camera_lock_fd.close()
                print("🔓 Released camera lock")
            except:
                pass
        print("🛑 Face monitor stopped")
            
    def _monitor_loop(self):
        """Monitor loop that captures frames from either USB camera or Raspberry Pi camera"""
        self.video_capture = None
        camera_type = None
        
        # Try picamera2 first (native Raspberry Pi camera support)
        # Following camtest.py pattern: use XRGB8888 format, size (1280, 720)
        print("🎥 Attempting to open camera...")
        camera_lock_fd = None
        if PICAMERA2_AVAILABLE:
            # Try to acquire camera lock (prevents multiple processes from using camera)
            try:
                CAMERA_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
                camera_lock_fd = open(CAMERA_LOCK_FILE, 'w')
                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(camera_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                print("🔒 Acquired camera lock - attempting to open picamera2")
            except (IOError, OSError) as lock_error:
                # Camera is locked by another process
                print(f"⚠️ Camera is already in use by another process - will use fallback")
                if camera_lock_fd:
                    camera_lock_fd.close()
                    camera_lock_fd = None
                # Don't try picamera2 if locked, skip to fallback
                pass
            else:
                # We have the lock, try to use picamera2
                try:
                    self.picam2 = Picamera2()
                    # Match camtest.py configuration exactly
                    config = self.picam2.create_video_configuration(
                        main={"format": 'XRGB8888', "size": (1280, 720)}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(0.5)  # Allow camera to warm up
                    # Test capture
                    test_frame = self.picam2.capture_array()
                    if test_frame is not None and test_frame.size > 0:
                        camera_type = "picamera2"
                        print(f"✅ Camera: Picamera2 (size: {test_frame.shape[1]}x{test_frame.shape[0]})")
                        # Store lock file descriptor for cleanup
                        self._camera_lock_fd = camera_lock_fd
                    else:
                        # Release lock if initialization failed
                        if camera_lock_fd:
                            fcntl.flock(camera_lock_fd.fileno(), fcntl.LOCK_UN)
                            camera_lock_fd.close()
                            camera_lock_fd = None
                except Exception as e:
                    import traceback
                    print(f"⚠️ Picamera2 failed: {e}")
                    print(f"   Error details: {type(e).__name__}")
                    # Release lock on error
                    if camera_lock_fd:
                        try:
                            fcntl.flock(camera_lock_fd.fileno(), fcntl.LOCK_UN)
                            camera_lock_fd.close()
                        except:
                            pass
                        camera_lock_fd = None
                    if self.picam2:
                        try:
                            self.picam2.stop()
                        except:
                            pass
                        self.picam2 = None
        
        # Fallback to OpenCV VideoCapture (USB webcams)
        if not camera_type:
            configs = [(0, cv2.CAP_V4L2, "V4L2/0"), (0, cv2.CAP_ANY, "ANY/0"), 
                       (1, cv2.CAP_V4L2, "V4L2/1"), (1, cv2.CAP_ANY, "ANY/1")]
            
            for idx, backend, name in configs:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            self.video_capture = cap
                            camera_type = "opencv"
                            print(f"✅ Camera: OpenCV {name} (size: {frame.shape[1]}x{frame.shape[0]})")
                            break
                        cap.release()
                except Exception as e:
                    pass
        
        if not camera_type:
            print("❌ Cannot open any camera")
            self.is_running = False
            return
        
        frame_count = 0
        while self.is_running:
            try:
                # Capture frame based on camera type
                if camera_type == "picamera2":
                    frame = self.picam2.capture_array()
                    # Convert XRGB to BGR for OpenCV compatibility
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ret = True
                else:
                    ret, frame = self.video_capture.read()
                
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue
                
                with self.lock:
                    self.current_frame = frame.copy()
                
                frame_count += 1
                
                # Process every 5th frame
                if frame_count % 5 == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Face Detection - now tracks ALL faces
                    face_locs = face_recognition.face_locations(rgb_frame)
                    detected_names: Set[str] = set()
                    
                    if len(face_locs) > 0:
                        encs = face_recognition.face_encodings(rgb_frame, face_locs)
                        
                        for i, enc in enumerate(encs):
                            match_name = "Unknown"
                            for kname, kencs in self.known_faces.items():
                                matches = face_recognition.compare_faces(kencs, enc, tolerance=0.5)
                                if True in matches: 
                                    match_name = kname
                                    break
                            detected_names.add(match_name)
                    
                    # YOLO Detection (for debug view)
                    yolo_dets = []
                    p_count = 0
                    if self.yolo_active:
                        yolo_dets = self.detector.detect_objects(frame)
                        p_count = sum(1 for d in yolo_dets if d['class'] == 'person')
                    
                    with self.lock:
                        # Update face cache (handles stability)
                        self._update_face_cache(detected_names)
                        
                        self.people_count = p_count
                        self._last_face_locs = face_locs
                        self._last_detected_names = detected_names
                        self._last_yolo_dets = yolo_dets
                        
                        # Update object cache using same YOLO results (every frame we process)
                        if self.yolo_active and yolo_dets:
                            self._update_object_cache(yolo_dets)
                
                # Debug Display
                if SHOW_DEBUG_VIDEO:
                    display = frame.copy()
                    flocs = getattr(self, '_last_face_locs', [])
                    detected = getattr(self, '_last_detected_names', set())
                    ydets = getattr(self, '_last_yolo_dets', [])
                    
                    # Draw faces with their names
                    names_list = list(detected) if detected else []
                    for i, (t, r, b, l) in enumerate(flocs):
                        cv2.rectangle(display, (l, t), (r, b), (255, 0, 0), 2)
                        name = names_list[i] if i < len(names_list) else "?"
                        cv2.putText(display, name, (l, b+25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                    
                    # Draw YOLO
                    for det in ydets:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        lbl = f"{det['class']} {det['confidence']:.1f}"
                        clr = (0, 255, 0) if det['class'] == 'person' else (255, 0, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), clr, 2)
                        cv2.putText(display, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
                    
                    # Status - show stable people
                    stable_str = ", ".join(self.current_people) if self.current_people else "None"
                    status = f"Stable: [{stable_str}] | YOLO People: {len([d for d in ydets if d['class']=='person'])}"
                    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Show face cache info
                    cache_info = f"Face cache: {len(self.face_cache)} entries, {FACE_CACHE_DURATION}s window"
                    cv2.putText(display, cache_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Show object cache info
                    cached_objs = self.get_recent_objects(seconds=5.0)
                    obj_names = [o['class'] for o in cached_objs if o['class'] != 'person'][:5]
                    obj_cache_str = f"Object cache: {', '.join(obj_names) if obj_names else 'None'}"
                    cv2.putText(display, obj_cache_str, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cv2.imshow("Debug View", display)
                    cv2.waitKey(1)
            
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
