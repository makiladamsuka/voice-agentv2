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
from typing import Dict, List, Optional, Set
from picamera2 import Picamera2
from object_detector import ObjectDetector

# --- DEBUG SETTINGS ---
SHOW_DEBUG_VIDEO = True  # Set True to show camera window on HDMI display
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
        
        # Object detection cache (last 5 seconds)
        self.object_cache = []  # List of (timestamp, detections)
        self.cache_duration = 5.0  # Keep last 5 seconds
        
        # YOLO disabled for better performance on Raspberry Pi
        print("🔍 YOLO disabled for better performance on Raspberry Pi")
        self.yolo_active = False
        self.detector = ObjectDetector(load_yolo=False)
    
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
            try:
                self.picam2.stop()
            except:
                pass
        if SHOW_DEBUG_VIDEO:
            cv2.destroyAllWindows()
        print("🛑 Face monitor stopped")
            
    def _monitor_loop(self):
        """Monitor loop using picamera2"""
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
            print(f"❌ Picamera2 initialization failed: {e}")
            self.is_running = False
            return
        
        frame_count = 0
        while self.is_running:
            try:
                # Capture frame from picamera2
                frame = self.picam2.capture_array()
                # Convert XRGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.current_frame = frame.copy()
                
                frame_count += 1
                
                # Process every 5th frame for face recognition
                if frame_count % 5 == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Face Detection - tracks ALL faces
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
                    
                    with self.lock:
                        # Update face cache (handles stability)
                        self._update_face_cache(detected_names)
                        self._last_face_locs = face_locs
                        self._last_detected_names = list(detected_names)
                
                # Debug Display on HDMI
                if SHOW_DEBUG_VIDEO:
                    display = frame.copy()
                    
                    # Draw faces with names
                    face_locs = getattr(self, '_last_face_locs', [])
                    names = getattr(self, '_last_detected_names', [])
                    
                    for i, (top, right, bottom, left) in enumerate(face_locs):
                        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
                        name = names[i] if i < len(names) else "?"
                        cv2.putText(display, name, (left, bottom + 25), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Status text
                    people_str = ", ".join(self.current_people) if self.current_people else "None"
                    cv2.putText(display, f"People: {people_str}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow("Face Monitor", display)
                    cv2.waitKey(1)
            
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                time.sleep(0.5)
                continue
