"""
Continuous Face Monitor
Background service that monitors webcam and tracks who's present.
Supports multi-person tracking with stability cache.
"""

import cv2
import face_recognition
import threading
import time
from typing import Dict, List, Optional, Set
from object_detector import ObjectDetector

# --- DEBUG SETTINGS ---
SHOW_DEBUG_VIDEO = True  # Toggle debug window
# ----------------------

# --- STABILITY SETTINGS ---
FACE_CACHE_DURATION = 2.0  # Seconds to keep face in memory (prevents flicker)
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
        
        # Face cache for stability (prevents flickering)
        self.face_cache: List[tuple] = []  # List of (timestamp, Set[names])
        
        # Object detection cache (last 5 seconds)
        self.object_cache = []  # List of (timestamp, detections)
        self.cache_duration = 5.0  # Keep last 5 seconds
        
        # Init YOLO for debug view AND object cache
        print("🔍 Initializing YOLO for debug view and object cache...")
        try:
            self.detector = ObjectDetector()
            self.yolo_active = True
        except Exception as e:
            print(f"⚠️ YOLO init failed: {e}")
            self.yolo_active = False
    
    # ==================== NEW MULTI-PERSON API ====================
    
    def get_current_people(self) -> Set[str]:
        """Get all people currently visible (stable, from cache)"""
        with self.lock:
            return self.current_people.copy()
    
    def get_new_arrivals(self) -> List[str]:
        """Get people who just appeared (weren't in previous check).
        Consumes the change - next call returns empty unless new people arrive."""
        with self.lock:
            arrivals = list(self.current_people - self.previous_people)
            # Update previous to current (consume the event)
            self.previous_people = self.current_people.copy()
            return arrivals
    
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
        
        # Compute stable set (anyone seen in last 2 seconds)
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
        if self.video_capture:
            self.video_capture.release()
        if SHOW_DEBUG_VIDEO:
            cv2.destroyAllWindows()
        print("🛑 Face monitor stopped")
            
    def _monitor_loop(self):
        # Try multiple camera backends
        self.video_capture = None
        configs = [(0, cv2.CAP_ANY, "ANY/0"), (1, cv2.CAP_ANY, "ANY/1"), (0, cv2.CAP_V4L2, "V4L2/0")]
        
        for idx, backend, name in configs:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.video_capture = cap
                        print(f"✅ Camera: {name}")
                        break
                    cap.release()
            except: pass
            
        if not self.video_capture or not self.video_capture.isOpened():
            print("❌ Cannot open webcam")
            self.is_running = False
            return
        
        frame_count = 0
        while self.is_running:
            ret, frame = self.video_capture.read()
            if not ret:
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
                
                # Update object cache every 10 frames
                if self.yolo_active and frame_count % 10 == 0:
                    cache_detections = self.detector.detect_objects(frame)
                    with self.lock:
                        self._update_object_cache(cache_detections)
                
                # Status - show stable people
                stable_str = ", ".join(self.current_people) if self.current_people else "None"
                status = f"Stable: [{stable_str}] | YOLO People: {len([d for d in ydets if d['class']=='person'])}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Show face cache info
                cache_info = f"Face cache: {len(self.face_cache)} entries, {FACE_CACHE_DURATION}s window"
                cv2.putText(display, cache_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                cv2.imshow("Debug View", display)
                cv2.waitKey(1)
