"""
Continuous Face Monitor
Background service that monitors webcam and tracks who's present.
"""

import cv2
import face_recognition
import threading
import time
from typing import Dict, List, Optional
from object_detector import ObjectDetector

# --- DEBUG SETTINGS ---
SHOW_DEBUG_VIDEO = True  # Toggle debug window
# ----------------------

class FaceMonitor:
    """Background face monitoring service"""
    
    def __init__(self, known_faces: Dict[str, List]):
        self.known_faces = known_faces
        self.current_person: Optional[str] = None
        self.previous_person: Optional[str] = None  # Track changes
        self.people_count = 0
        self.current_frame = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.video_capture = None
        
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
            # Detect transition from nobody to somebody
            was_nobody = self.previous_person is None
            is_somebody = self.current_person is not None
            return was_nobody and is_somebody
    
    def _update_object_cache(self, detections: List[Dict]):
        """Add detections to cache with timestamp"""
        current_time = time.time()
        
        # Add new detection (lock already held by caller)
        self.object_cache.append((current_time, detections))
        
        # Remove old detections (> cache_duration)
        cutoff_time = current_time - self.cache_duration
        self.object_cache = [(t, d) for t, d in self.object_cache 
                            if t > cutoff_time]
    
    def get_recent_objects(self, seconds: float = 5.0) -> List[Dict]:
        """Get all unique objects detected in last N seconds"""
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        with self.lock:
            # Collect all recent detections
            recent = []
            for timestamp, detections in self.object_cache:
                if timestamp > cutoff_time:
                    recent.extend(detections)
            
            # Return unique objects (by class name)
            # Keep highest confidence for each class
            unique_objects = {}
            for det in recent:
                class_name = det['class']
                if class_name not in unique_objects:
                    unique_objects[class_name] = det
                elif det['confidence'] > unique_objects[class_name]['confidence']:
                    unique_objects[class_name] = det
            
            return list(unique_objects.values())
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("🎥 Face monitor started")
    
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
                
                # Face Detection
                face_locs = face_recognition.face_locations(rgb_frame)
                current_name = None
                
                if len(face_locs) == 1:
                    encs = face_recognition.face_encodings(rgb_frame, face_locs)
                    if encs:
                        match_name = "Unknown"
                        for kname, kencs in self.known_faces.items():
                            matches = face_recognition.compare_faces(kencs, encs[0], tolerance=0.5)
                            if True in matches: 
                                match_name = kname
                                break
                        current_name = match_name
                elif len(face_locs) > 1:
                    current_name = "Multiple"
                
                # YOLO Detection (for debug view)
                yolo_dets = []
                p_count = 0
                if self.yolo_active:
                    yolo_dets = self.detector.detect_objects(frame)
                    p_count = sum(1 for d in yolo_dets if d['class'] == 'person')
                
                with self.lock:
                    self.current_person = current_name
                    self.people_count = p_count
                    self._last_face_locs = face_locs
                    self._last_name = current_name
                    self._last_yolo_dets = yolo_dets
            
            # Debug Display
            if SHOW_DEBUG_VIDEO:
                display = frame.copy()
                flocs = getattr(self, '_last_face_locs', [])
                fname = getattr(self, '_last_name', None)
                ydets = getattr(self, '_last_yolo_dets', [])
                
                # Draw faces
                for t, r, b, l in flocs:
                    cv2.rectangle(display, (l, t), (r, b), (255, 0, 0), 2)
                    if fname:
                        cv2.putText(display, fname, (l, b+25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                
                # Draw YOLO
                for det in ydets:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    lbl = f"{det['class']} {det['confidence']:.1f}"
                    clr = (0, 255, 0) if det['class'] == 'person' else (255, 0, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), clr, 2)
                    cv2.putText(display, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
                
                # YOLO detection every 5th frame for debug display
                if self.yolo_active and frame_count % 5 == 0:
                    yolo_detections = self.detector.detect_objects(frame)
                    
                    # Update people count
                    people = [d for d in yolo_detections if d['class'] == 'person']
                    self.people_count = len(people)
                    
                    # Store for debug display
                    for det in yolo_detections:
                        bbox = [int(x) for x in det['bbox']]
                        label = f"{det['class']} {int(det['confidence']*100)}%"
                        
                        # Draw bounding box
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(display, label, (bbox[0], bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update object cache every 10 frames (~0.3s) for continuous tracking
                if self.yolo_active and frame_count % 10 == 0:
                    cache_detections = self.detector.detect_objects(frame)
                    with self.lock: # Ensure thread safety for cache update
                        self._update_object_cache(cache_detections)
                
                # Status
                status = f"Face: {fname or 'None'} | People: {len([d for d in ydets if d['class']=='person'])}"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show cached objects (last 5 seconds)
                recent_objs = self.get_recent_objects(seconds=5.0)
                
                # Show title
                cv2.putText(display, "Cached Objects (5s):", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                if recent_objs:
                    # Show objects (exclude person, limit to 8)
                    non_person_objs = [obj for obj in recent_objs if obj['class'] != 'person'][:8]
                    y_offset = 85
                    for obj in non_person_objs:
                        obj_text = f"  {obj['class']} ({int(obj['confidence']*100)}%)"
                        cv2.putText(display, obj_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                        y_offset += 20
                else:
                    cv2.putText(display, "  (none)", (10, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
                
                
                cv2.imshow("Debug View", display)
                
                
                cv2.imshow("Debug View", display)
                cv2.waitKey(1)
