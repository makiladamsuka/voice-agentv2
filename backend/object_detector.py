"""
Object Detection Service using YOLOv8
Provides environmental awareness for the voice agent.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class ObjectDetector:
    """YOLO-based object detection for environmental awareness"""
    
    def __init__(self, model_name: str = "yolov8s.pt"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model to use (yolov8n.pt = nano, yolov8s.pt = small, etc.)
        """
        # Set models directory
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / model_name
        
        print(f"🔍 Loading YOLO model: {model_name}")
        print(f"📁 Models directory: {models_dir}")
        
        # Download to models/ folder if not exists
        self.model = YOLO(str(model_path))
        
        # COCO class names
        self.class_names = self.model.names
        print(f"✅ YOLO model loaded with {len(self.class_names)} classes")
    
    def detect_objects(self, frame, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Image frame (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            List of detected objects with format:
            [{'class': 'person', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}, ...]
        """
        if frame is None:
            return []
        
        # Run YOLO inference
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    'class': self.class_names[class_id],
                    'confidence': confidence,
                    'bbox': bbox  # [x1, y1, x2, y2]
                })
        
        return detections
    
    def count_people(self, frame) -> int:
        """Count number of people in frame"""
        detections = self.detect_objects(frame)
        people = [d for d in detections if d['class'] == 'person']
        return len(people)
    
    def find_object(self, frame, object_name: str) -> Optional[Dict]:
        """
        Find specific object in frame
        
        Args:
            frame: Image frame
            object_name: Object to find (e.g., "laptop", "bottle", "phone")
        
        Returns:
            Detection dict if found, None otherwise
        """
        detections = self.detect_objects(frame)
        
        # Normalize object name
        object_name = object_name.lower().strip()
        
        # Find matching detections
        matches = []
        for detection in detections:
            if object_name in detection['class'].lower():
                matches.append(detection)
        
        # Return highest confidence match
        if matches:
            return max(matches, key=lambda d: d['confidence'])
        return None
    
    def describe_scene(self, frame, max_objects: int = 5) -> Dict:
        """
        Get overall scene description
        
        Returns:
            {
                'people_count': int,
                'main_objects': ['laptop', 'phone', ...],
                'total_objects': int
            }
        """
        detections = self.detect_objects(frame)
        
        # Count people
        people = [d for d in detections if d['class'] == 'person']
        people_count = len(people)
        
        # Get non-person objects, sorted by confidence
        objects = [d for d in detections if d['class'] != 'person']
        objects.sort(key=lambda d: d['confidence'], reverse=True)
        
        # Get main objects (unique classes only)
        seen_classes = set()
        main_objects = []
        for obj in objects:
            if obj['class'] not in seen_classes:
                main_objects.append(obj['class'])
                seen_classes.add(obj['class'])
                if len(main_objects) >= max_objects:
                    break
        
        return {
            'people_count': people_count,
            'main_objects': main_objects,
            'total_objects': len(objects)
        }
    
    def get_object_region(self, frame, object_name: str) -> Optional[np.ndarray]:
        """
        Extract image region of specific object
        Useful for color analysis of specific items
        
        Returns:
            Cropped image region if object found, None otherwise
        """
        detection = self.find_object(frame, object_name)
        
        if detection:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop region
            region = frame[y1:y2, x1:x2]
            return region
        
        return None
