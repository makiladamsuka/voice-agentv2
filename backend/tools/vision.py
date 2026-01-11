import cv2
import numpy as np
import face_recognition
import pickle
from pathlib import Path
from collections import Counter
from livekit.agents import RunContext
from livekit.agents.llm import function_tool

class VisionTools:
    def __init__(self, face_monitor, object_detector_factory):
        self.face_monitor = face_monitor
        self.get_object_detector = object_detector_factory

    @property
    def object_detector(self):
        return self.get_object_detector()

    def _get_color_name(self, rgb, np):
        """Convert RGB to color name using HSV"""
        r, g, b = rgb
        rgb_pixel = np.uint8([[[r, g, b]]])
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
        h, s, v = hsv_pixel[0][0]
        
        if s < 30:
            if v > 200: return "white"
            elif v < 50: return "black"
            else: return "gray"
        
        if v < 50: return "black"
        
        if (h <= 10) or (h >= 170): return "red"
        elif h <= 25: return "orange"
        elif h <= 34: return "yellow"
        elif h <= 85: return "green"
        elif h <= 100: return "cyan"
        elif h <= 130: return "blue"
        elif h <= 155: return "purple"
        else: return "pink"

    # @function_tool  # Removed - called as helper method
    async def recognize_face(self, context: RunContext) -> str:
        """
        Identifies who is currently in front of the webcam.
        Use when user asks who is there or wants to be recognized.
        """
        print("🎥 Face recognition using continuous monitor!")
        
        if not self.face_monitor:
            return "Face monitoring is not active."
        
        current_person = self.face_monitor.get_current_person()
        
        if not current_person:
            return "I don't see anyone right now."
        elif current_person == "Unknown":
            return "I see someone but don't recognize them. Would you like to enroll?"
        elif current_person.startswith("Multiple"):
            return "I see multiple people. Please make sure only one person is in frame."
        else:
            return f"Hello {current_person}! Nice to see you."

    # @function_tool  # Removed - called as helper method
    async def enroll_new_face(self, person_name: str, context: RunContext) -> str:
        """
        Enroll a new person's face for recognition.
        Use when someone introduces themselves and you want to remember them.
        
        Args:
            person_name: The person's name to save
        """
        print(f"📝 Enrolling: {person_name}")
        
        if not self.face_monitor:
            return "Face monitoring is not active."
            
        frame = self.face_monitor.get_current_frame()
        
        if frame is None:
            return "I can't see anyone right now. Please make sure you're in front of the camera."
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locs = face_recognition.face_locations(rgb_frame)
        
        if len(face_locs) == 0:
            return "I don't see any faces. Please position yourself in front of the camera."
        
        if len(face_locs) > 1:
            return "I see multiple faces. Please make sure only you are in the frame."
        
        encodings = face_recognition.face_encodings(rgb_frame, face_locs)
        
        if not encodings:
            return "I couldn't get a clear face encoding. Please try again."
        
        # Save encoding - NOTE: We need to access the known_faces path. 
        # Assuming we stay relative to the module or use a config. 
        # For now, we'll traverse up from this file location.
        encodings_dir = Path(__file__).parent.parent / "known_faces"
        encodings_dir.mkdir(exist_ok=True)
        encodings_path = encodings_dir / "encodings.pkl"
        
        # Load existing
        if encodings_path.exists():
            with open(encodings_path, 'rb') as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}
        
        # Add new person
        if person_name in known_faces:
            known_faces[person_name].append(encodings[0])
        else:
            known_faces[person_name] = [encodings[0]]
        
        # Save
        with open(encodings_path, 'wb') as f:
            pickle.dump(known_faces, f)
        
        # Update runtime
        if self.face_monitor:
            self.face_monitor.known_faces = known_faces
        
        print(f"✅ Enrolled: {person_name}")
        return f"Great! I'll remember you, {person_name}. Nice to meet you!"

    # @function_tool  # Removed - called as helper method
    async def identify_color(self, context: RunContext) -> str:
        """
        Identifies the dominant color in the camera view.
        Use when user asks about colors they're wearing or showing.
        """
        print("🎨 Color detection using shared frame!")
        
        if not self.face_monitor:
            return "Camera monitoring is not active."
        
        frame = self.face_monitor.get_current_frame()
        if frame is None:
            return "No camera frame available."
        
        # Process frame
        small_frame = cv2.resize(frame, (150, 150))
        pixels = small_frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        label_counts = Counter(labels.flatten())
        dominant_label = label_counts.most_common(1)[0][0]
        dominant_color_bgr = centers[dominant_label]
        dominant_color_rgb = dominant_color_bgr[::-1]
        
        color_name = self._get_color_name(dominant_color_rgb, np)
        
        return f"The dominant color I see is {color_name}."

    # @function_tool  # Removed - called as helper method
    async def describe_environment(self, context: RunContext) -> str:
        """
        Describes the current environment - people count and visible objects.
        Use when user asks what you see or about the surroundings.
        """
        print("👁️ Describing environment (using cache)")
        
        if not self.face_monitor:
            return "Face monitoring is not active."

        recent_objects = self.face_monitor.get_recent_objects(seconds=3.0)
        print(f"   📦 Object cache returned: {len(recent_objects)} objects -> {[o['class'] for o in recent_objects]}")
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Count people
        people = [obj for obj in recent_objects if obj['class'] == 'person']
        people_count = len(people)
        
        # Get non-person objects (top 3)
        objects = [obj for obj in recent_objects if obj['class'] != 'person']
        object_names = [obj['class'] for obj in objects[:3]]
        
        response = []
        
        if people_count == 0:
            response.append("I don't see anyone")
        elif people_count == 1:
            response.append("I see 1 person")
        else:
            response.append(f"I see {people_count} people")
        
        if object_names:
            obj_list = ', '.join(object_names)
            response.append(f"and {obj_list}")
        
        return '. '.join(response) if response else "I don't see much right now"

    # @function_tool  # Removed - called as helper method
    async def identify_object(self, object_name: str, context: RunContext) -> str:
        """
        Finds a specific object and describes it.
        Use when user asks about a particular item (laptop, phone, cup, etc).
        
        Args:
            object_name: The object to find (e.g., "laptop", "phone", "bottle")
        """
        print(f"🔍 Looking for: {object_name} (using cache)")
        
        if not self.face_monitor:
            return "Face monitoring is not active."

        recent_objects = self.face_monitor.get_recent_objects(seconds=5.0)
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Find matching object
        object_name_lower = object_name.lower()
        matches = [obj for obj in recent_objects 
                   if object_name_lower in obj['class'].lower()]
        
        if matches:
            best_match = max(matches, key=lambda x: x['confidence'])
            confidence = int(best_match['confidence'] * 100)
            return f"Yes! I can see a {best_match['class']} ({confidence}% confidence)"
        else:
            return f"I haven't seen a {object_name} in the last few seconds"

    # @function_tool  # Removed - called as helper method
    async def count_people_in_room(self, context: RunContext) -> str:
        """
        Counts how many people are visible in the camera view.
        Use when user asks how many people are present.
        """
        print("👥 Counting people (using cache)")
        
        if not self.face_monitor:
            return "Face monitoring is not active."
            
        recent_objects = self.face_monitor.get_recent_objects(seconds=3.0)
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Count people from cache
        people = [obj for obj in recent_objects if obj['class'] == 'person']
        count = len(people)
        
        if count == 0:
            return "I don't see anyone right now"
        elif count == 1:
            return "I see 1 person"
        else:
            return f"I see {count} people"
