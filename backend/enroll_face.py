import cv2
import face_recognition
import pickle
import sys
import os
from pathlib import Path

def enroll_face(name):
    print(f"📸 Preparing to enroll: {name}")
    print("   Look at the camera...")
    
    # Try opening camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    # Capture a few frames to let auto-exposure settle
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture frame")
        return

    # Convert to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes = face_recognition.face_locations(rgb_frame)
    
    if len(boxes) == 0:
        print("❌ No face detected. Please try again.")
        return
    
    if len(boxes) > 1:
        print(f"⚠️ Multiple faces detected ({len(boxes)}). Please ensure only {name} is in frame.")
        return

    # Encode face
    encodings = face_recognition.face_encodings(rgb_frame, boxes)
    
    if len(encodings) > 0:
        new_encoding = encodings[0]
        
        # Load existing
        encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
        encodings_path.parent.mkdir(parents=True, exist_ok=True)
        
        known_faces = {}
        if encodings_path.exists():
            try:
                with open(encodings_path, 'rb') as f:
                    known_faces = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Start fresh (error loading: {e})")

        # Add new face
        if name in known_faces:
            print(f"   Updating existing face for {name}")
            known_faces[name].append(new_encoding)
        else:
            print(f"   Adding new face for {name}")
            known_faces[name] = [new_encoding]
            
        # Save
        with open(encodings_path, 'wb') as f:
            pickle.dump(known_faces, f)
            
        print(f"✅ Successfully enrolled {name}!")
    else:
        print("❌ Could not encode face.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python user_enroller.py <Name>")
    else:
        name = sys.argv[1]
        enroll_face(name)
