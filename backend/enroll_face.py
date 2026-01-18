#!/usr/bin/env python3
"""
Face Enrollment Script for Voice Agent V2
Captures photos and creates face encodings for recognition.
"""

import cv2
import face_recognition
import pickle
import os
import time
from pathlib import Path

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️ picamera2 not available - will use OpenCV only")

def enroll_face(name: str, num_samples: int = 5):
    """
    Capture face samples and create encodings.
    
    Args:
        name: Name of the person to enroll
        num_samples: Number of photos to capture
    """
    print(f"\n🎥 Face Enrollment for: {name}")
    print(f"📸 Capturing {num_samples} photos automatically...\n")
    
    # Initialize camera (picamera2 first, then fallback to OpenCV)
    video_capture = None
    picam2 = None
    camera_type = None
    
    # Try picamera2 first (native Raspberry Pi camera support)
    # Following camtest.py pattern: use XRGB8888 format, size (1280, 720)
    print("🎥 Attempting to open camera...")
    if PICAMERA2_AVAILABLE:
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"format": 'XRGB8888', "size": (1280, 720)}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)  # Allow camera to warm up
            # Test capture
            test_frame = picam2.capture_array()
            if test_frame is not None and test_frame.size > 0:
                camera_type = "picamera2"
                print(f"✅ Camera: Picamera2 (size: {test_frame.shape[1]}x{test_frame.shape[0]})")
        except Exception as e:
            print(f"⚠️ Picamera2 failed: {e}")
            if picam2:
                try:
                    picam2.stop()
                except:
                    pass
                picam2 = None
    
    # Fallback to OpenCV VideoCapture (USB webcams)
    if not camera_type:
        try:
            video_capture = cv2.VideoCapture(0)
            if video_capture.isOpened():
                ret, test_frame = video_capture.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    camera_type = "opencv"
                    print(f"✅ Camera: OpenCV (size: {test_frame.shape[1]}x{test_frame.shape[0]})")
                else:
                    video_capture.release()
                    video_capture = None
        except Exception as e:
            print(f"⚠️ OpenCV camera failed: {e}")
    
    if not camera_type:
        print("❌ Error: Could not open any camera!")
        return False
    
    encodings = []
    count = 0
    attempts = 0
    max_attempts = num_samples * 20
    
    print("📷 Look at the camera... capturing photos automatically")
    print("Press ESC to cancel\n")
    
    while count < num_samples and attempts < max_attempts:
        # Capture frame based on camera type
        if camera_type == "picamera2":
            frame = picam2.capture_array()
            # Convert XRGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret = True
        else:
            ret, frame = video_capture.read()
        
        attempts += 1
        
        if not ret or frame is None:
            continue
        
        # Display the frame
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Photo {count + 1}/{num_samples} - Hold still",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('Face Enrollment', display_frame)
        
        key = cv2.waitKey(100) & 0xFF
        
        # ESC to quit
        if key == 27:
            print("\n❌ Enrollment cancelled.")
            video_capture.release()
            cv2.destroyAllWindows()
            return False
        
        # Auto-capture every few frames
        if attempts % 10 == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) != 1:
                continue
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])
                count += 1
                print(f"✅ Photo {count}/{num_samples} captured!")
    
    # Cleanup camera
    if picam2:
        picam2.stop()
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()
    
    if len(encodings) < num_samples:
        print(f"\n⚠️  Only captured {len(encodings)} photos. Need {num_samples}.")
        return False
    
    # Save encodings
    encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
    encodings_path.parent.mkdir(exist_ok=True)
    
    # Load existing
    known_faces = {}
    if encodings_path.exists():
        with open(encodings_path, 'rb') as f:
            known_faces = pickle.load(f)
        print(f"\n📂 Loaded existing encodings for {len(known_faces)} people")
    
    # Add new
    known_faces[name] = encodings
    
    # Save
    with open(encodings_path, 'wb') as f:
        pickle.dump(known_faces, f)
    
    print(f"\n✅ Successfully enrolled {name} with {len(encodings)} face samples!")
    print(f"💾 Saved to: {encodings_path}")
    print(f"\n👥 Total people enrolled: {len(known_faces)}")
    print(f"   Names: {', '.join(known_faces.keys())}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("  FACE RECOGNITION ENROLLMENT")
    print("=" * 60)
    
    name = input("\nEnter your name: ").strip()
    
    if not name:
        print("❌ Name cannot be empty!")
        exit(1)
    
    num_samples = input("How many photos? (default: 5): ").strip()
    num_samples = int(num_samples) if num_samples.isdigit() else 5
    
    success = enroll_face(name, num_samples)
    
    if success:
        print("\n🎉 Enrollment complete! Restart the agent to use face recognition.")
    else:
        print("\n❌ Enrollment failed. Please try again.")
