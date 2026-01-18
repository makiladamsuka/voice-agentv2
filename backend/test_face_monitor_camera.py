#!/usr/bin/env python3
"""
Test FaceMonitor camera initialization
"""

import sys
import time
from pathlib import Path

print("=" * 60)
print("  FACEMONITOR CAMERA INITIALIZATION TEST")
print("=" * 60)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("\n📦 Importing FaceMonitor...")
try:
    from face_monitor import FaceMonitor
    print("✅ FaceMonitor imported successfully")
except Exception as e:
    print(f"❌ Failed to import FaceMonitor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎥 Creating FaceMonitor instance...")
try:
    # Create with empty known_faces dict
    monitor = FaceMonitor({})
    print("✅ FaceMonitor instance created")
except Exception as e:
    print(f"❌ Failed to create FaceMonitor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🚀 Starting FaceMonitor (this will attempt camera initialization)...")
try:
    monitor.start()
    print("✅ FaceMonitor started")
    
    # Wait a bit for camera initialization
    print("\n⏳ Waiting for camera initialization (3 seconds)...")
    time.sleep(3)
    
    # Check if camera was initialized
    if monitor.picam2 is not None:
        print("✅ picamera2 is active in FaceMonitor")
        print(f"   Camera type: picamera2")
    elif monitor.video_capture is not None:
        print("⚠️ Using OpenCV VideoCapture (not picamera2)")
        print(f"   Camera type: opencv (fallback)")
    else:
        print("❌ No camera initialized!")
        monitor.stop()
        sys.exit(1)
    
    # Try to get a frame
    print("\n📸 Testing frame capture from FaceMonitor...")
    time.sleep(1)  # Allow some frames to be captured
    frame = monitor.get_current_frame()
    
    if frame is not None:
        print(f"✅ Frame captured successfully!")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("⚠️ No frame available yet (may need more time)")
    
    # Check current people detection
    people = monitor.get_current_people()
    fresh_people = monitor.get_fresh_people()
    print(f"\n👥 Face detection status:")
    print(f"   Current people (stable): {list(people) if people else 'None'}")
    print(f"   Fresh people: {list(fresh_people) if fresh_people else 'None'}")
    
except Exception as e:
    print(f"❌ FaceMonitor test failed: {e}")
    import traceback
    traceback.print_exc()
    try:
        monitor.stop()
    except:
        pass
    sys.exit(1)

print("\n🧹 Stopping FaceMonitor...")
try:
    monitor.stop()
    print("✅ FaceMonitor stopped successfully")
except Exception as e:
    print(f"⚠️ Error stopping FaceMonitor: {e}")

print("\n" + "=" * 60)
print("  ✅ FACEMONITOR CAMERA TEST COMPLETE!")
print("=" * 60)
