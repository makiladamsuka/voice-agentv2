#!/usr/bin/env python3
"""
Test script to verify picamera2 is working with the backend code pattern
"""

import sys
import cv2
import time

print("=" * 60)
print("  PICAMERA2 BACKEND TEST")
print("=" * 60)

# Test 1: Check if picamera2 is available
print("\n📦 Test 1: Checking picamera2 availability...")
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    print("✅ picamera2 module is available")
except ImportError as e:
    PICAMERA2_AVAILABLE = False
    print(f"❌ picamera2 not available: {e}")
    print("   Install with: sudo apt install python3-picamera2")
    sys.exit(1)

# Test 2: Try to initialize and configure camera (matching camtest.py and backend pattern)
print("\n📷 Test 2: Initializing Picamera2 (matching backend pattern)...")
picam2 = None
try:
    picam2 = Picamera2()
    print("✅ Picamera2 object created")
    
    # Use same config as backend (matching camtest.py pattern)
    config = picam2.create_video_configuration(
        main={"format": 'XRGB8888', "size": (1280, 720)}
    )
    picam2.configure(config)
    print("✅ Camera configured: XRGB8888 format, 1280x720 size")
    
    picam2.start()
    print("✅ Camera started")
    
    # Allow camera to warm up
    time.sleep(0.5)
    print("✅ Camera warm-up complete")
    
except Exception as e:
    print(f"❌ Camera initialization failed: {e}")
    import traceback
    traceback.print_exc()
    if picam2:
        try:
            picam2.stop()
        except:
            pass
    sys.exit(1)

# Test 3: Capture a test frame
print("\n📸 Test 3: Capturing test frame...")
try:
    frame = picam2.capture_array()
    if frame is not None and frame.size > 0:
        print(f"✅ Frame captured successfully!")
        print(f"   Shape: {frame.shape}")
        print(f"   Size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Dtype: {frame.dtype}")
    else:
        print("❌ Frame captured but empty")
        sys.exit(1)
except Exception as e:
    print(f"❌ Frame capture failed: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 4: Convert frame format (matching backend conversion)
print("\n🔄 Test 4: Testing frame conversion (XRGB to BGR)...")
try:
    # Convert XRGB to BGR (matching face_monitor.py pattern)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    print(f"✅ Frame converted to BGR")
    print(f"   BGR shape: {bgr_frame.shape}")
    
    # Test RGB conversion for face recognition
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    print(f"✅ Frame converted to RGB for face recognition")
    print(f"   RGB shape: {rgb_frame.shape}")
    
except Exception as e:
    print(f"❌ Frame conversion failed: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 5: Capture multiple frames (simulating monitor loop)
print("\n🔁 Test 5: Testing continuous frame capture (5 frames)...")
try:
    for i in range(5):
        frame = picam2.capture_array()
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(f"   ✅ Frame {i+1}/5 captured and converted")
    print("✅ Continuous capture test passed")
except Exception as e:
    print(f"❌ Continuous capture failed: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Test 6: Test face_monitor pattern (without face recognition)
print("\n🎯 Test 6: Testing backend FaceMonitor camera pattern...")
try:
    # Simulate the exact pattern from face_monitor.py
    camera_type = "picamera2"
    frame_count = 0
    
    for i in range(3):
        if camera_type == "picamera2":
            frame = picam2.capture_array()
            # Convert XRGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret = True
        
        if ret and frame is not None:
            frame_count += 1
            print(f"   ✅ Monitor loop frame {frame_count} successful")
        else:
            print(f"   ❌ Monitor loop frame {frame_count} failed")
    
    print(f"✅ FaceMonitor pattern test passed ({frame_count} frames)")
    
except Exception as e:
    print(f"❌ FaceMonitor pattern test failed: {e}")
    import traceback
    traceback.print_exc()
    picam2.stop()
    sys.exit(1)

# Cleanup
print("\n🧹 Cleaning up...")
try:
    picam2.stop()
    print("✅ Camera stopped successfully")
except Exception as e:
    print(f"⚠️ Error stopping camera: {e}")

print("\n" + "=" * 60)
print("  ✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n📝 Summary:")
print("   ✅ picamera2 is available and working")
print("   ✅ Camera initialization matches backend pattern")
print("   ✅ Frame capture works correctly")
print("   ✅ Frame conversion (XRGB→BGR→RGB) works")
print("   ✅ Continuous capture works")
print("   ✅ FaceMonitor camera pattern works")
print("\n🎉 Backend can access picamera2 successfully!")
