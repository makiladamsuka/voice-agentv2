# Voice Agent v2 - AI Agent Instructions

## Architecture Overview

This is a **LiveKit-based voice agent** with face recognition and visual content delivery:

- **Backend** (Python): LiveKit voice agent with OpenCV-based face monitoring, YOLO object detection, and HTTP image server
- **Frontend** (Next.js/React): LiveKit client with real-time data packet handling for dynamic image display
- **Communication**: LiveKit WebRTC for voice/video, `room.local_participant.publish_data()` for backend→frontend messages

### Key Components

```
backend/
  greeting_agent.py         # Main LiveKit agent with LLM function tools
  face_monitor.py           # Background webcam thread tracking faces
  image_server.py           # HTTP server (port 8080) for assets  
  image_manager.py          # Fuzzy matching for events/maps
  object_detector.py        # YOLO (disabled by default on Pi)
  tools/
    vision.py               # Face/object/color detection tools
    content.py              # Event/map display tools

frontend/
  components/app/
    image-display.tsx       # Listens to room.on('dataReceived') for images
    app.tsx                 # SessionProvider wrapper
  api/connection-details/   # Token generation endpoint
```

## Critical Patterns

### 1. Backend-Frontend Communication

**Backend sends images via LiveKit data channel:**
```python
await self.room.local_participant.publish_data(
    json.dumps({
        "type": "image",
        "category": "event",  # or "map", "fallback"
        "url": "http://localhost:8080/assets/events/techfest.jpg",
        "caption": "Event: Tech Fest"
    }).encode()
)
```

**Frontend listens in [image-display.tsx](frontend/components/app/image-display.tsx#L30-L50):**
```tsx
room.on('dataReceived', (payload, participant, kind) => {
  const message = JSON.parse(decoder.decode(payload));
  if (message.type === 'image') {
    setImageData(message);
  }
});
```

### 2. Multi-Person Face Recognition

[face_monitor.py](backend/face_monitor.py) runs as background thread with **stability cache** (2s) to prevent flicker:

- `get_fresh_people()` - Most recent detection (for real-time context)
- `get_current_people()` - Cached stable list
- `get_new_arrivals()` - Filtered by 60s greeting cooldown
- `mark_greeted(name)` - Prevents re-greeting same person

**Agent greeting flow** ([greeting_agent.py](backend/greeting_agent.py#L250-L280)):
```python
new_people = self.face_monitor.get_new_arrivals()
for name in new_people:
    greeting = generate_greeting(name)
    await session.send_text(greeting)
    self.face_monitor.mark_greeted(name)
```

### 3. Modular Function Tools

Agent delegates to tool modules ([tools/vision.py](backend/tools/vision.py), [tools/content.py](backend/tools/content.py)):

```python
self.vision_tools = VisionTools(
    face_monitor=None,  # Set later in agent loop
    object_detector_factory=self.get_object_detector  # Lazy-loaded YOLO
)

@function_tool
async def enroll_new_face(self, person_name: str, context: RunContext) -> str:
    self.vision_tools.face_monitor = self.face_monitor
    return await self.vision_tools.enroll_new_face(person_name, context)
```

**Why this pattern?** Separates agent orchestration from tool implementation while allowing runtime dependency injection.

### 4. Asset Management

[image_manager.py](backend/image_manager.py) uses fuzzy matching (SequenceMatcher) for queries like "tech fest" → `techfest.jpg`:

```python
backend/assets/
  events/        # Event posters (fuzzy matched)
  maps/          # Location maps (fuzzy matched)
  fallback/      # Generic images
```

[image_server.py](backend/image_server.py) serves via HTTP (not base64) for performance. Frontend loads images with `crossOrigin="anonymous"`.

## Development Workflows

### Running Locally

```bash
# Backend (with webcam access) - WORKS ON RASPBERRY PI
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# IMPORTANT: Use 'dev' command, not direct execution
python greeting_agent.py dev  # Requires .env with LiveKit + OpenRouter API keys

# Frontend - REQUIRES NODE 20+ (not v18)
cd frontend
node --version  # Must be v20+
npm install && npm run dev
```

### Docker Setup

```bash
docker compose up  # Frontend:3000, Backend:8080
```

**Volume mounts** allow live asset updates without rebuild:
- `backend/known_faces` → face encodings
- `backend/assets` → event/map images

### Adding New Function Tools

1. Implement in [tools/vision.py](backend/tools/vision.py) or [tools/content.py](backend/tools/content.py)
2. Add `@function_tool` wrapper in [greeting_agent.py](backend/greeting_agent.py#L130-L180)
3. Update agent instructions with tool description

### Enrolling Faces

**Via agent auto-enrollment** (preferred):
```
User: "I'm Alex"
Agent: [calls enroll_new_face("Alex")] "Nice to meet you, Alex!"
```

**Manual enrollment:**
```bash
python backend/enroll_face.py "Alex"  # Captures from webcam
```

Encodings saved to `backend/known_faces/encodings.pkl`.

## Configuration

### Environment Variables

**Backend** ([.env](backend/.env)):
```env
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENROUTER_API_KEY=...
LLM_CHOICE=mistralai/devstral-2512:free  # Or any OpenRouter model
```

**Frontend** (passed via Docker build args or `.env.local`):
```env
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
AGENT_NAME=campus-greeting-agent
```

### Performance Tuning

**YOLO disabled by default** ([face_monitor.py](backend/face_monitor.py#L50-L52)) for Raspberry Pi:
```python
self.detector = ObjectDetector(load_yolo=False)  # Set True to enable
```

**Face cache duration** adjustable in [face_monitor.py](backend/face_monitor.py#L14-L16):
```python
FACE_CACHE_DURATION = 2.0     # Prevent flicker
GREETING_COOLDOWN = 60.0      # Re-greeting delay
```

## Raspberry Pi Deployment

### Known Issues & Solutions

#### 1. Frontend ARM Build Failures / Bus Errors

**Problem:** Next.js frontend crashes with "Bus error" on ARM64 (Raspberry Pi 4/5)

**Root cause:** Next.js 15.x has compatibility issues with ARM architecture on older Node versions (v18). Requires Node 20+.

**Solutions:**

```bash
# Option A: Upgrade Node.js to v20+ (RECOMMENDED)
# Using nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
cd frontend && npm install && npm run dev

# Using NodeSource:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
cd frontend && npm install && npm run dev

# Option B: Run frontend on separate x86 machine, backend on Pi
# Pi runs: backend only (works perfectly)
# x86 runs: frontend only
# Both point to same LIVEKIT_URL

# Option C: Use Docker with cross-compilation (slower but works)
docker buildx build --platform linux/arm64 -t voice-agent-frontend:arm64 ./frontend
```

**Current status:** 
- Backend works perfectly on Pi with Python 3.11
- Frontend requires Node 20+ (currently on v18.20.4 which causes Bus error)
- Workaround in [Dockerfile](frontend/Dockerfile#L14-L18): Extended npm timeouts + `--legacy-peer-deps`

#### 2. Raspberry Pi Camera v2 Access

**Problem:** OpenCV can't access Pi Camera v2 via default `cv2.VideoCapture(0)`

**Root cause:** Pi Camera v2 requires `libcamera` driver, not V4L2

**Solutions:**

```bash
# Enable legacy camera support (required for OpenCV)
sudo raspi-config
# Navigate to: Interface Options → Legacy Camera → Enable
sudo reboot

# Verify camera appears as /dev/video0
ls -l /dev/video*
v4l2-ctl --list-devices

# Test with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

**Code tries multiple backends** ([face_monitor.py](backend/face_monitor.py#L217-L225)):
```python
configs = [(0, cv2.CAP_V4L2, "V4L2/0"), (0, cv2.CAP_ANY, "ANY/0"), 
           (1, cv2.CAP_V4L2, "V4L2/1"), (1, cv2.CAP_ANY, "ANY/1")]
```

**If still failing:**
```bash
# Install picamera2 for native Pi camera support (alternative to OpenCV)
sudo apt install -y python3-picamera2

# Or use USB webcam instead (plug & play with V4L2)
```

#### 3. Performance Optimization for Pi

**Disable YOLO** (already default in [face_monitor.py](backend/face_monitor.py#L50-L52)):
```python
self.detector = ObjectDetector(load_yolo=False)  # Saves ~2GB RAM
```

**Reduce face detection frequency:**
```python
# In face_monitor.py _monitor_loop()
# Process every 5th frame (default) - increase to 10 for slower Pi models
if frame_count % 10 == 0:  # Less CPU usage
```

**Use smaller face recognition model:**
```bash
# Install dlib without CUDA (lighter weight)
pip install dlib --no-cache-dir
```

## Common Pitfalls

1. **"Agent not seeing images"** - Check HTTP server started at port 8080 (logs: `✅ Image server started`)
2. **"Multiple greetings"** - Use `mark_greeted()` after sending greeting
3. **"Stale face names"** - Use `get_fresh_people()` not `get_current_people()` for context injection
4. **"Frontend not receiving data"** - Verify `room.local_participant.publish_data()` called (not `room.publish_data()`)
5. **"Face encodings not loading"** - Check `backend/known_faces/encodings.pkl` exists and format matches pickle.load()
6. **"Camera not opening on Pi"** - Enable legacy camera in `raspi-config` or use USB webcam
7. **"Frontend build timeout on Pi"** - Build on x86 machine or run frontend separately

## Project-Specific Conventions

- **Emoji logging** - All components use emoji prefixes (🎥, 📋, ✅, ⚠️) for visual log parsing
- **Time-aware greetings** - [greetings.py](backend/greetings.py) generates context-aware messages based on time of day and last seen
- **Lazy loading** - YOLO and heavy models loaded on-demand via factory pattern
- **Tool naming** - Use verb_noun format (`enroll_new_face`, `show_event_poster`)
- **Frontend data protocol** - Always send `{type, category, url, caption}` JSON structure

## Quick Troubleshooting

### Backend Won't Start
```bash
# Check camera access
ls -l /dev/video*
# If no camera: backend still starts but face monitor fails

# Check dependencies
cd backend
pip install -r requirements.txt

# Check .env file
cat .env | grep LIVEKIT_URL
```

### Frontend Build Issues on ARM
```bash
# Check Node version (need 20+)
node --version

# If v18.x → causes Bus error on ARM
# Upgrade to Node 20:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Then clean install
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --legacy-peer-deps

# If timeout: increase Docker memory limit to 4GB+
```

### Backend CLI Changes (livekit-agents 1.3.10+)
```bash
# ❌ OLD (doesn't work):
python greeting_agent.py

# ✅ NEW (correct):
python greeting_agent.py dev   # Development mode
python greeting_agent.py start # Production mode

# The CLI now requires a command - just running the script fails with:
# "Missing command" error
```

### Camera Debug
```bash
# View camera logs from face_monitor
docker compose logs backend | grep "🎥"

# Expected output:
# ✅ Camera: V4L2/0 (size: 640x480)

# If you see: ❌ Cannot open any camera
# → Enable legacy camera or switch to USB webcam
```

### Image Display Not Working
```bash
# Test image server directly
curl http://localhost:8080/assets/events/

# Check frontend console for CORS errors
# Check room.on('dataReceived') listener attached
```
