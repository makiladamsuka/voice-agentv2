# Raspberry Pi Commands Reference

Quick reference for running the voice agent on Raspberry Pi.

## Running the Agent

```bash
# Backend only
python backend/agent.py dev

# With less verbose LiveKit logs
LIVEKIT_LOG_LEVEL=warn python backend/agent.py dev
```

## Docker Frontend

```bash
# Start frontend container
docker run -d -p 3000:3000 voice-agentv2-frontend:latest

# Check if running
docker ps

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>
```

## Accessing from Laptop

```bash
# SSH with port forwarding (enables microphone access)
ssh -L 3000:localhost:3000 nema@raspberrypi.local

# Then open in browser: http://localhost:3000
```

## Kiosk Mode (Pi Display)

```bash
# Start fullscreen browser
chromium-browser --kiosk --disable-gpu http://localhost:3000

# Exit kiosk: F11 or Alt+F4
# Force kill: pkill chromium
```

## System Monitoring

```bash
# CPU temperature
vcgencmd measure_temp

# CPU frequency (check overclock)
vcgencmd measure_clock arm

# Check for throttling/undervoltage
vcgencmd get_throttled
# 0x0 = All good
# 0x50005 = Undervoltage detected

# CPU usage
htop
```

## Overclocking (requires heatsink + fan)

Edit `/boot/config.txt`:
```
over_voltage=6
arm_freq=2000
gpu_freq=700
```
Then `sudo reboot`

## Git Branch Management

```bash
# Main branches
git checkout camerafix    # Camera + face recognition
git checkout rag-voice    # Camera + RAG event Q&A

# Pull latest
git pull origin <branch>

# Discard local changes
git checkout -- .
```

## Installing Dependencies

```bash
# Tesseract OCR (for event poster scanning)
sudo apt install tesseract-ocr

# Python packages
pip install -r backend/requirements.txt
```

## Debug Settings (face_monitor.py)

```python
SHOW_DEBUG_VIDEO = True    # Camera window on HDMI
DEBUG_LOG_INTERVAL = 5.0   # Status log every 5s (0 = disable)
```
