# Quick Start - Raspberry Pi Setup

Complete guide for setting up the voice agent on a fresh Raspberry Pi (Bookworm).

## Prerequisites

- Raspberry Pi 4 with Raspberry Pi OS Bookworm
- SSH access to the Pi
- LiveKit Cloud account with credentials

---

## Step 1: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in for group change
logout
```

## Step 2: Install System Dependencies

```bash
sudo apt update
sudo apt install -y cmake build-essential i2c-tools python3-pip python3-venv python3-picamera2

# Enable I2C for OLED display
sudo raspi-config
# Navigate: Interface Options → I2C → Enable

sudo reboot
```

## Step 3: Clone the Project

```bash
cd ~/Documents
git clone <your-repo-url> voice-agentv2
cd voice-agentv2
git checkout emooled  # or your working branch
```

## Step 4: Transfer Credentials from Laptop

From your **laptop** terminal:

```bash
# Transfer backend .env (contains LiveKit, OpenRouter, Deepgram keys)
scp /path/to/voice-agentv2/backend/.env nema@raspberrypi.local:/home/nema/Documents/voice-agentv2/backend/.env
```

## Step 5: Setup Python Backend

```bash
cd ~/Documents/voice-agentv2

# Create venv with system packages (required for picamera2)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# If numpy conflicts with picamera2:
pip uninstall numpy -y
# Let it use system numpy
```

## Step 6: Build Frontend Docker

```bash
cd ~/Documents/voice-agentv2

# Load credentials for build args
source backend/.env

# Build frontend with LiveKit credentials
docker build \
  --build-arg LIVEKIT_URL=$LIVEKIT_URL \
  --build-arg LIVEKIT_API_KEY=$LIVEKIT_API_KEY \
  --build-arg LIVEKIT_API_SECRET=$LIVEKIT_API_SECRET \
  --build-arg AGENT_NAME=${AGENT_NAME:-campus-greeting-agent} \
  -t voice-agent-frontend ./frontend
```

> ⚠️ This takes 15-20 minutes on Raspberry Pi

## Step 7: Run the Application

### Start Frontend (Docker)

```bash
docker run -d -p 3000:3000 --name voice-frontend voice-agent-frontend:latest
```

### Start Backend (Native Python)

```bash
cd ~/Documents/voice-agentv2/backend
source ../venv/bin/activate
export $(grep -v '^#' .env | xargs)
python voice_agent.py dev
```

## Step 8: Access the App

- **From Pi**: http://localhost:3000
- **From Laptop**: http://raspberrypi.local:3000

Or use SSH port forwarding for microphone access:
```bash
ssh -L 3000:localhost:3000 nema@raspberrypi.local
# Then open http://localhost:3000 on laptop
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Start frontend | `docker run -d -p 3000:3000 --name voice-frontend voice-agent-frontend:latest` |
| Stop frontend | `docker stop voice-frontend && docker rm voice-frontend` |
| View frontend logs | `docker logs -f voice-frontend` |
| Start backend | `cd backend && source ../venv/bin/activate && export $(grep -v '^#' .env \| xargs) && python voice_agent.py dev` |
| Check containers | `docker ps` |

---

## Troubleshooting

**picamera2 not found?**
```bash
rm -rf venv
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**numpy dtype error?**
```bash
pip uninstall numpy -y
# Uses system numpy compatible with picamera2
```

**LIVEKIT_URL not found?**
```bash
export $(grep -v '^#' backend/.env | xargs)
```

**dlib build fails?**
```bash
sudo apt install -y cmake build-essential
```

**Frontend build ESLint errors?**
Make sure you're on the correct git branch (`emooled`) and the Dockerfile is synced.

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| Backend .env | `backend/.env` | LiveKit, OpenRouter, Deepgram credentials |
| Root .env | `.env` | Docker Compose variables (optional) |
| Known faces | `backend/known_faces/` | Face recognition database |
| Assets | `backend/assets/` | Event posters, maps, etc. |
