·# Campus Greeting Robot - Voice Agent V2

A LiveKit-powered voice agent for campus assistance with face recognition, color detection, and image display capabilities.

## Features

- **Face Recognition**: Recognizes enrolled users via webcam
- **Color Detection**: Identifies dominant colors in camera view
- **Event Posters**: Displays event information with images
- **Location Maps**: Shows campus navigation maps
- **Natural Language**: Fuzzy matching for user queries

## Setup

### Docker Setup (Recommended for Multi-Platform Support)

**📦 For detailed Docker instructions, see [DOCKER.md](DOCKER.md)**

Docker setup supports both x86 (standard PCs) and ARM architectures (Raspberry Pi).

#### Prerequisites
- Docker Engine 20.10+ or Docker Desktop
- Docker Compose V2
- (Optional) Docker Buildx for multi-architecture builds

#### Quick Start - Development

1. Clone the repository:
```bash
git clone https://github.com/makiladamsuka/voice-agentv2.git
cd voice-agentv2
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your LiveKit credentials
```

3. Start all services:
```bash
docker compose up
```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8080

#### Quick Start - Production

1. Build and start services in detached mode:
```bash
docker compose up -d --build
```

2. View logs:
```bash
docker compose logs -f
```

3. Stop services:
```bash
docker compose down
```

#### Adding Face Encodings and Assets

The following directories are mounted as volumes for easy development:
- `backend/known_faces` - Add face encodings here
- `backend/assets/events` - Event posters
- `backend/assets/maps` - Location maps
- `backend/assets/fallback` - Fallback images

Simply add files to these directories and restart the backend:
```bash
docker compose restart backend
```

#### Building Multi-Architecture Images

To build images for both x86 and ARM (e.g., for Raspberry Pi):

1. Create a builder instance:
```bash
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap
```

2. Build multi-platform images:
```bash
# Backend
docker buildx build --platform linux/amd64,linux/arm64 -t voice-agent-backend:latest ./backend

# Frontend
docker buildx build --platform linux/amd64,linux/arm64 -t voice-agent-frontend:latest ./frontend
```

3. For pushing to a registry:
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t your-registry/voice-agent-backend:latest \
  --push ./backend
```

#### Troubleshooting Docker

- **Permission errors**: Ensure Docker has access to the mounted volumes
- **Port conflicts**: Change ports in `docker-compose.yml` if 3000 or 8080 are in use
- **Build failures on ARM**: Ensure you have sufficient memory (recommended 2GB+ for Raspberry Pi)
- **Slow builds**: Use `docker compose build --parallel` for faster builds

### Manual Setup (Alternative to Docker)

For local development without Docker:

#### Backend

1. Navigate to backend directory:
```bash
cd voice-agentv2/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure `.env` with your LiveKit credentials

4. (Optional) Add face encodings to `known_faces/encodings.pkl`

5. Add images:
   - Event posters → `assets/events/`
   - Location maps → `assets/maps/`
   - Fallback image → `assets/fallback/`

6. Run the agent:
```bash
python greeting_agent.py dev
```

#### Frontend

1. Navigate to frontend directory:
```bash
cd voice-agentv2/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure `.env.local` with LiveKit credentials

4. Run the development server:
```bash
npm run dev
```

5. Open http://localhost:3000

## Usage Examples

- "Who am I?" - Face recognition
- "What color is my shirt?" - Color detection
- "Show me the tech fest" - Event poster
- "Where is the DS lab?" - Location map

## Image Naming

Use descriptive names with underscores or hyphens:
- `tech_fest_2024.jpg`
- `ds-lab.png`
- `library_map.jpg`

The system uses fuzzy matching to find relevant images.
