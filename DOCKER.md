# Docker Setup Guide

This guide provides detailed instructions for running the voice-agentv2 project using Docker.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose V2
- (Optional) Docker Buildx for multi-architecture builds

## Quick Start

### 1. Environment Configuration

Copy the example environment file and configure your LiveKit credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your LiveKit credentials:
```env
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

### 2. Start Services

Development mode (with volume mounts):
```bash
docker compose up
```

Production mode (detached):
```bash
docker compose up -d
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8080

## Architecture

The Docker setup uses a multi-service architecture:

### Backend Service
- **Base Image**: `python:3.10-slim-bullseye` (multi-arch compatible)
- **Port**: 8080
- **Volumes**:
  - `./backend/known_faces` → `/app/known_faces` (face encodings)
  - `./backend/assets` → `/app/assets` (event posters, maps, fallback images)

### Frontend Service
- **Base Image**: `node:20-alpine` (multi-arch compatible)
- **Port**: 3000
- **Build**: Multi-stage build for optimized production images
- **Depends on**: backend service

## Volume Mounts

For development convenience, the following directories are mounted as volumes:

- `backend/known_faces` - Face recognition encodings
- `backend/assets/events` - Event posters
- `backend/assets/maps` - Location maps  
- `backend/assets/fallback` - Fallback images

Changes to these directories are reflected immediately without rebuilding.

## Development Workflow

### Making Code Changes

#### Backend Changes
```bash
# After modifying Python code, rebuild and restart backend
docker compose up -d --build backend
```

#### Frontend Changes
```bash
# After modifying frontend code, rebuild and restart frontend
docker compose up -d --build frontend
```

### Adding Face Encodings

1. Add face encoding files to `backend/known_faces/`
2. Restart the backend service:
```bash
docker compose restart backend
```

### Adding Assets

1. Add event posters to `backend/assets/events/`
2. Add location maps to `backend/assets/maps/`
3. Restart the backend service:
```bash
docker compose restart backend
```

## Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
```

## Stopping Services

```bash
# Stop and remove containers
docker compose down

# Stop, remove containers and volumes
docker compose down -v
```

## Multi-Architecture Builds

To build images for both x86 and ARM architectures (e.g., for Raspberry Pi):

### Setup Buildx

```bash
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap
```

### Build Multi-Platform Images

```bash
# Backend
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t voice-agent-backend:latest \
  ./backend

# Frontend
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t voice-agent-frontend:latest \
  ./frontend
```

### Push to Registry

If you want to push to a container registry:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/voice-agent-backend:latest \
  --push \
  ./backend

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/voice-agent-frontend:latest \
  --push \
  ./frontend
```

## Troubleshooting

### Port Already in Use

If ports 3000 or 8080 are already in use, modify `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8081:8080"  # Changed from 8080:8080
  
  frontend:
    ports:
      - "3001:3000"  # Changed from 3000:3000
```

### Permission Errors

Ensure Docker has permission to access mounted volumes:

```bash
# Linux/macOS
sudo chown -R $USER:$USER backend/known_faces backend/assets
```

### Build Failures on ARM

If building on Raspberry Pi with limited memory:

1. Increase swap space
2. Build with limited parallelism:
```bash
docker compose build --parallel 1
```

### Container Exits Immediately

Check logs for errors:
```bash
docker compose logs backend
docker compose logs frontend
```

Common issues:
- Missing environment variables
- Invalid LiveKit credentials
- Missing required files

## Production Deployment

### Recommended Configuration

1. Use environment-specific `.env` files
2. Remove volume mounts in production for immutable deployments
3. Use health checks:

```yaml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Security Considerations

1. Never commit `.env` files with real credentials
2. Use Docker secrets for sensitive data in production
3. Run containers with non-root users (already configured in frontend)
4. Keep base images updated

### Performance Optimization

1. Use multi-stage builds (already implemented in frontend)
2. Minimize layer count
3. Use `.dockerignore` to exclude unnecessary files (already configured)
4. Consider using Docker BuildKit for faster builds

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [LiveKit Documentation](https://docs.livekit.io/)
