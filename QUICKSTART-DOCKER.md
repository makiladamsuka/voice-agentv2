# Quick Start - Docker

This is a quick reference for getting started with Docker. For comprehensive documentation, see [DOCKER.md](DOCKER.md).

## TL;DR

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your LiveKit credentials

# 2. Start services
docker compose up

# 3. Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8080
```

## Common Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Rebuild after code changes
docker compose up -d --build

# Stop services
docker compose down

# Restart a service
docker compose restart backend
```

## Multi-Platform Build

```bash
# Build for both x86 and ARM
docker buildx build --platform linux/amd64,linux/arm64 -t voice-agent-backend:latest ./backend
docker buildx build --platform linux/amd64,linux/arm64 -t voice-agent-frontend:latest ./frontend
```

## Troubleshooting

**Port already in use?**
Edit `docker-compose.yml` to change port mappings.

**Permission errors?**
```bash
sudo chown -R $USER:$USER backend/known_faces backend/assets
```

**Need more help?**
See the full [DOCKER.md](DOCKER.md) documentation.
