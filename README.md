·# Campus Greeting Robot - Voice Agent V2

A LiveKit-powered voice agent for campus assistance with face recognition, color detection, and image display capabilities.

## Features

- **Face Recognition**: Recognizes enrolled users via webcam
- **Color Detection**: Identifies dominant colors in camera view
- **Event Posters**: Displays event information with images
- **Location Maps**: Shows campus navigation maps
- **Natural Language**: Fuzzy matching for user queries

## Setup

### Backend

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

### Frontend

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
