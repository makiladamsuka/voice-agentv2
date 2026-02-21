from pathlib import Path
import json
import base64
from openai import OpenAI
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def index_posters(assets_dir: Path):
    """
    Scans the assets_dir for image files, sends them to a VLM (OpenAI) 
    to extract event details, and returns a list of event dictionaries.
    """
    events_dir = assets_dir / "events"
    if not events_dir.exists():
        print(f"⚠️ Events directory not found: {events_dir}")
        return []

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    events = []
    
    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    print(f"🔍 Scanning for posters in {events_dir}...")

    for file_path in events_dir.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            print(f"   Processing {file_path.name}...")
            try:
                base64_image = encode_image(file_path)
                
                response = client.chat.completions.create(
                    model="google/gemini-2.0-flash-001", # Good, cheap vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract event details from this poster. Return JSON with keys: title, date, time, location, description. If not an event poster, return null."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"} 
                )
                
                content = response.choices[0].message.content
                if content:
                    event_data = json.loads(content)
                    if event_data:
                        event_data['source_file'] = file_path.name
                        events.append(event_data)
                        print(f"   ✅ Extracted: {event_data.get('title', 'Unknown Event')}")
            
            except Exception as e:
                print(f"   ❌ Failed to process {file_path.name}: {e}")

    return events
