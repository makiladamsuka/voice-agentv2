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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    events = []

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    categories = ["events", "competitions", "posts"]

    for category in categories:
        cat_dir = assets_dir / category
        if not cat_dir.exists():
            cat_dir.mkdir(parents=True, exist_ok=True)
            
    print(f"\n🔍 Scanning for posters in {assets_dir} (events, competitions, posts)...")

    for category in ["events", "competitions", "posts"]:
        for file_path in (assets_dir / category).iterdir():
            if file_path.suffix.lower() in valid_extensions:
                print(f"\n⏳ [AI OCR] Analyzing '{file_path.name}' ({category}) via Gemini 2.0 Flash...")
                print(f"   [AI OCR] Extracting details and preparing for vectorization...")
                try:
                    base64_image = encode_image(file_path)
                    
                    response = client.chat.completions.create(
                        model="google/gemini-2.5-flash", # Good, cheap vision model
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract details from this poster/image. Return JSON with keys: title, date, time, location, description. Do your best to extract any relevant information."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=1000
                    )
                    
                    content = response.choices[0].message.content
                    if content:
                        event_data = json.loads(content)
                        if event_data:
                            event_data['source_file'] = file_path.name
                            event_data['category'] = category
                            events.append(event_data)
                            print(f"   ✅ [AI OCR] Successfully extracted: {event_data.get('title', 'Unknown Event')}")
                            print(f"   🧠 [Vector DB] Vectorizing document and updating knowledge base...")
            
                except Exception as e:
                    print(f"   ❌ [AI OCR] Failed to process {file_path.name}: {e}")

    print("\n🏁 [Vector DB] Finished analyzing and vectorizing all posters.")
    return events
