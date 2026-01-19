"""
Poster Indexer - Extracts text from event posters using OCR
Uses pytesseract (Tesseract OCR) which is lighter and compatible with Pi.
"""

import pytesseract
from PIL import Image
from pathlib import Path
from typing import Dict, List
import re

class PosterIndexer:
    """Extracts text from event poster images using Tesseract OCR"""
    
    def __init__(self, assets_dir: Path = None):
        if assets_dir is None:
            assets_dir = Path(__file__).parent / "assets"
        self.assets_dir = assets_dir
        self.events_dir = assets_dir / "events"
        print("📖 OCR engine ready (Tesseract)")
    
    def extract_text(self, image_path: Path) -> str:
        """Extract all text from an image"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            print(f"⚠️ OCR error for {image_path}: {e}")
            return ""
    
    def extract_event_info(self, image_path: Path) -> Dict:
        """
        Extract structured event info from poster.
        Returns dict with: name, date, time, venue, description
        """
        raw_text = self.extract_text(image_path)
        
        # Basic extraction - get event name from filename
        event_name = image_path.stem.replace("-", " ").replace("_", " ").title()
        
        # Try to find date patterns (e.g., "Jan 25", "25/01/2026", "25th January")
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # 25/01/2026
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b',  # 25th January
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?)\b',  # January 25th
        ]
        
        date_found = None
        for pattern in date_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                date_found = match.group(1)
                break
        
        # Try to find time patterns (e.g., "10:00 AM", "10am - 6pm")
        time_pattern = r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?(?:\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)?)\b'
        time_match = re.search(time_pattern, raw_text, re.IGNORECASE)
        time_found = time_match.group(1) if time_match else None
        
        # Try to find venue (look for keywords)
        venue_keywords = ['hall', 'auditorium', 'room', 'lab', 'ground', 'court', 'building', 'center', 'centre']
        venue_found = None
        for keyword in venue_keywords:
            pattern = rf'\b(\w+\s+{keyword}|\w+\s+\w+\s+{keyword})\b'
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                venue_found = match.group(1)
                break
        
        return {
            "name": event_name,
            "filename": image_path.name,
            "raw_text": raw_text,
            "date": date_found,
            "time": time_found,
            "venue": venue_found,
            "description": raw_text[:200] if len(raw_text) > 200 else raw_text
        }
    
    def index_all_posters(self) -> List[Dict]:
        """Scan all event posters and extract info"""
        events = []
        
        if not self.events_dir.exists():
            print(f"⚠️ Events directory not found: {self.events_dir}")
            return events
        
        # Scan all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        for image_path in self.events_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                print(f"📸 Scanning: {image_path.name}")
                event_info = self.extract_event_info(image_path)
                events.append(event_info)
                print(f"   → Found: {event_info['name']}")
                if event_info['date']:
                    print(f"   → Date: {event_info['date']}")
        
        print(f"✅ Indexed {len(events)} event posters")
        return events


# Test if run directly
if __name__ == "__main__":
    indexer = PosterIndexer()
    events = indexer.index_all_posters()
    
    print("\n" + "="*50)
    print("INDEXED EVENTS:")
    print("="*50)
    for event in events:
        print(f"\n📌 {event['name']}")
        print(f"   Date: {event['date'] or 'Not found'}")
        print(f"   Time: {event['time'] or 'Not found'}")
        print(f"   Venue: {event['venue'] or 'Not found'}")
        print(f"   Text preview: {event['raw_text'][:100]}...")
