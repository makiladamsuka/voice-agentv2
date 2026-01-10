"""
Image Manager for Voice Agent V2
Handles fuzzy matching and serving of event posters and location maps.
"""

import os
from pathlib import Path
from typing import Optional, List
import base64
from difflib import SequenceMatcher

class ImageManager:
    """Manages event posters and location maps with fuzzy matching"""
    
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.events_dir = assets_dir / "events"
        self.maps_dir = assets_dir / "maps"
        self.fallback_dir = assets_dir / "fallback"
        
        # Ensure directories exist
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ ImageManager initialized")
        print(f"   Events: {self.events_dir}")
        print(f"   Maps: {self.maps_dir}")
        print(f"   Fallback: {self.fallback_dir}")
    
    def _fuzzy_match(self, query: str, candidates: List[str], threshold: float = 0.5) -> Optional[str]:
        """
        Fuzzy match query against candidate strings.
        Returns best match if similarity > threshold.
        """
        query_lower = query.lower().strip()
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            # Remove file extension and convert to lowercase
            candidate_name = Path(candidate).stem.lower()
            
            # Replace underscores and hyphens with spaces for better matching
            candidate_clean = candidate_name.replace('_', ' ').replace('-', ' ')
            
            # Calculate similarity
            score = SequenceMatcher(None, query_lower, candidate_clean).ratio()
            
            # Also check if query is a substring
            if query_lower in candidate_clean:
                score = max(score, 0.8)
            
            print(f"  Matching '{query}' vs '{candidate_name}': score = {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= threshold:
            print(f"  ✅ Best match: {best_match} (score: {best_score:.2f})")
            return best_match
        else:
            print(f"  ❌ No match found (best score: {best_score:.2f})")
            return None
    
    def _get_all_images(self, directory: Path) -> List[str]:
        """Get all image files in a directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        images = []
        
        if directory.exists():
            for file in directory.iterdir():
                if file.suffix.lower() in image_extensions:
                    images.append(file.name)
        
        return images
    
    def find_event_image(self, query: str) -> Optional[Path]:
        """
        Find an event poster based on natural language query.
        
        Args:
            query: Natural language description (e.g., "tech fest", "cultural night")
        
        Returns:
            Path to image file or None if not found
        """
        print(f"🔍 Searching for event image: '{query}'")
        
        available_events = self._get_all_images(self.events_dir)
        print(f"   Available events: {available_events}")
        
        if not available_events:
            print("   ⚠️  No event images found in directory")
            return None
        
        matched_file = self._fuzzy_match(query, available_events)
        
        if matched_file:
            return self.events_dir / matched_file
        
        return None
    
    def list_available_events(self) -> List[str]:
        """
        List all available event posters.
        
        Returns:
            List of human-readable event names
        """
        available_events = self._get_all_images(self.events_dir)
        
        # Convert filenames to human-readable format
        event_names = []
        for filename in available_events:
            # Remove extension and convert to title case
            name = Path(filename).stem
            # Replace hyphens and underscores with spaces
            name = name.replace('-', ' ').replace('_', ' ')
            # Capitalize each word
            name = ' '.join(word.capitalize() for word in name.split())
            event_names.append(name)
        
        return event_names
    
    def find_location_map(self, query: str) -> Optional[Path]:
        """
        Find a location map based on natural language query.
        
        Args:
            query: Natural language query (e.g., "DS lab", "library", "cafeteria")
        
        Returns:
            Path to image file or None if not found
        """
        print(f"🗺️  Searching for location map: '{query}'")
        
        available_maps = self._get_all_images(self.maps_dir)
        print(f"   Available maps: {available_maps}")
        
        if not available_maps:
            print("   ⚠️  No map images found in directory")
            return None
        
        matched_file = self._fuzzy_match(query, available_maps)
        
        if matched_file:
            return self.maps_dir / matched_file
        
        return None
    
    def get_fallback_image(self) -> Optional[Path]:
        """Get the fallback 'no image found' image"""
        fallback_files = self._get_all_images(self.fallback_dir)
        
        if fallback_files:
            return self.fallback_dir / fallback_files[0]
        
        return None
    
    def encode_image(self, image_path: Path) -> str:
        """
        Load and base64 encode an image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded string
        """
        print(f"📸 Encoding image: {image_path}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        encoded = base64.b64encode(image_data).decode('utf-8')
        print(f"   ✅ Encoded {len(image_data)} bytes → {len(encoded)} chars")
        
        return encoded
