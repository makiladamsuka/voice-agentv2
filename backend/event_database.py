import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
import hashlib
from event_indexer import index_posters

class EventDatabase:
    def __init__(self, persist_directory):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="campus_events",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def add_events(self, events):
        if not events:
            return

        ids = []
        documents = []
        metadatas = []

        for i, event in enumerate(events):
            # Create a rich text description for embedding
            text_desc = f"{event.get('title', '')} on {event.get('date', '')} at {event.get('time', '')}. {event.get('description', '')}"
            
            ids.append(f"event_{i}_{event.get('source_file', 'unknown')}")
            documents.append(text_desc)
            
            # Metadata must be simple types
            meta = {k: str(v) for k, v in event.items()}
            metadatas.append(meta)

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"💾 Added {len(ids)} events to database")

    def query_events(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Format results for the agent
        formatted_events = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for meta in results['metadatas'][0]:
                formatted_events.append(meta)
        
        return formatted_events

    def has_data(self) -> bool:
        """Returns True if the collection already has indexed events."""
        return self.collection.count() > 0


def _compute_events_manifest(assets_dir: Path) -> dict:
    """Compute a dict of {filename: md5_hash} for all images in the folders."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    manifest = {}
    categories = ["events", "competitions", "posts"]
    
    for category in categories:
        cat_dir = assets_dir / category
        if cat_dir.exists():
            for f in sorted(cat_dir.iterdir()):
                if f.suffix.lower() in valid_extensions:
                    md5 = hashlib.md5(f.read_bytes()).hexdigest()
                    manifest[f"{category}/{f.name}"] = md5
    return manifest


def build_event_database(assets_dir: Path):
    """
    Builds or updates the event database from posters.
    Skips re-indexing if the events folder hasn't changed since last run.
    """
    db_path = Path(__file__).parent / "event_db"
    db_path.mkdir(exist_ok=True)
    manifest_path = db_path / "event_manifest.json"

    db = EventDatabase(db_path)

    # Compute current state of the folders
    current_manifest = _compute_events_manifest(assets_dir)

    # Load previously saved manifest (if any)
    saved_manifest = {}
    if manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text())
        except Exception:
            saved_manifest = {}

    # Skip re-indexing if nothing changed AND DB already has data
    if current_manifest == saved_manifest and db.has_data():
        print(f"✅ Event DB up-to-date ({len(current_manifest)} posters, skipping re-index)")
        return db

    # Something changed (or first run) — re-index
    changed = set(current_manifest) ^ set(saved_manifest)
    print(f"🔄 Events changed ({len(changed)} file(s) differ). Re-indexing...")
    events = index_posters(assets_dir)
    db.add_events(events)

    # Save extracted events to JSON for the frontend to read
    extracted_events_path = db_path / "extracted_events.json"
    extracted_events_path.write_text(json.dumps(events, indent=2))

    # Save the new manifest
    manifest_path.write_text(json.dumps(current_manifest, indent=2))
    print(f"💾 Manifest saved ({len(current_manifest)} files tracked)")

    return db
