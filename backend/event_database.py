import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from poster_indexer import index_posters

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

def build_event_database(assets_dir: Path):
    """
    Builds or updates the event database from posters.
    """
    db_path = Path(__file__).parent / "event_db"
    db_path.mkdir(exist_ok=True)
    
    db = EventDatabase(db_path)
    
    # Index posters
    events = index_posters(assets_dir)
    
    # Add to DB
    db.add_events(events)
    
    return db
