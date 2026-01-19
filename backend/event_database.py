"""
Event Database - ChromaDB vector store for event information
Stores OCR-extracted text from posters for semantic search.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
from poster_indexer import PosterIndexer

class EventDatabase:
    """ChromaDB-based vector store for event Q&A"""
    
    def __init__(self, persist_dir: Path = None):
        if persist_dir is None:
            persist_dir = Path(__file__).parent / "event_db"
        
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistence
        print("📚 Initializing event database...")
        self.client = chromadb.Client(Settings(
            persist_directory=str(persist_dir),
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="events",
            metadata={"description": "Event information from posters"}
        )
        print(f"✅ Event database ready! ({self.collection.count()} events indexed)")
    
    def index_events(self, events: List[Dict]):
        """Add events to the vector database"""
        if not events:
            print("⚠️ No events to index")
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, event in enumerate(events):
            # Create searchable document from event info
            doc_parts = [
                f"Event: {event['name']}",
                f"Description: {event.get('description', '')}",
            ]
            if event.get('date'):
                doc_parts.append(f"Date: {event['date']}")
            if event.get('time'):
                doc_parts.append(f"Time: {event['time']}")
            if event.get('venue'):
                doc_parts.append(f"Venue: {event['venue']}")
            
            document = " | ".join(doc_parts)
            documents.append(document)
            
            # Metadata for structured access
            metadatas.append({
                "name": event['name'],
                "filename": event.get('filename', ''),
                "date": event.get('date') or '',
                "time": event.get('time') or '',
                "venue": event.get('venue') or '',
            })
            
            ids.append(f"event_{i}_{event['name'].lower().replace(' ', '_')}")
        
        # Clear existing and add new
        try:
            # Delete existing events
            existing = self.collection.get()
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except:
            pass
        
        # Add events
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Indexed {len(events)} events in database")
    
    def query(self, question: str, n_results: int = 3) -> List[Dict]:
        """
        Search for events matching the question.
        Returns list of matching events with relevance.
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_results, self.collection.count())
        )
        
        # Format results
        matches = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                matches.append({
                    "document": doc,
                    "name": metadata.get('name', 'Unknown'),
                    "date": metadata.get('date', ''),
                    "time": metadata.get('time', ''),
                    "venue": metadata.get('venue', ''),
                    "filename": metadata.get('filename', ''),
                    "relevance": 1 - (distance / 2)  # Convert distance to relevance score
                })
        
        return matches
    
    def get_all_events(self) -> List[Dict]:
        """Get all indexed events"""
        results = self.collection.get()
        events = []
        if results['metadatas']:
            for metadata in results['metadatas']:
                events.append(metadata)
        return events


def build_event_database(assets_dir: Path = None) -> EventDatabase:
    """
    Build event database from scratch by scanning posters.
    Called at startup to ensure database is up to date.
    """
    print("\n" + "="*50)
    print("🔄 Building event database from posters...")
    print("="*50 + "\n")
    
    # Initialize indexer and scan posters
    indexer = PosterIndexer(assets_dir)
    events = indexer.index_all_posters()
    
    # Store in database
    db = EventDatabase()
    db.index_events(events)
    
    return db


# Test if run directly
if __name__ == "__main__":
    db = build_event_database()
    
    print("\n" + "="*50)
    print("TESTING QUERIES:")
    print("="*50)
    
    test_questions = [
        "When is the art exhibition?",
        "What events are happening?",
        "Where is the sports meet?",
        "Tell me about freshers",
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        results = db.query(q)
        if results:
            for r in results:
                print(f"   → {r['name']}: {r['date'] or 'Date TBD'} at {r['venue'] or 'Venue TBD'}")
        else:
            print("   → No matching events found")
