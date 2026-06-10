import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from event_indexer import index_posters
assets_dir = Path("assets")
print("Starting indexer...")
events = index_posters(assets_dir)
print("Finished. Extracted:", events)
