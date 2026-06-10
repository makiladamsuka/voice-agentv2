import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)
print("URL:", os.getenv("LIVEKIT_URL"))
print("KEY:", os.getenv("LIVEKIT_API_KEY"))
print("SECRET:", os.getenv("LIVEKIT_API_SECRET"))
