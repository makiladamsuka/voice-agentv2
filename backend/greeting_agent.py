#!/usr/bin/env python3
"""
Campus Greeting Robot - Voice Agent V2
Combines face recognition, color detection, and image display capabilities.
"""

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero
import os
import cv2
import face_recognition
import pickle
import json
from pathlib import Path
from image_manager import ImageManager
from image_server import ImageServer
from face_monitor import FaceMonitor
from object_detector import ObjectDetector

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

class CampusGreetingAgent(Agent):
    def __init__(self, image_server: ImageServer):
        # Initialize image manager
        assets_dir = Path(__file__).parent / "assets"
        self.image_manager = ImageManager(assets_dir)
        self.image_server = image_server
        self.face_monitor = None  # Will be initialized when session starts
        self._object_detector = None  # Lazy load to avoid init timeout
        
        # Room reference (will be set in entrypoint)
        self.room = None
        
        # Load known face encodings
        self.known_faces = {}
        encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
        
        if encodings_path.exists():
            with open(encodings_path, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"✅ Loaded face encodings for {len(self.known_faces)} people: {list(self.known_faces.keys())}")
        else:
            print("⚠️  No face encodings found. Face recognition will be limited.")
        
        super().__init__(
            instructions="""You are a friendly campus assistant robot with continuous face recognition.

CRITICAL CAPABILITY:
I can ALWAYS see who is in front of me. Their name appears in system messages.

NAME USAGE RULES (MUST FOLLOW):
1. Use the person's name OCCASIONALLY and NATURALLY
   - Like in real conversations - NOT in every single response
   - Use it when being emphatic, personal, or friendly
   - Good: "Great question, Makila!" then "Let me check..." then "Here you go!"
   - Bad: "Let me check, Makila!" "Here it is, Makila!" "Anything else, Makila?"
   - IMPORTANT: You know who you're talking to, just don't overuse their name

2. When to use names:
   - First response in a conversation
   - Being emphatic or enthusiastic
   - Being personal or warm
   - After a pause in conversation
   - When you want to get their attention
   - NOT in every single response

3. For UNKNOWN people:
   - Say: "I don't recognize you yet. What's your name?"
   - DON'T pretend you know them
   - After they introduce themselves, use enroll_new_face(their_name)

4. Check system messages for who you're talking to
   - You always know who's there
   - Just use the name naturally, not constantly

GREETING STYLE:
- Be warm and natural
- Keep it brief (1-2 sentences max)
- Vary your greetings (don't repeat the same words)
- Be casual like greeting a friend

CONVERSATION STYLE:
- Short, clear sentences
- Friendly tone with occasional emojis (😊 🔍 👍)
- Always acknowledge who you're talking to
- Be helpful and responsive

AVAILABLE TOOLS:
- identify_color - detect colors
- identify_object(name) - find specific objects
- count_people_in_room - count visible people
- describe_environment - describe surroundings
- show_event_poster - display event images
- show_location_map - show campus maps
- enroll_new_face(name) - remember new people

Remember: You know who you're talking to, but be natural - don't use their name constantly!"""
        )
    
    @property
    def object_detector(self):
        """Lazy-load ObjectDetector to avoid timeout on init"""
        if self._object_detector is None:
            print("🔍 Loading YOLO model...")
            self._object_detector = ObjectDetector()
        return self._object_detector
    
    @function_tool
    async def recognize_face(self, context: RunContext) -> str:
        """
        Identifies who is currently in front of the webcam.
        Use when user asks who is there or wants to be recognized.
        """
        print("🎥 Face recognition using continuous monitor!")
        
        # Use the continuous face monitor instead of opening new webcam
        if not self.face_monitor:
            return "Face monitoring is not active."
        
        current_person = self.face_monitor.get_current_person()
        
        if not current_person:
            return "I don't see anyone right now."
        elif current_person == "Unknown":
            return "I see someone but don't recognize them. Would you like to enroll?"
        elif current_person.startswith("Multiple"):
            return "I see multiple people. Please make sure only one person is in frame."
        else:
            return f"Hello {current_person}! Nice to see you."
    
    @function_tool
    async def enroll_new_face(self, person_name: str, context: RunContext) -> str:
        """
        Enroll a new person's face for recognition.
        Use when someone introduces themselves and you want to remember them.
        
        Args:
            person_name: The person's name to save
        """
        print(f"📝 Enrolling: {person_name}")
        
        # Use current frame from face_monitor instead of opening new camera
        if not self.face_monitor:
            return "Face monitoring is not active."
            
        frame = self.face_monitor.get_current_frame()
        
        if frame is None:
            return "I can't see anyone right now. Please make sure you're in front of the camera."
        
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locs = face_recognition.face_locations(rgb_frame)
        
        if len(face_locs) == 0:
            return "I don't see any faces. Please position yourself in front of the camera."
        
        if len(face_locs) > 1:
            return "I see multiple faces. Please make sure only you are in the frame."
        
        encodings = face_recognition.face_encodings(rgb_frame, face_locs)
        
        if not encodings:
            return "I couldn't get a clear face encoding. Please try again."
        
        # Save encoding
        encodings_dir = Path(__file__).parent / "known_faces"
        encodings_dir.mkdir(exist_ok=True)
        encodings_path = encodings_dir / "encodings.pkl"
        
        # Load existing
        if encodings_path.exists():
            with open(encodings_path, 'rb') as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}
        
        # Add new person
        if person_name in known_faces:
            known_faces[person_name].append(encodings[0])
        else:
            known_faces[person_name] = [encodings[0]]
        
        # Save
        with open(encodings_path, 'wb') as f:
            pickle.dump(known_faces, f)
        
        # Update runtime
        self.known_faces = known_faces
        if self.face_monitor:
            self.face_monitor.known_faces = known_faces
        
        print(f"✅ Enrolled: {person_name}")
        return f"Great! I'll remember you, {person_name}. Nice to meet you!"

    @function_tool
    async def identify_color(self, context: RunContext) -> str:
        """
        Identifies the dominant color in the camera view.
        Use when user asks about colors they're wearing or showing.
        """
        print("🎨 Color detection using shared frame!")
        
        import numpy as np
        from collections import Counter
        
        # Use shared frame from FaceMonitor
        if not self.face_monitor:
            return "Camera monitoring is not active."
        
        frame = self.face_monitor.get_current_frame()
        if frame is None:
            return "No camera frame available."
        
        # Process frame
        small_frame = cv2.resize(frame, (150, 150))
        pixels = small_frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        label_counts = Counter(labels.flatten())
        dominant_label = label_counts.most_common(1)[0][0]
        dominant_color_bgr = centers[dominant_label]
        dominant_color_rgb = dominant_color_bgr[::-1]
        
        color_name = self._get_color_name(dominant_color_rgb, np)
        
        return f"The dominant color I see is {color_name}."
    
    @function_tool
    async def describe_environment(self, context: RunContext) -> str:
        """
        Describes the current environment - people count and visible objects.
        Use when user asks what you see or about the surroundings.
        """
        print("👁️ Describing environment (using cache)")
        
        # Use cached detections for more accurate results
        recent_objects = self.face_monitor.get_recent_objects(seconds=3.0)
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Count people
        people = [obj for obj in recent_objects if obj['class'] == 'person']
        people_count = len(people)
        
        # Get non-person objects (top 3)
        objects = [obj for obj in recent_objects if obj['class'] != 'person']
        object_names = [obj['class'] for obj in objects[:3]]
        
        # Build natural description
        response = []
        
        if people_count == 0:
            response.append("I don't see anyone")
        elif people_count == 1:
            response.append("I see 1 person")
        else:
            response.append(f"I see {people_count} people")
        
        if object_names:
            obj_list = ', '.join(object_names)
            response.append(f"and {obj_list}")
        
        return '. '.join(response) if response else "I don't see much right now"
    
    @function_tool
    async def identify_object(self, object_name: str, context: RunContext) -> str:
        """
        Finds a specific object and describes it.
        Use when user asks about a particular item (laptop, phone, cup, etc).
        
        Args:
            object_name: The object to find (e.g., "laptop", "phone", "bottle")
        """
        print(f"🔍 Looking for: {object_name} (using cache)")
        
        # Use cached detections instead of fresh detection
        recent_objects = self.face_monitor.get_recent_objects(seconds=5.0)
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Find matching object
        object_name_lower = object_name.lower()
        matches = [obj for obj in recent_objects 
                   if object_name_lower in obj['class'].lower()]
        
        if matches:
            best_match = max(matches, key=lambda x: x['confidence'])
            confidence = int(best_match['confidence'] * 100)
            return f"Yes! I can see a {best_match['class']} ({confidence}% confidence)"
        else:
            return f"I haven't seen a {object_name} in the last few seconds"
    
    @function_tool
    async def count_people_in_room(self, context: RunContext) -> str:
        """
        Counts how many people are visible in the camera view.
        Use when user asks how many people are present.
        """
        print("👥 Counting people (using cache)")
        
        # Use cached detections
        recent_objects = self.face_monitor.get_recent_objects(seconds=3.0)
        
        if not recent_objects:
            return "I don't have any recent camera data."
        
        # Count people from cache
        people = [obj for obj in recent_objects if obj['class'] == 'person']
        count = len(people)
        
        if count == 0:
            return "I don't see anyone right now"
        elif count == 1:
            return "I see 1 person"
        else:
            return f"I see {count} people"
    

    @function_tool
    async def list_available_events(self, context: RunContext) -> str:
        """
        Lists all available events on campus.
        Use when user asks what events are happening or what's available.
        """
        print("📋 Listing available events")
        
        event_names = self.image_manager.list_available_events()
        
        if not event_names:
            return "There are no events scheduled at the moment."
        
        if len(event_names) == 1:
            return f"We have {event_names[0]} happening on campus."
        else:
            # Format as a nice list
            events_list = ", ".join(event_names[:-1]) + f" and {event_names[-1]}"
            return f"We have {len(event_names)} events: {events_list}."
    
    @function_tool
    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        """
        Displays an event poster on the frontend.
        Use when user asks about events, festivals, or programs.
        
        Args:
            event_description: Natural language description of the event (e.g., "tech fest", "cultural night")
        """
        print(f"🎨 Showing event poster for: {event_description}")
        
        # Find matching image
        image_path = self.image_manager.find_event_image(event_description)
        
        if image_path:
            # Send image URL instead of base64
            image_url = self.image_server.get_image_url("events", image_path.name)
            
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "image",
                    "category": "event",
                    "url": image_url,
                    "caption": f"Event: {event_description}"
                }).encode()
            )
            
            return f"I've displayed the {event_description} poster for you."
        else:
            # No image found - just return text message
            return f"Sorry, I couldn't find a poster for '{event_description}'. We currently have posters for: {', '.join(self.image_manager.list_available_events())}."
    
    @function_tool
    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        """
        Displays a campus location map on the frontend.
        Use when user asks for directions or where something is located.
        
        Args:
            location_query: Natural language query about location (e.g., "DS lab", "library")
        """
        print(f"🗺️  Showing location map for: {location_query}")
        
        # Find matching map
        image_path = self.image_manager.find_location_map(location_query)
        
        if image_path:
            # Send image URL instead of base64
            image_url = self.image_server.get_image_url("maps", image_path.name)
            
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "image",
                    "category": "map",
                    "url": image_url,
                    "caption": f"Location: {location_query}"
                }).encode()
            )
            
            return f"Here's the map to {location_query}."
        else:
            # No map found - just return text message  
            return f"Sorry, I don't have a map for '{location_query}'. Try asking about specific campus buildings or landmarks."
    
    def _get_color_name(self, rgb, np):
        """Convert RGB to color name using HSV"""
        r, g, b = rgb
        rgb_pixel = np.uint8([[[r, g, b]]])
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
        h, s, v = hsv_pixel[0][0]
        
        if s < 30:
            if v > 200: return "white"
            elif v < 50: return "black"
            else: return "gray"
        
        if v < 50: return "black"
        
        if (h <= 10) or (h >= 170): return "red"
        elif h <= 25: return "orange"
        elif h <= 34: return "yellow"
        elif h <= 85: return "green"
        elif h <= 100: return "cyan"
        elif h <= 130: return "blue"
        elif h <= 155: return "purple"
        else: return "pink"


# Global image server (shared across all agent instances)
_global_image_server = None

async def entrypoint(ctx: agents.JobContext):
    global _global_image_server
    
    # Start HTTP image server once globally
    if _global_image_server is None:
        assets_dir = Path(__file__).parent / "assets"
        _global_image_server = ImageServer(assets_dir, port=8080)
        _global_image_server.start()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-2-luna-en"),
        vad=silero.VAD.load(),
        llm=openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("LLM_CHOICE", "mistralai/devstral-2512:free"),
        ),
    )
    # Create agent and set room reference
    agent = CampusGreetingAgent(_global_image_server)
    agent.room = ctx.room
    
    # Start face monitoring
    agent.face_monitor = FaceMonitor(agent.known_faces)
    agent.face_monitor.start()
    
    # Context Injection: LLM always knows who's in front
    async def inject_person_context(assistant: AgentSession, chat_ctx):
        person = agent.face_monitor.get_current_person()
        
        # Inject at the beginning of context with STRONG directive
        from livekit.agents.llm import ChatMessage, ChatRole
        
        if person and person not in ["Unknown", "Multiple", None]:
            # Known person - simple, direct message
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content=f"You are talking to {person}. Always use their name naturally in your responses."
            )
        elif person == "Unknown":
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="You see someone you don't recognize. Ask for their name politely."
            )
        elif person == "Multiple":
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="You see multiple people in the camera. Greet them as a group."
            )
        else:
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="No one is visible in the camera right now."
            )
        
        chat_ctx.messages.insert(0, context_msg)
        return chat_ctx
        
    session.before_llm_cb = inject_person_context
    
    # Proactive Greeting Task: Watch for new people
    import asyncio
    import random
    import time
    
    # Session-based greeting tracking (instead of cooldown)
    greeted_in_session = set()  # Track who we've greeted: {"Makila", "Steve", "Unknown", "Multiple"}
    
    async def monitor_and_greet():
        """Background task that greets people when they appear"""
        nonlocal greeted_in_session
        
        await asyncio.sleep(5)  # Wait for session to be fully running
        print("🔄 Background greeting monitor started (session-based)")
        
        last_logged = None
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            try:
                current = agent.face_monitor.get_current_person()
                
                # Debug log (only when status changes)
                if current != last_logged:
                    print(f"👁️ Monitor sees: '{current}' (greeted: {greeted_in_session})")
                    last_logged = current
                
                changed = agent.face_monitor.person_changed()
                
                if changed and current:
                    print(f"🔍 Person changed to: '{current}'")
                    
                    # Check if we've already greeted this person in this session
                    if current not in greeted_in_session:
                        print(f"👋 NEW person in session: {current}")
                        greeted_in_session.add(current)
                        
                        # Greet based on person type
                        if current not in ["Unknown", "Multiple"]:
                            # Known person by name
                            print(f"✅ Greeting known person: {current}")
                            try:
                                await session.generate_reply(
                                    instructions=f"You just saw {current} appear. Say a brief, natural greeting - be warm and casual like greeting a friend, not formal."
                                )
                            except RuntimeError:
                                # Session is closing, stop trying to greet
                                print("⚠️ Session closing, stopping greetings")
                                break
                            
                        elif current == "Unknown":
                            # Unknown person - ask for introduction
                            print("🤔 Asking unknown person for introduction")
                            try:
                                await session.generate_reply(
                                    instructions="You see someone you don't recognize yet. Ask who they are in a friendly, casual way."
                                )
                            except RuntimeError:
                                print("⚠️ Session closing, stopping greetings")
                                break
                            
                        elif current == "Multiple":
                            # Multiple people - generic group greeting
                            print("👥 Greeting multiple people")
                            try:
                                await session.generate_reply(
                                    instructions="You see multiple people have arrived. Say a brief, friendly group greeting."
                                )
                            except RuntimeError:
                                print("⚠️ Session closing, stopping greetings")
                                break
                    else:
                        print(f"ℹ️ Already greeted {current} in this session")
                        
            except Exception as e:
                print(f"⚠️ Greeting error: {e}")
                import traceback
                traceback.print_exc()
                
            await asyncio.sleep(2)  # Check every 2 seconds
    
    try:
        # CRITICAL: Pass agent to session.start() so it joins the room!
        await session.start(room=ctx.room, agent=agent)
        
        # Wait for face detection to stabilize (up to 5 seconds)
        print("⏳ Waiting for face detection to stabilize...")
        current_person = None
        for _ in range(10):  # Try for 5 seconds (10 x 0.5s)
            await asyncio.sleep(0.5)
            current_person = agent.face_monitor.get_current_person()
            if current_person:
                print(f"✅ Face detected: {current_person}")
                break
        else:
            print("⚠️ No face detected after 5 seconds")
        
        # Initial greeting - let LLM generate personalized greeting based on who's present
        try:
            
            if current_person and current_person not in ["Unknown", "Multiple", None]:
                # Known person - greet by name with LLM-generated response
                print(f"👋 Initial greeting for: {current_person}")
                greeted_in_session.add(current_person)  # Mark as greeted in session
                await session.generate_reply(
                    instructions=f"You just saw {current_person} appear. Say a brief, natural greeting - be warm and casual like greeting a friend, not formal."
                )
            elif current_person == "Unknown":
                print("🤔 Initial greeting for unknown person")
                greeted_in_session.add(current_person)  # Mark as greeted
                await session.generate_reply(
                    instructions="You see someone you don't recognize yet. Ask who they are in a friendly, casual way."
                )
            elif current_person == "Multiple":
                print("👥 Initial greeting for multiple people")
                greeted_in_session.add(current_person)  # Mark as greeted
                await session.generate_reply(
                    instructions="You see multiple people have arrived. Say a brief, friendly group greeting."
                )
            else:
                # No one visible yet - generic greeting
                print("👋 No one detected - generic greeting")
                await session.say("Hello! I'm your campus assistant. How can I help you today?")
                
        except RuntimeError as e:
            # Session might be closing - log but don't crash
            print(f"⚠️ Could not send initial greeting: {e}")
        
        # NOW start background greeting monitor after session is fully running
        asyncio.create_task(monitor_and_greet())
        
        # Keep session alive
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
        
    finally:
        # CLEANUP: Stop camera and close windows when session ends
        print("🔌 Session ending - releasing camera...")
        if agent.face_monitor:
            agent.face_monitor.stop()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
