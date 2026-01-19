from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero
import os
import pickle
import json
import asyncio
from pathlib import Path
from image_manager import ImageManager
from image_server import ImageServer
from face_monitor import FaceMonitor
from object_detector import ObjectDetector
from greetings import generate_greeting, generate_group_greeting
from event_database import EventDatabase, build_event_database

# Import modular tools
from tools.vision import VisionTools
from tools.content import ContentTools
from tools.system import SystemTools

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

class CampusGreetingAgent(Agent):
    def __init__(self, image_server, event_db=None):
        # Initialize image manager
        assets_dir = Path(__file__).parent / "assets"
        self.image_manager = ImageManager(assets_dir)
        self.image_server = image_server
        self.event_db = event_db  # Event database for Q&A
        self.face_monitor = None
        self._object_detector = None
        
        # Room reference
        self.room = None
        
        # Initialize tool helpers
        self.vision_tools = VisionTools(
            face_monitor=None,  # Will set later
            object_detector_factory=self.get_object_detector
        )
        self.content_tools = ContentTools(
            image_manager=self.image_manager,
            image_server=self.image_server,
            room_provider=lambda: self.room
        )
        self.system_tools = SystemTools()
        
        # Load known face encodings
        self.known_faces = {}
        encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
        
        if encodings_path.exists():
            try:
                with open(encodings_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Loaded face encodings for {len(self.known_faces)} people: {list(self.known_faces.keys())}")
            except (ModuleNotFoundError, AttributeError, ImportError) as e:
                # Handle numpy version incompatibility or other pickle loading issues
                print(f"⚠️  Failed to load face encodings due to version incompatibility: {e}")
                print("   The encodings.pkl file may have been created with a different numpy version.")
                print("   Face recognition will start fresh. Re-enroll faces if needed.")
                # Backup the old file and create a new empty one
                backup_path = encodings_path.with_suffix('.pkl.backup')
                try:
                    import shutil
                    shutil.move(encodings_path, backup_path)
                    print(f"   Old encodings backed up to: {backup_path}")
                except Exception as backup_error:
                    print(f"   Could not backup old encodings: {backup_error}")
                self.known_faces = {}
            except Exception as e:
                print(f"⚠️  Error loading face encodings: {e}")
                print("   Face recognition will start fresh.")
                self.known_faces = {}
        else:
            print("⚠️  No face encodings found. Face recognition will be limited.")
        
        super().__init__(
            instructions="""You are a friendly campus assistant robot with continuous face recognition.

CRITICAL CAPABILITY:
I can ALWAYS see who is in front of me. Their name appears in system messages.

AUTO-ENROLLMENT (VERY IMPORTANT):
When someone introduces themselves (says "I'm [Name]" or "My name is [Name]"):
1. IMMEDIATELY call enroll_new_face(their_name) to remember them
2. Then respond warmly like "Nice to meet you, [Name]! I'll remember you now."
3. Example: User says "I'm Alex" → call enroll_new_face("Alex") → "Great to meet you Alex!"

NAME USAGE RULES (MUST FOLLOW):
1. Use the person's name OCCASIONALLY and NATURALLY
   - Like in real conversations - NOT in every single response
   - Use it when being emphatic, personal, or friendly
   - IMPORTANT: You know who you're talking to, just don't overuse their name

2. For UNKNOWN people:
   - Ask: "Hi there! I don't think we've met. What's your name?"
   - When they tell you, ALWAYS use enroll_new_face(name) to remember them

3. For KNOWN people:
   - Greet them by name: "Hey [Name]! Good to see you!"

GREETING STYLE:
- Be warm and natural
- Keep it brief (1-2 sentences max)
- Be casual like greeting a friend

CONVERSATION STYLE:
- Short, clear sentences
- Friendly tone
- Be helpful and responsive

AVAILABLE TOOLS:
- identify_color - detect colors
- identify_object(name) - find specific objects
- count_people_in_room - count visible people
- describe_environment - describe surroundings  
- show_event_poster - display event images
- show_location_map - show campus maps
- enroll_new_face(name) - ALWAYS USE when someone introduces themselves!
- ask_about_events(question) - answer questions about events (dates, times, venues)
- get_cpu_temperature - get CPU temperature
- get_system_info - get comprehensive system information
- get_cpu_usage - get CPU usage percentage
- get_memory_usage - get memory/RAM usage

Remember: Auto-enroll new people when they introduce themselves!"""
        )
    
    def get_object_detector(self):
        """Lazy-load ObjectDetector"""
        if self._object_detector is None:
            print("🔍 Loading YOLO model...")
            self._object_detector = ObjectDetector()
        return self._object_detector

    # --- Delegate to Tool Modules ---

    @function_tool
    async def recognize_face(self, context: RunContext) -> str:
        """Identifies who is currently in front of the webcam."""
        print("🎥 [TOOL] recognize_face called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.recognize_face(context)
    
    @function_tool
    async def enroll_new_face(self, person_name: str, context: RunContext) -> str:
        """Enroll a new person's face for recognition."""
        print(f"📝 [TOOL] enroll_new_face called for: {person_name}")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.enroll_new_face(person_name, context)

    @function_tool
    async def identify_color(self, context: RunContext) -> str:
        """Identifies the dominant color in the camera view."""
        print("🎨 [TOOL] identify_color called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.identify_color(context)
    
    @function_tool
    async def describe_environment(self, context: RunContext) -> str:
        """Describes the current environment - people count and visible objects."""
        print("👁️ [TOOL] describe_environment called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.describe_environment(context)
    
    @function_tool
    async def identify_object(self, object_name: str, context: RunContext) -> str:
        """Finds a specific object and describes it."""
        print(f"🔍 [TOOL] identify_object called for: {object_name}")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.identify_object(object_name, context)
    
    @function_tool
    async def count_people_in_room(self, context: RunContext) -> str:
        """Counts how many people are visible in the camera view."""
        print("👥 [TOOL] count_people_in_room called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.count_people_in_room(context)

    @function_tool
    async def list_available_events(self, context: RunContext) -> str:
        """Lists all available events on campus."""
        print("📋 [TOOL] list_available_events called")
        return await self.content_tools.list_available_events(context)
    
    @function_tool
    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        """Displays an event poster on the frontend."""
        print(f"🎨 [TOOL] show_event_poster called for: {event_description}")
        return await self.content_tools.show_event_poster(event_description, context)
    
    @function_tool
    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        """Displays a campus location map on the frontend."""
        print(f"🗺️ [TOOL] show_location_map called for: {location_query}")
        return await self.content_tools.show_location_map(location_query, context)
    
    @function_tool
    async def get_cpu_temperature(self, context: RunContext) -> str:
        """Gets the CPU temperature of the Raspberry Pi."""
        print("🌡️ [TOOL] get_cpu_temperature called")
        return await self.system_tools.get_cpu_temperature(context)
    
    @function_tool
    async def get_system_info(self, context: RunContext) -> str:
        """Gets comprehensive system information including CPU temperature, usage, memory, disk, and uptime."""
        print("💻 [TOOL] get_system_info called")
        return await self.system_tools.get_system_info(context)
    
    @function_tool
    async def get_cpu_usage(self, context: RunContext) -> str:
        """Gets the CPU usage percentage."""
        print("⚡ [TOOL] get_cpu_usage called")
        return await self.system_tools.get_cpu_usage(context)
    
    @function_tool
    async def get_memory_usage(self, context: RunContext) -> str:
        """Gets the memory (RAM) usage information."""
        print("🧠 [TOOL] get_memory_usage called")
        return await self.system_tools.get_memory_usage(context)
    
    @function_tool
    async def ask_about_events(self, question: str, context: RunContext) -> str:
        """
        Answers questions about campus events by searching poster information.
        Use for questions like: "When is the art exhibition?", "What events are happening?", "Where is the sports meet?"
        
        Args:
            question: The question about events to answer
        """
        print(f"📅 [TOOL] ask_about_events called: {question}")
        
        if not self.event_db:
            return "Event database is not available. I can't answer event questions right now."
        
        results = self.event_db.query(question, n_results=3)
        
        if not results:
            return "I don't have information about that event. Try asking about art exhibition, freshers sportmeet, or openhouse."
        
        # Format response
        responses = []
        for r in results:
            parts = [f"**{r['name']}**"]
            if r.get('date'):
                parts.append(f"Date: {r['date']}")
            if r.get('time'):
                parts.append(f"Time: {r['time']}")
            if r.get('venue'):
                parts.append(f"Venue: {r['venue']}")
            responses.append(" - ".join(parts))
        
        return "Here's what I found:\\n" + "\\n".join(responses)

# Global services (shared across all agent instances)
_global_face_monitor = None
_global_image_server = None
_global_event_db = None

def _load_known_faces():
    """Load face encodings from file"""
    known_faces = {}
    encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
    
    if encodings_path.exists():
        try:
            with open(encodings_path, 'rb') as f:
                known_faces = pickle.load(f)
            print(f"✅ Loaded face encodings for {len(known_faces)} people: {list(known_faces.keys())}")
        except Exception as e:
            print(f"⚠️ Error loading face encodings: {e}")
    else:
        print("⚠️ No face encodings found.")
    
    return known_faces

def _init_globals():
    """Initialize global services (camera, image server, event database) before any connection"""
    global _global_face_monitor, _global_image_server, _global_event_db
    
    # Start image server for posters/maps
    if _global_image_server is None:
        assets_dir = Path(__file__).parent / "assets"
        _global_image_server = ImageServer(assets_dir, port=8080)
        _global_image_server.start()
    
    # Build event database from posters (OCR)
    if _global_event_db is None:
        try:
            assets_dir = Path(__file__).parent / "assets"
            _global_event_db = build_event_database(assets_dir)
        except Exception as e:
            print(f"⚠️ Could not build event database: {e}")
            _global_event_db = None
    
    # Start camera
    if _global_face_monitor is None:
        print("🎥 Starting camera early (before any connection)...")
        known_faces = _load_known_faces()
        _global_face_monitor = FaceMonitor(known_faces)
        _global_face_monitor.start()
        import time
        time.sleep(1)
        print("✅ Camera ready!")


async def entrypoint(ctx: agents.JobContext):
    global _global_face_monitor, _global_image_server, _global_event_db
    
    # Initialize globals (camera, image server, event db) ONCE before any connection
    _init_globals()
    
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
    agent = CampusGreetingAgent(_global_image_server, _global_event_db)
    agent.room = ctx.room
    
    # Use global face monitor (already running)
    agent.face_monitor = _global_face_monitor
    agent.known_faces = _global_face_monitor.known_faces
    
    # Context Injection: LLM always knows who's in front
    async def inject_person_context(assistant: AgentSession, chat_ctx):
        # Use thread-safe FRESH people getter (most recent detection)
        fresh = agent.face_monitor.get_fresh_people()
        
        # Categorize
        known = [p for p in fresh if p != "Unknown"]
        unknown_count = sum(1 for p in fresh if p == "Unknown")
        
        from livekit.agents.llm import ChatMessage, ChatRole
        
        # Debug: log what we're injecting
        print(f"🎯 Context injection - Fresh: {fresh}, Known: {known}")
        
        if known:
            # Known people - list their names with STRONG emphasis
            names = ", ".join(known)
            if unknown_count:
                context_msg = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"CURRENT PERSON IN FRONT OF YOU: {names}. There's also someone you don't recognize. When asked 'who am I', answer with: {names}"
                )
            else:
                context_msg = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"CURRENT PERSON IN FRONT OF YOU: {names}. When asked 'who am I', answer with: {names}"
                )
        elif unknown_count:
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="CURRENT: Unknown person. You don't recognize them. Ask for their name."
            )
        else:
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="No one is visible right now."
            )
        
        chat_ctx.messages.insert(0, context_msg)
        return chat_ctx
        
    session.before_llm_cb = inject_person_context
    
    # Proactive Greeting Task: Watch for new people
    
    async def monitor_and_greet():
        """Background task that greets people when they appear"""
        await asyncio.sleep(5)  # Wait for session to be fully running
        print("🔄 Background greeting monitor started (multi-person mode)")
        
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            try:
                # Get new arrivals (FaceMonitor handles 5-second cooldown)
                arrivals = agent.face_monitor.get_new_arrivals()
                
                if arrivals:
                    print(f"👋 New arrivals: {arrivals}")
                    
                    # Categorize arrivals
                    known_people = [p for p in arrivals if p != "Unknown"]
                    unknown_count = arrivals.count("Unknown")
                    
                    # Mark all as greeted (5-second cooldown in FaceMonitor)
                    for p in arrivals:
                        agent.face_monitor.mark_greeted(p)
                    
                    # Build greeting instruction based on who arrived
                    try:
                        if len(known_people) > 0 and unknown_count == 0:
                            # Only known people
                            if len(known_people) == 1:
                                name = known_people[0]
                                greeting = generate_greeting(name, is_known=True)
                                print(f"✅ Greeting known person: {name} -> {greeting}")
                                await session.say(greeting)
                            else:
                                greeting = generate_group_greeting(known_people, 0)
                                print(f"✅ Greeting multiple known people -> {greeting}")
                                await session.say(greeting)
                        
                        elif known_people and unknown_count > 0:
                            # Mix of known and unknown
                            greeting = generate_group_greeting(known_people, unknown_count)
                            print(f"🤔 Greeting mix -> {greeting}")
                            await session.say(greeting)
                        
                        elif unknown_count == 1:
                            # Single unknown person - ask for name
                            greeting = generate_greeting("Unknown", is_known=False)
                            print(f"🤔 Greeting unknown person -> {greeting}")
                            await session.say(greeting)
                        
                        else:
                            # Multiple unknown people
                            greeting = generate_group_greeting([], unknown_count)
                            print(f"👥 Greeting {unknown_count} unknown people -> {greeting}")
                            await session.say(greeting)
                            
                    except RuntimeError:
                        print("⚠️ Session closing, stopping greetings")
                        break
                        
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
        people = set()
        for i in range(10):  # Try for 5 seconds (10 x 0.5s)
            await asyncio.sleep(0.5)
            # Check both stable and fresh detection
            stable = agent.face_monitor.get_current_people()
            fresh = agent.face_monitor.fresh_people if hasattr(agent.face_monitor, 'fresh_people') else set()
            print(f"  [{i+1}/10] Stable: {stable}, Fresh: {fresh}")
            
            # Use fresh if available, otherwise stable
            people = fresh if fresh else stable
            if people:
                print(f"✅ Faces detected: {people}")
                break
        else:
            print("⚠️ No faces detected after 5 seconds")
        
        # Initial greeting based on who's present
        try:
            known_people = [p for p in people if p != "Unknown"]
            unknown_count = sum(1 for p in people if p == "Unknown")
            
            # Mark everyone as greeted (5-second cooldown)
            for p in people:
                agent.face_monitor.mark_greeted(p)
            
            if known_people and not unknown_count:
                # All known people - direct greeting
                if len(known_people) == 1:
                    greeting = generate_greeting(known_people[0], is_known=True)
                else:
                    greeting = generate_group_greeting(known_people, 0)
                print(f"👋 Initial greeting -> {greeting}")
                await session.say(greeting)
            elif known_people and unknown_count:
                # Mix of known and unknown
                greeting = generate_group_greeting(known_people, unknown_count)
                print(f"👋 Initial greeting (mix) -> {greeting}")
                await session.say(greeting)
            elif unknown_count == 1:
                # Single unknown person
                greeting = generate_greeting("Unknown", is_known=False)
                print(f"🤔 Initial greeting (unknown) -> {greeting}")
                await session.say(greeting)
            elif unknown_count > 1:
                # Multiple unknown people
                greeting = generate_group_greeting([], unknown_count)
                print(f"👥 Initial greeting (unknowns) -> {greeting}")
                await session.say(greeting)
            else:
                # No one visible
                print("👋 No one detected - generic greeting")
                await session.say("Hello! I'm your campus assistant. How can I help you today?")
                
        except RuntimeError as e:
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
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="campus-greeting-agent",  # Must match frontend AGENT_NAME
    ))
