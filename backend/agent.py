from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero
import os
import pickle
import json
import asyncio
import re
from pathlib import Path
from image_manager import ImageManager
from image_server import ImageServer
from face_monitor import FaceMonitor
from object_detector import ObjectDetector
from greetings import generate_greeting, generate_group_greeting
from event_database import EventDatabase, build_event_database
from emotion_parser import parse_emotion, get_emotion_for_context
from emotion_sync import get_emotion_for_text, analyze_emotion
import oled_display
from oled_display import EmotionMode

# Import modular tools
from tools.vision import VisionTools
from tools.content import ContentTools
from tools.system import SystemTools

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class EmotionSpeechWrapper:
    """
    Wrapper that provides emotional speech with VADER sentiment analysis.
    Analyzes text segments and syncs OLED emotions with speech.
    """
    
    @staticmethod
    async def speak_with_emotion(session, text: str):
        """
        Speak text with synchronized emotions.
        Analyzes each sentence and shows matching emotion while speaking.
        
        Args:
            session: AgentSession to use for speaking
            text: Full text to speak
        """
        # Get emotionally segmented text
        segments = get_emotion_for_text(text)
        
        print(f"\n🎭 === EMOTION SYNC DEBUG ===")
        print(f"📝 Full text: {text}")
        print(f"📊 Segments: {len(segments)}")
        for i, seg in enumerate(segments):
            print(f"   {i+1}. [{seg['emotion']}] {seg['text']}")
        print(f"🎭 ===========================\n")
        
        for segment in segments:
            emotion = segment["emotion"]
            segment_text = segment["text"]
            
            print(f"🎤 NOW SPEAKING: [{emotion}] {segment_text}")
            
            # Start emotion (looping mode)
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.start_emotion(emotion)
                    print(f"👀 OLED: Started {emotion} emotion")
            except Exception as e:
                print(f"⚠️ OLED error: {e}")
            
            # Speak the segment
            try:
                await session.say(segment_text)
            except Exception as e:
                print(f"⚠️ Speech error: {e}")
            
            # Small pause between segments
            await asyncio.sleep(0.1)
        
        # Return to idle after all speech
        try:
            if oled_display.DISPLAY_RUNNING:
                oled_display.stop_emotion()
                print(f"👀 OLED: Returned to idle")
        except Exception as e:
            print(f"⚠️ OLED error: {e}")


# Convenience function for easier use
async def emotional_say(session, text: str):
    """Convenience function to speak with emotions."""
    await EmotionSpeechWrapper.speak_with_emotion(session, text)


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

ABOUT YOUR EMOTIONS:
Your facial expressions are AUTOMATIC! They sync with what you say.
- Say something happy → happy eyes appear
- Say "sorry" → sad eyes appear
- No need to manage emotions - just speak naturally!

CRITICAL CAPABILITY:
I can ALWAYS see who is in front of me. Their name appears in system messages.

AUTO-ENROLLMENT:
When someone introduces themselves (says "I'm [Name]" or "My name is [Name]"):
1. IMMEDIATELY call enroll_new_face(their_name) 
2. Then respond warmly: "Nice to meet you, [Name]! I'll remember you now."

NAME USAGE:
- Use names occasionally and naturally, not every response
- For UNKNOWN: "Hi there! I don't think we've met. What's your name?"
- For KNOWN: "Hey [Name]! Good to see you!"

CONVERSATION STYLE:
- Short, clear sentences
- Friendly, warm tone
- Be helpful and responsive

AVAILABLE TOOLS:
- identify_color, identify_object, count_people_in_room, describe_environment
- show_event_poster, show_location_map, ask_about_events
- enroll_new_face(name)
- get_cpu_temperature, get_system_info, get_cpu_usage, get_memory_usage"""
        )
    
    def get_object_detector(self):
        """Lazy-load ObjectDetector"""
        if self._object_detector is None:
            print("🔍 Loading YOLO model...")
            self._object_detector = ObjectDetector()
        return self._object_detector
    
    async def tts_node(self, text_stream, model_settings):
        """
        Override tts_node to intercept ALL text going to TTS.
        Applies VADER sentiment analysis for real-time emotion sync.
        
        This catches both LLM responses AND session.say() calls.
        """
        accumulated_text = ""
        current_emotion = "idle"
        
        async def emotion_aware_text_stream():
            nonlocal accumulated_text, current_emotion
            
            async for text_chunk in text_stream:
                accumulated_text += text_chunk
                
                # Analyze emotion every chunk
                emotion = analyze_emotion(accumulated_text)
                if emotion != current_emotion:
                    current_emotion = emotion
                    print(f"🎭 TTS emotion: [{emotion}] | Text: ...{accumulated_text[-40:]}")
                    try:
                        if oled_display.DISPLAY_RUNNING:
                            oled_display.start_emotion(emotion)
                    except Exception as e:
                        print(f"⚠️ OLED error: {e}")
                
                yield text_chunk
            
            # Final debug output
            print(f"\n📝 TTS Complete: {accumulated_text[:80]}{'...' if len(accumulated_text) > 80 else ''}")
            print(f"🎭 Final emotion: [{current_emotion}]\n")
        
        # Call parent's tts_node with our emotion-aware stream
        async for audio_frame in super().tts_node(emotion_aware_text_stream(), model_settings):
            yield audio_frame

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
    async def set_emotion(self, emotion: str, context: RunContext) -> str:
        """
        Set the robot's emotional expression on the OLED eyes.
        MUST be called before every response to show the appropriate emotion.
        
        Args:
            emotion: One of: idle, happy, smile, looking, sad, angry, boring
        """
        print(f"😊 [TOOL] set_emotion called: {emotion}")
        
        valid_emotions = ["idle", "happy", "smile", "looking", "sad", "angry", "boring"]
        emotion_clean = emotion.lower().strip()
        
        if emotion_clean not in valid_emotions:
            return f"Invalid emotion. Use one of: {', '.join(valid_emotions)}"
        
        # Trigger OLED emotion display
        try:
            if oled_display.DISPLAY_RUNNING:
                oled_display.display_emotion(emotion_clean)
                return f"Emotion set to: {emotion_clean}"
            else:
                return f"Emotion received: {emotion_clean} (display not running)"
        except Exception as e:
            print(f"⚠️ OLED error: {e}")
            return f"Emotion received: {emotion_clean} (display error)"

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
_is_ready = False  # Flag to track if heavy components are loaded

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

def _init_lightweight():
    """Lightweight init - only start fast services for immediate connection"""
    global _global_image_server
    
    # Start image server for posters/maps (fast)
    if _global_image_server is None:
        assets_dir = Path(__file__).parent / "assets"
        _global_image_server = ImageServer(assets_dir, port=8080)
        _global_image_server.start()
        print("✅ Image server started")
    
    # Start OLED display (I2C must run on main thread, but it's fast)
    try:
        oled_display.setup_and_start_display()
        print("✅ OLED display started")
    except Exception as e:
        print(f"⚠️ Could not start OLED display: {e}")

async def _init_heavy_async(agent):
    """Background initialization of heavy ML components"""
    global _global_face_monitor, _global_event_db, _is_ready
    
    print("🔄 Starting background initialization of ML components...")
    
    # Run heavy init in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    
    # 2. Build event database from posters (OCR) - can be slow
    if _global_event_db is None:
        try:
            assets_dir = Path(__file__).parent / "assets"
            _global_event_db = await loop.run_in_executor(
                None, build_event_database, assets_dir
            )
            print("✅ Event database ready")
        except Exception as e:
            print(f"⚠️ Could not build event database: {e}")
            _global_event_db = None
    
    # 3. Start camera and face recognition (slowest)
    if _global_face_monitor is None:
        print("🎥 Starting camera...")
        known_faces = await loop.run_in_executor(None, _load_known_faces)
        _global_face_monitor = FaceMonitor(known_faces)
        await loop.run_in_executor(None, _global_face_monitor.start)
        await asyncio.sleep(1)  # Give camera time to warm up
        print("✅ Camera ready!")
    
    # Update agent with initialized components
    agent.face_monitor = _global_face_monitor
    agent.known_faces = _global_face_monitor.known_faces
    agent.event_db = _global_event_db
    
    _is_ready = True
    print("🎉 All components initialized!")



async def entrypoint(ctx: agents.JobContext):
    global _global_face_monitor, _global_image_server, _global_event_db, _is_ready
    
    # LIGHTWEIGHT init - only start fast services
    _init_lightweight()
    
    # Create session immediately (no waiting for ML models)
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
    
    # Create agent without heavy components (will be set later)
    agent = CampusGreetingAgent(_global_image_server, None)  # event_db set later
    agent.room = ctx.room
    agent.face_monitor = None  # Will be set after background init
    
    # Context Injection: LLM always knows who's in front (handles None face_monitor)
    async def inject_person_context(assistant: AgentSession, chat_ctx):
        # Check if face monitor is ready
        if not _is_ready or agent.face_monitor is None:
            from livekit.agents.llm import ChatMessage, ChatRole
            context_msg = ChatMessage(
                role=ChatRole.SYSTEM,
                content="System is still initializing. Face recognition not yet available."
            )
            chat_ctx.messages.insert(0, context_msg)
            return chat_ctx
            
        # Use thread-safe FRESH people getter (most recent detection)
        fresh = agent.face_monitor.get_fresh_people()
        
        # Categorize
        known = [p for p in fresh if p != "Unknown"]
        unknown_count = sum(1 for p in fresh if p == "Unknown")
        
        from livekit.agents.llm import ChatMessage, ChatRole
        
        # Debug: log what we're injecting
        print(f"🎯 Context injection - Fresh: {fresh}, Known: {known}")
        
        if known:
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
    
    # Proactive Greeting Task: Watch for new people (only runs after init completes)
    async def monitor_and_greet():
        """Background task that greets people when they appear"""
        # Wait for initialization to complete
        while not _is_ready:
            await asyncio.sleep(1)
        
        await asyncio.sleep(2)  # Additional delay after init
        print("🔄 Background greeting monitor started (multi-person mode)")
        
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            try:
                if agent.face_monitor is None:
                    await asyncio.sleep(2)
                    continue
                    
                # Get new arrivals (FaceMonitor handles 5-second cooldown)
                arrivals = agent.face_monitor.get_new_arrivals()
                
                if arrivals:
                    print(f"👋 New arrivals: {arrivals}")
                    
                    # Categorize arrivals
                    known_people = [p for p in arrivals if p != "Unknown"]
                    unknown_count = arrivals.count("Unknown")
                    
                    # Mark all as greeted
                    for p in arrivals:
                        agent.face_monitor.mark_greeted(p)
                    
                    try:
                        if len(known_people) > 0 and unknown_count == 0:
                            if len(known_people) == 1:
                                name = known_people[0]
                                greeting = generate_greeting(name, is_known=True)
                                print(f"✅ Greeting known person: {name} -> {greeting}")
                                await emotional_say(session, greeting)
                            else:
                                greeting = generate_group_greeting(known_people, 0)
                                print(f"✅ Greeting multiple known people -> {greeting}")
                                await emotional_say(session, greeting)
                        
                        elif known_people and unknown_count > 0:
                            greeting = generate_group_greeting(known_people, unknown_count)
                            print(f"🤔 Greeting mix -> {greeting}")
                            await emotional_say(session, greeting)
                        
                        elif unknown_count == 1:
                            greeting = generate_greeting("Unknown", is_known=False)
                            print(f"🤔 Greeting unknown person -> {greeting}")
                            await emotional_say(session, greeting)
                        
                        else:
                            greeting = generate_group_greeting([], unknown_count)
                            print(f"👥 Greeting {unknown_count} unknown people -> {greeting}")
                            await emotional_say(session, greeting)
                            
                    except RuntimeError:
                        print("⚠️ Session closing, stopping greetings")
                        break
                        
            except Exception as e:
                print(f"⚠️ Greeting error: {e}")
                import traceback
                traceback.print_exc()
                
            await asyncio.sleep(2)
    
    try:
        # START SESSION IMMEDIATELY (before heavy init)
        await session.start(room=ctx.room, agent=agent)
        
        # Send loading message right away
        print("💬 Sending loading message...")
        await session.say("Give me a moment to wake up. I'm loading my systems...")
        
        # Start background initialization
        print("🔄 Starting background initialization...")
        await _init_heavy_async(agent)
        
        # Announce readiness
        print("🎉 Initialization complete - announcing readiness")
        await session.say("I'm ready! How can I help you today?")
        
        # NOW start background greeting monitor
        asyncio.create_task(monitor_and_greet())
        
        # Keep session alive
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
        
    finally:
        # CLEANUP
        print("🔌 Session ending - releasing camera...")
        if agent.face_monitor:
            agent.face_monitor.stop()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="campus-greeting-agent",  # Must match frontend AGENT_NAME
        initialize_process_timeout=120,  # 2 minutes for slow devices like Raspberry Pi
    ))
