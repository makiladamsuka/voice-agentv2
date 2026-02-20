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
import signal
import numpy as np
from pathlib import Path
from image_manager import ImageManager
from image_server import ImageServer
from face_monitor import FaceMonitor
# from object_detector import ObjectDetector
from greetings import generate_greeting, generate_group_greeting
from event_database import EventDatabase, build_event_database
# from emotion_parser import parse_emotion, get_emotion_for_context
# from emotion_sync import get_emotion_for_text, analyze_emotion
# OLED/TFT display — now uses procedural rendering internally
import oled_display  # start_emotion(), stop_emotion(), etc.

# Import modular tools
from tools.vision import VisionTools
from tools.content import ContentTools
from tools.system import SystemTools

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


# class EmotionSpeechWrapper:
#     """
#     Wrapper that provides emotional speech with VADER sentiment analysis.
#     Analyzes text segments and syncs OLED emotions with speech.
#     """
#     
#     @staticmethod
#     async def speak_with_emotion(session, text: str):
#         """
#         Speak text with synchronized emotions.
#         Analyzes each sentence and shows matching emotion while speaking.
#         
#         Args:
#             session: AgentSession to use for speaking
#             text: Full text to speak
#         """
#         # Get emotionally segmented text
#         # segments = get_emotion_for_text(text)
#         
#         # print(f"\n🎭 === EMOTION SYNC DEBUG ===")
#         # print(f"📝 Full text: {text}")
#         # print(f"📊 Segments: {len(segments)}")
#         # for i, seg in enumerate(segments):
#         #     print(f"   {i+1}. [{seg['emotion']}] {seg['text']}")
#         # print(f"🎭 ===========================\n")
#         
#         # for segment in segments:
#         #     emotion = segment["emotion"]
#         #     segment_text = segment["text"]
#             
#         #     print(f"🎤 NOW SPEAKING: [{emotion}] {segment_text}")
#             
#         #     # Start emotion (looping mode) - DISABLED here, handled by tts_node for better sync
#         #     # try:
#         #     #     if oled_display.DISPLAY_RUNNING:
#         #     #         oled_display.start_emotion(emotion)
#         #     #         print(f"👀 OLED: Started {emotion} emotion")
#         #     # except Exception as e:
#         #     #     print(f"⚠️ OLED error: {e}")
#             
#         #     # Speak the segment
#         #     try:
#         #         await session.say(segment_text)
#         #     except Exception as e:
#         #         print(f"⚠️ Speech error: {e}")
#             
#         #     # Small pause between segments
#         #     await asyncio.sleep(0.1)
#         
#         # Return to idle after all speech - DISABLED here, handled by tts_node and session events
#         # try:
#         #     if oled_display.DISPLAY_RUNNING:
#         #         oled_display.stop_emotion()
#         #         print(f"👀 OLED: Returned to idle")
#         # except Exception as e:
#         #     print(f"⚠️ OLED error: {e}")
#         pass





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
            object_detector_factory=None # self.get_object_detector (Removed)
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
        
        from prompt import SYSTEM_INSTRUCTIONS
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS
        )
    

    


    # --- Delegate to Tool Modules ---

    # =========================================================================
    # 🛠️ GROUP 1: VISION & PERCEPTION TOOLS (User Requested)
    # =========================================================================

    @function_tool
    async def recognize_face(self, mode: str = "identify", context: RunContext = None) -> str:
        """Identifies who is currently in front of the webcam (Manual Trigger).
        
        Args:
            mode: Recognition mode: 'identify' (default) or 'detailed'
        """
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
    async def identify_color(self, detail_level: str = "basic", context: RunContext = None) -> str:
        """Identifies the dominant color in the camera view.
        
        Args:
            detail_level: Detail level: 'basic' (default) or 'detailed'
        """
        print("🎨 [TOOL] identify_color called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.identify_color(context)
    
    @function_tool
    async def describe_environment(self, detail_level: str = "full", context: RunContext = None) -> str:
        """Describes the current environment - people count and visible objects.
        
        Args:
            detail_level: Detail level: 'full' (default), 'people', or 'objects'
        """
        print("👁️ [TOOL] describe_environment called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.describe_environment(context)
    
    @function_tool
    async def identify_object(self, object_name: str, context: RunContext) -> str:
        """Finds a specific object and describes it (Feature Removed)."""
        print(f"🔍 [TOOL] identify_object called for: {object_name} (FEATURE REMOVED)")
        return "I'm sorry, my object detection system has been disabled."
        # self.vision_tools.face_monitor = self.face_monitor
        # return await self.vision_tools.identify_object(object_name, context)
    
    @function_tool
    async def count_people_in_room(self, include_details: str = "count", context: RunContext = None) -> str:
        """Counts how many people are visible in the camera view.
        
        Args:
            include_details: Output type: 'count' (default) or 'detailed'
        """
        print("👥 [TOOL] count_people_in_room called")
        self.vision_tools.face_monitor = self.face_monitor
        return await self.vision_tools.count_people_in_room(context)
    
    # =========================================================================
    # 🛠️ GROUP 2: CAMPUS INFORMATION TOOLS (User Requested)
    # =========================================================================

    @function_tool
    async def list_available_events(self, filter_type: str = "all", context: RunContext = None) -> str:
        """Lists all available events on campus.
        
        Args:
            filter_type: Filter type: 'all' (default), 'today', or 'upcoming'
        """
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
    async def get_cpu_temperature(self, unit: str = "celsius", context: RunContext = None) -> str:
        """Gets the CPU temperature of the Raspberry Pi.
        
        Args:
            unit: Temperature unit: 'celsius' (default) or 'fahrenheit'
        """
        print("🌡️ [TOOL] get_cpu_temperature called")
        return await self.system_tools.get_cpu_temperature(context)
    
    @function_tool
    async def get_system_info(self, detail_level: str = "full", context: RunContext = None) -> str:
        """Gets comprehensive system information including CPU temperature, usage, memory, disk, and uptime.
        
        Args:
            detail_level: Detail level: 'full' (default), 'brief', or 'performance'
        """
        print("💻 [TOOL] get_system_info called")
        return await self.system_tools.get_system_info(context)
    
    @function_tool
    async def get_cpu_usage(self, format_type: str = "percentage", context: RunContext = None) -> str:
        """Gets the CPU usage percentage.
        
        Args:
            format_type: Output format: 'percentage' (default) or 'detailed'
        """
        print("⚡ [TOOL] get_cpu_usage called")
        return await self.system_tools.get_cpu_usage(context)
    
    @function_tool
    async def get_memory_usage(self, format_type: str = "standard", context: RunContext = None) -> str:
        """Gets the memory (RAM) usage information.
        
        Args:
            format_type: Output format: 'standard' (default), 'percentage', or 'detailed'
        """
        print("🧠 [TOOL] get_memory_usage called")
        return await self.system_tools.get_memory_usage(context)
    
    @function_tool
    async def ask_about_events(self, question: str, context: RunContext) -> str:
        """Answers questions about campus events using the vector database."""
        print(f"📅 [TOOL] ask_about_events called: {question}")
        
        if not self.event_db:
            return "I'm sorry, the event database is not available right now."
            
        # Query the database
        results = self.event_db.query_events(question)
        
        if not results:
            return "I couldn't find any specific events matching your question."
            
        # Format context for LLM
        context_str = "Found these relevant events:\n"
        for i, event in enumerate(results):
            context_str += f"{i+1}. {event.get('title', 'Event')} on {event.get('date', 'Unknown Date')}: {event.get('description', '')}\n"
            
        print(f"   found {len(results)} events")
        return context_str

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
    if _global_event_db is None and build_event_database:
        try:
            print("📅 Building event database...")
            assets_dir = Path(__file__).parent / "assets"
            # Run in executor to avoid blocking
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


def _handle_signal(sig, frame):
    """Handle termination signals for graceful shutdown"""
    print(f"\n🛑 Received signal {sig}, shutting down...")
    try:
        if oled_display.DISPLAY_RUNNING:
            print("👋 OLED: Shutdown requested via signal")
            oled_display.stop_display()
    except Exception as e:
        print(f"⚠️ Shutdown signal error: {e}")
    
    # Allow natural exit
    # sys.exit(0) is not needed as LiveKit runner handles it, but we ensured OLED stop


async def entrypoint(ctx: agents.JobContext):
    global _global_face_monitor, _global_image_server, _global_event_db, _is_ready
    
    # Register signal handlers for Ctrl+C and termination
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_wrapper()))
        except (NotImplementedError, ValueError):
            # Fallback for systems where add_signal_handler isn't available
            signal.signal(sig, _handle_signal)
            
    async def shutdown_wrapper():
        """Clean shutdown transition"""
        print("🧼 Performing final cleanup...")
        try:
            if oled_display.DISPLAY_RUNNING:
                oled_display.stop_display()
        except:
            pass
        # Give a small moment for I2C to settle
        await asyncio.sleep(0.5)
        # Note: We don't exit here, we let the runner clean up the rest

    
    # LIGHTWEIGHT init - only start fast services
    _init_lightweight()
    
    # Create session immediately (no waiting for ML models)
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-luna-en"),
        vad=silero.VAD.load(),
        llm=openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # User request: "openrouter/free" routes to available free models
            model="openrouter/free"
        ),
    )
    
    # Create agent without heavy components (will be set later)
    agent = CampusGreetingAgent(_global_image_server, None)  # event_db set later
    agent.room = ctx.room
    agent.face_monitor = None  # Will be set after background init
    agent.is_speaking = False  # Track speaking state for emotion logic
    
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
    # (Unused on_user_speech removed)
    
    # Proactive Greeting Task: Watch for new people (only runs after init completes)
    async def monitor_and_greet():
        """Background task that greets people and TRACKS FACES"""
        # Wait for initialization to complete
        while not _is_ready:
            await asyncio.sleep(1)
        
        await asyncio.sleep(2)  # Additional delay after init
        
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            try:
                if agent.face_monitor is None:
                    await asyncio.sleep(2)
                    continue
                    
                # 1. Face Tracking (High frequency)
                if agent.face_monitor:
                    face_center = agent.face_monitor.get_face_center()
                    
                    if oled_display.DISPLAY_RUNNING:
                        if face_center:
                            oled_display.update_face_target(face_center[0], face_center[1])
                            # Show "Happy" if seeing someone (and not busy doing something else)
                            # Only override "idle" states. Don't override "thinking", "listening" (idle2), or "talking" (happy)
                            if oled_display.current_emotion in ["idle", "idle1"]:
                                oled_display.start_emotion("happy")
                        else:
                            oled_display.update_face_target(0.0, 0.0)
                            # If lost face and was "happy" (and NOT speaking), go back to idle
                            if oled_display.current_emotion == "happy" and not agent.is_speaking:
                                oled_display.start_emotion("idle")

                # 2. Greeting Logic (Lower frequency)
                # Check for new arrivals
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
                                await session.say(greeting)
                            else:
                                greeting = generate_group_greeting(known_people, 0)
                                print(f"✅ Greeting multiple known people -> {greeting}")
                                await session.say(greeting)
                        
                        elif known_people and unknown_count > 0:
                            greeting = generate_group_greeting(known_people, unknown_count)
                            print(f"🤔 Greeting mix -> {greeting}")
                            await session.say(greeting)
                        
                        elif unknown_count == 1:
                            greeting = generate_greeting("Unknown", is_known=False)
                            print(f"🤔 Greeting unknown person -> {greeting}")
                            await session.say(greeting)
                        
                        else:
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
                
            await asyncio.sleep(0.1)
    
    try:
        # --- Register event listeners BEFORE session.start() ---
        
        # Register user state callback for idle2 (listening) emotion
        @session.on("user_started_speaking")
        def on_user_started_speaking(*args):
            """Show idle2 when user starts speaking"""
            print("👂 User speaking - showing idle2")
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.start_emotion("idle2")
            except Exception as e:
                print(f"⚠️ User speech start error: {e}")

        @session.on("user_stopped_speaking")
        def on_user_stopped_speaking(*args):
            """Return to idle1 when user stops speaking"""
            print("👀 User stopped - returning to idle1")
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.stop_emotion()
            except Exception as e:
                print(f"⚠️ User speech stop error: {e}")

        # Agent THOUGHT start (When LLM starts generating)
        @session.on("agent_speech_committed")
        def on_agent_speech_committed(*args):
            print("🤔 Agent thinking - EMOTION: thinking")
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.start_emotion("thinking")
            except Exception as e:
                print(f"⚠️ OLED error: {e}")

        # Agent SPEECH start
        @session.on("agent_speech_started")
        def on_agent_speech_started(*args):
            print("🗣️ Agent speaking - EMOTION: happy")
            agent.is_speaking = True
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.start_emotion("happy")  # Talking state
            except Exception as e:
                print(f"⚠️ OLED error: {e}")

        # Precise emotion finish listeners
        @session.on("agent_speech_stopped")
        @session.on("agent_speech_finished")
        def on_agent_speech_finished(*args):
            print(f"🔊 Agent finished speaking - EMOTION: idle")
            agent.is_speaking = False
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.stop_emotion()  # Return to idle
            except Exception as e:
                print(f"⚠️ OLED error: {e}")

        @session.on("agent_speech_interrupted")
        def on_agent_speech_interrupted(*args):
            print("🔊 Agent interrupted - EMOTION: idle")
            agent.is_speaking = False
            try:
                if oled_display.DISPLAY_RUNNING:
                    oled_display.stop_emotion()
            except Exception as e:
                print(f"⚠️ OLED error: {e}")

        # START SESSION
        print("🚀 Starting LiveKit session...")
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
        
        # Start audio amplitude monitor for speech-reactive eyes
        async def audio_amplitude_monitor():
            """Reads agent audio output and drives eye reactivity in real time."""
            print("🎵 Audio amplitude monitor started")
            smooth_amp = 0.0
            
            try:
                # Get the agent's audio output track via session
                # We subscribe to audio frames published by the local participant
                audio_stream = None
                
                # Wait for audio track to appear
                for _ in range(20):
                    for pub in ctx.room.local_participant.track_publications.values():
                        if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                            audio_stream = rtc.AudioStream(pub.track)
                            break
                    if audio_stream:
                        break
                    await asyncio.sleep(0.5)
                
                if not audio_stream:
                    print("⚠️ Audio track not found for amplitude monitor.")
                    return
                
                print("✅ Audio stream found - monitoring amplitude")
                
                async for event in audio_stream:
                    frame = event.frame
                    # Convert raw PCM to numpy for RMS computation
                    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                    if len(samples) == 0:
                        continue
                    rms = np.sqrt(np.mean(samples ** 2))
                    # Normalize roughly: 16-bit PCM max = 32768
                    norm = min(rms / 8000.0, 1.0)
                    # Smooth: fast attack, slow decay
                    if norm > smooth_amp:
                        smooth_amp = smooth_amp * 0.3 + norm * 0.7  # Fast attack
                    else:
                        smooth_amp = smooth_amp * 0.85 + norm * 0.15  # Slow decay
                    
                    if oled_display.DISPLAY_RUNNING:
                        oled_display.set_speech_amplitude(smooth_amp)
                    
            except Exception as e:
                print(f"⚠️ Audio amplitude monitor error: {e}")
        
        asyncio.create_task(audio_amplitude_monitor())
        
        # Keep session alive
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
        
    finally:
        # CLEANUP - Show sad emotion on disconnect, then shutdown
        print("🔌 Session ending...")
        
        # Show sad emotion when disconnecting
        try:
            if oled_display.DISPLAY_RUNNING:
                print("😢 Showing sad emotion for disconnect...")
                oled_display.display_emotion("sad")
                await asyncio.sleep(2)  # Let it play for 2 seconds
                oled_display.stop_display()
                print("👀 OLED display stopped safely")
        except Exception as e:
            print(f"⚠️ OLED shutdown error: {e}")
        
        # Release camera
        if agent.face_monitor:
            agent.face_monitor.stop()
            print("📷 Camera released")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="campus-greeting-agent",  # Must match frontend AGENT_NAME
        initialize_process_timeout=120,  # 2 minutes for slow devices like Raspberry Pi
    ))
