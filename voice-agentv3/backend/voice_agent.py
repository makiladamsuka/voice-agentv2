from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, RunContext
from livekit.plugins import openai, deepgram, silero
import os
import asyncio
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

class SimpleVoiceAgent(Agent):
    def __init__(self):
        from prompt import SYSTEM_INSTRUCTIONS
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS
        )

async def entrypoint(ctx: agents.JobContext):
    # Create session immediately 
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        tts=deepgram.TTS(model="aura-luna-en"),
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3, 
            prefix_padding_duration=0.2
        ),
        llm=openai.LLM(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openrouter/auto"
        ),
    )
    
    # Create agent
    agent = SimpleVoiceAgent()
    
    # START SESSION
    print("🚀 Starting LiveKit session...")
    await session.start(room=ctx.room, agent=agent)
    
    await session.say("Hello! I am your simplified voice assistant. How can I help you today?")
    
    # Keep session alive
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
