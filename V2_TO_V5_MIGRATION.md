# V2 to V5 Migration Log

This file tracks all modifications made in `voice-agentv2` that need to be migrated to `voice-agentv5` (the real robot).
For each set of changes, a new entry will be added with the file paths and a description of what was changed.

---

## 1. Recent Feature & UI Updates (Up to `9740f21`)

### Frontend Changes
- **SiriGlow Update:** Activated SiriGlow on the mic section exclusively when the agent is in the "thinking" state (`1cb5c0a`).
- **Chat Transcript Fixes:** 
  - Hid transcript from the chat area while ensuring it shows in the mic area (`0c38c7c`).
  - Removed duplicate transcript merging that caused crazy scrolling behavior (`fa39112`).

### Backend Changes
- **Agent Context & Capabilities:**
  - Injected event summary directly into the LLM context (`b371f66`).
  - Added `get_current_time` tool and fixed OpenRouter model id (`e3e49d4`).
  - Switched to `gpt-4o-mini` to reduce turn-taking latency (`9b28548`).
  - Disabled verbal proactive greetings and reverted STT language detection (`9740f21`).

## 2. LLM Context Injection Fix

### Backend Changes (`backend/voice_agent.py`)
- **Fixed System Prompt Duplication & Precedence:**
  - Cleaned up old dynamic system messages (events and person context) on each turn to prevent the context array from growing infinitely with duplicates.
  - Re-ordered the injection so that the dynamic person context and event summary are inserted *after* the main system prompt (index 1 and 2). This ensures LLMs (like `gpt-4o-mini`) prioritize the events and current person data over the lengthy background persona instructions.
