# Synchronized Emotional Expression System

A comprehensive guide to making your voice robot feel alive with real-time emotional expressions.

## Overview

### The Problem
Current implementation shows one emotion for the entire response. This feels robotic because:
- Emotion starts before speech
- Emotion ends before speech finishes
- No variation within a response
- Abrupt transitions

### The Goal
**Synchronized emotions** - the robot's expression changes precisely when speaking emotional content:
```
"Hello there!"           → happy eyes
"I'm sorry, I couldn't"  → transition to sad
"find that event."       → sad eyes
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Response                             │
│  "Hello! I'm sorry, I couldn't find that event."               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Emotion Segmenter                            │
│  Split into emotional segments:                                 │
│  [{ text: "Hello!", emotion: "happy" },                        │
│   { text: "I'm sorry, I couldn't find that event.",            │
│     emotion: "sad" }]                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Sequential Segment Player                          │
│  For each segment:                                              │
│    1. Set OLED emotion                                         │
│    2. TTS speak text                                           │
│    3. Wait for completion                                      │
│    4. Next segment                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────┐    ┌──────────────────────┐
│      OLED Display    │    │     TTS (Deepgram)   │
│   Shows emotion      │    │   Speaks text        │
└──────────────────────┘    └──────────────────────┘
```

---

## Sentiment Analysis Options

### Option 1: Keyword-Based (Recommended for Pi)
**Pros:** Fast, no dependencies, works offline
**Cons:** Limited accuracy

```python
EMOTION_KEYWORDS = {
    "happy": ["hello", "hi", "great", "wonderful", "nice", "welcome", 
              "glad", "pleased", "excellent", "awesome", "love"],
    "sad": ["sorry", "apologize", "unfortunately", "can't", "couldn't", 
            "unable", "regret", "sad", "disappointed"],
    "angry": ["error", "wrong", "failed", "broken", "frustrated"],
    "looking": ["who", "what", "where", "hmm", "interesting", "curious"],
    "smile": ["sure", "of course", "happy to", "let me", "here"],
    "boring": ["again", "already", "told you", "repeated"]
}

def analyze_emotion(text: str) -> str:
    text_lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return emotion
    return "idle"
```

### Option 2: VADER Sentiment (Better accuracy)
**Library:** `vaderSentiment`
**Pros:** Trained on social media, handles nuances
**Cons:** Extra dependency (~2MB)

```bash
pip install vaderSentiment
```

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_emotion_vader(text: str) -> str:
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.5:
        return "happy"
    elif compound >= 0.1:
        return "smile"
    elif compound <= -0.5:
        return "sad"
    elif compound <= -0.1:
        return "looking"  # neutral/curious
    else:
        return "idle"
```

### Option 3: Hybrid (Recommended)
Combine keyword detection for speed + sentiment for nuance:

```python
def analyze_emotion_hybrid(text: str) -> str:
    # First check keywords (fast path)
    keyword_emotion = analyze_emotion(text)
    if keyword_emotion != "idle":
        return keyword_emotion
    
    # Fallback to sentiment analysis
    return analyze_emotion_vader(text)
```

---

## Text Segmentation

### By Sentences (Recommended)
Split at sentence boundaries, analyze each:

```python
import re

def segment_by_emotion(text: str) -> list:
    """Split text into emotionally coherent segments."""
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    segments = []
    for sentence in sentences:
        if sentence.strip():
            emotion = analyze_emotion(sentence)
            segments.append({
                "text": sentence.strip(),
                "emotion": emotion
            })
    
    return segments
```

---

## OLED Display Modes

### Current: One-Shot
Plays animation once, returns to idle. Not suitable for speech.

### Needed: Looping Mode
Keep emotion playing while speech is ongoing:

```python
# Modified display_emotion function:
def display_emotion(emotion: str, loop: bool = True):
    global current_mode, current_emotion
    current_emotion = emotion
    if loop:
        # Keep looping until changed
        video_queue.append({"emotion": emotion, "loop": True})
    else:
        video_queue.append({"emotion": emotion, "loop": False})
```

---

## Integration with Speech

### Sequential Segment Player

```python
async def speak_with_emotion(session, text: str):
    """Speak text with synchronized emotions."""
    segments = segment_by_emotion(text)
    
    for segment in segments:
        # 1. Set emotion (starts looping)
        oled_display.display_emotion(segment["emotion"])
        
        # 2. Speak the text segment
        await session.say(segment["text"])
        
        # 3. Small pause between segments
        await asyncio.sleep(0.1)
    
    # 4. Return to idle after all speech
    oled_display.display_emotion("idle")
```

---

## Making Idle Feel Alive

### Random Blinking
Add occasional eye "blinks" during idle:

```python
import random
import time

BLINK_INTERVAL_MIN = 3  # seconds
BLINK_INTERVAL_MAX = 7  # seconds

def _play_idle_with_life():
    last_blink = time.time()
    next_blink = random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
    
    while current_emotion == "idle":
        # Normal idle frame
        _display_frame(idle_frames[current_frame])
        
        # Check for blink
        if time.time() - last_blink > next_blink:
            _play_blink_animation()
            last_blink = time.time()
            next_blink = random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
```

---

## Dependencies Summary

| Component | Library | Purpose | Install |
|-----------|---------|---------|---------|
| Sentiment | vaderSentiment | Analyze text emotion | `pip install vaderSentiment` |
| Fallback | (built-in) | Keyword matching | None |
| OLED | luma.oled | Display control | `pip install luma.oled` |
| TTS | deepgram | Speech synthesis | Via LiveKit |

---

## File Structure

```
backend/
├── agent.py              # Main agent, LLM integration
├── emotion_sync.py       # NEW: Segmentation + analysis
├── oled_display.py       # MODIFIED: Looping mode, life
└── emotion_parser.py     # Existing keyword parser
```

---

## Next Steps

1. **Implement emotion_sync.py** with segment_by_emotion()
2. **Modify oled_display.py** for looping mode
3. **Add idle life** with blinking/movements
4. **Wrap session.say** with emotional_say()
5. **Test and tune** emotion detection

Shall I implement these changes?
