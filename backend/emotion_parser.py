"""
Emotion Parser - Extracts emotion tags from LLM responses
Parses [emotion] tags and returns clean text for TTS/frontend.
"""

import re
from typing import Tuple, Optional

# Valid emotions that the OLED can display
VALID_EMOTIONS = {"idle", "looking", "happy", "sad", "angry", "boring", "smile"}
DEFAULT_EMOTION = "idle"

# Regex pattern to match [emotion] at start of response
EMOTION_PATTERN = re.compile(r'^\s*\[(\w+)\]\s*', re.IGNORECASE)


def parse_emotion(text: str) -> Tuple[str, str]:
    """
    Extract emotion tag from LLM response.
    
    Args:
        text: LLM response like "[happy] Great to see you!"
        
    Returns:
        Tuple of (emotion, clean_text)
        - emotion: The detected emotion or "idle" if none found
        - clean_text: Text with emotion tag removed
        
    Examples:
        >>> parse_emotion("[happy] Hello!")
        ('happy', 'Hello!')
        
        >>> parse_emotion("[sad] I couldn't find that.")
        ('sad', "I couldn't find that.")
        
        >>> parse_emotion("Hello!")  # No tag
        ('idle', 'Hello!')
    """
    if not text:
        return DEFAULT_EMOTION, ""
    
    match = EMOTION_PATTERN.match(text)
    
    if match:
        emotion = match.group(1).lower()
        clean_text = text[match.end():].strip()
        
        # Validate emotion
        if emotion in VALID_EMOTIONS:
            return emotion, clean_text
        else:
            # Unknown emotion tag - treat as idle but still remove the tag
            print(f"⚠️ Unknown emotion tag: [{emotion}]")
            return DEFAULT_EMOTION, clean_text
    
    # No emotion tag found
    return DEFAULT_EMOTION, text.strip()


def get_emotion_for_context(context: str) -> str:
    """
    Suggest emotion based on context (for non-LLM triggers like face detection).
    
    Args:
        context: One of "unknown_face", "known_face", "no_face", "error"
        
    Returns:
        Appropriate emotion string
    """
    context_emotions = {
        "unknown_face": "looking",
        "known_face": "happy",
        "no_face": "idle",
        "error": "sad",
        "greeting": "smile",
    }
    return context_emotions.get(context, DEFAULT_EMOTION)


# Test when run directly
if __name__ == "__main__":
    test_cases = [
        "[happy] Great to see you, Makila!",
        "[sad] I'm sorry, I couldn't find that event.",
        "[looking] Hi there! I don't think we've met. What's your name?",
        "[angry] That's frustrating!",
        "[boring] You already asked me that.",
        "[smile] Sure, let me help you with that!",
        "Hello, how can I help?",  # No tag
        "[unknown_emotion] This has invalid tag",
    ]
    
    print("Testing emotion parser:\n")
    for text in test_cases:
        emotion, clean = parse_emotion(text)
        print(f"Input:   {text[:50]}...")
        print(f"Emotion: {emotion}")
        print(f"Clean:   {clean[:50]}...")
        print()
