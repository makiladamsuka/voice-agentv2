"""
Emotion Sync Module

Analyzes text for emotional content and segments it for synchronized
expression on OLED displays.

Uses VADER sentiment analysis with keyword fallback for accurate emotion detection.
"""

import re
from typing import List, Dict, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
_analyzer = SentimentIntensityAnalyzer()

# Keyword mapping for fast emotion detection
EMOTION_KEYWORDS = {
    "happy": [
        "hello", "hi", "hey", "great", "wonderful", "nice", "welcome",
        "glad", "pleased", "excellent", "awesome", "amazing",
        "fantastic", "good to see", "happy", "excited", "yay"
    ],
    "loving": [
        "love", "adore", "care", "sweet", "dear", "affection",
        "miss you", "appreciate", "thank you so much", "means a lot"
    ],
    "sad": [
        "sorry", "apologize", "unfortunately", "can't", "couldn't",
        "unable", "regret", "sad", "disappointed", "don't know",
        "not sure", "afraid", "bad news"
    ],
    "angry": [
        "error", "wrong", "failed", "broken", "frustrated", "annoyed",
        "stop", "no way", "impossible", "angry", "furious", "mad", "upset"
    ],
    "looking": [
        "hmm", "interesting", "curious", "let me see", "checking",
        "looking", "searching", "finding", "think", "thinking"
    ],
    "smile": [
        "sure", "of course", "happy to", "let me help", "here you go",
        "certainly", "absolutely", "no problem", "alright", "okay"
    ],
    "boring": [
        "again", "already", "told you", "repeated", "same thing",
        "just said", "already answered"
    ]
}

# Negative contractions and words to check for negation
NEGATIONS = ["not", "no", "don't", "doesn't", "wasn't", "isn't", "aren't", "ain't", "never"]

# Valid emotions for OLED display
# idle1 = default/no one talking, idle2 = user is talking/listening
VALID_EMOTIONS = ["idle1", "idle2", "happy", "smile", "looking", "sad", "angry", "boring", "loving"]


def is_negated(text: str, keyword: str) -> bool:
    """
    Check if a keyword is negated in the text.
    Example: "not angry" -> True
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Very simple check: is there a negation word within 3 words before the keyword?
    words = text_lower.split()
    if keyword_lower not in words:
        # Check if keyword is part of a word?
        found_idx = -1
        for i, w in enumerate(words):
            if keyword_lower in w:
                found_idx = i
                break
        if found_idx == -1: return False
    else:
        found_idx = words.index(keyword_lower)
        
    start_idx = max(0, found_idx - 3)
    for i in range(start_idx, found_idx):
        if words[i] in NEGATIONS:
            return True
            
    return False


def analyze_emotion_keywords(text: str) -> str:
    """
    Fast keyword-based emotion detection.
    Returns emotion if keywords found, else "idle1".
    """
    text_lower = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                # Check for negation if it's a negative emotion
                if emotion in ["sad", "angry", "boring"]:
                    if is_negated(text_lower, kw):
                        continue # Skip this keyword
                return emotion
    return "idle1"


def analyze_emotion_vader(text: str) -> str:
    """
    VADER sentiment-based emotion detection.
    Maps compound sentiment score to emotions.
    """
    # Strip emojis for VADER to get better text-based score if needed, 
    # but VADER actually handles emojis well.
    scores = _analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Map sentiment to emotion
    if compound >= 0.5:
        return "happy"
    elif compound >= 0.2:
        return "smile"
    elif compound <= -0.6:  # very negative
        return "angry"
    elif compound <= -0.2:  # slightly negative
        return "sad"
    elif compound <= -0.1:
        return "looking"  # Thinking/concerned
    else:
        return "idle1"


def analyze_emotion(text: str) -> str:
    """
    Hybrid emotion analysis: keywords first (fast), then VADER (accurate).
    
    Args:
        text: Text to analyze
        
    Returns:
        Emotion string: one of VALID_EMOTIONS
    """
    # First try keyword detection (fast path)
    keyword_emotion = analyze_emotion_keywords(text)
    if keyword_emotion != "idle1":
        return keyword_emotion
    
    # Fallback to VADER sentiment
    return analyze_emotion_vader(text)


def segment_by_sentence(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Full text to split
        
    Returns:
        List of sentence strings
    """
    # Split by sentence-ending punctuation, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def segment_by_emotion(text: str) -> List[Dict[str, str]]:
    """
    Split text into emotionally coherent segments.
    
    Each segment has:
    - text: The sentence/phrase to speak
    - emotion: The detected emotion for that segment
    
    Args:
        text: Full text to segment
        
    Returns:
        List of dicts with 'text' and 'emotion' keys
    
    Example:
        >>> segment_by_emotion("Hello there! Sorry, I couldn't find that.")
        [
            {"text": "Hello there!", "emotion": "happy"},
            {"text": "Sorry, I couldn't find that.", "emotion": "sad"}
        ]
    """
    sentences = segment_by_sentence(text)
    
    segments = []
    for sentence in sentences:
        emotion = analyze_emotion(sentence)
        segments.append({
            "text": sentence,
            "emotion": emotion
        })
    
    return segments


def merge_adjacent_emotions(segments: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge adjacent segments with the same emotion.
    Reduces the number of emotion transitions for smoother display.
    
    Args:
        segments: List of segment dicts
        
    Returns:
        Merged list of segment dicts
    """
    if not segments:
        return []
    
    merged = [segments[0].copy()]
    
    for segment in segments[1:]:
        if segment["emotion"] == merged[-1]["emotion"]:
            # Same emotion - merge text
            merged[-1]["text"] += " " + segment["text"]
        else:
            # Different emotion - add new segment
            merged.append(segment.copy())
    
    return merged


def get_emotion_for_text(text: str, merge: bool = True) -> List[Dict[str, str]]:
    """
    Main function: Analyze text and return emotion segments.
    
    Args:
        text: Full text to analyze
        merge: If True, merge adjacent segments with same emotion
        
    Returns:
        List of dicts with 'text' and 'emotion' keys
    """
    segments = segment_by_emotion(text)
    
    if merge:
        segments = merge_adjacent_emotions(segments)
    
    return segments


# Quick test
if __name__ == "__main__":
    test_texts = [
        "Hello there! Great to see you!",
        "I'm sorry, I couldn't find that event. Let me try again.",
        "Hi! Welcome to campus! Unfortunately, the library is closed today.",
        "Hmm, that's interesting. Let me check that for you."
    ]
    
    print("Testing Emotion Sync Module\n" + "=" * 40)
    for text in test_texts:
        print(f"\nInput: {text}")
        segments = get_emotion_for_text(text)
        for seg in segments:
            print(f"  [{seg['emotion']}] {seg['text']}")
