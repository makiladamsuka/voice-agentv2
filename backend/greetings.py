"""
Natural Greeting Generator
Generates time-aware and context-aware greetings to feel more human.
"""

import random
from datetime import datetime
from typing import Dict, Optional

# Track when we last saw each person (persisted in memory for session)
_last_seen: Dict[str, datetime] = {}

def get_time_of_day() -> str:
    """Get period of day based on current time."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

def get_context(name: str) -> str:
    """
    Determine the context based on when we last saw this person.
    Returns: 'first_time', 'same_session', 'returning'
    """
    now = datetime.now()
    
    if name not in _last_seen:
        return "first_time"
    
    last = _last_seen[name]
    diff = (now - last).total_seconds()
    
    if diff < 300:  # Less than 5 minutes
        return "same_session"
    elif diff < 3600:  # Less than 1 hour
        return "short_break"
    else:
        return "returning"

def mark_seen(name: str):
    """Record that we've seen this person now."""
    _last_seen[name] = datetime.now()

def generate_greeting(name: str, is_known: bool = True) -> str:
    """
    Generate a natural greeting based on time and context.
    
    Args:
        name: Person's name (or None for unknown)
        is_known: Whether this is a known person
    """
    time_of_day = get_time_of_day()
    
    if not is_known or name == "Unknown":
        # Unknown person greetings
        unknown_greetings = [
            "Hi there! I don't think we've met. What's your name?",
            "Hey! I don't recognize you yet. Mind introducing yourself?",
            "Hello! I'm not sure we've been introduced. What should I call you?",
        ]
        if time_of_day == "morning":
            unknown_greetings.append("Good morning! I don't think I know you yet. What's your name?")
        elif time_of_day == "afternoon":
            unknown_greetings.append("Good afternoon! We haven't met, have we? What's your name?")
        elif time_of_day == "evening":
            unknown_greetings.append("Good evening! I don't believe we've met. What's your name?")
        
        return random.choice(unknown_greetings)
    
    # Known person - check context
    context = get_context(name)
    mark_seen(name)
    
    if context == "same_session":
        # They just stepped away briefly
        back_greetings = [
            f"Oh, {name}! Back again?",
            f"Hey {name}, you're back!",
            f"Welcome back, {name}!",
            f"{name}! Miss me?",
        ]
        return random.choice(back_greetings)
    
    elif context == "short_break":
        # They were gone for a bit
        short_break_greetings = [
            f"Hey {name}! How's it going?",
            f"Oh, {name}! Good to see you again.",
            f"{name}! What's up?",
        ]
        return random.choice(short_break_greetings)
    
    elif context == "returning":
        # They've been gone for a while
        if time_of_day == "morning":
            returning_greetings = [
                f"Good morning, {name}!",
                f"Morning, {name}! How are you?",
                f"Hey {name}! Starting your day?",
            ]
        elif time_of_day == "afternoon":
            returning_greetings = [
                f"Good afternoon, {name}!",
                f"Hey {name}! How's your day going?",
                f"{name}! Nice to see you this afternoon.",
            ]
        elif time_of_day == "evening":
            returning_greetings = [
                f"Good evening, {name}!",
                f"Hey {name}! Still around this evening?",
                f"{name}! How was your day?",
            ]
        else:  # night
            returning_greetings = [
                f"Hey {name}! Working late?",
                f"{name}! Burning the midnight oil?",
                f"Oh, {name}! You're here late.",
            ]
        return random.choice(returning_greetings)
    
    else:  # first_time seeing them this session
        if time_of_day == "morning":
            first_greetings = [
                f"Good morning, {name}!",
                f"Morning, {name}! Great to see you!",
                f"Hey {name}! Good morning!",
            ]
        elif time_of_day == "afternoon":
            first_greetings = [
                f"Good afternoon, {name}!",
                f"Hey {name}! Good to see you!",
                f"{name}! How's it going?",
            ]
        elif time_of_day == "evening":
            first_greetings = [
                f"Good evening, {name}!",
                f"Hey {name}! Nice to see you!",
                f"{name}! There you are!",
            ]
        else:
            first_greetings = [
                f"Hey {name}!",
                f"{name}! What brings you here?",
                f"Oh, {name}! Hi there!",
            ]
        return random.choice(first_greetings)

def generate_group_greeting(known_names: list, unknown_count: int) -> str:
    """Generate greeting for multiple people."""
    time_of_day = get_time_of_day()
    
    # Mark all known people as seen
    for name in known_names:
        mark_seen(name)
    
    if known_names and unknown_count == 0:
        # All known
        names_str = " and ".join(known_names)
        greetings = [
            f"Hey {names_str}! Good to see you all!",
            f"{names_str}! What's up?",
            f"Oh, {names_str}! How's everyone doing?",
        ]
        if time_of_day == "morning":
            greetings.append(f"Good morning, {names_str}!")
        elif time_of_day == "afternoon":
            greetings.append(f"Good afternoon, {names_str}!")
        return random.choice(greetings)
    
    elif known_names and unknown_count > 0:
        # Mix of known and unknown
        names_str = " and ".join(known_names)
        greetings = [
            f"Hey {names_str}! And who's your friend?",
            f"{names_str}! Good to see you. Who's this?",
            f"Oh, {names_str}! And I see you brought someone new!",
        ]
        return random.choice(greetings)
    
    else:
        # All unknown
        greetings = [
            "Hey everyone! I don't think we've met.",
            "Hello there! I'm not sure I know any of you yet.",
            "Hi everyone! Welcome! Who am I talking to?",
        ]
        return random.choice(greetings)
