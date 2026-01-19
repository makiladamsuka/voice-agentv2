import json
from livekit.agents import RunContext
from livekit.agents.llm import function_tool

class ContentTools:
    def __init__(self, image_manager, room_provider):
        self.image_manager = image_manager
        self._room_provider = room_provider

    @property
    def room(self):
        return self._room_provider()

    async def list_available_events(self, context: RunContext) -> str:
        """Lists all available events on campus."""
        print("📋 Listing available events")
        
        event_names = self.image_manager.list_available_events()
        
        if not event_names:
            return "There are no events scheduled at the moment."
        
        if len(event_names) == 1:
            return f"We have {event_names[0]} happening on campus."
        else:
            events_list = ", ".join(event_names[:-1]) + f" and {event_names[-1]}"
            return f"We have {len(event_names)} events: {events_list}."

    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        """Displays an event poster on the frontend."""
        print(f"🎨 Showing event poster for: {event_description}")
        
        # Find matching image
        image_path = self.image_manager.find_event_image(event_description)
        
        if image_path:
            return f"I found the {event_description} poster! It's available at: {image_path.name}"
        else:
            return f"Sorry, I couldn't find a poster for '{event_description}'. We have: {', '.join(self.image_manager.list_available_events())}."

    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        """Displays a campus location map on the frontend."""
        print(f"🗺️  Showing location map for: {location_query}")
        
        # Find matching map
        image_path = self.image_manager.find_location_map(location_query)
        
        if image_path:
            return f"I found a map for {location_query}! Check {image_path.name}"
        else:
            return f"Sorry, I don't have a map for '{location_query}'."
