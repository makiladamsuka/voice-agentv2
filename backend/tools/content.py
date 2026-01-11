import json
from livekit.agents import RunContext
from livekit.agents.llm import function_tool

class ContentTools:
    def __init__(self, image_manager, image_server, room_provider):
        self.image_manager = image_manager
        self.image_server = image_server
        self._room_provider = room_provider  # function that returns current room

    @property
    def room(self):
        return self._room_provider()

    # @function_tool  # Removed - called as helper method
    async def list_available_events(self, context: RunContext) -> str:
        """
        Lists all available events on campus.
        Use when user asks what events are happening or what's available.
        """
        print("📋 Listing available events")
        
        event_names = self.image_manager.list_available_events()
        
        if not event_names:
            return "There are no events scheduled at the moment."
        
        if len(event_names) == 1:
            return f"We have {event_names[0]} happening on campus."
        else:
            # Format as a nice list
            events_list = ", ".join(event_names[:-1]) + f" and {event_names[-1]}"
            return f"We have {len(event_names)} events: {events_list}."

    # @function_tool  # Removed - called as helper method
    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        """
        Displays an event poster on the frontend.
        Use when user asks about events, festivals, or programs.
        
        Args:
            event_description: Natural language description of the event (e.g., "tech fest", "cultural night")
        """
        print(f"🎨 Showing event poster for: {event_description}")
        
        if not self.room:
            return "I am not connected to a room right now."

        # Find matching image
        image_path = self.image_manager.find_event_image(event_description)
        
        if image_path:
            # Send image URL instead of base64
            image_url = self.image_server.get_image_url("events", image_path.name)
            
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "image",
                    "category": "event",
                    "url": image_url,
                    "caption": f"Event: {event_description}"
                }).encode()
            )
            
            return f"I've displayed the {event_description} poster for you."
        else:
            # No image found - just return text message
            return f"Sorry, I couldn't find a poster for '{event_description}'. We currently have posters for: {', '.join(self.image_manager.list_available_events())}."

    # @function_tool  # Removed - called as helper method
    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        """
        Displays a campus location map on the frontend.
        Use when user asks for directions or where something is located.
        
        Args:
            location_query: Natural language query about location (e.g., "DS lab", "library")
        """
        print(f"🗺️  Showing location map for: {location_query}")
        
        if not self.room:
            return "I am not connected to a room right now."
        
        # Find matching map
        image_path = self.image_manager.find_location_map(location_query)
        
        if image_path:
            # Send image URL instead of base64
            image_url = self.image_server.get_image_url("maps", image_path.name)
            
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "image",
                    "category": "map",
                    "url": image_url,
                    "caption": f"Location: {location_query}"
                }).encode()
            )
            
            return f"Here's the map to {location_query}."
        else:
            # No map found - just return text message  
            return f"Sorry, I don't have a map for '{location_query}'. Try asking about specific campus buildings or landmarks."
