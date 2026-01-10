"""
Simple HTTP server for serving event and map images.
Runs alongside the LiveKit agent.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
from pathlib import Path
import os

class ImageServer:
    """Simple HTTP server to serve images from assets directory"""
    
    def __init__(self, assets_dir: Path, port: int = 8080):
        self.assets_dir = assets_dir
        self.port = port
        self.server = None
        self.thread = None
        
    def start(self):
        """Start the HTTP server in a background thread"""
        # Don't start if already running - check both server and thread
        if self.server is not None and self.thread is not None and self.thread.is_alive():
            print(f"⚠️  Image server already running on port {self.port}")
            return
        
        # Store parent directory of assets
        parent_dir = self.assets_dir.parent
        
        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                # Serve from parent directory so /assets/... works
                super().__init__(*args, directory=str(parent_dir), **kwargs)
            
            def end_headers(self):
                # Add CORS headers to allow frontend to load images
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                super().end_headers()
            
            def log_message(self, format, *args):
                # Customize logging
                print(f"📁 Image server: {args[0]}")
        
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), CustomHandler)
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"⚠️  Port {self.port} already in use (another worker owns it)")
                return
            else:
                raise
        
        def serve():
            print(f"✅ Image server started on http://localhost:{self.port}")
            print(f"   Serving from: {parent_dir}")
            print(f"   Assets at: /assets/")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Image server stopped")
    
    def get_image_url(self, category: str, filename: str) -> str:
        """
        Get the URL for an image
        
        Args:
            category: 'events', 'maps', or 'fallback'
            filename: Image filename
        
        Returns:
            Full URL to the image
        """
        return f"http://localhost:{self.port}/assets/{category}/{filename}"
