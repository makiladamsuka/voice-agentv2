"""
Simple HTTP server for serving event and map images.
Runs alongside the LiveKit agent.
Also serves camera frames for debugging.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path
import os
import socket
import cv2
import io
from typing import Optional

class ImageServer:
    """Simple HTTP server to serve images from assets directory and camera frames"""
    
    def __init__(self, assets_dir: Path, port: int = 8080, host: str = "0.0.0.0"):
        self.assets_dir = assets_dir
        self.port = port
        self.host = host
        self.server = None
        self.thread = None
        self._server_host = None  # Will be set to actual network IP or localhost
        self.face_monitor = None  # Will be set by greeting_agent to provide camera frames
        
    def _get_local_ip(self):
        """Get the local network IP address"""
        try:
            # Connect to external server to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"  # Fallback to localhost
    
    def start(self):
        """Start the HTTP server in a background thread"""
        # Don't start if already running - check both server and thread
        if self.server is not None and self.thread is not None and self.thread.is_alive():
            print(f"⚠️  Image server already running on port {self.port}")
            return
        
        # Determine the server host for URL generation
        if self.host == "0.0.0.0":
            self._server_host = self._get_local_ip()
        else:
            self._server_host = self.host
        
        # Store parent directory of assets
        parent_dir = self.assets_dir.parent
        server_instance = self  # Capture self for use in handler
        
        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests"""
                # Handle camera frame endpoint
                if self.path == '/camera/frame.jpg' or self.path == '/camera/frame':
                    self._serve_camera_frame()
                    return
                
                # Handle camera debug page
                if self.path in ['/camera', '/camera/', '/camera/debug']:
                    self._serve_camera_debug_page()
                    return
                
                # Default: serve static files from assets directory
                self._serve_static_file(parent_dir)
            
            def _serve_camera_frame(self):
                """Serve the current camera frame as JPEG"""
                try:
                    # Get current frame from face monitor
                    if server_instance.face_monitor:
                        frame = server_instance.face_monitor.get_current_frame()
                        if frame is not None:
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            jpg_bytes = buffer.tobytes()
                            
                            # Send response
                            self.send_response(200)
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpg_bytes)))
                            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                            self.send_header('Pragma', 'no-cache')
                            self.send_header('Expires', '0')
                            self.end_headers()
                            self.wfile.write(jpg_bytes)
                            return
                    
                    # No frame available
                    self.send_response(503)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Camera frame not available')
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(f'Error: {str(e)}'.encode())
            
            def _serve_camera_debug_page(self):
                """Serve an HTML page that auto-refreshes the camera frame"""
                html = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Debug View</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
            font-family: monospace;
        }
        h1 {
            margin-top: 0;
        }
        #frame {
            border: 2px solid #444;
            max-width: 100%;
            height: auto;
        }
        .info {
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }
        .controls {
            margin: 10px 0;
        }
        button {
            background: #444;
            color: #fff;
            border: 1px solid #666;
            padding: 8px 16px;
            cursor: pointer;
            margin-right: 10px;
        }
        button:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <h1>📹 Camera Debug View</h1>
    <div class="controls">
        <button onclick="location.reload()">Refresh</button>
        <button onclick="toggleAutoRefresh()" id="autoBtn">Auto-refresh: ON</button>
    </div>
    <img id="frame" src="/camera/frame.jpg" alt="Camera Frame">
    <div class="info">
        <p>Frame updates every <span id="interval">2</span> seconds</p>
        <p>Last update: <span id="lastUpdate">-</span></p>
    </div>
    
    <script>
        let autoRefresh = true;
        let interval = 2000;
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            document.getElementById('autoBtn').textContent = 
                'Auto-refresh: ' + (autoRefresh ? 'ON' : 'OFF');
        }
        
        function updateFrame() {
            const img = document.getElementById('frame');
            const timestamp = new Date().toLocaleTimeString();
            img.src = '/camera/frame.jpg?t=' + Date.now();
            document.getElementById('lastUpdate').textContent = timestamp;
        }
        
        // Update frame on load
        updateFrame();
        
        // Auto-refresh
        setInterval(() => {
            if (autoRefresh) {
                updateFrame();
            }
        }, interval);
        
        // Update timestamp periodically
        setInterval(() => {
            const now = new Date();
            document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
        }, 1000);
    </script>
</body>
</html>
                """
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _serve_static_file(self, base_dir):
                """Serve static files from the base directory"""
                try:
                    # Map /assets/* to assets directory
                    if self.path.startswith('/assets/'):
                        file_path = base_dir / self.path[1:]  # Remove leading /
                    else:
                        file_path = base_dir / self.path.lstrip('/')
                    
                    # Security: ensure path is within base_dir
                    try:
                        file_path.resolve().relative_to(base_dir.resolve())
                    except ValueError:
                        self.send_response(403)
                        self.end_headers()
                        return
                    
                    if file_path.is_file():
                        # Determine content type
                        if file_path.suffix == '.jpg' or file_path.suffix == '.jpeg':
                            content_type = 'image/jpeg'
                        elif file_path.suffix == '.png':
                            content_type = 'image/png'
                        elif file_path.suffix == '.gif':
                            content_type = 'image/gif'
                        else:
                            content_type = 'application/octet-stream'
                        
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        self.send_response(200)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b'File not found')
                        
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f'Error: {str(e)}'.encode())
            
            def do_OPTIONS(self):
                """Handle OPTIONS for CORS"""
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.end_headers()
            
            def log_message(self, format, *args):
                # Customize logging
                print(f"📁 Image server: {args[0]}")
        
        try:
            self.server = HTTPServer((self.host, self.port), CustomHandler)
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"⚠️  Port {self.port} already in use (another worker owns it)")
                return
            else:
                raise
        
        def serve():
            print(f"✅ Image server started")
            print(f"   Local access: http://localhost:{self.port}")
            print(f"   Network access: http://{self._server_host}:{self.port}")
            print(f"   Serving from: {parent_dir}")
            print(f"   Assets at: /assets/")
            print(f"   Camera debug: http://localhost:{self.port}/camera/debug")
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
            Full URL to the image (accessible from network)
        """
        # Use the actual server host determined at startup
        host = self._server_host or self._get_local_ip()
        return f"http://{host}:{self.port}/assets/{category}/{filename}"
