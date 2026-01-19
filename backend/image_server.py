"""
Simple HTTP server for serving event posters and map images.
Runs alongside the LiveKit agent.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path
import socket

class ImageServer:
    """Simple HTTP server to serve images from assets directory"""
    
    def __init__(self, assets_dir: Path, port: int = 8080, host: str = "0.0.0.0"):
        self.assets_dir = assets_dir
        self.port = port
        self.host = host
        self.server = None
        self.thread = None
        self._server_host = None
        
    def _get_local_ip(self):
        """Get the local network IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def start(self):
        """Start the HTTP server in a background thread"""
        if self.server is not None and self.thread is not None and self.thread.is_alive():
            print(f"⚠️  Image server already running on port {self.port}")
            return
        
        if self.host == "0.0.0.0":
            self._server_host = self._get_local_ip()
        else:
            self._server_host = self.host
        
        parent_dir = self.assets_dir.parent
        
        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests - serve static files"""
                self._serve_static_file(parent_dir)
            
            def _serve_static_file(self, base_dir):
                """Serve static files from the base directory"""
                try:
                    if self.path.startswith('/assets/'):
                        file_path = base_dir / self.path[1:]
                    else:
                        file_path = base_dir / self.path.lstrip('/')
                    
                    try:
                        file_path.resolve().relative_to(base_dir.resolve())
                    except ValueError:
                        self.send_response(403)
                        self.end_headers()
                        return
                    
                    if file_path.is_file():
                        if file_path.suffix in ['.jpg', '.jpeg']:
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
                        self.send_header('Access-Control-Allow-Origin', '*')
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
                pass  # Suppress logging
        
        try:
            self.server = HTTPServer((self.host, self.port), CustomHandler)
        except OSError as e:
            if e.errno == 98:
                print(f"⚠️  Port {self.port} already in use")
                return
            else:
                raise
        
        def serve():
            print(f"✅ Image server started: http://{self._server_host}:{self.port}")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Image server stopped")
    
    def get_image_url(self, category: str, filename: str) -> str:
        """Get the URL for an image"""
        host = self._server_host or self._get_local_ip()
        return f"http://{host}:{self.port}/assets/{category}/{filename}"
