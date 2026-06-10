"""
Simple HTTP server for serving media assets (posters, maps).
Runs alongside the LiveKit agent.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path
import socket
import os

class ImageServer:
    """Simple HTTP server to serve media from assets directory"""
    
    def __init__(self, assets_dir: Path, port: int = 8080, host: str = "0.0.0.0"):
        self.assets_dir = assets_dir
        self.port = port
        self.host = host
        self.server = None
        self._server_host = None
        self.face_monitor = None
        
    def set_face_monitor(self, monitor):
        """Link face monitor for live streaming"""
        self.face_monitor = monitor
        if self.server:
            self.server.face_monitor = monitor
        
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
            print(f"⚠️  Media server already running on port {self.port}")
            return
        
        if os.getenv("IMAGE_SERVER_HOST"):
            self._server_host = os.getenv("IMAGE_SERVER_HOST")
        elif self.host == "0.0.0.0":
            self._server_host = self._get_local_ip()
        else:
            self._server_host = self.host
        
        parent_dir = self.assets_dir.parent
        
        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests - serve static files or camera stream"""
                if self.path == '/':
                    self._serve_landing_page()
                elif self.path == '/camera/mjpeg':
                    self._serve_mjpeg_stream()
                elif self.path == '/camera/live.jpg':
                    self._serve_single_frame()
                else:
                    self._serve_static_file(parent_dir)

            def do_POST(self):
                """Handle POST requests - e.g., triggering a re-index"""
                if self.path == '/trigger-index':
                    try:
                        import sys
                        import importlib
                        from pathlib import Path
                        # Ensure backend is in path
                        backend_dir = Path(__file__).parent
                        if str(backend_dir) not in sys.path:
                            sys.path.insert(0, str(backend_dir))

                        # Force-reload modules so code changes are picked up without restart
                        import event_indexer
                        import event_database
                        importlib.reload(event_indexer)
                        importlib.reload(event_database)
                        from event_database import build_event_database

                        # Delete manifest to force full re-scan
                        manifest_path = backend_dir / "event_db" / "event_manifest.json"
                        if manifest_path.exists():
                            manifest_path.unlink()

                        print("🔄 Manual re-index triggered via API")
                        build_event_database(Path(self.server.assets_dir))
                        
                        self.send_response(200)
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(b"OK")
                    except Exception as e:
                        print(f"❌ Error during manual re-index: {e}")
                        self.send_response(500)
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_landing_page(self):
                """Serve a simple HTML page with links to available services"""
                html = f"""
                <html>
                <head>
                    <title>AI Voice Agent - Media Server</title>
                    <style>
                        body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: 0 auto; background: #f4f4f9; color: #333; }}
                        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                        .status {{ font-weight: bold; padding: 5px 10px; border-radius: 4px; display: inline-block; }}
                        .online {{ background: #d4edda; color: #155724; }}
                        .offline {{ background: #f8d7da; color: #721c24; }}
                        a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
                        a:hover {{ text-decoration: underline; }}
                        code {{ background: #eee; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
                    </style>
                </head>
                <body>
                    <h1>🤖 AI Voice Agent - Diagnostics</h1>
                    
                    <div class="card">
                        <h2>🎥 Camera Stream</h2>
                        <p><strong>MJPEG Stream (Live):</strong> <a href="/camera/mjpeg">/camera/mjpeg</a></p>
                        <p><strong>Single Frame (.jpg):</strong> <a href="/camera/live.jpg">/camera/live.jpg</a></p>
                        
                        <p>Status: <span class="status {'online' if self.server.face_monitor and self.server.face_monitor.current_frame is not None else 'offline'}">
                            {'READY' if self.server.face_monitor and self.server.face_monitor.current_frame is not None else 'NOT READY'}
                        </span></p>
                    </div>

                    <div class="card">
                        <h2>🛠️ Troubleshooting</h2>
                        <ul>
                            <li>If the stream is empty/black, check the terminal logs for <code>❌ Could not open USB webcam</code>.</li>
                            <li>Ensure SSH port forwarding is active: <code>ssh -L 8080:localhost:8080 nema@raspberrypi.local</code></li>
                            <li>Try reloading this page to check the status indicator above.</li>
                        </ul>
                    </div>
                </body>
                </html>
                """
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())

            def _serve_mjpeg_stream(self):
                """Serve a continuous MJPEG stream from the camera"""
                if not hasattr(self.server, 'face_monitor') or self.server.face_monitor is None:
                    self.send_error(503, "Camera not initialized")
                    return

                try:
                    import cv2
                    import time
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--frame')
                    self.end_headers()

                    while True:
                        frame = self.server.face_monitor.get_current_frame()
                        if frame is not None:
                            # Encode frame as JPEG
                            _, jpeg = cv2.imencode('.jpg', frame)
                            
                            # Manually write multipart boundary and frame headers
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                            self.wfile.write(b'\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        
                        time.sleep(0.1)  # Limit to 10 FPS
                except (ConnectionResetError, BrokenPipeError):
                    pass # Client disconnected
                except Exception as e:
                    print(f"⚠️ MJPEG Stream Error: {e}")

            def _serve_single_frame(self):
                """Serve a single JPEG frame from the camera"""
                if not hasattr(self.server, 'face_monitor') or self.server.face_monitor is None:
                    self.send_error(503, "Camera not initialized")
                    return

                try:
                    import cv2
                    frame = self.server.face_monitor.get_current_frame()
                    if frame is not None:
                        _, jpeg = cv2.imencode('.jpg', frame)
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg)))
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                    else:
                        self.send_error(503, "No frame available")
                except Exception as e:
                    self.send_error(500, str(e))

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
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress logging
        
        try:
            self.server = HTTPServer((self.host, self.port), CustomHandler)
            self.server.face_monitor = self.face_monitor # Pass monitor to server instance object
            self.server.assets_dir = self.assets_dir
        except OSError as e:
            if e.errno == 98:
                print(f"⚠️  Port {self.port} already in use")
                return
            else:
                raise
        
        def serve():
            print(f"✅ Media server started: http://{self._server_host}:{self.port}")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Media server stopped")
    
    def get_image_url(self, category: str, filename: str) -> str:
        """Get the URL for an image"""
        host = self._server_host or self._get_local_ip()
        return f"http://{host}:{self.port}/assets/{category}/{filename}"
