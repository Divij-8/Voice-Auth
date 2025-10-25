"""
Simple HTTP server to serve the frontend HTML file on port 3000
This allows the frontend to be accessed while FastAPI backend runs on port 8000
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 3000
HANDLER = http.server.SimpleHTTPRequestHandler

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-type')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print("\n" + "="*70)
        print("VOICE GUARDIAN - FRONTEND SERVER")
        print("="*70)
        print(f"\n✓ Frontend server running on http://localhost:{PORT}")
        print(f"✓ Serving index.html from: {os.getcwd()}")
        print(f"\n📌 Open your browser to: http://localhost:{PORT}")
        print("\n⚠️  Make sure the FastAPI backend is also running on port 8000")
        print("="*70 + "\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✓ Frontend server stopped")
