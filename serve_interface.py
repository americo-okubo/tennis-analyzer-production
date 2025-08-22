#!/usr/bin/env python3
"""
Simple HTTP server to serve the web interface
"""
import http.server
import socketserver
import os
import webbrowser
import time
import threading

def serve_interface():
    PORT = 8001
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"Starting web server at http://localhost:{PORT}")
    print("This will serve web_interface.html properly")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print("Open http://localhost:8001/web_interface.html in your browser")
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{PORT}/web_interface.html')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped")

if __name__ == "__main__":
    serve_interface()