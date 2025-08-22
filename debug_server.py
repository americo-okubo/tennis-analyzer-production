#!/usr/bin/env python3
"""
Debug server for Railway deployment
"""
import os
import sys
from pathlib import Path

print("=== DEBUG SERVER START ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment variables:")
for key, value in os.environ.items():
    if 'PORT' in key or 'RAILWAY' in key:
        print(f"  {key}: {value}")

print("=== CHECKING FILES ===")
project_root = Path(__file__).parent
print(f"Project root: {project_root}")
print(f"Files in project root:")
for item in sorted(project_root.iterdir()):
    print(f"  {item.name}")

print("=== CHECKING API DIRECTORY ===")
api_dir = project_root / "api"
if api_dir.exists():
    print(f"API directory exists: {api_dir}")
    print(f"Files in API directory:")
    for item in sorted(api_dir.iterdir()):
        print(f"  {item.name}")
else:
    print("API directory NOT FOUND!")

print("=== TESTING IMPORTS ===")
try:
    sys.path.insert(0, str(project_root))
    from api.main import app
    print("✅ Successfully imported app from api.main")
except Exception as e:
    print(f"❌ Failed to import app: {e}")
    import traceback
    traceback.print_exc()

print("=== STARTING UVICORN ===")
try:
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
except Exception as e:
    print(f"❌ Failed to start server: {e}")
    import traceback
    traceback.print_exc()