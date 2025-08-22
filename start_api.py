#!/usr/bin/env python3
"""
Start script for Tennis Analyzer API
Simplified startup without multiprocessing issues
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the API server"""
    os.environ["ENVIRONMENT"] = "development"
    
    print("Starting Tennis Analyzer API...")
    print("API will be available at: http://localhost:8002")
    print("API documentation at: http://localhost:8002/docs")
    print("Press Ctrl+C to stop the server")
    
    # Start server with simpler configuration
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8006,  # Changed to 8006 with fixed validation
        reload=False,  # Disable reload to avoid multiprocessing issues
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()