#!/usr/bin/env python3
"""
Simplest possible FastAPI server for Railway testing
"""
import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tennis Analyzer - Test")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "Tennis Analyzer is working!", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tennis-analyzer"}

@app.get("/test")
async def test():
    return {
        "message": "Test endpoint working",
        "port": os.environ.get("PORT", "not_set"),
        "environment": "railway" if "RAILWAY_" in str(os.environ) else "local"
    }

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/.well-known/health")
async def wellknown_health():
    return {"status": "ok"}

@app.get("/robots.txt")  
async def robots():
    return JSONResponse(content="User-agent: *\nDisallow: /", media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on host=0.0.0.0, port={port}")
    logger.info(f"Environment variables: PORT={os.environ.get('PORT')}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        access_log=True,
        log_level="info"
    )