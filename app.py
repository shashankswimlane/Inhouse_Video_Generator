"""
Application Entrypoint Proxy
Exposes the main FastAPI app instance from main.py for backwards compatibility.
"""

from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
