from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from pathlib import Path

# Import from our existing logic
from app import init_model, generate, get_model_info, MODEL_INFO

app = FastAPI(title="Small Language Model API")

# Initialize the model on startup
@app.on_event("startup")
async def startup_event():
    print("Initializing model...")
    init_model()
    print("Model loaded successfully.")

# Mount static files for HTML/CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    max_tokens: int = 128
    rep_penalty: float = 1.15

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Generate is blocking, run in executor
        loop = asyncio.get_event_loop()
        answer, source = await loop.run_in_executor(
            None, 
            generate, 
            req.message, 
            req.temperature, 
            req.top_k, 
            req.top_p, 
            req.max_tokens, 
            req.rep_penalty
        )
        return {"answer": answer, "source": source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info")
async def info_endpoint():
    return MODEL_INFO

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
