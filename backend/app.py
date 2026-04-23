import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.planner import create_plan
from agent.executor import execute_step
from agent.validator import validate_step
from agent.memory import retrieve_similar_task, store_task
from rag.retrieve import init_retriever

app = FastAPI(title="Autonomous Agent API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    request: str

@app.on_event("startup")
async def startup_event():
    print("[System] Initializing RAG embeddings...")
    init_retriever()

@app.get("/")
async def root():
    return {"status": "online", "message": "Autonomous Agent API is running"}

@app.post("/chat")
async def chat(body: RequestBody):
    user_request = body.request
    
    # Check Memory
    cached_output = retrieve_similar_task(user_request)
    if cached_output:
        return {"output": cached_output, "cached": True}
        
    # Generating plan (This is a simplified synchronous version for now)
    # We will later move this to a WebSocket for real-time progress.
    plan = create_plan(user_request)
    if not plan:
        raise HTTPException(status_code=500, detail="Failed to generate a plan.")
        
    context = ""
    last_output = ""
    
    for i, step in enumerate(plan):
        # In a real app, we'd stream these steps via WebSockets
        step_output = execute_step(step, context)
        is_valid, reason = validate_step(step, step_output)
        
        if not is_valid:
            # Simple retry logic for now
            step_output = execute_step(f"{step} (Previous failed: {reason})", context)
            
        context += f"\n--- Output of Step {i+1}: {step} ---\n{step_output}\n"
        last_output = step_output
        
    store_task(user_request, last_output)
    return {"output": last_output, "cached": False, "plan": plan}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
