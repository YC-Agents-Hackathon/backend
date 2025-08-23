"""
Detective LLM Backend - Simplified Implementation
"""

import json
import time
import subprocess
import tempfile
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

MODEL = "openai/gpt-4.1"


# Data Models
class Message(BaseModel):
    role: str
    content: str
    ts: int


class UploadRequest(BaseModel):
    evidence: List[str]


class UploadResponse(BaseModel):
    totalEvidenceCount: int
    message: str


class ChatRequest(BaseModel):
    notebookId: str
    userMessage: str


class ChatResponse(BaseModel):
    response: str


class ReportResponse(BaseModel):
    report: str


class CodeExecutionRequest(BaseModel):
    code: str
    language: str


class CodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time: float


# In-Memory Storage
class NotebookState:
    def __init__(self):
        self.messages: List[Message] = []


# Global storage and Dedalus client
notebooks: Dict[str, NotebookState] = {}
global_evidence: List[str] = []  # Global evidence storage
client = None
runner = None

# System Prompts
INVESTIGATOR_PROMPT = """
You are a meticulous detective AI assisting an investigation. Use only the provided evidence and conversation. 
Be concise, actionable, and avoid speculation. 
If information is missing, ask a short clarifying question. 
Prefer bullet points and short paragraphs. Keep responses under 200 words.
You MUST only respond with the answer to the question, no other text.
"""

REPORT_PROMPT = """
You are a seasoned detective writing a clear narrative report for non-technical readers. 
Summarize facts, key actors, likely sequence of events, and find a conclusion to the case. 
Keep it confident but avoid speculation. 
Output plain text with short section headers (Summary, Timeline, Key Actors, Evidence Highlights, Findings, Conclusion). 
Keep under 800 words.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, runner
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    yield


app = FastAPI(
    title="Detective LLM Backend",
    description="Multi-thread detective assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_notebook(notebook_id: str) -> NotebookState:
    """Get or create notebook"""
    if notebook_id not in notebooks:
        notebooks[notebook_id] = NotebookState()
    return notebooks[notebook_id]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/evidence")
async def get_evidence():
    """Debug endpoint to check global evidence"""
    return {
        "totalEvidenceCount": len(global_evidence),
        "evidence": global_evidence[:5] if global_evidence else [],  # Show first 5 items
        "message": f"Global evidence contains {len(global_evidence)} items"
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_evidence(request: UploadRequest):
    global global_evidence
    global_evidence.extend(request.evidence)
    print(f"Uploaded evidence to global storage: {len(global_evidence)} total items")
    print(f"Latest evidence: {request.evidence}")
    return UploadResponse(
        totalEvidenceCount=len(global_evidence),
        message=f"Successfully uploaded {len(request.evidence)} evidence items to global storage"
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not runner:
        raise HTTPException(status_code=500, detail="Client not ready")

    notebook = get_notebook(request.notebookId)

    # Build context
    context_parts = [INVESTIGATOR_PROMPT]
    if global_evidence:
        context_parts.append("\nEVIDENCE:\n" + "\n".join(global_evidence))

    # Add conversation history
    for msg in notebook.messages:
        role = "User" if msg.role == "user" else "Assistant"
        context_parts.append(f"{role}: {msg.content}")

    context_parts.append(f"User: {request.userMessage}")
    context = "\n\n".join(context_parts)

    try:
        response = await runner.run(input=context, model=MODEL)

        # Store messages
        notebook.messages.append(
            Message(role="user", content=request.userMessage, ts=int(time.time()))
        )
        notebook.messages.append(
            Message(
                role="assistant", content=response.final_output, ts=int(time.time())
            )
        )

        return ChatResponse(response=response.final_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not runner:
        raise HTTPException(status_code=500, detail="Client not ready")

    async def generate():
        try:
            notebook = get_notebook(request.notebookId)

            # Build context
            context_parts = [INVESTIGATOR_PROMPT]
            if global_evidence:
                context_parts.append("\nEVIDENCE:\n" + "\n".join(global_evidence))

            for msg in notebook.messages:
                role = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content}")

            context_parts.append(f"User: {request.userMessage}")
            context = "\n\n".join(context_parts)

            print("CONTEXT: ", context)

            # Use the correct Dedalus streaming API
            result = runner.run(input=context, model=MODEL, stream=True)
            accumulated = ""

            # Stream the response using correct StreamChunk structure
            async for chunk in result:
                # StreamChunk has choices[0].delta.content structure
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Check if this chunk has content
                    if delta.content:
                        accumulated += delta.content
                        yield f"data: {json.dumps({'type': 'content', 'content': delta.content})}\n\n"
                    
                    # Check if streaming is finished
                    if choice.finish_reason == 'stop':
                        print(f"Streaming finished. Total accumulated: {accumulated}")
                        yield f"data: {json.dumps({'type': 'final', 'content': accumulated})}\n\n"
                        break

            # Store messages
            notebook.messages.append(
                Message(role="user", content=request.userMessage, ts=int(time.time()))
            )
            notebook.messages.append(
                Message(role="assistant", content=accumulated, ts=int(time.time()))
            )
            print(f"Final accumulated: {accumulated}")

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return EventSourceResponse(generate())


@app.post("/api/report", response_model=ReportResponse)
async def generate_report():
    if not runner:
        raise HTTPException(status_code=500, detail="Client not ready")

    # Build report context with ALL conversations from all notebooks
    context_parts = [REPORT_PROMPT]
    
    # Use global evidence
    if global_evidence:
        context_parts.append("\nEVIDENCE:\n" + "\n".join(global_evidence))
    
    # Add ALL conversations from ALL notebooks as context
    all_conversations = []
    for notebook_id, notebook in notebooks.items():
        if notebook.messages:
            all_conversations.append(f"\n--- Conversation from Notebook {notebook_id} ---")
            for msg in notebook.messages:
                role = "User" if msg.role == "user" else "Assistant"
                all_conversations.append(f"{role}: {msg.content}")
    
    if all_conversations:
        context_parts.append("\nALL CONVERSATION HISTORY:")
        context_parts.extend(all_conversations)

    context_parts.append(
        "\nGenerate a comprehensive detective report and solve the case."
    )
    context = "\n\n".join(context_parts)

    try:
        print(f"Report context length: {len(context)}")  # Debug
        response = await runner.run(input=context, model=MODEL)
        print(f"Report response type: {type(response)}")  # Debug
        print(f"Report final_output: {response.final_output}")  # Debug

        if not response.final_output:
            print("Warning: Empty final_output, using fallback")
            report_text = "Report generation completed but no content was returned."
        else:
            report_text = response.final_output

        return ReportResponse(report=report_text)
    except Exception as e:
        print(f"Report error: {str(e)}")  # Debug
        raise HTTPException(
            status_code=500, detail=f"Report generation failed: {str(e)}"
        )


def execute_python_code(code: str) -> tuple[str, str, float]:
    """Execute Python code safely in a temporary file with timeout."""
    start_time = time.time()
    
    try:
        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the code with timeout (10 seconds)
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return result.stdout, "", execution_time
            else:
                return result.stdout, result.stderr, execution_time
                
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return "", "Code execution timed out (10 seconds limit)", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return "", f"Execution error: {str(e)}", execution_time


@app.post("/api/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """Execute code safely with basic security restrictions."""
    
    # Basic security checks
    dangerous_imports = [
        'os', 'subprocess', 'sys', 'shutil', 'glob', 'pickle', 
        'socket', 'urllib', 'requests', '__import__', 'eval', 'exec'
    ]
    
    code_lower = request.code.lower()
    for dangerous in dangerous_imports:
        if dangerous in code_lower:
            return CodeExecutionResponse(
                output="",
                error=f"Security restriction: '{dangerous}' is not allowed",
                execution_time=0.0
            )
    
    if request.language == "python":
        output, error, exec_time = execute_python_code(request.code)
        return CodeExecutionResponse(
            output=output,
            error=error if error else None,
            execution_time=exec_time
        )
    else:
        return CodeExecutionResponse(
            output="",
            error=f"Language '{request.language}' is not supported yet. Only Python is currently available.",
            execution_time=0.0
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
