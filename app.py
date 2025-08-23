"""
Detective LLM Backend - Simplified Implementation
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dedalus_labs import AsyncDedalus, DedalusRunner
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
    notebookId: str
    evidence: List[str]


class UploadResponse(BaseModel):
    notebookId: str
    totalEvidenceCount: int


class ChatRequest(BaseModel):
    notebookId: str
    userMessage: str


class ChatResponse(BaseModel):
    response: str


class ReportResponse(BaseModel):
    report: str


# In-Memory Storage
class NotebookState:
    def __init__(self):
        self.evidence: List[str] = []
        self.messages: List[Message] = []


# Global storage and Dedalus client
notebooks: Dict[str, NotebookState] = {}
client = None
runner = None

# System Prompts
INVESTIGATOR_PROMPT = """
You are a meticulous detective AI assisting an investigation. Use only the provided evidence and conversation. 
Be concise, actionable, and avoid speculation. 
If information is missing, ask a short clarifying question. 
Prefer bullet points and short paragraphs. Keep responses under 200 words.
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


@app.post("/api/upload", response_model=UploadResponse)
async def upload_evidence(request: UploadRequest):
    notebook = get_notebook(request.notebookId)
    notebook.evidence.extend(request.evidence)
    return UploadResponse(
        notebookId=request.notebookId, totalEvidenceCount=len(notebook.evidence)
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not runner:
        raise HTTPException(status_code=500, detail="Client not ready")

    notebook = get_notebook(request.notebookId)

    # Build context
    context_parts = [INVESTIGATOR_PROMPT]
    if notebook.evidence:
        context_parts.append("\nEVIDENCE:\n" + "\n".join(notebook.evidence))

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
            if notebook.evidence:
                context_parts.append("\nEVIDENCE:\n" + "\n".join(notebook.evidence))

            for msg in notebook.messages:
                role = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content}")

            context_parts.append(f"User: {request.userMessage}")
            context = "\n\n".join(context_parts)

            print("CONTEXT: ", context)

            result = runner.run(input=context, model=MODEL, stream=True)
            accumulated = ""

            async for chunk in result:
                if hasattr(chunk, "content") and chunk.content:
                    accumulated += chunk.content
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

                if hasattr(chunk, "final_output") and chunk.final_output:
                    accumulated = chunk.final_output
                    yield f"data: {json.dumps({'type': 'final', 'content': chunk.final_output})}\n\n"

            # Store messages
            notebook.messages.append(
                Message(role="user", content=request.userMessage, ts=int(time.time()))
            )
            notebook.messages.append(
                Message(role="assistant", content=accumulated, ts=int(time.time()))
            )

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return EventSourceResponse(generate())


@app.post("/api/report", response_model=ReportResponse)
async def generate_report():
    if not runner:
        raise HTTPException(status_code=500, detail="Client not ready")

    # Build report context with ALL conversations from all notebooks
    context_parts = [REPORT_PROMPT]
    
    # Find the first notebook with evidence and use it
    evidence_used = False
    for notebook_id, notebook in notebooks.items():
        if notebook.evidence and not evidence_used:
            context_parts.append("\nEVIDENCE:\n" + "\n".join(notebook.evidence))
            evidence_used = True
            break
    
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
