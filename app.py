"""
Detective LLM Backend - Simplified Implementation
"""

import asyncio
import json
import time
import subprocess
import tempfile
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set

from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Import our multi-agent system
from agents.orchestrator import global_orchestrator, OrchestrationStatus
from agents.detective_agents import AGENT_REGISTRY
from agents.base_agent import AgentOutput
from supabase_client import SupabaseEvidenceClient

load_dotenv()

MODEL = "openai/gpt-4.1"


# Data Models
class Message(BaseModel):
    role: str
    content: str
    ts: int


class UploadRequest(BaseModel):
    evidence: List[str]
    notebookId: Optional[str] = None


class UploadResponse(BaseModel):
    totalEvidenceCount: int
    message: str


class CaseFileUpload(BaseModel):
    filename: str
    content: str
    file_size: int


class CaseUploadRequest(BaseModel):
    case_id: str
    files: List[CaseFileUpload]


class CaseUploadResponse(BaseModel):
    success_count: int
    total_count: int
    failed_files: List[str]
    message: str
    case_id: str


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


class MultiAgentAnalysisRequest(BaseModel):
    case_id: Optional[str] = None
    evidence: List[str]
    agent_types: List[str] = ["pattern_recognition", "timeline_reconstruction", "entity_relationship", "financial_analysis", "communication_analysis", "evidence_validation"]
    create_notebook: bool = True
    notebook_title: Optional[str] = "Multi-Agent Analysis"


class MultiAgentAnalysisResponse(BaseModel):
    run_id: str
    status: str
    message: str
    agent_count: int
    websocket_url: str


# In-Memory Storage
class NotebookState:
    def __init__(self):
        self.messages: List[Message] = []


# Global storage and Dedalus client
notebooks: Dict[str, NotebookState] = {}
global_evidence: List[str] = []  # Global evidence storage
client = None
runner = None
supabase_client = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()
        self.active_connections[run_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            self.active_connections[run_id].discard(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
    
    async def send_to_run(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    disconnected.add(connection)
            # Remove disconnected connections
            for conn in disconnected:
                self.active_connections[run_id].discard(conn)

websocket_manager = ConnectionManager()

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
    global client, runner, supabase_client
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    supabase_client = SupabaseEvidenceClient()
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


@app.post("/api/upload-case-files", response_model=CaseUploadResponse)
async def upload_case_files(request: CaseUploadRequest):
    """Simplified case-level file upload endpoint"""
    global global_evidence
    
    success_count = 0
    failed_files = []
    case_id = request.case_id
    
    try:
        print(f"Processing {len(request.files)} files for case {case_id}")
        
        for file_item in request.files:
            try:
                # Store file content with case context
                evidence_entry = f"=== CASE: {case_id} | FILE: {file_item.filename} | SIZE: {file_item.file_size} bytes ===\n{file_item.content}\n=== END FILE ===\n"
                global_evidence.append(evidence_entry)
                success_count += 1
                print(f"✓ Successfully processed: {file_item.filename} ({file_item.file_size} bytes)")
                
            except Exception as e:
                error_msg = f"{file_item.filename}: {str(e)}"
                failed_files.append(error_msg)
                print(f"✗ Failed to process {file_item.filename}: {str(e)}")
        
        total_count = len(request.files)
        message = f"Case {case_id}: Processed {success_count}/{total_count} files. Total evidence: {len(global_evidence)} items"
        
        print(f"Upload complete: {message}")
        
        return CaseUploadResponse(
            success_count=success_count,
            total_count=total_count,
            failed_files=failed_files,
            message=message,
            case_id=case_id
        )
        
    except Exception as e:
        error_msg = f"Case file upload failed for case {case_id}: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend connectivity"""
    return {
        "status": "healthy",
        "message": "Backend is running",
        "evidence_count": len(global_evidence),
        "timestamp": int(time.time())
    }


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


@app.post("/api/multi-agent-analysis", response_model=MultiAgentAnalysisResponse)
async def start_multi_agent_analysis(request: MultiAgentAnalysisRequest):
    """Start a multi-agent analysis of the evidence"""
    if not runner or not supabase_client:
        raise HTTPException(status_code=500, detail="Client not ready")
    
    # Validate agent types
    invalid_agents = [agent for agent in request.agent_types if agent not in AGENT_REGISTRY]
    if invalid_agents:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid agent types: {invalid_agents}. Available: {list(AGENT_REGISTRY.keys())}"
        )
    
    try:
        # Determine evidence source
        if request.case_id:
            # Get evidence from Supabase for the specific case
            print(f"Retrieving evidence for case: {request.case_id}")
            evidence_files = await supabase_client.get_case_evidence(request.case_id)
            
            if not evidence_files:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No evidence files found for case {request.case_id}. Please upload evidence files first."
                )
            
            # Process evidence files into text suitable for AI analysis
            evidence = await supabase_client.process_evidence_to_text(evidence_files)
            
            # Get case information for context
            case_info = await supabase_client.get_case_info(request.case_id)
            case_context = f"Case: {case_info.get('title', 'Unknown')} - {case_info.get('description', 'No description')}" if case_info else ""
            
            print(f"Retrieved {len(evidence_files)} evidence files from case {request.case_id}")
            
        else:
            # Fallback to provided evidence or global evidence
            evidence = request.evidence if request.evidence else global_evidence
            case_context = ""
            
            if not evidence:
                raise HTTPException(status_code=400, detail="No evidence provided and no case_id specified")
        
        # Start multi-agent analysis
        run_id = f"case_{request.case_id}_{int(time.time())}" if request.case_id else f"run_{int(time.time())}"
        
        # Context for agents
        context = {
            "runner": runner,
            "case_id": request.case_id,
            "case_context": case_context,
            "create_notebook": request.create_notebook,
            "notebook_title": request.notebook_title or f"Multi-Agent Analysis - {run_id}",
            "evidence_count": len(evidence)
        }
        
        # Status callback for real-time updates
        async def status_callback(agent_output: AgentOutput):
            await websocket_manager.send_to_run(run_id, {
                "type": "agent_status",
                "agent_id": agent_output.agent_id,
                "agent_type": agent_output.agent_type,
                "status": agent_output.status.value,
                "progress": agent_output.progress,
                "current_step": agent_output.current_step,
                "error": agent_output.error
            })
        
        # Progress callback for real-time updates  
        async def progress_callback(agent_output: AgentOutput):
            await websocket_manager.send_to_run(run_id, {
                "type": "agent_progress",
                "agent_id": agent_output.agent_id,
                "progress": agent_output.progress,
                "current_step": agent_output.current_step
            })
        
        multi_agent_run = await global_orchestrator.start_multi_agent_analysis(
            run_id=run_id,
            evidence=evidence,
            agent_types=request.agent_types,
            context=context,
            status_callback=status_callback,
            progress_callback=progress_callback
        )
        
        return MultiAgentAnalysisResponse(
            run_id=run_id,
            status=multi_agent_run.status.value,
            message=f"Started analysis with {len(request.agent_types)} agents for {len(evidence)} evidence items",
            agent_count=len(request.agent_types),
            websocket_url=f"/ws/multi-agent/{run_id}"
        )
        
    except Exception as e:
        print(f"Error starting multi-agent analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start multi-agent analysis: {str(e)}")


@app.get("/api/multi-agent-analysis/{run_id}")
async def get_multi_agent_status(run_id: str):
    """Get status of a multi-agent analysis run"""
    run = global_orchestrator.get_run_status(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    
    return run.get_summary()


@app.get("/api/multi-agent-analysis/{run_id}/notebook")
async def get_multi_agent_notebook(run_id: str):
    """Get generated notebook from multi-agent analysis"""
    run = global_orchestrator.get_run_status(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    
    if run.status != OrchestrationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not yet completed")
    
    notebook_cells = run.get_all_notebook_cells()
    
    return {
        "run_id": run_id,
        "title": f"Multi-Agent Analysis - {run_id}",
        "cells": notebook_cells,
        "created_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None
    }


@app.get("/api/agent-types")
async def get_available_agent_types():
    """Get list of available agent types"""
    agents_info = []
    for agent_type, agent_class in AGENT_REGISTRY.items():
        # Create temporary instance to get metadata
        temp_agent = agent_class()
        agents_info.append({
            "type": agent_type,
            "name": temp_agent.agent_name,
            "description": temp_agent.agent_description
        })
    
    return {
        "agent_types": agents_info,
        "total_count": len(agents_info)
    }


@app.websocket("/ws/multi-agent/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time multi-agent updates"""
    await websocket_manager.connect(websocket, run_id)
    
    try:
        # Send initial status
        run = global_orchestrator.get_run_status(run_id)
        if run:
            await websocket.send_text(json.dumps({
                "type": "run_status",
                "run_id": run_id,
                "status": run.status.value,
                "progress": run.progress,
                "summary": run.get_summary()
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Analysis run not found"
            }))
            return
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for any message from client (keepalive)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send periodic status updates
                run = global_orchestrator.get_run_status(run_id)
                if run:
                    await websocket.send_text(json.dumps({
                        "type": "run_status",
                        "run_id": run_id,
                        "status": run.status.value,
                        "progress": run.progress
                    }))
                    
                    # If run is completed, send final update and close
                    if run.status in [OrchestrationStatus.COMPLETED, OrchestrationStatus.FAILED, OrchestrationStatus.CANCELLED]:
                        await websocket.send_text(json.dumps({
                            "type": "run_complete",
                            "run_id": run_id,
                            "status": run.status.value,
                            "summary": run.get_summary()
                        }))
                        break
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(websocket, run_id)


@app.delete("/api/multi-agent-analysis/{run_id}")
async def cancel_multi_agent_analysis(run_id: str):
    """Cancel a running multi-agent analysis"""
    run = global_orchestrator.get_run_status(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    
    if run.status not in [OrchestrationStatus.RUNNING, OrchestrationStatus.PENDING]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed analysis")
    
    await global_orchestrator.cancel_run(run_id)
    
    # Notify WebSocket clients
    await websocket_manager.send_to_run(run_id, {
        "type": "run_cancelled",
        "run_id": run_id,
        "message": "Analysis cancelled by user"
    })
    
    return {"message": "Analysis cancelled successfully"}


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


# Cleanup task for completed runs
@app.on_event("startup")
async def startup_event():
    # Schedule periodic cleanup of completed runs
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodically clean up completed runs"""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        try:
            global_orchestrator.cleanup_completed_runs()
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
