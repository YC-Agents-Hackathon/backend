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
from typing import Dict, List, Optional, Set, Any
from pathlib import Path, Set, Any
from pathlib import Path

from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.requests import Request
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
    notebook_data: Optional[List[Dict[str, Any]]] = None


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
You are an elite detective AI with expertise in criminal investigation, forensic analysis, and case management. You serve as an interactive analysis copilot for ongoing investigations.

PRIMARY RESPONSIBILITIES:
- Analyze provided evidence with forensic precision
- Identify patterns, connections, and inconsistencies in data
- Generate actionable investigative leads and hypotheses
- Answer specific questions about suspects, timelines, locations, and evidence
- Cross-reference information across multiple evidence sources

ANALYSIS APPROACH:
- Start with the provided evidence as your foundation
- If critical information is missing, search the web for current/supplementary data
- Apply logical deduction and investigative reasoning
- Identify gaps that require additional evidence collection
- Highlight contradictions or anomalies that need investigation

RESPONSE FORMAT:
- Use bullet points for clarity and quick scanning
- Provide specific details (dates, times, names, locations)
- Include confidence levels when making assessments (High/Medium/Low confidence)
- Suggest next investigative steps when relevant
- Keep responses focused and actionable (under 200 words)
- CRITICAL: Only provide the direct answer to the question asked - no preamble or closing remarks

EVIDENCE ANALYSIS PRIORITIES:
1. Timeline reconstruction and sequence of events
2. Suspect identification and behavioral analysis  
3. Location analysis and geographical patterns
4. Communication patterns and relationship mapping
5. Financial transactions and resource tracking
6. Digital footprints and technological evidence
"""

REPORT_PROMPT = """
You are a seasoned detective sergeant with 20+ years of investigative experience, preparing a comprehensive case report for law enforcement leadership, prosecutors, and stakeholders. This report will be used for case briefings, court proceedings, and decision-making.

REPORT OBJECTIVE:
Create a clear, authoritative narrative that synthesizes all available evidence into a coherent case assessment. Present findings with professional confidence while maintaining investigative objectivity.

WRITING STYLE:
- Professional law enforcement tone
- Clear, concise sentences accessible to non-technical readers
- Factual and evidence-based conclusions
- Avoid speculation - clearly distinguish between facts and assessments
- Use active voice and definitive statements where evidence supports them

REQUIRED STRUCTURE (Use these exact section headers):

**EXECUTIVE SUMMARY**
- 2-3 sentence overview of the case and primary findings
- Key conclusion about case resolution or status

**CASE TIMELINE** 
- Chronological sequence of key events with specific dates/times
- Focus on pivotal moments and turning points
- Include both confirmed facts and significant gaps

**KEY ACTORS**
- Primary suspects: background, involvement level, evidence connections
- Witnesses: credibility assessment and testimony summary  
- Victims: relevant background and circumstances
- Other significant persons: roles and relevance

**EVIDENCE ANALYSIS**
- Physical evidence: significance and forensic findings
- Digital evidence: communications, financial records, digital footprints
- Testimonial evidence: witness statements and reliability assessment
- Documentary evidence: reports, records, correspondence

**INVESTIGATIVE FINDINGS**
- Patterns identified across evidence sources
- Connections between actors, events, and locations
- Contradictions or inconsistencies requiring resolution
- Assessment of evidence strength (strong/moderate/weak)

**CONCLUSIONS AND RECOMMENDATIONS**
- Case resolution status (solved/unsolved/pending)
- Confidence level in findings (high/medium/low with justification)
- Recommended next steps or actions
- Areas requiring additional investigation

CRITICAL REQUIREMENTS:
- Maximum 800 words total
- Use only information from provided evidence and conversations
- No external research or assumptions beyond evidence
- Include specific details (names, dates, locations, amounts)
- Maintain objectivity - present facts, not theories
- End with clear assessment of case status and next steps
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, runner, supabase_client
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    supabase_client = SupabaseEvidenceClient()
    yield
    save_evidence()  # Save evidence on shutdown


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
        "evidence": global_evidence,  # Show ALL evidence items
        "message": f"Global evidence contains {len(global_evidence)} items"
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_evidence(request: UploadRequest):
    global global_evidence
    global_evidence.extend(request.evidence)
    save_evidence()  # Save to file immediately
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
        response = await runner.run(input=context, model=MODEL, mcp_servers=[PERPLEXITY_MCP_SERVER])

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

            print("\n" + "="*80)
            print("FULL CONTEXT BEING SENT TO MODEL:")
            print("="*80)
            print(context)
            print("="*80)
            print(f"Context length: {len(context)} characters")
            print(f"Evidence items: {len(global_evidence)}")
            print("="*80 + "\n")

            # Use the correct Dedalus streaming API with MCP servers
            result = runner.run(input=context, model=MODEL, stream=True, mcp_servers=[PERPLEXITY_MCP_SERVER])
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
        # Determine evidence source - prioritize provided evidence over case_id lookup
        if request.evidence and len(request.evidence) > 0:
            # Use provided evidence directly (sent from frontend)
            evidence = request.evidence
            print(f"Using provided evidence: {len(evidence)} items")
            
            # Get case information for context if case_id is provided
            if request.case_id:
                case_info = await supabase_client.get_case_info(request.case_id)
                if case_info:
                    case_context = f"Case: {case_info.get('title', 'Unknown')} - {case_info.get('description', 'No description')}"
                else:
                    case_context = f"Case ID: {request.case_id} (details not found in database)"
                    print(f"Warning: Case {request.case_id} not found in database, proceeding with analysis")
            else:
                case_context = ""
            
        elif request.case_id:
            # Fallback: Get evidence from Supabase for the specific case
            print(f"No evidence provided, retrieving from Supabase for case: {request.case_id}")
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
            # Last resort: use global evidence
            evidence = global_evidence
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
        
        # Wait for analysis to complete if create_notebook is true
        if request.create_notebook:
            # Wait a bit for agents to start, then poll for completion
            await asyncio.sleep(1)  # Let agents start
            
            max_wait_time = 120  # 2 minutes max
            check_interval = 2   # Check every 2 seconds
            waited = 0
            
            while waited < max_wait_time:
                current_run = global_orchestrator.get_run_status(run_id)
                if current_run and current_run.status in [OrchestrationStatus.COMPLETED, OrchestrationStatus.FAILED]:
                    break
                await asyncio.sleep(check_interval)
                waited += check_interval
            
            # Get final results
            final_run = global_orchestrator.get_run_status(run_id)
            if final_run and final_run.status == OrchestrationStatus.COMPLETED:
                notebook_cells = final_run.get_all_notebook_cells()
                return MultiAgentAnalysisResponse(
                    run_id=run_id,
                    status=final_run.status.value,
                    message=f"Completed analysis with {len(request.agent_types)} agents",
                    agent_count=len(request.agent_types),
                    websocket_url=f"/ws/multi-agent/{run_id}",
                    notebook_data=notebook_cells
                )
            else:
                # Analysis didn't complete in time or failed
                return MultiAgentAnalysisResponse(
                    run_id=run_id,
                    status=final_run.status.value if final_run else "timeout",
                    message=f"Analysis status: {final_run.status.value if final_run else 'timeout'}",
                    agent_count=len(request.agent_types),
                    websocket_url=f"/ws/multi-agent/{run_id}"
                )
        else:
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


@app.get("/workflow-analysis/{case_id}")
async def workflow_analysis_page(case_id: str, request: Request):
    """Serve workflow analysis page for monitoring multi-agent runs"""
    # Find active or recent runs for this case
    active_runs = []
    for run_id, run in global_orchestrator.active_runs.items():
        # Check if this run is related to the case (you might want to add case_id to run context)
        if run.context and run.context.get("case_id") == case_id:
            active_runs.append({
                "run_id": run_id,
                "status": run.status.value,
                "progress": run.progress,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "agents": [
                    {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type,
                        "agent_name": agent.agent_name,
                        "status": agent.status.value,
                        "progress": agent.progress,
                        "current_step": agent.current_step,
                        "error": agent.error
                    }
                    for agent in run.agents
                ]
            })
    
    # Check if this is an API request (JSON) or browser request (HTML)
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header:
        # Return JSON for API requests
        return {"runs": active_runs}
    
    # Return HTML page with real-time updates
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Agent Workflow Analysis - Case {case_id}</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {{
                theme: {{
                    extend: {{
                        animation: {{
                            'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        }}
                    }}
                }}
            }}
        </script>
        <style>
            .status-pending {{ background-color: #f3f4f6; color: #374151; }}
            .status-running {{ background-color: #dbeafe; color: #1e40af; }}
            .status-completed {{ background-color: #dcfce7; color: #166534; }}
            .status-failed {{ background-color: #fecaca; color: #dc2626; }}
            .status-cancelled {{ background-color: #f3f4f6; color: #6b7280; }}
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <div class="mb-8">
                <h1 class="text-3xl font-bold text-gray-900 mb-2">Multi-Agent Workflow Analysis</h1>
                <p class="text-gray-600">Real-time monitoring for Case ID: <span class="font-mono bg-gray-100 px-2 py-1 rounded">{case_id}</span></p>
            </div>
            
            <div id="runs-container">
                <!-- Runs will be populated here -->
            </div>
            
            <div id="no-runs" class="text-center py-12 bg-white rounded-lg shadow-sm border" style="display: none;">
                <div class="text-gray-400 text-6xl mb-4">🤖</div>
                <h2 class="text-xl font-semibold text-gray-700 mb-2">No Active Analysis Runs</h2>
                <p class="text-gray-500">Start a multi-agent analysis from the case view to see real-time progress here.</p>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket(runId) {{
                const wsUrl = `ws://localhost:8000/ws/multi-agent/${{runId}}`;
                ws = new WebSocket(wsUrl);
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    updateRunDisplay(data);
                }};
                
                ws.onerror = function(error) {{
                    console.log('WebSocket error:', error);
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket connection closed');
                    // Try to reconnect after 3 seconds
                    setTimeout(() => connectWebSocket(runId), 3000);
                }};
            }}
            
            function updateRunDisplay(data) {{
                // Update the UI with new data
                const runElement = document.getElementById(`run-${{data.run_id}}`);
                if (runElement) {{
                    // Update progress bars, status badges, etc.
                    const progressBar = runElement.querySelector('.progress-bar');
                    if (progressBar) {{
                        progressBar.style.width = `${{data.progress * 100}}%`;
                    }}
                    
                    const statusBadge = runElement.querySelector('.status-badge');
                    if (statusBadge) {{
                        statusBadge.textContent = data.status.toUpperCase();
                        statusBadge.className = `status-badge px-3 py-1 rounded-full text-xs font-medium status-${{data.status}}`;
                    }}
                }}
            }}
            
            function renderRuns(runs) {{
                const container = document.getElementById('runs-container');
                const noRuns = document.getElementById('no-runs');
                
                if (runs.length === 0) {{
                    container.style.display = 'none';
                    noRuns.style.display = 'block';
                    return;
                }}
                
                container.style.display = 'block';
                noRuns.style.display = 'none';
                
                container.innerHTML = runs.map(run => `
                    <div id="run-${{run.run_id}}" class="bg-white rounded-lg shadow-sm border mb-6 overflow-hidden">
                        <div class="px-6 py-4 border-b bg-gray-50">
                            <div class="flex items-center justify-between">
                                <h3 class="text-lg font-semibold text-gray-900">Analysis Run: ${{run.run_id}}</h3>
                                <div class="flex items-center space-x-3">
                                    <span class="status-badge px-3 py-1 rounded-full text-xs font-medium status-${{run.status}}">${{run.status.toUpperCase()}}</span>
                                    <span class="text-sm text-gray-500">${{Math.round(run.progress * 100)}}% Complete</span>
                                </div>
                            </div>
                            <div class="mt-3">
                                <div class="bg-gray-200 rounded-full h-2">
                                    <div class="progress-bar bg-blue-500 h-2 rounded-full transition-all duration-500" style="width: ${{run.progress * 100}}%"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="p-6">
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                ${{run.agents.map(agent => `
                                    <div class="border rounded-lg p-4">
                                        <div class="flex items-center justify-between mb-2">
                                            <h4 class="font-medium text-gray-900">${{agent.agent_name}}</h4>
                                            <span class="px-2 py-1 rounded text-xs font-medium status-${{agent.status}}">${{agent.status.toUpperCase()}}</span>
                                        </div>
                                        <p class="text-sm text-gray-600 mb-2">${{agent.current_step || 'Initializing...'}}</p>
                                        <div class="bg-gray-200 rounded-full h-1.5">
                                            <div class="bg-green-500 h-1.5 rounded-full transition-all duration-300" style="width: ${{agent.progress * 100}}%"></div>
                                        </div>
                                        ${{agent.error ? `<p class="text-xs text-red-600 mt-2">${{agent.error}}</p>` : ''}}
                                    </div>
                                `).join('')}}
                            </div>
                        </div>
                    </div>
                `).join('');
                
                // Connect WebSocket for the first active run
                const activeRun = runs.find(r => r.status === 'running');
                if (activeRun && !ws) {{
                    connectWebSocket(activeRun.run_id);
                }}
            }}
            
            // Initial load
            const initialRuns = {json.dumps(active_runs)};
            renderRuns(initialRuns);
            
            // Refresh every 5 seconds
            setInterval(async () => {{
                try {{
                    const response = await fetch(`/workflow-analysis/{case_id}`);
                    const data = await response.json();
                    if (data.runs) {{
                        renderRuns(data.runs);
                    }}
                }} catch (error) {{
                    console.error('Failed to refresh runs:', error);
                }}
            }}, 5000);
        </script>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


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
