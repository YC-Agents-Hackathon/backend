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
from pathlib import Path

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

# Evidence persistence
EVIDENCE_FILE = Path("evidence_storage.json")

def load_evidence():
    """Load evidence from file on startup"""
    global global_evidence
    if EVIDENCE_FILE.exists():
        try:
            with open(EVIDENCE_FILE, 'r') as f:
                data = json.load(f)
                global_evidence = data.get('evidence', [])
                print(f"Loaded {len(global_evidence)} evidence items from storage")
        except Exception as e:
            print(f"Error loading evidence: {e}")

def save_evidence():
    """Save evidence to file"""
    try:
        with open(EVIDENCE_FILE, 'w') as f:
            json.dump({'evidence': global_evidence}, f, indent=2)
        print(f"Saved {len(global_evidence)} evidence items to storage")
    except Exception as e:
        print(f"Error saving evidence: {e}")

# MCP Server Configuration for Perplexity
PERPLEXITY_MCP_SERVER = "akakak/sonar"

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
    global client, runner
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    load_evidence()  # Load evidence on startup
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
