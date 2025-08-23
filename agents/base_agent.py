"""
Base Agent Class for Detective AI System
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentOutput(BaseModel):
    agent_id: str
    agent_type: str
    status: AgentStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    outputs: Dict[str, Any] = {}
    notebook_cells: List[Dict[str, Any]] = []
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None


class BaseAgent(ABC):
    """Base class for all detective AI agents"""
    
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.status = AgentStatus.PENDING
        self.progress = 0.0
        self.current_step = "Initializing"
        self.outputs = {}
        self.notebook_cells = []
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.execution_time = None
        
        # Callbacks for real-time updates
        self.status_callback = None
        self.progress_callback = None
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Unique identifier for the agent type"""
        pass
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Human-readable name for the agent"""
        pass
    
    @property
    @abstractmethod
    def agent_description(self) -> str:
        """Description of what the agent does"""
        pass
    
    @abstractmethod
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        """Main execution method for the agent"""
        pass
    
    def set_callbacks(self, status_callback=None, progress_callback=None):
        """Set callbacks for real-time updates"""
        self.status_callback = status_callback
        self.progress_callback = progress_callback
    
    def update_status(self, status: AgentStatus, step: str = None):
        """Update agent status and notify callbacks"""
        self.status = status
        if step:
            self.current_step = step
            
        if self.status_callback:
            asyncio.create_task(self.status_callback(self.get_output()))
    
    def update_progress(self, progress: float, step: str = None):
        """Update progress and notify callbacks"""
        self.progress = max(0.0, min(1.0, progress))
        if step:
            self.current_step = step
            
        if self.progress_callback:
            asyncio.create_task(self.progress_callback(self.get_output()))
    
    def add_notebook_cell(self, cell_type: str, content: Dict[str, Any]):
        """Add a cell to the generated notebook"""
        cell = {
            "id": str(uuid.uuid4()),
            "type": cell_type,
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        }
        self.notebook_cells.append(cell)
    
    def get_output(self) -> AgentOutput:
        """Get current agent output state"""
        return AgentOutput(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            outputs=self.outputs,
            notebook_cells=self.notebook_cells,
            error=self.error,
            started_at=self.started_at,
            completed_at=self.completed_at,
            execution_time=self.execution_time
        )
    
    async def run(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        """Main run method with timing and error handling"""
        try:
            self.started_at = datetime.utcnow()
            self.update_status(AgentStatus.RUNNING, "Starting analysis")
            
            start_time = time.time()
            result = await self.execute(evidence, context)
            end_time = time.time()
            
            self.execution_time = end_time - start_time
            self.completed_at = datetime.utcnow()
            
            if self.status != AgentStatus.FAILED:
                self.update_status(AgentStatus.COMPLETED, "Analysis complete")
            
            return self.get_output()
            
        except Exception as e:
            self.error = str(e)
            self.completed_at = datetime.utcnow()
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise