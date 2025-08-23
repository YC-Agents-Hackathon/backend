"""
Multi-Agent Orchestration System
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum

from .base_agent import BaseAgent, AgentStatus, AgentOutput
from .detective_agents import AGENT_REGISTRY


class OrchestrationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentOrchestrator:
    """Manages multiple agents and their execution"""
    
    def __init__(self, max_concurrent_agents: int = 6):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_runs: Dict[str, "MultiAgentRun"] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        
    async def start_multi_agent_analysis(
        self,
        run_id: str,
        evidence: List[str],
        agent_types: List[str],
        context: Dict[str, Any],
        status_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> "MultiAgentRun":
        """Start a multi-agent analysis run"""
        
        # Validate agent types
        invalid_agents = [agent for agent in agent_types if agent not in AGENT_REGISTRY]
        if invalid_agents:
            raise ValueError(f"Unknown agent types: {invalid_agents}")
        
        # Create agent instances
        agents = []
        for agent_type in agent_types:
            agent_class = AGENT_REGISTRY[agent_type]
            agent = agent_class()
            
            # Set up callbacks for real-time updates
            if status_callback:
                agent.set_callbacks(status_callback=status_callback)
            if progress_callback:
                agent.set_callbacks(progress_callback=progress_callback)
                
            agents.append(agent)
            self.agent_instances[agent.agent_id] = agent
        
        # Create multi-agent run
        run = MultiAgentRun(
            run_id=run_id,
            agents=agents,
            evidence=evidence,
            context=context,
            max_concurrent=self.max_concurrent_agents
        )
        
        self.active_runs[run_id] = run
        
        # Start execution
        asyncio.create_task(run.execute())
        
        return run
    
    def get_run_status(self, run_id: str) -> Optional["MultiAgentRun"]:
        """Get status of a multi-agent run"""
        return self.active_runs.get(run_id)
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentOutput]:
        """Get status of a specific agent"""
        agent = self.agent_instances.get(agent_id)
        return agent.get_output() if agent else None
    
    async def cancel_run(self, run_id: str):
        """Cancel a multi-agent run"""
        run = self.active_runs.get(run_id)
        if run:
            await run.cancel()
    
    def cleanup_completed_runs(self):
        """Clean up completed runs to free memory"""
        completed_runs = [
            run_id for run_id, run in self.active_runs.items()
            if run.status in [OrchestrationStatus.COMPLETED, OrchestrationStatus.FAILED, OrchestrationStatus.CANCELLED]
        ]
        
        for run_id in completed_runs:
            run = self.active_runs[run_id]
            # Remove agent instances
            for agent in run.agents:
                self.agent_instances.pop(agent.agent_id, None)
            # Remove run
            del self.active_runs[run_id]


class MultiAgentRun:
    """Represents a single multi-agent analysis run"""
    
    def __init__(
        self,
        run_id: str,
        agents: List[BaseAgent],
        evidence: List[str],
        context: Dict[str, Any],
        max_concurrent: int = 6
    ):
        self.run_id = run_id
        self.agents = agents
        self.evidence = evidence
        self.context = context
        self.max_concurrent = max_concurrent
        
        self.status = OrchestrationStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0.0
        
        self.agent_results: Dict[str, AgentOutput] = {}
        self.failed_agents: Set[str] = set()
        self.completed_agents: Set[str] = set()
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cancelled = False
        
    async def execute(self):
        """Execute all agents concurrently"""
        try:
            self.status = OrchestrationStatus.RUNNING
            self.started_at = datetime.utcnow()
            
            # Create tasks for all agents
            tasks = []
            for agent in self.agents:
                task = asyncio.create_task(self._run_agent(agent))
                tasks.append(task)
            
            # Wait for all agents to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                agent = self.agents[i]
                if isinstance(result, Exception):
                    self.failed_agents.add(agent.agent_id)
                    print(f"Agent {agent.agent_id} failed: {result}")
                else:
                    self.agent_results[agent.agent_id] = result
                    self.completed_agents.add(agent.agent_id)
            
            # Determine final status
            if self._cancelled:
                self.status = OrchestrationStatus.CANCELLED
            elif self.failed_agents:
                self.status = OrchestrationStatus.FAILED if len(self.failed_agents) == len(self.agents) else OrchestrationStatus.COMPLETED
            else:
                self.status = OrchestrationStatus.COMPLETED
            
            self.completed_at = datetime.utcnow()
            self.progress = 1.0
            
        except Exception as e:
            self.status = OrchestrationStatus.FAILED
            self.completed_at = datetime.utcnow()
            print(f"Multi-agent run {self.run_id} failed: {e}")
    
    async def _run_agent(self, agent: BaseAgent) -> AgentOutput:
        """Run a single agent with concurrency control"""
        async with self._semaphore:
            if self._cancelled:
                agent.update_status(AgentStatus.CANCELLED)
                return agent.get_output()
            
            try:
                result = await agent.run(self.evidence, self.context)
                self._update_overall_progress()
                return result
            except Exception as e:
                agent.error = str(e)
                agent.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
                self._update_overall_progress()
                raise
    
    def _update_overall_progress(self):
        """Update overall progress based on agent completion"""
        total_agents = len(self.agents)
        completed_count = len(self.completed_agents) + len(self.failed_agents)
        self.progress = completed_count / total_agents if total_agents > 0 else 0.0
    
    async def cancel(self):
        """Cancel the multi-agent run"""
        self._cancelled = True
        self.status = OrchestrationStatus.CANCELLED
        
        # Cancel all pending agents
        for agent in self.agents:
            if agent.status == AgentStatus.RUNNING:
                agent.update_status(AgentStatus.CANCELLED)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the multi-agent run"""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_agents": len(self.agents),
            "completed_agents": len(self.completed_agents),
            "failed_agents": len(self.failed_agents),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "status": agent.status.value,
                    "progress": agent.progress,
                    "current_step": agent.current_step,
                    "error": agent.error
                }
                for agent in self.agents
            ]
        }
    
    def get_all_notebook_cells(self) -> List[Dict[str, Any]]:
        """Get all notebook cells from all completed agents"""
        all_cells = []
        
        # Add overview cell
        all_cells.append({
            "id": str(uuid.uuid4()),
            "type": "markdown",
            "content": {
                "content": f"""# Multi-Agent Investigation Analysis

**Run ID**: {self.run_id}
**Started**: {self.started_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self.started_at else 'Unknown'}
**Status**: {self.status.value.upper()}
**Agents**: {len(self.agents)} total, {len(self.completed_agents)} completed, {len(self.failed_agents)} failed

## Analysis Overview
This notebook contains the results from multiple specialized AI agents analyzing the provided evidence.
Each agent focused on a specific aspect of the investigation.
"""
            },
            "created_at": datetime.utcnow().isoformat()
        })
        
        # Add cells from each successful agent
        for agent in self.agents:
            if agent.agent_id in self.completed_agents:
                # Add agent header
                all_cells.append({
                    "id": str(uuid.uuid4()),
                    "type": "markdown",
                    "content": {
                        "content": f"""---

# {agent.agent_name}

**Agent Type**: {agent.agent_type}
**Status**: {agent.status.value.upper()}
**Execution Time**: {agent.execution_time:.2f if agent.execution_time else 'Unknown'}s

{agent.agent_description}
"""
                    },
                    "created_at": datetime.utcnow().isoformat()
                })
                
                # Add agent's notebook cells
                all_cells.extend(agent.notebook_cells)
        
        return all_cells


# Global orchestrator instance
global_orchestrator = AgentOrchestrator()