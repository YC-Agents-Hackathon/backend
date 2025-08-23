"""
Specialized Detective AI Agents
"""

import asyncio
import json
from typing import Dict, List, Any
from dedalus_labs import DedalusRunner
from .base_agent import BaseAgent, AgentOutput, AgentStatus


class PatternRecognitionAgent(BaseAgent):
    """Agent specialized in identifying patterns and anomalies in data"""
    
    @property
    def agent_type(self) -> str:
        return "pattern_recognition"
    
    @property
    def agent_name(self) -> str:
        return "Pattern Recognition Agent"
    
    @property
    def agent_description(self) -> str:
        return "Analyzes data for suspicious patterns, anomalies, and trends"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are a forensic data analyst specializing in pattern recognition. 
Identify suspicious patterns, anomalies, and correlations in the provided data. 
Focus on statistical outliers, behavioral patterns, and temporal correlations. 
Present findings as bullet points with confidence scores (0-100%).

Format your response as JSON with the following structure:
{
    "patterns": [
        {"pattern": "description", "confidence": 85, "evidence": ["supporting evidence"]},
    ],
    "anomalies": [
        {"anomaly": "description", "severity": "high/medium/low", "details": "..."},
    ],
    "correlations": [
        {"variables": ["var1", "var2"], "strength": 0.85, "significance": "description"},
    ],
    "summary": "Brief summary of key findings"
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.2, "Analyzing patterns")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.8, "Processing results")
            
            # Try to parse JSON response
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in basic structure
                result = {
                    "patterns": [],
                    "anomalies": [],
                    "correlations": [],
                    "summary": response.final_output
                }
            
            self.outputs["analysis"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": f"# Pattern Recognition Analysis\n\n{result.get('summary', '')}"
            })
            
            if result.get("patterns"):
                patterns_content = "## Identified Patterns\n\n"
                for p in result["patterns"]:
                    patterns_content += f"- **{p.get('pattern', 'Unknown')}** (Confidence: {p.get('confidence', 0)}%)\n"
                    if p.get('evidence'):
                        patterns_content += f"  - Evidence: {', '.join(p['evidence'])}\n"
                
                self.add_notebook_cell("markdown", {"content": patterns_content})
            
            if result.get("anomalies"):
                anomalies_content = "## Anomalies Detected\n\n"
                for a in result["anomalies"]:
                    anomalies_content += f"- **{a.get('anomaly', 'Unknown')}** (Severity: {a.get('severity', 'unknown')})\n"
                    anomalies_content += f"  - {a.get('details', '')}\n"
                
                self.add_notebook_cell("markdown", {"content": anomalies_content})
            
            self.update_progress(1.0, "Pattern analysis complete")
            
        except Exception as e:
            self.error = f"Pattern analysis failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


class TimelineReconstructionAgent(BaseAgent):
    """Agent specialized in reconstructing chronological sequences"""
    
    @property
    def agent_type(self) -> str:
        return "timeline_reconstruction"
    
    @property
    def agent_name(self) -> str:
        return "Timeline Reconstruction Agent"
    
    @property
    def agent_description(self) -> str:
        return "Creates precise chronological sequences from evidence"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are a timeline reconstruction specialist. Create precise chronological sequences from evidence. 
Identify gaps, inconsistencies, and key temporal relationships. Output structured timelines with timestamps, events, and confidence levels.

Format your response as JSON:
{
    "timeline": [
        {"timestamp": "2024-01-01T10:00:00Z", "event": "description", "confidence": 90, "source": "evidence reference"},
    ],
    "gaps": [
        {"period": "2024-01-01 to 2024-01-02", "description": "Missing information about..."},
    ],
    "inconsistencies": [
        {"description": "Conflicting timestamps", "details": "..."},
    ],
    "key_periods": [
        {"start": "2024-01-01", "end": "2024-01-02", "significance": "Critical period because..."},
    ]
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.3, "Reconstructing timeline")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.8, "Analyzing temporal relationships")
            
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                result = {"timeline": [], "gaps": [], "inconsistencies": [], "summary": response.final_output}
            
            self.outputs["timeline"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": "# Timeline Reconstruction\n\nChronological analysis of events based on available evidence."
            })
            
            if result.get("timeline"):
                timeline_content = "## Event Timeline\n\n"
                for event in result["timeline"]:
                    timeline_content += f"**{event.get('timestamp', 'Unknown time')}**: {event.get('event', '')}\n"
                    timeline_content += f"- Confidence: {event.get('confidence', 0)}%\n"
                    timeline_content += f"- Source: {event.get('source', 'Unknown')}\n\n"
                
                self.add_notebook_cell("markdown", {"content": timeline_content})
            
            # Add gaps and inconsistencies
            if result.get("gaps") or result.get("inconsistencies"):
                issues_content = "## Timeline Issues\n\n"
                
                if result.get("gaps"):
                    issues_content += "### Information Gaps\n"
                    for gap in result["gaps"]:
                        issues_content += f"- **{gap.get('period', '')}**: {gap.get('description', '')}\n"
                
                if result.get("inconsistencies"):
                    issues_content += "\n### Inconsistencies\n"
                    for inc in result["inconsistencies"]:
                        issues_content += f"- **{inc.get('description', '')}**: {inc.get('details', '')}\n"
                
                self.add_notebook_cell("markdown", {"content": issues_content})
            
            self.update_progress(1.0, "Timeline reconstruction complete")
            
        except Exception as e:
            self.error = f"Timeline reconstruction failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


class EntityRelationshipAgent(BaseAgent):
    """Agent specialized in mapping entity relationships"""
    
    @property
    def agent_type(self) -> str:
        return "entity_relationship"
    
    @property
    def agent_name(self) -> str:
        return "Entity Relationship Agent"
    
    @property
    def agent_description(self) -> str:
        return "Maps connections between people, organizations, locations, and assets"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are a relationship mapping expert. Identify and map all entities (people, organizations, locations, assets) and their relationships. 
Create network graphs showing connection strength and relationship types. Flag suspicious associations.

Format your response as JSON:
{
    "entities": {
        "people": [{"name": "John Doe", "role": "suspect", "attributes": {...}}],
        "organizations": [{"name": "Company ABC", "type": "corporation", "attributes": {...}}],
        "locations": [{"name": "123 Main St", "type": "address", "attributes": {...}}],
        "assets": [{"name": "Bank Account", "type": "financial", "attributes": {...}}]
    },
    "relationships": [
        {"from": "John Doe", "to": "Company ABC", "type": "employee", "strength": 0.9, "suspicious": false},
    ],
    "networks": [
        {"name": "Financial Network", "entities": ["..."], "description": "..."},
    ],
    "suspicious_connections": [
        {"description": "Unusual connection between...", "risk_level": "high"},
    ]
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.25, "Identifying entities")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.7, "Mapping relationships")
            
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                result = {"entities": {}, "relationships": [], "networks": [], "summary": response.final_output}
            
            self.outputs["entity_map"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": "# Entity Relationship Analysis\n\nMapping of entities and their connections in the case."
            })
            
            # Entities summary
            if result.get("entities"):
                entities_content = "## Identified Entities\n\n"
                for entity_type, entities in result["entities"].items():
                    if entities:
                        entities_content += f"### {entity_type.title()}\n"
                        for entity in entities:
                            entities_content += f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'unknown')})\n"
                
                self.add_notebook_cell("markdown", {"content": entities_content})
            
            # Relationships
            if result.get("relationships"):
                rel_content = "## Key Relationships\n\n"
                for rel in result["relationships"]:
                    rel_content += f"- **{rel.get('from', '')}** → **{rel.get('to', '')}**\n"
                    rel_content += f"  - Type: {rel.get('type', '')}\n"
                    rel_content += f"  - Strength: {rel.get('strength', 0)}\n"
                    if rel.get('suspicious'):
                        rel_content += f"  - ⚠️ Flagged as suspicious\n"
                    rel_content += "\n"
                
                self.add_notebook_cell("markdown", {"content": rel_content})
            
            self.update_progress(1.0, "Entity mapping complete")
            
        except Exception as e:
            self.error = f"Entity relationship analysis failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


class FinancialAnalysisAgent(BaseAgent):
    """Agent specialized in financial forensics"""
    
    @property
    def agent_type(self) -> str:
        return "financial_analysis"
    
    @property
    def agent_name(self) -> str:
        return "Financial Analysis Agent"
    
    @property
    def agent_description(self) -> str:
        return "Analyzes financial transactions, flows, and anomalies"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are a financial forensics specialist. Analyze transaction patterns, cash flows, and financial anomalies. 
Identify potential money laundering, fraud indicators, and suspicious financial behaviors. Quantify risk levels.

Format your response as JSON:
{
    "transactions": [
        {"id": "tx123", "amount": 10000, "date": "2024-01-01", "risk_score": 0.8, "flags": ["large_amount", "unusual_timing"]},
    ],
    "flow_analysis": {
        "total_inflow": 100000,
        "total_outflow": 95000,
        "net_flow": 5000,
        "unusual_patterns": ["..."]
    },
    "risk_indicators": [
        {"indicator": "Structuring", "description": "Multiple transactions below reporting threshold", "risk": "high"},
    ],
    "money_laundering_indicators": [
        {"pattern": "Layering", "confidence": 0.75, "description": "..."},
    ],
    "recommendations": [
        "Further investigation needed for transactions above $10k"
    ]
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.3, "Analyzing transactions")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.8, "Identifying financial risks")
            
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                result = {"transactions": [], "flow_analysis": {}, "risk_indicators": [], "summary": response.final_output}
            
            self.outputs["financial_analysis"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": "# Financial Analysis\n\nForensic analysis of financial transactions and patterns."
            })
            
            # Flow analysis summary
            if result.get("flow_analysis"):
                flow = result["flow_analysis"]
                flow_content = f"""## Cash Flow Summary

- **Total Inflow**: ${flow.get('total_inflow', 0):,.2f}
- **Total Outflow**: ${flow.get('total_outflow', 0):,.2f}
- **Net Flow**: ${flow.get('net_flow', 0):,.2f}

"""
                if flow.get("unusual_patterns"):
                    flow_content += "### Unusual Patterns\n"
                    for pattern in flow["unusual_patterns"]:
                        flow_content += f"- {pattern}\n"
                
                self.add_notebook_cell("markdown", {"content": flow_content})
            
            # Risk indicators
            if result.get("risk_indicators"):
                risk_content = "## Risk Indicators\n\n"
                for indicator in result["risk_indicators"]:
                    risk_level = indicator.get('risk', 'unknown').upper()
                    risk_content += f"- **{indicator.get('indicator', '')}** ({risk_level})\n"
                    risk_content += f"  - {indicator.get('description', '')}\n"
                
                self.add_notebook_cell("markdown", {"content": risk_content})
            
            self.update_progress(1.0, "Financial analysis complete")
            
        except Exception as e:
            self.error = f"Financial analysis failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


class CommunicationAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing communications"""
    
    @property
    def agent_type(self) -> str:
        return "communication_analysis"
    
    @property
    def agent_name(self) -> str:
        return "Communication Analysis Agent"
    
    @property
    def agent_description(self) -> str:
        return "Processes communications for insights, sentiment, and deception indicators"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are a communication analysis expert. Extract key information from communications, identify sentiment changes, 
detect deception indicators, and map communication networks. Highlight critical conversations and code words.

Format your response as JSON:
{
    "key_communications": [
        {"id": "msg123", "date": "2024-01-01", "participants": ["A", "B"], "summary": "...", "importance": "high"},
    ],
    "sentiment_analysis": [
        {"period": "2024-01-01 to 2024-01-07", "trend": "negative", "key_events": ["..."]},
    ],
    "deception_indicators": [
        {"message_id": "msg123", "indicators": ["evasion", "contradiction"], "confidence": 0.8},
    ],
    "code_words": [
        {"word": "meeting", "likely_meaning": "illegal transaction", "frequency": 15},
    ],
    "communication_network": [
        {"from": "Person A", "to": "Person B", "frequency": 25, "urgency": "high"},
    ]
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.35, "Analyzing communications")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.8, "Detecting patterns")
            
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                result = {"key_communications": [], "sentiment_analysis": [], "deception_indicators": [], "summary": response.final_output}
            
            self.outputs["communication_analysis"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": "# Communication Analysis\n\nAnalysis of communication patterns, sentiment, and deception indicators."
            })
            
            # Key communications
            if result.get("key_communications"):
                comm_content = "## Critical Communications\n\n"
                for comm in result["key_communications"]:
                    importance = comm.get('importance', 'medium').upper()
                    comm_content += f"**{comm.get('date', '')}** - {', '.join(comm.get('participants', []))} ({importance})\n"
                    comm_content += f"- {comm.get('summary', '')}\n\n"
                
                self.add_notebook_cell("markdown", {"content": comm_content})
            
            # Deception indicators
            if result.get("deception_indicators"):
                deception_content = "## Deception Indicators\n\n"
                for indicator in result["deception_indicators"]:
                    deception_content += f"- **Message {indicator.get('message_id', '')}** (Confidence: {indicator.get('confidence', 0)*100:.0f}%)\n"
                    deception_content += f"  - Indicators: {', '.join(indicator.get('indicators', []))}\n"
                
                self.add_notebook_cell("markdown", {"content": deception_content})
            
            self.update_progress(1.0, "Communication analysis complete")
            
        except Exception as e:
            self.error = f"Communication analysis failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


class EvidenceValidationAgent(BaseAgent):
    """Agent specialized in evidence validation"""
    
    @property
    def agent_type(self) -> str:
        return "evidence_validation"
    
    @property
    def agent_name(self) -> str:
        return "Evidence Validation Agent"
    
    @property
    def agent_description(self) -> str:
        return "Cross-references and validates evidence integrity and reliability"
    
    async def execute(self, evidence: List[str], context: Dict[str, Any]) -> AgentOutput:
        runner = context.get("runner")
        if not runner:
            raise ValueError("DedalusRunner not provided in context")
        
        prompt = """You are an evidence validation specialist. Cross-reference evidence for consistency, identify contradictions, 
assess reliability, and flag potential fabricated or tampered evidence. Provide credibility scores.

Format your response as JSON:
{
    "evidence_items": [
        {"id": "ev123", "type": "document", "credibility_score": 0.9, "reliability": "high", "issues": []},
    ],
    "contradictions": [
        {"items": ["ev123", "ev456"], "description": "Conflicting statements about...", "severity": "medium"},
    ],
    "corroborations": [
        {"items": ["ev123", "ev789"], "description": "Both sources confirm...", "strength": "strong"},
    ],
    "tampering_indicators": [
        {"evidence_id": "ev123", "indicators": ["metadata_inconsistency"], "likelihood": 0.3},
    ],
    "reliability_assessment": {
        "overall_score": 0.85,
        "high_reliability_count": 15,
        "medium_reliability_count": 3,
        "low_reliability_count": 1
    }
}

EVIDENCE:
""" + "\n".join(evidence)
        
        self.update_progress(0.4, "Validating evidence")
        
        try:
            response = await runner.run(input=prompt, model="openai/gpt-4o")
            self.update_progress(0.8, "Cross-referencing sources")
            
            try:
                result = json.loads(response.final_output)
            except json.JSONDecodeError:
                result = {"evidence_items": [], "contradictions": [], "corroborations": [], "summary": response.final_output}
            
            self.outputs["validation_report"] = result
            
            # Create notebook cells
            self.add_notebook_cell("markdown", {
                "content": "# Evidence Validation Report\n\nAssessment of evidence reliability, consistency, and integrity."
            })
            
            # Overall assessment
            if result.get("reliability_assessment"):
                assessment = result["reliability_assessment"]
                assessment_content = f"""## Overall Reliability Assessment

**Overall Score**: {assessment.get('overall_score', 0)*100:.1f}%

- High Reliability: {assessment.get('high_reliability_count', 0)} items
- Medium Reliability: {assessment.get('medium_reliability_count', 0)} items  
- Low Reliability: {assessment.get('low_reliability_count', 0)} items
"""
                self.add_notebook_cell("markdown", {"content": assessment_content})
            
            # Contradictions
            if result.get("contradictions"):
                contradiction_content = "## Evidence Contradictions\n\n"
                for contradiction in result["contradictions"]:
                    severity = contradiction.get('severity', 'unknown').upper()
                    contradiction_content += f"**{severity} SEVERITY**\n"
                    contradiction_content += f"- Items: {', '.join(contradiction.get('items', []))}\n"
                    contradiction_content += f"- Issue: {contradiction.get('description', '')}\n\n"
                
                self.add_notebook_cell("markdown", {"content": contradiction_content})
            
            # Tampering indicators
            if result.get("tampering_indicators"):
                tampering_content = "## Potential Tampering Indicators\n\n"
                for indicator in result["tampering_indicators"]:
                    likelihood = indicator.get('likelihood', 0) * 100
                    tampering_content += f"- **Evidence {indicator.get('evidence_id', '')}** (Likelihood: {likelihood:.0f}%)\n"
                    tampering_content += f"  - Indicators: {', '.join(indicator.get('indicators', []))}\n"
                
                self.add_notebook_cell("markdown", {"content": tampering_content})
            
            self.update_progress(1.0, "Evidence validation complete")
            
        except Exception as e:
            self.error = f"Evidence validation failed: {str(e)}"
            self.update_status(AgentStatus.FAILED, f"Error: {str(e)}")
            raise
        
        return self.get_output()


# Registry of all available agents
AGENT_REGISTRY = {
    "pattern_recognition": PatternRecognitionAgent,
    "timeline_reconstruction": TimelineReconstructionAgent,
    "entity_relationship": EntityRelationshipAgent,
    "financial_analysis": FinancialAnalysisAgent,
    "communication_analysis": CommunicationAnalysisAgent,
    "evidence_validation": EvidenceValidationAgent,
}