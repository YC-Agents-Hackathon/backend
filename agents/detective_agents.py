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
            
            # Create comprehensive markdown content
            full_content = f"## Pattern Recognition Analysis\n\n**Summary**: {result.get('summary', 'No summary available')}\n\n"
            
            if result.get("patterns"):
                full_content += "### 🔍 Identified Patterns\n\n"
                for p in result["patterns"]:
                    full_content += f"**Pattern**: {p.get('pattern', 'Unknown')} \n"
                    full_content += f"**Confidence**: {p.get('confidence', 0)}% \n"
                    if p.get('evidence'):
                        full_content += f"**Evidence**: {', '.join(p['evidence'])} \n"
                    full_content += "\n"
            
            if result.get("anomalies"):
                full_content += "### ⚠️ Anomalies Detected\n\n"
                for a in result["anomalies"]:
                    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(a.get('severity', 'unknown'), "⚪")
                    full_content += f"{severity_emoji} **{a.get('anomaly', 'Unknown')}** (Severity: {a.get('severity', 'unknown')})\n"
                    full_content += f"   {a.get('details', '')}\n\n"
            
            if result.get("correlations"):
                full_content += "### 📊 Correlations\n\n"
                for c in result["correlations"]:
                    full_content += f"**Variables**: {', '.join(c.get('variables', []))}\n"
                    full_content += f"**Strength**: {c.get('strength', 0)}\n"
                    full_content += f"**Significance**: {c.get('significance', '')}\n\n"
            
            # Add single comprehensive cell instead of multiple cells
            self.add_notebook_cell("markdown", {"content": full_content})
            
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
            
            # Create comprehensive timeline analysis
            full_content = "## Timeline Reconstruction\n\n**Analysis**: Chronological sequence of events based on available evidence.\n\n"
            
            if result.get("timeline"):
                full_content += "### 📅 Event Timeline\n\n"
                for i, event in enumerate(result["timeline"], 1):
                    timestamp = event.get('timestamp', 'Unknown time')
                    confidence = event.get('confidence', 0)
                    confidence_indicator = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
                    
                    full_content += f"**{i}.** `{timestamp}` {confidence_indicator}\n"
                    full_content += f"   **Event**: {event.get('event', 'No description')}\n"
                    full_content += f"   **Confidence**: {confidence}% | **Source**: {event.get('source', 'Unknown')}\n\n"
            
            if result.get("key_periods"):
                full_content += "### ⭐ Key Time Periods\n\n"
                for period in result["key_periods"]:
                    full_content += f"**{period.get('start', '')} → {period.get('end', '')}**\n"
                    full_content += f"   {period.get('significance', '')}\n\n"
            
            if result.get("gaps"):
                full_content += "### ❓ Information Gaps\n\n"
                for gap in result["gaps"]:
                    full_content += f"**Period**: {gap.get('period', '')}\n"
                    full_content += f"**Gap**: {gap.get('description', '')}\n\n"
            
            if result.get("inconsistencies"):
                full_content += "### ⚠️ Timeline Inconsistencies\n\n"
                for inc in result["inconsistencies"]:
                    full_content += f"**Issue**: {inc.get('description', '')}\n"
                    full_content += f"**Details**: {inc.get('details', '')}\n\n"
            
            # Add single comprehensive cell
            self.add_notebook_cell("markdown", {"content": full_content})
            
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
            
            # Create comprehensive entity relationship analysis
            full_content = "## Entity Relationship Analysis\n\n**Analysis**: Network mapping of entities and their connections in the investigation.\n\n"
            
            if result.get("entities"):
                full_content += "### 👥 Identified Entities\n\n"
                for entity_type, entities in result["entities"].items():
                    if entities:
                        type_emoji = {"people": "👤", "organizations": "🏢", "locations": "📍", "assets": "💰"}.get(entity_type, "📋")
                        full_content += f"**{type_emoji} {entity_type.title()}**\n"
                        for entity in entities:
                            name = entity.get('name', 'Unknown')
                            entity_type_detail = entity.get('type', 'unknown')
                            role = entity.get('role', '')
                            full_content += f"• **{name}** ({entity_type_detail})"
                            if role:
                                full_content += f" - {role}"
                            full_content += "\n"
                        full_content += "\n"
            
            if result.get("relationships"):
                full_content += "### 🔗 Key Relationships\n\n"
                for rel in result["relationships"]:
                    from_entity = rel.get('from', '')
                    to_entity = rel.get('to', '')
                    rel_type = rel.get('type', '')
                    strength = rel.get('strength', 0) * 100
                    suspicious = rel.get('suspicious', False)
                    
                    strength_indicator = "🟢" if strength >= 80 else "🟡" if strength >= 60 else "🔴"
                    suspicious_flag = " ⚠️" if suspicious else ""
                    
                    full_content += f"**{from_entity}** → **{to_entity}**{suspicious_flag}\n"
                    full_content += f"   **Type**: {rel_type} | **Strength**: {strength:.0f}% {strength_indicator}\n\n"
            
            if result.get("networks"):
                full_content += "### 🕸️ Network Analysis\n\n"
                for network in result["networks"]:
                    full_content += f"**{network.get('name', 'Unknown Network')}**\n"
                    full_content += f"   {network.get('description', 'No description')}\n"
                    entities = network.get('entities', [])
                    if entities:
                        full_content += f"   **Members**: {', '.join(entities[:5])}"
                        if len(entities) > 5:
                            full_content += f" (+{len(entities)-5} more)"
                    full_content += "\n\n"
            
            if result.get("suspicious_connections"):
                full_content += "### 🚨 Suspicious Connections\n\n"
                for conn in result["suspicious_connections"]:
                    risk_level = conn.get('risk_level', 'unknown').upper()
                    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
                    full_content += f"{risk_emoji} **{risk_level} RISK**: {conn.get('description', 'No description')}\n\n"
            
            # Add single comprehensive cell
            self.add_notebook_cell("markdown", {"content": full_content})
            
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
            
            # Create comprehensive financial analysis
            full_content = "## Financial Analysis\n\n**Analysis**: Forensic examination of financial transactions, flows, and anomalies.\n\n"
            
            if result.get("flow_analysis"):
                flow = result["flow_analysis"]
                full_content += "### 💰 Cash Flow Summary\n\n"
                
                total_inflow = flow.get('total_inflow', 0)
                total_outflow = flow.get('total_outflow', 0)
                net_flow = flow.get('net_flow', 0)
                
                flow_indicator = "🟢" if net_flow > 0 else "🔴" if net_flow < 0 else "🟡"
                
                full_content += f"**Total Inflow**: ${total_inflow:,.2f}\n"
                full_content += f"**Total Outflow**: ${total_outflow:,.2f}\n"
                full_content += f"**Net Flow**: ${net_flow:,.2f} {flow_indicator}\n\n"
                
                if flow.get("unusual_patterns"):
                    full_content += "**⚠️ Unusual Patterns Detected:**\n"
                    for pattern in flow["unusual_patterns"]:
                        full_content += f"• {pattern}\n"
                    full_content += "\n"
            
            if result.get("transactions"):
                high_risk_txns = [tx for tx in result["transactions"] if tx.get('risk_score', 0) >= 0.7]
                if high_risk_txns:
                    full_content += "### 🚨 High-Risk Transactions\n\n"
                    for tx in high_risk_txns[:10]:  # Limit to top 10
                        amount = tx.get('amount', 0)
                        date = tx.get('date', 'Unknown')
                        risk_score = tx.get('risk_score', 0) * 100
                        flags = tx.get('flags', [])
                        
                        full_content += f"**${amount:,.2f}** on `{date}` (Risk: {risk_score:.0f}%)\n"
                        if flags:
                            full_content += f"   🚩 Flags: {', '.join(flags)}\n"
                        full_content += "\n"
            
            if result.get("risk_indicators"):
                full_content += "### 🔍 Risk Indicators\n\n"
                for indicator in result["risk_indicators"]:
                    risk_level = indicator.get('risk', 'unknown').upper()
                    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(risk_level, "⚪")
                    
                    full_content += f"{risk_emoji} **{indicator.get('indicator', 'Unknown')}** ({risk_level})\n"
                    full_content += f"   {indicator.get('description', 'No description')}\n\n"
            
            if result.get("money_laundering_indicators"):
                full_content += "### 💵 Money Laundering Indicators\n\n"
                for ml in result["money_laundering_indicators"]:
                    confidence = ml.get('confidence', 0) * 100
                    confidence_indicator = "🟢" if confidence >= 75 else "🟡" if confidence >= 50 else "🔴"
                    
                    full_content += f"**{ml.get('pattern', 'Unknown Pattern')}** {confidence_indicator}\n"
                    full_content += f"   **Confidence**: {confidence:.0f}%\n"
                    full_content += f"   **Details**: {ml.get('description', 'No details')}\n\n"
            
            if result.get("recommendations"):
                full_content += "### 📝 Recommendations\n\n"
                for rec in result["recommendations"]:
                    full_content += f"• {rec}\n"
                full_content += "\n"
            
            # Add single comprehensive cell
            self.add_notebook_cell("markdown", {"content": full_content})
            
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
            
            # Create comprehensive communication analysis
            full_content = "## Communication Analysis\n\n**Analysis**: Deep examination of communication patterns, sentiment trends, and deception indicators.\n\n"
            
            if result.get("key_communications"):
                full_content += "### 💬 Critical Communications\n\n"
                for comm in result["key_communications"]:
                    importance = comm.get('importance', 'medium').upper()
                    importance_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(importance, "⚪")
                    
                    date = comm.get('date', 'Unknown date')
                    participants = ', '.join(comm.get('participants', []))
                    summary = comm.get('summary', 'No summary')
                    
                    full_content += f"{importance_emoji} **{date}** - {participants} ({importance})\n"
                    full_content += f"   {summary}\n\n"
            
            if result.get("sentiment_analysis"):
                full_content += "### 📊 Sentiment Analysis\n\n"
                for sentiment in result["sentiment_analysis"]:
                    period = sentiment.get('period', 'Unknown period')
                    trend = sentiment.get('trend', 'unknown')
                    trend_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(trend, "⚪")
                    
                    full_content += f"**{period}** {trend_emoji}\n"
                    full_content += f"   **Trend**: {trend.title()}\n"
                    key_events = sentiment.get('key_events', [])
                    if key_events:
                        full_content += f"   **Key Events**: {', '.join(key_events)}\n"
                    full_content += "\n"
            
            if result.get("deception_indicators"):
                full_content += "### 🔍 Deception Indicators\n\n"
                for indicator in result["deception_indicators"]:
                    message_id = indicator.get('message_id', 'Unknown')
                    confidence = indicator.get('confidence', 0) * 100
                    confidence_indicator = "🟢" if confidence >= 75 else "🟡" if confidence >= 50 else "🔴"
                    indicators = indicator.get('indicators', [])
                    
                    full_content += f"**Message {message_id}** {confidence_indicator}\n"
                    full_content += f"   **Confidence**: {confidence:.0f}%\n"
                    if indicators:
                        full_content += f"   **Indicators**: {', '.join(indicators)}\n"
                    full_content += "\n"
            
            if result.get("code_words"):
                full_content += "### 🔑 Suspected Code Words\n\n"
                for code_word in result["code_words"]:
                    word = code_word.get('word', 'Unknown')
                    meaning = code_word.get('likely_meaning', 'Unknown meaning')
                    frequency = code_word.get('frequency', 0)
                    
                    full_content += f"**\"{word}\"** → *{meaning}* (Used {frequency} times)\n"
                full_content += "\n"
            
            if result.get("communication_network"):
                full_content += "### 🕸️ Communication Network\n\n"
                for conn in result["communication_network"]:
                    from_person = conn.get('from', 'Unknown')
                    to_person = conn.get('to', 'Unknown')
                    frequency = conn.get('frequency', 0)
                    urgency = conn.get('urgency', 'unknown').upper()
                    urgency_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(urgency, "⚪")
                    
                    full_content += f"**{from_person}** → **{to_person}** {urgency_emoji}\n"
                    full_content += f"   **Frequency**: {frequency} messages | **Urgency**: {urgency}\n\n"
            
            # Add single comprehensive cell
            self.add_notebook_cell("markdown", {"content": full_content})
            
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
            
            # Create comprehensive evidence validation report
            full_content = "## Evidence Validation Report\n\n**Analysis**: Cross-reference validation of evidence integrity, reliability, and authenticity.\n\n"
            
            if result.get("reliability_assessment"):
                assessment = result["reliability_assessment"]
                overall_score = assessment.get('overall_score', 0) * 100
                score_indicator = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
                
                high_count = assessment.get('high_reliability_count', 0)
                medium_count = assessment.get('medium_reliability_count', 0)
                low_count = assessment.get('low_reliability_count', 0)
                
                full_content += f"### 📉 Overall Reliability Assessment\n\n"
                full_content += f"**Overall Score**: {overall_score:.1f}% {score_indicator}\n\n"
                full_content += f"🟢 **High Reliability**: {high_count} items\n"
                full_content += f"🟡 **Medium Reliability**: {medium_count} items\n"
                full_content += f"🔴 **Low Reliability**: {low_count} items\n\n"
            
            if result.get("evidence_items"):
                high_cred_items = [item for item in result["evidence_items"] if item.get('credibility_score', 0) >= 0.8]
                low_cred_items = [item for item in result["evidence_items"] if item.get('credibility_score', 0) < 0.6]
                
                if high_cred_items:
                    full_content += "### ✅ High-Credibility Evidence\n\n"
                    for item in high_cred_items[:5]:  # Top 5
                        score = item.get('credibility_score', 0) * 100
                        full_content += f"**{item.get('id', 'Unknown')}** ({item.get('type', 'unknown')}) - {score:.0f}%\n"
                    full_content += "\n"
                
                if low_cred_items:
                    full_content += "### ⚠️ Low-Credibility Evidence\n\n"
                    for item in low_cred_items[:5]:  # Top 5 problematic
                        score = item.get('credibility_score', 0) * 100
                        issues = item.get('issues', [])
                        full_content += f"**{item.get('id', 'Unknown')}** ({item.get('type', 'unknown')}) - {score:.0f}%\n"
                        if issues:
                            full_content += f"   🚩 Issues: {', '.join(issues)}\n"
                        full_content += "\n"
            
            if result.get("contradictions"):
                full_content += "### ⚔️ Evidence Contradictions\n\n"
                for contradiction in result["contradictions"]:
                    severity = contradiction.get('severity', 'unknown').upper()
                    severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
                    items = ', '.join(contradiction.get('items', []))
                    description = contradiction.get('description', 'No description')
                    
                    full_content += f"{severity_emoji} **{severity} SEVERITY**\n"
                    full_content += f"   **Items**: {items}\n"
                    full_content += f"   **Issue**: {description}\n\n"
            
            if result.get("corroborations"):
                full_content += "### 🤝 Evidence Corroborations\n\n"
                for corr in result["corroborations"]:
                    strength = corr.get('strength', 'unknown').upper()
                    strength_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "WEAK": "🔴"}.get(strength, "⚪")
                    items = ', '.join(corr.get('items', []))
                    description = corr.get('description', 'No description')
                    
                    full_content += f"{strength_emoji} **{strength} CORROBORATION**\n"
                    full_content += f"   **Items**: {items}\n"
                    full_content += f"   **Details**: {description}\n\n"
            
            if result.get("tampering_indicators"):
                full_content += "### 🛡️ Potential Tampering Indicators\n\n"
                for indicator in result["tampering_indicators"]:
                    evidence_id = indicator.get('evidence_id', 'Unknown')
                    likelihood = indicator.get('likelihood', 0) * 100
                    likelihood_indicator = "🔴" if likelihood >= 70 else "🟡" if likelihood >= 40 else "🟢"
                    indicators = indicator.get('indicators', [])
                    
                    full_content += f"**Evidence {evidence_id}** {likelihood_indicator}\n"
                    full_content += f"   **Tampering Likelihood**: {likelihood:.0f}%\n"
                    if indicators:
                        full_content += f"   **Indicators**: {', '.join(indicators)}\n"
                    full_content += "\n"
            
            # Add single comprehensive cell
            self.add_notebook_cell("markdown", {"content": full_content})
            
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