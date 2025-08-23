"""
Supabase client for the backend to retrieve case evidence
"""

import os
import json
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseEvidenceClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        self.supabase: Client = create_client(url, key)
    
    async def get_case_evidence(self, case_id: str) -> List[Dict[str, Any]]:
        """Get all evidence files for a case"""
        try:
            response = self.supabase.table("evidence_files").select("*").eq("case_id", case_id).execute()
            return response.data or []
        except Exception as e:
            print(f"Error fetching case evidence: {e}")
            return []
    
    async def get_case_info(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get case information"""
        try:
            response = self.supabase.table("cases").select("*").eq("id", case_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            else:
                print(f"No case found with id: {case_id}")
                return None
        except Exception as e:
            print(f"Error fetching case info: {e}")
            return None
    
    async def process_evidence_to_text(self, evidence_files: List[Dict[str, Any]]) -> List[str]:
        """Convert evidence file records to text suitable for AI analysis"""
        processed_evidence = []
        
        for file_record in evidence_files:
            try:
                # Extract key information from the evidence file record
                evidence_text = self._format_evidence_record(file_record)
                processed_evidence.append(evidence_text)
            except Exception as e:
                print(f"Error processing evidence file {file_record.get('filename', 'unknown')}: {e}")
                # Add error information as evidence
                processed_evidence.append(f"Evidence file processing error: {file_record.get('filename', 'unknown')} - {str(e)}")
        
        return processed_evidence
    
    def _format_evidence_record(self, file_record: Dict[str, Any]) -> str:
        """Format a single evidence file record into text for AI analysis"""
        filename = file_record.get("filename", "Unknown file")
        evidence_type = file_record.get("evidence_type", "unknown")
        file_size = file_record.get("file_size", 0)
        mime_type = file_record.get("mime_type", "unknown")
        metadata = file_record.get("metadata", {})
        analysis = file_record.get("analysis", "")
        created_at = file_record.get("created_at", "")
        
        # Format file size
        size_str = self._format_file_size(file_size)
        
        # Format metadata
        metadata_str = ""
        if metadata:
            try:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                metadata_lines = [f"  {k}: {v}" for k, v in metadata.items()]
                metadata_str = f"\nMetadata:\n" + "\n".join(metadata_lines)
            except:
                metadata_str = f"\nMetadata: {metadata}"
        
        # Format previous analysis
        analysis_str = f"\nPrevious Analysis: {analysis}" if analysis else ""
        
        # Create structured evidence text
        evidence_text = f"""=== Evidence File: {filename} ===
File Type: {evidence_type}
Size: {size_str}
MIME Type: {mime_type}
Created: {created_at}{metadata_str}{analysis_str}

Content: [File content would be extracted here in a full implementation]
Note: This evidence file record contains metadata and references but actual file content extraction is not yet implemented.
For {evidence_type} files, this would include:
- Document files: Full text extraction, tables, formatted content
- Image files: OCR text extraction, visual analysis, EXIF data
- Audio files: Speech-to-text transcript, speaker identification, audio analysis
- Video files: Audio transcript, visual scene analysis, object detection

=== End of {filename} ===
"""
        
        return evidence_text
    
    def _format_file_size(self, bytes_size: int) -> str:
        """Format file size in human readable format"""
        if bytes_size == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        size = float(bytes_size)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.1f} {units[unit_index]}"
    
    async def save_agent_run_to_case(self, case_id: str, run_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Save agent run results back to the case (placeholder for future implementation)"""
        try:
            # In the future, this would save the analysis results to a case_analysis table
            # or update the case with analysis metadata
            print(f"Would save analysis results for case {case_id}, run {run_id}")
            print(f"Analysis data keys: {list(analysis_data.keys())}")
            return True
        except Exception as e:
            print(f"Error saving analysis to case: {e}")
            return False