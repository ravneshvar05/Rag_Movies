"""
SRT file parser for movie transcripts.
"""
import re
from pathlib import Path
from typing import List, Optional
from src.models.schemas import SRTEntry, TimeStamp
from src.utils.time_utils import parse_srt_time_range
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class SRTParser:
    """Parse SRT subtitle files into structured data."""
    
    def __init__(self):
        self.logger = logger
    
    def parse_file(self, file_path: str) -> List[SRTEntry]:
        """
        Parse an SRT file into a list of entries.
        
        Args:
            file_path: Path to the SRT file
            
        Returns:
            List of SRTEntry objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SRT file not found: {file_path}")
        
        self.logger.info(f"Parsing SRT file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        entries = self._parse_content(content)
        self.logger.info(f"Parsed {len(entries)} subtitle entries")
        
        return entries
    
    def _parse_content(self, content: str) -> List[SRTEntry]:
        """
        Parse SRT content into entries.
        
        Args:
            content: Raw SRT file content
            
        Returns:
            List of SRTEntry objects
        """
        entries = []
        
        # Split into blocks (separated by double newlines)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            
            try:
                entry = self._parse_block(block)
                if entry:
                    entries.append(entry)
            except Exception as e:
                self.logger.warning(f"Failed to parse block: {e}")
                continue
        
        return entries
    
    def _parse_block(self, block: str) -> Optional[SRTEntry]:
        """
        Parse a single SRT block.
        
        Format:
        1
        00:00:01,000 --> 00:00:04,000
        This is the subtitle text
        
        Args:
            block: Single subtitle block
            
        Returns:
            SRTEntry or None if parsing fails
        """
        lines = block.strip().split('\n')
        
        if len(lines) < 3:
            return None
        
        # Parse index
        try:
            index = int(lines[0].strip())
        except ValueError:
            return None
        
        # Parse timestamps
        try:
            start_time, end_time = parse_srt_time_range(lines[1])
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamp: {e}")
            return None
        
        # Parse text (may be multiple lines)
        text = '\n'.join(lines[2:]).strip()
        
        # Clean HTML tags and formatting
        text = self._clean_text(text)
        
        if not text:
            return None
        
        return SRTEntry(
            index=index,
            start_time=start_time,
            end_time=end_time,
            text=text
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean subtitle text by removing HTML tags and extra whitespace.
        
        Args:
            text: Raw subtitle text
            
        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special formatting
        text = re.sub(r'\{[^}]+\}', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_speakers(self, entries: List[SRTEntry]) -> List[SRTEntry]:
        """
        Attempt to extract speaker names from subtitle text.
        Common patterns: "NAME: dialogue" or "[NAME] dialogue"
        
        Args:
            entries: List of SRT entries
            
        Returns:
            Updated list with speaker information
        """
        speaker_patterns = [
            r'^([A-Z][A-Za-z\s]+?):\s*(.+)$',  # NAME: dialogue
            r'^\[([A-Za-z\s]+?)\]\s*(.+)$',     # [NAME] dialogue
            r'^\(([A-Za-z\s]+?)\)\s*(.+)$',     # (NAME) dialogue
        ]
        
        for entry in entries:
            for pattern in speaker_patterns:
                match = re.match(pattern, entry.text)
                if match:
                    # Note: speaker info would need to be added to SRTEntry schema
                    # For now, we keep it in the text
                    break
        
        return entries