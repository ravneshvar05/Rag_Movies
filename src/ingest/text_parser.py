"""
Text file parser for movie transcripts with strict timestamp validation.
"""
import re
from pathlib import Path
from typing import List, Optional, Tuple
from src.models.schemas import SRTEntry, TimeStamp
from src.utils.logger import MovieRAGLogger
# from src.utils.time_utils import parse_timestamp


logger = MovieRAGLogger.get_logger(__name__)


class TextParser:
    """
    Parse generic text files with timestamps into structured data.
    Supported formats:
    - [HH:MM:SS] Text...
    - HH:MM:SS - Text...
    - Speaker (HH:MM:SS): Text...
    """
    
    # Regex for finding timestamps: HH:MM:SS or MM:SS
    # Matches: "01:23:45", "1:23:45", "12:34"
    TIMESTAMP_PATTERN = r'(?:\[|\(|^|\s)(\d{1,2}:\d{2}(?::\d{2})?(?:,\d{3})?)(?:\]|\)|$|\s)'
    
    def __init__(self):
        self.logger = logger
    
    def parse_file(self, file_path: str) -> List[SRTEntry]:
        """
        Parse a text file into a list of entries.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of SRTEntry objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Parsing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        entries = self._parse_content(content)
        
        if not entries:
            raise ValueError(f"No valid timestamps found in {file_path}. File must contain timestamps (e.g., [00:01:30]).")
        
        self.logger.info(f"Parsed {len(entries)} entries from text file")
        return entries
    
    def _parse_content(self, content: str) -> List[SRTEntry]:
        """
        Parse text content into entries.
        
        Args:
            content: Raw file content
            
        Returns:
            List of SRTEntry objects
        """
        entries = []
        lines = content.split('\n')
        index_counter = 1
        
        for line in lines:
            if not line.strip():
                continue
            
            # Find all timestamps in the line
            timestamps = re.findall(self.TIMESTAMP_PATTERN, line)
            
            if not timestamps:
                continue
            
            # Use the first timestamp found as start time
            start_time_str = timestamps[0]
            try:
                start_time = self._parse_flexible_time(start_time_str)
            except ValueError:
                continue
            
            # If there's a second timestamp, use it as end time. Otherwise, default duration.
            end_time = None
            if len(timestamps) > 1:
                try:
                    end_time_candidate = self._parse_flexible_time(timestamps[1])
                    if end_time_candidate.to_seconds() > start_time.to_seconds():
                        end_time = end_time_candidate
                except ValueError:
                    pass
            
            if not end_time:
                # Default duration of 2 seconds if no end time provided
                end_time = TimeStamp.from_seconds(start_time.to_seconds() + 2.0)
            
            # Clean text: Remove the timestamp strings from the line
            clean_text = line
            for ts in timestamps:
                clean_text = clean_text.replace(ts, "")
            
            # Remove brackets/parentheses that might have surrounded the timestamp
            clean_text = re.sub(r'\[\s*\]|\(\s*\)', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if not clean_text:
                continue
            
            entries.append(SRTEntry(
                index=index_counter,
                start_time=start_time,
                end_time=end_time,
                text=clean_text
            ))
            index_counter += 1
            
        return entries

    def _parse_flexible_time(self, time_str: str) -> TimeStamp:
        """
        Parse a flexible date string into TimeStamp.
        Handles HH:MM:SS, MM:SS, etc.
        """
        # Clean string
        time_str = time_str.strip()
        parts = time_str.split(':')
        
        hours = 0
        minutes = 0
        seconds = 0
        milliseconds = 0
        
        try:
            if len(parts) == 3: # HH:MM:SS
                hours = int(parts[0])
                minutes = int(parts[1])
                # Handle seconds with milliseconds (SS,mmm)
                if ',' in parts[2]:
                    sec_parts = parts[2].split(',')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1])
                elif '.' in parts[2]:
                    sec_parts = parts[2].split('.')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1].ljust(3, '0')[:3])
                else:
                    seconds = int(parts[2])
            elif len(parts) == 2: # MM:SS
                minutes = int(parts[0])
                # Handle seconds with milliseconds
                if ',' in parts[1]:
                    sec_parts = parts[1].split(',')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1])
                elif '.' in parts[1]:
                    sec_parts = parts[1].split('.')
                    seconds = int(sec_parts[0])
                    milliseconds = int(sec_parts[1].ljust(3, '0')[:3])
                else:
                    seconds = int(parts[1])
            else:
                raise ValueError(f"Invalid format: {time_str}")
                
            return TimeStamp(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds
            )
        except ValueError as e:
            raise ValueError(f"Could not parse timestamp '{time_str}': {e}")
