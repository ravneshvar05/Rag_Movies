"""
Time and timestamp utilities.
"""
import re
from typing import Tuple
from src.models.schemas import TimeStamp


def parse_srt_timestamp(timestamp_str: str) -> TimeStamp:
    """
    Parse SRT timestamp format: HH:MM:SS,mmm
    
    Args:
        timestamp_str: Timestamp string like "01:23:45,678"
        
    Returns:
        TimeStamp object
    """
    # Handle both comma and dot as millisecond separator
    timestamp_str = timestamp_str.replace(',', '.')
    
    pattern = r'(\d+):(\d+):(\d+)\.(\d+)'
    match = re.match(pattern, timestamp_str)
    
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
    hours, minutes, seconds, milliseconds = match.groups()
    
    return TimeStamp(
        hours=int(hours),
        minutes=int(minutes),
        seconds=int(seconds),
        milliseconds=int(milliseconds)
    )


def parse_srt_time_range(time_range: str) -> Tuple[TimeStamp, TimeStamp]:
    """
    Parse SRT time range: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
    
    Args:
        time_range: Time range string
        
    Returns:
        Tuple of (start_time, end_time)
    """
    parts = time_range.split('-->')
    if len(parts) != 2:
        raise ValueError(f"Invalid time range format: {time_range}")
    
    start_time = parse_srt_timestamp(parts[0].strip())
    end_time = parse_srt_timestamp(parts[1].strip())
    
    return start_time, end_time


def format_timestamp(timestamp: TimeStamp) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: TimeStamp object
        
    Returns:
        Formatted string like "01:23:45"
    """
    return f"{timestamp.hours:02d}:{timestamp.minutes:02d}:{timestamp.seconds:02d}"


def timestamps_overlap(
    start1: TimeStamp,
    end1: TimeStamp,
    start2: TimeStamp,
    end2: TimeStamp
) -> bool:
    """
    Check if two time ranges overlap.
    
    Args:
        start1, end1: First time range
        start2, end2: Second time range
        
    Returns:
        True if ranges overlap
    """
    s1 = start1.to_seconds()
    e1 = end1.to_seconds()
    s2 = start2.to_seconds()
    e2 = end2.to_seconds()
    
    return not (e1 < s2 or e2 < s1)


def calculate_duration(start: TimeStamp, end: TimeStamp) -> float:
    """
    Calculate duration in seconds.
    
    Args:
        start: Start timestamp
        end: End timestamp
        
    Returns:
        Duration in seconds
    """
    return end.to_seconds() - start.to_seconds()