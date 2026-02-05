"""
Scene-aware chunking for movie transcripts.
"""
from typing import List
from src.models.schemas import SRTEntry, TranscriptChunk, TimeStamp
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class TranscriptChunker:
    """
    Chunk movie transcripts intelligently with scene awareness.
    """
    
    def __init__(self, config: dict):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Chunking configuration from YAML
        """
        self.config = config
        self.max_chunk_seconds = config.get('max_chunk_seconds', 120)
        self.min_chunk_seconds = config.get('min_chunk_seconds', 10)
        self.overlap_seconds = config.get('overlap_seconds', 5)
        self.scene_break_threshold = config.get('scene_break_threshold', 3.0)
        self.strategy = config.get('chunking_strategy', 'scene_aware')
        self.logger = logger
    
    def chunk_entries(
        self,
        entries: List[SRTEntry],
        movie_id: str
    ) -> List[TranscriptChunk]:
        """
        Chunk SRT entries into transcript chunks.
        
        Args:
            entries: List of SRT entries
            movie_id: Unique movie identifier
            
        Returns:
            List of TranscriptChunk objects
        """
        if self.strategy == 'scene_aware':
            return self._scene_aware_chunking(entries, movie_id)
        else:
            return self._fixed_time_chunking(entries, movie_id)
    
    def _scene_aware_chunking(
        self,
        entries: List[SRTEntry],
        movie_id: str
    ) -> List[TranscriptChunk]:
        """
        Create chunks based on scene breaks (detected by gaps in subtitles).
        
        Args:
            entries: List of SRT entries
            movie_id: Movie identifier
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_entries = []
        current_scene_id = 1
        chunk_id_counter = 1
        
        for i, entry in enumerate(entries):
            current_chunk_entries.append(entry)
            
            # Check if we should break
            should_break = False
            
            # Check for scene break (gap in subtitles)
            if i < len(entries) - 1:
                next_entry = entries[i + 1]
                gap = next_entry.start_time.to_seconds() - entry.end_time.to_seconds()
                
                if gap > self.scene_break_threshold:
                    should_break = True
                    self.logger.debug(f"Scene break detected: {gap:.2f}s gap")
            
            # Check for max duration
            if current_chunk_entries:
                duration = self._calculate_chunk_duration(current_chunk_entries)
                if duration >= self.max_chunk_seconds:
                    should_break = True
            
            # Force break at end
            if i == len(entries) - 1:
                should_break = True
            
            if should_break and current_chunk_entries:
                # Create chunk if it meets minimum duration
                duration = self._calculate_chunk_duration(current_chunk_entries)
                
                if duration >= self.min_chunk_seconds or i == len(entries) - 1:
                    chunk = self._create_chunk(
                        current_chunk_entries,
                        movie_id,
                        f"{movie_id}_chunk_{chunk_id_counter:04d}",
                        f"scene_{current_scene_id:04d}"
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
                    
                    # Start new chunk with overlap
                    overlap_entries = self._get_overlap_entries(
                        current_chunk_entries,
                        self.overlap_seconds
                    )
                    current_chunk_entries = overlap_entries
                    current_scene_id += 1
                else:
                    # Chunk too short, keep accumulating
                    pass
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(entries)} entries")
        return chunks
    
    def _fixed_time_chunking(
        self,
        entries: List[SRTEntry],
        movie_id: str
    ) -> List[TranscriptChunk]:
        """
        Create fixed-duration chunks with overlap.
        
        Args:
            entries: List of SRT entries
            movie_id: Movie identifier
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_entries = []
        chunk_id_counter = 1
        
        for i, entry in enumerate(entries):
            current_chunk_entries.append(entry)
            
            duration = self._calculate_chunk_duration(current_chunk_entries)
            
            if duration >= self.max_chunk_seconds or i == len(entries) - 1:
                chunk = self._create_chunk(
                    current_chunk_entries,
                    movie_id,
                    f"{movie_id}_chunk_{chunk_id_counter:04d}",
                    None
                )
                chunks.append(chunk)
                chunk_id_counter += 1
                
                # Start new chunk with overlap
                overlap_entries = self._get_overlap_entries(
                    current_chunk_entries,
                    self.overlap_seconds
                )
                current_chunk_entries = overlap_entries
        
        return chunks
    
    def _create_chunk(
        self,
        entries: List[SRTEntry],
        movie_id: str,
        chunk_id: str,
        scene_id: str = None
    ) -> TranscriptChunk:
        """
        Create a TranscriptChunk from SRT entries.
        
        Args:
            entries: List of SRT entries
            movie_id: Movie identifier
            chunk_id: Unique chunk ID
            scene_id: Optional scene identifier
            
        Returns:
            TranscriptChunk object
        """
        if not entries:
            raise ValueError("Cannot create chunk from empty entries")
        
        # Combine text
        text = ' '.join(entry.text for entry in entries)
        
        # Get time range
        start_time = entries[0].start_time
        end_time = entries[-1].end_time
        
        # Extract speaker if available (simplified)
        speaker = self._extract_dominant_speaker(entries)
        
        return TranscriptChunk(
            chunk_id=chunk_id,
            movie_id=movie_id,
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker=speaker,
            scene_id=scene_id,
            metadata={
                'num_subtitles': len(entries),
                'duration_seconds': self._calculate_chunk_duration(entries)
            }
        )
    
    def _calculate_chunk_duration(self, entries: List[SRTEntry]) -> float:
        """Calculate total duration of entries in seconds."""
        if not entries:
            return 0.0
        return entries[-1].end_time.to_seconds() - entries[0].start_time.to_seconds()
    
    def _get_overlap_entries(
        self,
        entries: List[SRTEntry],
        overlap_seconds: float
    ) -> List[SRTEntry]:
        """
        Get entries for overlap with next chunk.
        
        Args:
            entries: Current chunk entries
            overlap_seconds: Desired overlap duration
            
        Returns:
            List of entries for overlap
        """
        if not entries or overlap_seconds <= 0:
            return []
        
        chunk_end = entries[-1].end_time.to_seconds()
        overlap_start = chunk_end - overlap_seconds
        
        overlap_entries = []
        for entry in reversed(entries):
            if entry.start_time.to_seconds() >= overlap_start:
                overlap_entries.insert(0, entry)
            else:
                break
        
        return overlap_entries
    
    def _extract_dominant_speaker(self, entries: List[SRTEntry]) -> str:
        """
        Extract dominant speaker from entries (if available in text).
        This is a placeholder - real implementation would be more sophisticated.
        
        Args:
            entries: List of SRT entries
            
        Returns:
            Speaker name or None
        """
        # Placeholder - could implement speaker detection from text patterns
        return None