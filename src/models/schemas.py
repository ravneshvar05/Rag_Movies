"""
Data models and schemas for the movie RAG system.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import timedelta


class TimeStamp(BaseModel):
    """Represents a timestamp in the movie."""
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    milliseconds: int = 0
    
    @classmethod
    def from_seconds(cls, total_seconds: float) -> "TimeStamp":
        """Create timestamp from total seconds."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return cls(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    
    def to_seconds(self) -> float:
        """Convert to total seconds."""
        return self.hours * 3600 + self.minutes * 60 + self.seconds + self.milliseconds / 1000
    
    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}.{self.milliseconds:03d}"


class TranscriptChunk(BaseModel):
    """A chunk of movie transcript with metadata."""
    chunk_id: str
    movie_id: str
    start_time: TimeStamp
    end_time: TimeStamp
    text: str
    speaker: Optional[str] = None
    scene_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "movie_id": self.movie_id,
            "start_time": self.start_time.to_seconds(),
            "end_time": self.end_time.to_seconds(),
            "text": self.text,
            "speaker": self.speaker,
            "scene_id": self.scene_id,
            "metadata": self.metadata
        }


class SRTEntry(BaseModel):
    """Single entry from an SRT file."""
    index: int
    start_time: TimeStamp
    end_time: TimeStamp
    text: str


class RetrievedContext(BaseModel):
    """Retrieved context for a query."""
    query: str
    chunks: List[TranscriptChunk]
    semantic_scores: Optional[List[float]] = None
    keyword_matches: Optional[List[str]] = None
    
    
class QuestionRoute(BaseModel):
    """Question classification result."""
    category: str  # WHY, BEFORE, AFTER, WHAT, QUOTE, SUMMARY
    requires_full_narrative: bool = False
    key_entities: List[str] = Field(default_factory=list)
    token_usage: Optional["TokenUsage"] = None
    
    
class RelevanceJudgment(BaseModel):
    """Relevance judgment for retrieved chunks."""
    relevant_chunk_ids: List[str]
    confidence_scores: Optional[List[float]] = None
    token_usage: Optional["TokenUsage"] = None
    

class DistilledContext(BaseModel):
    """Distilled and compressed context."""
    original_chunk_count: int
    distilled_text: str
    preserved_timestamps: List[str]
    key_facts: List[str] = Field(default_factory=list)
    token_usage: Optional["TokenUsage"] = None
    

class Answer(BaseModel):
    """Final answer to the question."""
    question: str
    answer: str
    supporting_timestamps: List[str] = Field(default_factory=list)
    confidence: Optional[str] = None  # high, medium, low
    source_chunk_ids: List[str] = Field(default_factory=list)
    model_used: str = ""
    token_usage: Optional["TokenUsage"] = None
    
    
class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: "TokenUsage"):
        """Add another usage object to this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


class RAGResult(BaseModel):
    """Complete RAG pipeline result."""
    question: str
    route: QuestionRoute
    retrieved_chunks: int
    relevant_chunks: int
    answer: Answer
    processing_time: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = Field(default_factory=dict)