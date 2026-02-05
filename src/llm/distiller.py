"""
LLM-based context distiller for compressing chunks.
"""
import os
from typing import List
from huggingface_hub import InferenceClient
from src.models.schemas import TranscriptChunk, DistilledContext
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class ContextDistiller:
    """
    Compresses and distills transcript chunks while preserving key information.
    """
    
    def __init__(self, config: dict):
        """
        Initialize context distiller.
        
        Args:
            config: LLM configuration dict
        """
        self.config = config
        self.logger = logger
        
        self.model_name = os.getenv('DISTILLER_MODEL', config.get('model'))
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.2)
        self.compression_ratio = config.get('compression_ratio', 0.6)
        self.timeout = config.get('timeout', 60)
        
        # Initialize HF Inference Client
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        
        self.client = InferenceClient(token=hf_token)
        self.logger.info(f"Distiller initialized with model: {self.model_name}")
    
    def distill(
        self,
        question: str,
        chunks: List[TranscriptChunk],
        system_prompt: str = None
    ) -> DistilledContext:
        """
        Distill transcript chunks into compressed context.
        
        Args:
            question: User question (for focused distillation)
            chunks: List of relevant chunks
            system_prompt: Optional system prompt override
            
        Returns:
            DistilledContext object
        """
        if not chunks:
            return DistilledContext(
                original_chunk_count=0,
                distilled_text="",
                preserved_timestamps=[],
                key_facts=[]
            )
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        self.logger.info(f"Distilling {len(chunks)} chunks")
        
        # If chunks are few, might not need distillation
        if len(chunks) <= 3:
            distilled_text = self._format_chunks_simple(chunks)
            return DistilledContext(
                original_chunk_count=len(chunks),
                distilled_text=distilled_text,
                preserved_timestamps=[str(c.start_time) for c in chunks],
                key_facts=[]
            )
        
        user_prompt = self._build_user_prompt(question, chunks)
        
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            distilled_text = response.choices[0].message.content.strip()
            
            # Extract timestamps mentioned in distilled text
            preserved_timestamps = self._extract_timestamps(distilled_text, chunks)
            
            result = DistilledContext(
                original_chunk_count=len(chunks),
                distilled_text=distilled_text,
                preserved_timestamps=preserved_timestamps,
                key_facts=[]  # Could extract key facts if needed
            )
            
            self.logger.info(
                f"Distilled {len(chunks)} chunks "
                f"(compression ~{len(distilled_text) / self._total_text_length(chunks):.1%})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Distillation failed: {e}")
            # Fallback: simple formatting
            distilled_text = self._format_chunks_simple(chunks)
            return DistilledContext(
                original_chunk_count=len(chunks),
                distilled_text=distilled_text,
                preserved_timestamps=[str(c.start_time) for c in chunks],
                key_facts=[]
            )
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for distillation."""
        return """You are a context distiller for a movie transcript QA system.
Your task is to compress transcript chunks while preserving:
1. Key facts, dialogue, and events
2. Character names and their actions
3. Timestamps (keep as [HH:MM:SS] format)
4. Causal relationships between events

Be concise but complete. Remove redundancy and filler while keeping all relevant information.
Format timestamps as [HH:MM:SS] in your output."""
    
    def _build_user_prompt(
        self,
        question: str,
        chunks: List[TranscriptChunk]
    ) -> str:
        """Build user prompt for distillation."""
        prompt_parts = [
            f"Question: {question}\n\n",
            "Compress the following transcript chunks while preserving information relevant to the question:\n\n"
        ]
        
        for chunk in chunks:
            timestamp = f"[{chunk.start_time}]"
            prompt_parts.append(f"{timestamp} {chunk.text}\n\n")
        
        prompt_parts.append(
            "Provide a compressed version that maintains key information and timestamps."
        )
        
        return ''.join(prompt_parts)
    
    def _format_chunks_simple(self, chunks: List[TranscriptChunk]) -> str:
        """Simple formatting without LLM compression."""
        parts = []
        for chunk in chunks:
            timestamp = f"[{chunk.start_time}]"
            parts.append(f"{timestamp} {chunk.text}")
        
        return "\n\n".join(parts)
    
    def _extract_timestamps(
        self,
        text: str,
        chunks: List[TranscriptChunk]
    ) -> List[str]:
        """Extract timestamps mentioned in distilled text."""
        timestamps = []
        for chunk in chunks:
            timestamp_str = str(chunk.start_time)
            if timestamp_str in text or f"[{timestamp_str}]" in text:
                timestamps.append(timestamp_str)
        
        return timestamps
    
    def _total_text_length(self, chunks: List[TranscriptChunk]) -> int:
        """Calculate total text length of chunks."""
        return sum(len(c.text) for c in chunks)