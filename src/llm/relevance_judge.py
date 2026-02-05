"""
LLM-based relevance judge for filtering chunks.
"""
import json
import os
from typing import List
from huggingface_hub import InferenceClient
from src.models.schemas import TranscriptChunk, RelevanceJudgment
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class RelevanceJudge:
    """
    Judges which retrieved chunks are relevant to the question.
    """
    
    def __init__(self, config: dict):
        """
        Initialize relevance judge.
        
        Args:
            config: LLM configuration dict
        """
        self.config = config
        self.logger = logger
        
        self.model_name = os.getenv('JUDGE_MODEL', config.get('model'))
        self.max_tokens = config.get('max_tokens', 500)
        self.temperature = config.get('temperature', 0.1)
        self.batch_size = config.get('batch_size', 5)
        self.timeout = config.get('timeout', 45)
        
        # Initialize HF Inference Client
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        
        self.client = InferenceClient(token=hf_token)
        self.logger.info(f"Judge initialized with model: {self.model_name}")
    
    def judge(
        self,
        question: str,
        chunks: List[TranscriptChunk],
        system_prompt: str = None
    ) -> RelevanceJudgment:
        """
        Judge which chunks are relevant to the question.
        
        Args:
            question: User question
            chunks: Retrieved chunks
            system_prompt: Optional system prompt override
            
        Returns:
            RelevanceJudgment with relevant chunk IDs
        """
        if not chunks:
            return RelevanceJudgment(relevant_chunk_ids=[])
        
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        self.logger.info(f"Judging relevance for {len(chunks)} chunks")
        
        # Process in batches if needed
        if len(chunks) > 20:
            # For large numbers, judge in batches
            all_relevant = []
            for i in range(0, len(chunks), 20):
                batch = chunks[i:i+20]
                judgment = self._judge_batch(question, batch, system_prompt)
                all_relevant.extend(judgment.relevant_chunk_ids)
            
            return RelevanceJudgment(relevant_chunk_ids=all_relevant)
        else:
            return self._judge_batch(question, chunks, system_prompt)
    
    def _judge_batch(
        self,
        question: str,
        chunks: List[TranscriptChunk],
        system_prompt: str
    ) -> RelevanceJudgment:
        """Judge a batch of chunks."""
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
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            judgment_data = self._parse_response(response_text, chunks)
            
            judgment = RelevanceJudgment(**judgment_data)
            
            self.logger.info(
                f"Judged {len(judgment.relevant_chunk_ids)} / {len(chunks)} "
                f"chunks as relevant"
            )
            
            return judgment
            
        except Exception as e:
            self.logger.error(f"Judge failed: {e}")
            # Fallback: consider all chunks relevant
            return RelevanceJudgment(
                relevant_chunk_ids=[c.chunk_id for c in chunks]
            )
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for judging."""
        return """You are a relevance judge for a movie transcript QA system.
Given a question and transcript chunks, identify which chunks contain information relevant to answering the question.

A chunk is relevant if it contains:
- Direct information to answer the question
- Context necessary to understand the answer
- Related events, characters, or dialogue mentioned in the question

Return ONLY valid JSON with this exact format:
{"relevant_chunk_ids": ["chunk_0001", "chunk_0003", "chunk_0010"]}

Do not include explanations, only the JSON with chunk IDs."""
    
    def _build_user_prompt(
        self,
        question: str,
        chunks: List[TranscriptChunk]
    ) -> str:
        """Build user prompt with question and chunks."""
        prompt_parts = [f"Question: {question}\n\nTranscript Chunks:\n"]
        
        for i, chunk in enumerate(chunks):
            timestamp = f"{chunk.start_time}"
            prompt_parts.append(
                f"\nChunk ID: {chunk.chunk_id}\n"
                f"Timestamp: {timestamp}\n"
                f"Text: {chunk.text[:300]}...\n"  # Truncate for context window
            )
        
        prompt_parts.append(
            "\n\nIdentify which chunk IDs are relevant to answering the question. "
            "Return JSON only."
        )
        
        return ''.join(prompt_parts)
    
    def _parse_response(
        self,
        response_text: str,
        chunks: List[TranscriptChunk]
    ) -> dict:
        """
        Parse LLM response to extract relevant chunk IDs.
        
        Args:
            response_text: Raw LLM response
            chunks: Original chunks for validation
            
        Returns:
            Dict with relevant_chunk_ids
        """
        text = response_text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            lines = text.split('\n')
            json_lines = []
            in_code = False
            for line in lines:
                if line.startswith('```'):
                    in_code = not in_code
                    continue
                if in_code or ('{' in line or '}' in line or '"' in line):
                    json_lines.append(line)
            text = '\n'.join(json_lines)
        
        try:
            data = json.loads(text)
            
            if 'relevant_chunk_ids' not in data:
                raise ValueError("Missing 'relevant_chunk_ids' field")
            
            # Validate chunk IDs
            valid_ids = {c.chunk_id for c in chunks}
            relevant_ids = [
                cid for cid in data['relevant_chunk_ids']
                if cid in valid_ids
            ]
            
            return {'relevant_chunk_ids': relevant_ids}
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse judge response: {e}")
            self.logger.debug(f"Response text: {text}")
            
            # Fallback: try to extract chunk IDs from text
            chunk_ids = []
            for chunk in chunks:
                if chunk.chunk_id in text:
                    chunk_ids.append(chunk.chunk_id)
            
            if chunk_ids:
                return {'relevant_chunk_ids': chunk_ids}
            
            # Ultimate fallback: all chunks
            return {'relevant_chunk_ids': [c.chunk_id for c in chunks]}