"""
Final answer generation using LLaMA models with fallback.
"""
import os
import time
from typing import Optional
from huggingface_hub import InferenceClient
from groq import Groq
from src.models.schemas import DistilledContext, Answer, TokenUsage
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class Answerer:
    """
    Generates final answer using LLaMA models via Groq (preferred) or HuggingFace.
    """
    
    def __init__(self, config: dict):
        """
        Initialize answerer.
        
        Args:
            config: LLM configuration dict
        """
        self.config = config
        self.logger = logger
        
        self.primary_model = os.getenv(
            'ANSWERER_MODEL',
            config.get('primary_model')
        )
        self.fallback_model = os.getenv(
            'FALLBACK_MODEL',
            config.get('fallback_model')
        )
        
        self.max_tokens = config.get('max_tokens', 1500)
        self.temperature = config.get('temperature', 0.3)
        self.max_retries = config.get('max_retries', 2)
        self.timeout = config.get('timeout', 90)
        
        # Initialize Clients
        self.groq_client = None
        self.hf_client = None
        
        # 1. Try Groq First
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
                self.logger.info("Initialized Groq Client for Answerer")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Groq client: {e}")
        
        # 2. Setup HF Client (for fallback or primary if Groq missing)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            self.hf_client = InferenceClient(token=hf_token)
            self.logger.info("Initialized HuggingFace Client")
        else:
            self.logger.warning("HF_TOKEN not set. HuggingFace fallback invalid.")

        if not self.groq_client and not self.hf_client:
             raise ValueError("No valid LLM credentials found (HF_TOKEN or GROQ_API_KEY required)")
            
        self.logger.info(
            f"Answerer initialized: primary={self.primary_model}, "
            f"fallback={self.fallback_model}"
        )
    
    def answer(
        self,
        question: str,
        context: DistilledContext,
        system_prompt: str = None
    ) -> Answer:
        """
        Generate final answer.
        
        Args:
            question: User question
            context: Distilled context
            system_prompt: Optional system prompt override
            
        Returns:
            Answer object
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        self.logger.info(f"Generating answer for question: {question[:100]}")
        
        # Try primary model
        answer = self._generate_with_model(
            question,
            context,
            system_prompt,
            self.primary_model
        )
        
        if answer:
            answer.model_used = self.primary_model
            return answer
        
        # Fallback to secondary model
        self.logger.warning(f"Primary model failed, using fallback: {self.fallback_model}")
        
        answer = self._generate_with_model(
            question,
            context,
            system_prompt,
            self.fallback_model
        )
        
        if answer:
            answer.model_used = self.fallback_model
            return answer
        
        # Ultimate fallback
        self.logger.error("Both models failed, returning fallback answer")
        return Answer(
            question=question,
            answer="I apologize, but I'm unable to generate an answer at this time due to technical issues.",
            supporting_timestamps=[],
            confidence="low",
            source_chunk_ids=[],
            model_used="fallback"
        )
    
    def _generate_with_model(
        self,
        question: str,
        context: DistilledContext,
        system_prompt: str,
        model_name: str,
        retry_count: int = 0
    ) -> Optional[Answer]:
        """
        Generate answer with specific model (Groq or HF).
        """
        user_prompt = self._build_user_prompt(question, context)
        
        try:
            # 1. Try Groq Generation
            if self.groq_client and ("llama" in model_name.lower() or "mixtral" in model_name.lower()):
                 # Assuming model names are compatible or set correctly in .env
                 # Basic mapping if needed, but assuming user sets correct Groq model ID
                 response = self.groq_client.chat.completions.create(
                     model=model_name,
                     messages=[
                         {"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}
                     ],
                     max_tokens=self.max_tokens,
                     temperature=self.temperature,
                 )
                 answer_text = response.choices[0].message.content.strip()

            # 2. Try HuggingFace Generation
            elif self.hf_client:
                response = self.hf_client.chat_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                answer_text = response.choices[0].message.content.strip()
            
            else:
                self.logger.error("No compatible client found for generation")
                return None
            
            # Extract timestamps from answer
            
            # Extract timestamps from answer
            supporting_timestamps = self._extract_timestamps(
                answer_text,
                context.preserved_timestamps
            )
            
            # Determine confidence
            confidence = self._assess_confidence(answer_text, context)
            
            # Extract token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            answer = Answer(
                question=question,
                answer=answer_text,
                supporting_timestamps=supporting_timestamps,
                confidence=confidence,
                source_chunk_ids=[],  # Could be populated from context
                model_used=model_name,
                token_usage=token_usage
            )
            
            self.logger.info(f"Answer generated successfully with {model_name}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate with {model_name}: {e}")
            
            # Retry if available
            if retry_count < self.max_retries:
                self.logger.info(f"Retrying... (attempt {retry_count + 1}/{self.max_retries})")
                time.sleep(2)  # Brief delay before retry
                return self._generate_with_model(
                    question,
                    context,
                    system_prompt,
                    model_name,
                    retry_count + 1
                )
            
            return None
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for answering."""
        return """You are a movie transcript QA assistant. Answer questions based ONLY on the provided transcript context.

Rules:
1. Answer ONLY from the given context - do not use external knowledge
2. Always cite timestamps when possible using [HH:MM:SS] format
3. If the answer is not in the context, say "I cannot find this information in the transcript"
4. Be specific and accurate
5. If multiple timestamps are relevant, cite all of them
6. Keep answers concise but complete

Format your answer clearly and cite timestamps."""
    
    def _build_user_prompt(
        self,
        question: str,
        context: DistilledContext
    ) -> str:
        """Build user prompt with question and context."""
        return f"""Context from movie transcript:
{context.distilled_text}

Question: {question}

Answer the question based on the context above. Cite timestamps when relevant."""
    
    def _extract_timestamps(
        self,
        answer_text: str,
        available_timestamps: list
    ) -> list:
        """Extract timestamps mentioned in answer."""
        mentioned = []
        for ts in available_timestamps:
            if ts in answer_text or f"[{ts}]" in answer_text:
                mentioned.append(ts)
        
        return mentioned
    
    def _assess_confidence(
        self,
        answer_text: str,
        context: DistilledContext
    ) -> str:
        """
        Assess answer confidence based on content.
        
        Args:
            answer_text: Generated answer
            context: Context used
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        answer_lower = answer_text.lower()
        
        # Low confidence indicators
        if any(phrase in answer_lower for phrase in [
            "i cannot find",
            "not in the transcript",
            "unclear",
            "not mentioned",
            "i don't know"
        ]):
            return "low"
        
        # High confidence indicators
        if len(self._extract_timestamps(answer_text, context.preserved_timestamps)) >= 2:
            return "high"
        
        if any(phrase in answer_lower for phrase in [
            "at",
            "during",
            "when",
            "specifically"
        ]):
            return "medium"
        
        return "medium"