"""
LLM-based question router for classification.
"""
import json
import os
from typing import Dict, Any
from huggingface_hub import InferenceClient
from src.models.schemas import QuestionRoute
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class QuestionRouter:
    """
    Routes questions to appropriate categories using LLM.
    """
    
    def __init__(self, config: dict):
        """
        Initialize question router.
        
        Args:
            config: LLM configuration dict
        """
        self.config = config
        self.logger = logger
        
        self.model_name = os.getenv('ROUTER_MODEL', config.get('model'))
        self.max_tokens = config.get('max_tokens', 200)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout', 30)
        
        # Initialize HF Inference Client
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        
        self.client = InferenceClient(token=hf_token)
        self.logger.info(f"Router initialized with model: {self.model_name}")
    
    def route(self, question: str, system_prompt: str = None) -> QuestionRoute:
        """
        Classify the question into a category.
        
        Args:
            question: User question
            system_prompt: Optional system prompt override
            
        Returns:
            QuestionRoute object
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        user_prompt = self._build_user_prompt(question)
        
        self.logger.debug(f"Routing question: {question[:100]}")
        
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            route_data = self._parse_response(response_text)
            
            route = QuestionRoute(**route_data)
            
            self.logger.info(f"Question routed to: {route.category}")
            
            return route
            
        except Exception as e:
            self.logger.error(f"Router failed: {e}")
            # Fallback to default
            return QuestionRoute(
                category="WHAT",
                requires_full_narrative=False
            )
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for routing."""
        return """You are a question classifier for a movie transcript QA system.
Classify the question into ONE category:
- WHY: Questions about motivations, reasons, causes
- BEFORE: Questions about what happened before an event
- AFTER: Questions about what happened after an event
- WHAT: General what/who/when questions about events, characters, actions
- QUOTE: Questions asking for exact quotes or dialogue
- SUMMARY: Questions asking for summaries or overviews

Also determine if the question requires evidence from multiple parts of the narrative (beginning and end).

Return ONLY valid JSON with this exact format:
{"category": "WHAT", "requires_full_narrative": false, "key_entities": ["character1", "character2"]}

Do not include any explanation, only the JSON."""
    
    def _build_user_prompt(self, question: str) -> str:
        """Build user prompt."""
        return f"""Question: {question}

Classify this question and return JSON only."""
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed dict
        """
        # Try to extract JSON from response
        # Sometimes models wrap JSON in markdown code blocks
        text = response_text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            # Find the actual JSON content
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
        
        # Try to parse
        try:
            data = json.loads(text)
            
            # Validate required fields
            if 'category' not in data:
                raise ValueError("Missing 'category' field")
            
            # Set defaults for optional fields
            if 'requires_full_narrative' not in data:
                data['requires_full_narrative'] = False
            
            if 'key_entities' not in data:
                data['key_entities'] = []
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Response text: {text}")
            
            # Fallback: try to extract category from text
            text_upper = text.upper()
            for category in ['WHY', 'BEFORE', 'AFTER', 'WHAT', 'QUOTE', 'SUMMARY']:
                if category in text_upper:
                    return {
                        'category': category,
                        'requires_full_narrative': False,
                        'key_entities': []
                    }
            
            # Ultimate fallback
            return {
                'category': 'WHAT',
                'requires_full_narrative': False,
                'key_entities': []
            }