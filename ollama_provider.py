import json
import logging
from typing import Dict, List, Optional, Any

try:
    import ollama
except ImportError:
    ollama = None

from base_provider import LLMProvider, PredictionRequest, PredictionResponse, TaskType, load_prompt, compute_prompt_hash

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation"""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        if ollama is None:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        
        self.host = kwargs.get('host', 'http://localhost:11434')
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        try:
            if self.host != 'http://localhost:11434':
                self.client = ollama.Client(host=self.host)
            else:
                self.client = ollama
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None
    
    @property
    def provider_name(self) -> str:
        return f"ollama-{self.model_name}"
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        if self.client is None:
            return False
        
        try:
            response = self.client.list()
            logger.debug(f"Ollama list response: {response}")
            
            models = response.get('models', [])
            logger.debug(f"Models found: {models}")
            
            available_models = []
            for model in models:
                # Handle different possible field names
                if 'name' in model:
                    available_models.append(model['name'])
                elif 'model' in model:
                    available_models.append(model['model'])
                else:
                    logger.warning(f"Unknown model structure: {model}")
            
            logger.debug(f"Available model names: {available_models}")
            logger.debug(f"Looking for model: {self.model_name}")
            
            # Check if our model name matches any available models
            model_found = any(self.model_name in model_name for model_name in available_models)
            logger.debug(f"Model found: {model_found}")
            
            return model_found
            
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            return False
    
    def _get_system_prompt(self, task_type: TaskType) -> str:
        """Get system prompt based on task type"""
        try:
            prompt = load_prompt()
        except Exception:
            prompt = "You are an expert in competitive Pokémon ADV OU (Advance Generation OverUsed) tier. You have deep knowledge of team building, movesets, and strategy for this format."

        if task_type == TaskType.COUNTERS:
            return f"{prompt} Your task is to predict the top 10 Pokémon that most effectively counter a given target Pokémon."
        elif task_type == TaskType.PARTNERS:
            return f"{prompt} Your task is to predict the top 10 Pokémon that work best as teammates with a given target Pokémon."
        elif task_type == TaskType.MOVES:
            return f"{prompt} Your task is to predict the top 10 most commonly used moves for a given Pokémon."
        
        return prompt
    
    def _get_user_prompt(self, request: PredictionRequest) -> str:
        """Generate user prompt based on the request"""
        task_name = request.task_type.value
        pokemon = request.pokemon
        
        context_info = request.get_context_info()
        
        prompt = f"""Predict the top 10 {task_name} for {pokemon} in ADV OU format.

{context_info}

Return your response as a valid JSON object with a single key "{task_name}" containing an array of exactly 10 items in order of likelihood/effectiveness.

Example format:
{{""{task_name}"": [""Item1"", ""Item2"", ""Item3"", ""Item4"", ""Item5"", ""Item6"", ""Item7"", ""Item8"", ""Item9"", ""Item10""]}}

Only return the JSON object, no additional text."""
        
        return prompt
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction using Ollama"""
        if not self.is_available():
            raise RuntimeError(f"Ollama model {self.model_name} is not available")
        
        system_prompt = self._get_system_prompt(request.task_type)
        user_prompt = self._get_user_prompt(request)
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            
            # Ollama client may return nested structures; handle safely
            if isinstance(response, dict) and 'message' in response:
                raw_response = response['message'].get('content', '')
            else:
                raw_response = getattr(response, 'content', '') if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                json_response = json.loads(raw_response)
                task_key = request.task_type.value
                predictions = json_response.get(task_key, [])
                
                if not isinstance(predictions, list):
                    raise ValueError("Predictions should be a list")
                
                # Ensure we have exactly 10 predictions
                predictions = predictions[:10]
                while len(predictions) < 10:
                    predictions.append(f"Unknown_{len(predictions)+1}")
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {e}. Raw response: {raw_response}")
                # Fallback: try to extract list-like content
                predictions = self._extract_list_from_text(raw_response)
            
            prompt_hash = compute_prompt_hash(system_prompt + "\n" + user_prompt)
            return PredictionResponse(
                predictions=predictions,
                raw_response=raw_response,
                cost=None,  # Ollama is free
                tokens_used=None,
                prompt_hash=prompt_hash
            )
            
        except Exception as e:
            logger.error(f"Error making prediction with Ollama: {e}")
            raise
    
    def _extract_list_from_text(self, text: str) -> List[str]:
        """Fallback method to extract a list from unstructured text"""
        import re
        
        # Try to find items in quotes or bullet points
        patterns = [
            r'"([^"]+)"',  # Items in quotes
            r'- ([^\n]+)',  # Bullet points with dashes
            r'\d+\.\s*([^\n]+)',  # Numbered list
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Clean and limit to 10
                cleaned = [match.strip() for match in matches if match.strip()]
                return cleaned[:10] + [f"Unknown_{i+1}" for i in range(len(cleaned), 10)]
        
        # Ultimate fallback
        return [f"Unknown_{i+1}" for i in range(10)]