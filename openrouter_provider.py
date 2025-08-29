import os
import json
import logging
from typing import Dict, List, Optional, Any

try:
    import openai
except ImportError:
    openai = None

from base_provider import LLMProvider, PredictionRequest, PredictionResponse, TaskType, load_prompt, load_metagame_analysis_prompt, compute_prompt_hash

logger = logging.getLogger(__name__)

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation using REST API"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        if openai is None:
            raise ImportError("openai package required. Run: pip install openai")

        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required in env or api_key param")

        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.api_key = api_key
        
        # Configure OpenAI client for OpenRouter
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=kwargs.get('host', 'https://openrouter.ai/api/v1')
            )
            self.client_configured = True
        except Exception as e:
            logger.error(f"Failed to configure OpenRouter client: {e}")
            self.client_configured = False

    @property
    def provider_name(self) -> str:
        # Use the raw model_name to match config (e.g., "openai/gpt-5-mini")
        return self.model_name

    def is_available(self) -> bool:
        """Check if OpenRouter API is accessible"""
        if not self.client_configured:
            return False
        
        try:
            # Test with a simple message
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=20,
                temperature=0.1
            )
            return True
        except Exception as e:
            logger.error(f"Error checking OpenRouter availability: {e}")
            return False

    def _get_system_prompt(self, task_type: TaskType) -> str:
        try:
            prompt = load_prompt()
        except Exception:
            prompt = "You are an expert in competitive Pokémon ADV OU tier."

        if task_type == TaskType.COUNTERS:
            return f"{prompt} Your task is to predict the top 10 Pokémon that most effectively counter a given target Pokémon."
        elif task_type == TaskType.PARTNERS:
            return f"{prompt} Your task is to predict the top 10 Pokémon that work best as teammates with a given target Pokémon."
        elif task_type == TaskType.MOVES:
            return f"{prompt} Your task is to predict the top 10 most commonly used moves for a given Pokémon."
        return prompt

    def _get_user_prompt(self, request: PredictionRequest, metagame_analysis: Optional[str] = None) -> str:
        """Generate user prompt based on the request and optional metagame analysis"""
        task_name = request.task_type.value
        pokemon = request.pokemon
        context_info = request.get_context_info()
        
        # Include metagame analysis if provided
        analysis_section = ""
        if metagame_analysis:
            analysis_section = f"""
METAGAME ANALYSIS:
{metagame_analysis}

Based on this analysis, """
        else:
            analysis_section = ""

        prompt = f"""{analysis_section}Predict the top 10 {task_name} for {pokemon} in ADV OU format.

{context_info}
Return your response as a valid JSON object with a single key "{task_name}" containing an array of exactly 10 items in order of likelihood/effectiveness.

Example format:
{{""{task_name}"": [""Item1"", ""Item2"", ""Item3"", ""Item4"", ""Item5"", ""Item6"", ""Item7"", ""Item8"", ""Item9"", ""Item10""]}}

Only return the JSON object, no additional text or explanation."""
        return prompt

    def _generate_metagame_analysis(self, pokemon: str) -> str:
        """Generate metagame analysis for a Pokemon using the comprehensive prompt"""
        try:
            analysis_prompt = load_metagame_analysis_prompt()
            user_prompt = f"Analyze {pokemon} in the ADV OU metagame:"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens * 2,  # Allow longer response for analysis
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Failed to generate metagame analysis for {pokemon}: {e}")
            return f"Unable to generate detailed analysis for {pokemon}."

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self.is_available():
            raise RuntimeError(f"OpenRouter model {self.model_name} is not available")

        metagame_analysis = None
        
        # Step 1: Generate metagame analysis if requested
        if request.use_metagame_analysis:
            metagame_analysis = self._generate_metagame_analysis(request.pokemon)

        system_prompt = self._get_system_prompt(request.task_type)
        user_prompt = self._get_user_prompt(request, metagame_analysis)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}  # Request JSON response
            )

            raw_response = response.choices[0].message.content

            # Parse JSON response
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
                cost=None,  # OpenRouter cost tracking not implemented
                tokens_used=None,
                prompt_hash=prompt_hash,
                metagame_analysis=metagame_analysis
            )
        except Exception as e:
            logger.error(f"Error making prediction with OpenRouter: {e}")
            raise



    def _extract_list_from_text(self, text: str) -> List[str]:
        import re
        patterns = [r'"([^\"]+)"', r'- ([^\n]+)', r'\d+\.\s*([^\n]+)']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                cleaned = [m.strip() for m in matches if m.strip()]
                return cleaned[:10] + [f"Unknown_{i+1}" for i in range(len(cleaned), 10)]
        return [f"Unknown_{i+1}" for i in range(10)]
