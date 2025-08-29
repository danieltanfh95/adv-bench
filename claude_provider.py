import json
import logging
import os
from typing import Dict, List, Optional, Any

try:
    import anthropic
except ImportError:
    anthropic = None

from base_provider import LLMProvider, PredictionRequest, PredictionResponse, TaskType, load_prompt, load_metagame_analysis_prompt, compute_prompt_hash

logger = logging.getLogger(__name__)

class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation"""
    
    # Model pricing per 1M tokens (input, output)
    MODEL_PRICING = {
        'claude-sonnet-4-20250514': (3.00, 15.00),
        'claude-opus-4-20241022': (15.00, 75.00),
        'claude-3-5-sonnet-20241022': (3.00, 15.00),
        'claude-3-opus-20240229': (15.00, 75.00),
    }
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")
        
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self.client = None
    
    @property
    def provider_name(self) -> str:
        return f"claude-{self.model_name.split('-')[-1]}"
    
    def is_available(self) -> bool:
        """Check if Claude API is accessible"""
        if self.client is None:
            return False
        
        try:
            # Test with a simple message
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Error checking Claude availability: {e}")
            return False
    
    
    def _get_system_prompt(self, task_type: TaskType) -> str:
        """Get system prompt based on task type"""
        # Prefer loading the canonical prompt from prompt.txt if present
        try:
            prompt = load_prompt()
        except Exception:
            prompt = "You are an expert in competitive Pokémon ADV OU (Advance Generation OverUsed) tier. You have deep knowledge of team building, movesets, and strategy for this format based on historical usage statistics and competitive analysis."

        # Optionally customize with a short task-specific blurb
        if task_type == TaskType.COUNTERS:
            return f"{prompt}\n\nYour task is to predict the top 10 Pokémon that most effectively counter a given target Pokémon. Consider: type matchups, movesets, stats, abilities, meta-game positioning and defensive capabilities."
        elif task_type == TaskType.PARTNERS:
            return f"{prompt}\n\nYour task is to predict the top 10 Pokémon that work best as teammates with a given target Pokémon. Consider synergy, role complementarity, archetypes, hazards, and momentum."
        elif task_type == TaskType.MOVES:
            return f"{prompt}\n\nYour task is to predict the top 10 most commonly used moves for a given Pokémon. Consider STAB, coverage, utility moves, common sets, and item interactions."
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
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens * 2,  # Allow longer response for analysis
                temperature=self.temperature,
                system=analysis_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.warning(f"Failed to generate metagame analysis for {pokemon}: {e}")
            return f"Unable to generate detailed analysis for {pokemon}."
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction using Claude with optional metagame analysis"""
        if not self.is_available():
            raise RuntimeError(f"Claude model {self.model_name} is not available")
        
        metagame_analysis = None
        total_cost = 0.0
        total_tokens = 0
        
        # Step 1: Generate metagame analysis if requested
        if request.use_metagame_analysis:
            metagame_analysis = self._generate_metagame_analysis(request.pokemon)
            # Note: We're not tracking the cost/tokens of the analysis step for simplicity
            # In a production system, you'd want to track this separately
        
        system_prompt = self._get_system_prompt(request.task_type)
        user_prompt = self._get_user_prompt(request, metagame_analysis)
        
        try:
            # Use messages API with both system and user roles to better align model
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            raw_response = response.content[0].text
            
            # Calculate cost
            cost = self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
            
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
                cost=cost,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                prompt_hash=prompt_hash,
                metagame_analysis=metagame_analysis
            )
            
        except Exception as e:
            logger.error(f"Error making prediction with Claude: {e}")
            raise

    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        if self.model_name not in self.MODEL_PRICING:
            return 0.0
        
        input_cost_per_1m, output_cost_per_1m = self.MODEL_PRICING[self.model_name]
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        
        return input_cost + output_cost
    
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