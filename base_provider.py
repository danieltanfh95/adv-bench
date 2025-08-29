from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import hashlib

class TaskType(Enum):
    COUNTERS = "counters"
    PARTNERS = "partners" 
    MOVES = "moves"

@dataclass
class PredictionRequest:
    pokemon: str
    task_type: TaskType
    context: Optional[Dict[str, Any]] = None
    use_metagame_analysis: bool = False
    
    def get_context_info(self) -> str:
        """Format context information for the prompt"""
        if not self.context:
            return ""
        
        context_info = []
        if 'known_moves' in self.context:
            context_info.append(f"Known moves: {', '.join(self.context['known_moves'])}")
        if 'team_members' in self.context:
            context_info.append(f"Team members: {', '.join(self.context['team_members'])}")
        if 'role' in self.context:
            context_info.append(f"Role: {self.context['role']}")
        if 'team_archetype' in self.context:
            context_info.append(f"Team archetype: {self.context['team_archetype']}")
        
        if context_info:
            return "\n".join(context_info) + "\n"
        return ""

@dataclass 
class PredictionResponse:
    predictions: List[str]
    raw_response: Optional[str] = None
    cost: Optional[float] = None
    tokens_used: Optional[int] = None
    prompt_hash: Optional[str] = None
    metagame_analysis: Optional[str] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a prediction for the given request"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider"""
        pass
    

def load_prompt(prompt_path: Optional[str] = None) -> str:
    if prompt_path is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_metagame_analysis_prompt(prompt_path: Optional[str] = None) -> str:
    if prompt_path is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'metagame_analysis_prompt.txt')
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Metagame analysis prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
