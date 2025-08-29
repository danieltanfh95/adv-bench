import yaml
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 1000
    host: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True

@dataclass
class BenchmarkConfig:
    dataset_path: str = "2025-07-moveset-gen3ou-1500.json"
    output_dir: str = "results"
    models: List[ModelConfig] = None
    test_subset: Optional[List[str]] = None
    tasks: List[str] = None
    parallel: bool = True
    save_raw_responses: bool = True
    use_metagame_analysis: bool = False
    
    def __post_init__(self):
        if self.models is None:
            self.models = []
        if self.tasks is None:
            self.tasks = ["counters", "partners", "moves"]

def load_config(config_path: str) -> BenchmarkConfig:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Parse model configurations
    models = []
    for model_data in config_data.get('models', []):
        # Handle environment variable substitution for API keys
        api_key = model_data.get('api_key')
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)
        
        models.append(ModelConfig(
            provider=model_data['provider'],
            model_name=model_data['model_name'],
            temperature=model_data.get('temperature', 0.1),
            max_tokens=model_data.get('max_tokens', 1000),
            host=model_data.get('host'),
            api_key=api_key,
            enabled=model_data.get('enabled', True)
        ))
    
    return BenchmarkConfig(
        dataset_path=config_data.get('dataset_path', "2025-07-moveset-gen3ou-1500.json"),
        output_dir=config_data.get('output_dir', "results"),
        models=models,
        test_subset=config_data.get('test_subset'),
        tasks=config_data.get('tasks', ["counters", "partners", "moves"]),
        parallel=config_data.get('parallel', True),
        save_raw_responses=config_data.get('save_raw_responses', True),
        use_metagame_analysis=config_data.get('use_metagame_analysis', False)
    )

def save_default_config(config_path: str = "benchmark_config.yaml"):
    """Save a default configuration file"""
    default_config = {
        'dataset_path': '2025-07-moveset-gen3ou-1500.json',
        'output_dir': 'results',
        'parallel': True,
        'save_raw_responses': True,
        'use_metagame_analysis': False,  # Set to True to enable 2-step chain-of-thought analysis
        'tasks': ['counters', 'partners', 'moves'],
        'test_subset': None,  # Set to list of Pokemon names to test subset
        'models': [
            {
                'provider': 'ollama',
                'model_name': 'llama3.3:70b',
                'temperature': 0.1,
                'max_tokens': 1000,
                'host': 'http://localhost:11434',
                'enabled': True
            },
            {
                'provider': 'ollama', 
                'model_name': 'qwen2.5:72b',
                'temperature': 0.1,
                'max_tokens': 1000,
                'host': 'http://localhost:11434',
                'enabled': True
            },
            {
                'provider': 'claude',
                'model_name': 'claude-sonnet-4-20250514',
                'temperature': 0.1,
                'max_tokens': 1000,
                'api_key': '${ANTHROPIC_API_KEY}',
                'enabled': True
            },
            {
                'provider': 'claude',
                'model_name': 'claude-3-5-sonnet-20241022',
                'temperature': 0.1,
                'max_tokens': 1000,
                'api_key': '${ANTHROPIC_API_KEY}',
                'enabled': False
            }
        ]
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {config_path}")
    print("Remember to:")
    print("1. Set your ANTHROPIC_API_KEY environment variable")
    print("2. Ensure Ollama is running with the specified models")
    print("3. Adjust model names and settings as needed")