# ADV OU Pokemon Benchmark

A comprehensive benchmark system for evaluating Large Language Models (LLMs) on competitive Pokémon ADV OU (Advance Generation OverUsed) tier prediction tasks.

## Overview

This benchmark evaluates AI systems' ability to predict competitive Pokémon ADV OU (Advance Generation OverUsed) tier strategies by predicting:
- **Counters**: Top 10 Pokémon that effectively counter a given target Pokémon
- **Partners**: Top 10 Pokémon that synergize well as teammates with a given Pokémon  
- **Moves**: Top 10 most commonly used moves for a given Pokémon

Based on the specification in `spec.md`, this benchmark follows competitive Pokémon meta-game analysis and uses Smogon usage statistics as ground truth.

## Supported Models

### Ollama (Local Models)
- Llama 3.3 70B
- Qwen 2.5 72B
- And any other models available through Ollama

### Anthropic Claude (API)
- Claude Sonnet 4
- Claude 3.5 Sonnet
- Claude Opus 4
- Claude 3 Opus

### OpenRouter (API)
- GPT-5 and GPT-5 Mini
- Claude Sonnet 4 and Claude Opus 4.1
- Grok-4 and Grok Code Fast
- Gemini 2.5 Pro
- o3
- And any other models available through OpenRouter

## Installation

1. Clone/download this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - For Claude: Set `ANTHROPIC_API_KEY` environment variable
   - For OpenRouter: Set `OPENROUTER_API_KEY` environment variable
   - For Ollama: Ensure Ollama is running locally

## Quick Start

1. **Generate default configuration:**
```bash
python benchmark.py --create-config
```

2. **Edit the configuration file** (`benchmark_config.yaml`) to:
   - Enable/disable specific models
   - Set API keys (or use environment variables)
   - Adjust test parameters

3. **Run the benchmark:**
```bash
python benchmark.py --config benchmark_config.yaml
```

4. **View results** in the `results/` directory

## Configuration

The `benchmark_config.yaml` file controls all aspects of the benchmark:

```yaml
dataset_path: '2025-07-moveset-gen3ou-1500.json'
output_dir: 'results'
parallel: true
save_raw_responses: true
tasks: ['counters', 'partners', 'moves']
test_subset: null  # Set to list of Pokemon names to test subset

models:
  - provider: 'ollama'
    model_name: 'llama3.3:70b'
    temperature: 0.1
    max_tokens: 1000
    host: 'http://localhost:11434'
    enabled: true
  
  - provider: 'claude'
    model_name: 'claude-sonnet-4-20250514'
    temperature: 0.1
    max_tokens: 1000
    api_key: '${ANTHROPIC_API_KEY}'
    enabled: true
  
  - provider: 'openrouter'
    model_name: 'openai/gpt-5'
    temperature: 0.1
    max_tokens: 1000
    api_key: '${OPENROUTER_API_KEY}'
    enabled: true
```

### Configuration Options

- **dataset_path**: Path to the Pokemon dataset JSON file
- **output_dir**: Directory to save benchmark results
- **parallel**: Run evaluations in parallel (recommended for multiple models)
- **save_raw_responses**: Include raw model responses in results
- **tasks**: List of tasks to evaluate (`counters`, `partners`, `moves`)
- **test_subset**: Limit evaluation to specific Pokemon (useful for testing)

### Model Configuration

Each model entry supports:
- **provider**: `'ollama'`, `'claude'`, or `'openrouter'`
- **model_name**: Specific model identifier
- **temperature**: Sampling temperature (0.0-1.0)
- **max_tokens**: Maximum response tokens
- **enabled**: Whether to include this model in the benchmark
- **host**: (Ollama only) Ollama server URL
- **api_key**: (Claude/OpenRouter only) API key or environment variable reference

## Evaluation Metrics

The benchmark calculates multiple metrics for comprehensive evaluation:

### Primary Metrics
- **Top-k Accuracy** (k=1,3,5,10): Percentage of correct predictions in top-k
- **Mean Reciprocal Rank (MRR)**: Quality of ranking
- **Normalized Discounted Cumulative Gain (NDCG)**: Relevance-weighted ranking quality

### Weighted Scoring
Based on the benchmark specification:
- **Counter Prediction**: 40% weight (Top-5 accuracy > 60% success criteria)
- **Partner Prediction**: 35% weight (Top-5 accuracy > 50% success criteria)  
- **Move Prediction**: 25% weight (Top-3 accuracy > 40% success criteria)

## Results Output

The benchmark generates several output files:

### JSON Results
Complete results with all metrics, raw responses, and metadata:
```
results/benchmark_results_YYYYMMDD_HHMMSS.json
```

### CSV Summary  
Tabular summary for easy analysis:
```
results/benchmark_summary_YYYYMMDD_HHMMSS.csv
```

### Results Structure
```json
{
  "raw_results": {...},
  "summary": {
    "provider_name": {
      "task_metrics": {...},
      "overall_score": 0.75,
      "total_cost": 2.50,
      "total_tokens": 50000,
      "valid_predictions": 150,
      "failed_predictions": 0
    }
  },
  "comparative_analysis": {
    "ranking": {...},
    "cost_analysis": {...}
  }
}
```

## Example Usage

### Basic Benchmark
```bash
python benchmark.py --config benchmark_config.yaml
```

### Verbose Output
```bash
python benchmark.py --config benchmark_config.yaml --verbose
```

### Test with Subset
Edit `benchmark_config.yaml`:
```yaml
test_subset: ['Tyranitar', 'Metagross', 'Swampert']
```

### Quick API Test
Set up a minimal config with one model and a small subset to test your API setup.

## Dataset Format

The benchmark uses Pokemon data in JSON format following the specification in `spec.md`:
```json
{
  "Tyranitar": {
    "counters": ["Medicham", "Marowak", "Heracross", "Machamp", "Dugtrio", "Swampert", "Metagross", "Aerodactyl", "Claydol", "Magneton"],
    "partners": ["Swampert", "Skarmory", "Gengar", "Zapdos", "Celebi", "Magneton", "Dugtrio", "Forretress", "Aerodactyl", "Claydol"], 
    "moves": ["Rock Slide", "Earthquake", "Hidden Power Bug", "Dragon Dance", "Fire Blast", "Pursuit", "Crunch", "Substitute", "Rest", "Sleep Talk"]
  }
}
```

## Cost Management

For API-based models (Claude):
- The benchmark tracks token usage and costs
- Start with small test subsets to estimate costs
- Use `test_subset` parameter to limit evaluation scope
- Monitor the cost analysis in results

## Troubleshooting

### Ollama Issues
- Ensure Ollama is running: `ollama list`
- Check model availability: `ollama pull llama3.3:70b`
- Verify host URL in configuration

### Claude API Issues  
- Check API key: `echo $ANTHROPIC_API_KEY`
- Verify model name matches Anthropic's naming
- Check rate limits and quotas

### OpenRouter API Issues
- Check API key: `echo $OPENROUTER_API_KEY`
- Verify model name matches OpenRouter's model identifiers (e.g., `openai/gpt-5`, `anthropic/claude-sonnet-4`)
- Check rate limits and account credits
- Visit OpenRouter dashboard for usage monitoring

### General Issues
- Run with `--verbose` for detailed logging
- Check `benchmark.log` for error details
- Ensure dataset file exists and is valid JSON

## Performance Tips

1. **Use parallel execution** for multiple models
2. **Start with small subsets** to test configuration
3. **Monitor system resources** when running local models
4. **Use appropriate temperatures** (0.1 recommended for consistent results)

## Extending the Benchmark

To add support for new LLM providers:

1. Create a new provider class inheriting from `LLMProvider`
2. Implement required methods: `predict()`, `is_available()`, `provider_name`
3. Add provider initialization in `BenchmarkRunner._initialize_providers()`
4. Update configuration schema in `config.py`

## License

This benchmark is designed for research and evaluation purposes. Please respect the terms of service for any APIs used.