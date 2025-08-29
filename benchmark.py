#!/usr/bin/env python3

import json
import os
import sys
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import traceback
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import BenchmarkConfig, ModelConfig, load_config, save_default_config
from base_provider import LLMProvider, PredictionRequest, TaskType
from ollama_provider import OllamaProvider
from claude_provider import ClaudeProvider
from openrouter_provider import OpenRouterProvider
from deepseek_provider import DeepSeekProvider
from metrics import calculate_all_metrics, aggregate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self.dataset = self._load_dataset()
        self._setup_output_dir()
        self._initialize_providers()
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the Pokemon dataset"""
        if not os.path.exists(self.config.dataset_path):
            logger.error(f"Dataset not found: {self.config.dataset_path}")
            sys.exit(1)
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded dataset with {len(dataset)} Pokemon")
        return dataset
    
    def _setup_output_dir(self):
        """Create output directory structure"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, 'metrics'), exist_ok=True)
    
    def _initialize_providers(self):
        """Initialize LLM providers based on configuration"""
        for model_config in self.config.models:
            if not model_config.enabled:
                continue
            
            try:
                if model_config.provider.lower() == 'ollama':
                    provider = OllamaProvider(
                        model_name=model_config.model_name,
                        host=model_config.host,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens
                    )
                elif model_config.provider.lower() == 'claude':
                    provider = ClaudeProvider(
                        model_name=model_config.model_name,
                        api_key=model_config.api_key,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens
                    )
                elif model_config.provider.lower() == 'openrouter':
                    print(model_config)
                    provider = OpenRouterProvider(
                        model_name=model_config.model_name,
                        api_key=model_config.api_key,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens
                    )
                elif model_config.provider.lower() == 'deepseek':
                    provider = DeepSeekProvider(
                        model_name=model_config.model_name,
                        api_key=model_config.api_key,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens
                    )
                else:
                    logger.error(f"Unknown provider: {model_config.provider}")
                    continue
                
                if provider.is_available():
                    self.providers[provider.provider_name] = provider
                    logger.info(f"Initialized provider: {provider.provider_name}")
                else:
                    logger.warning(f"Provider not available: {model_config.provider} - {model_config.model_name}")
            
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Failed to initialize {model_config.provider} - {model_config.model_name}: {e}")
    
    def _get_test_pokemon(self) -> List[str]:
        """Get list of Pokemon to test"""
        if self.config.test_subset:
            # Filter to only Pokemon that exist in the dataset
            test_pokemon = [p for p in self.config.test_subset if p in self.dataset]
            if not test_pokemon:
                logger.warning("None of the test_subset Pokemon found in dataset, using full dataset")
                return list(self.dataset.keys())
            return test_pokemon
        return list(self.dataset.keys())
    
    def _make_single_prediction(
        self, 
        provider: LLMProvider, 
        pokemon: str, 
        task_type: TaskType
    ) -> Tuple[str, str, TaskType, Dict[str, Any]]:
        """Make a single prediction and return results"""
        try:
            request = PredictionRequest(
                pokemon=pokemon,
                task_type=task_type,
                use_metagame_analysis=self.config.use_metagame_analysis
            )
            
            start_time = time.time()
            response = provider.predict(request)
            end_time = time.time()
            
            # Get ground truth
            ground_truth = self.dataset[pokemon].get(task_type.value, [])
            
            # Calculate metrics
            metrics = calculate_all_metrics(response.predictions, ground_truth)
            
            run_id = str(uuid.uuid4())
            # Persist raw run to JSONL
            out_entry = {
                'run_id': run_id,
                'model': provider.provider_name,
                'model_name': provider.model_name,
                'target': pokemon,
                'task': task_type.value,
                'predictions': response.predictions,
                'ground_truth': ground_truth,
                'metrics': metrics,
                'response_time': end_time - start_time,
                'cost': response.cost,
                'tokens_used': response.tokens_used,
                'raw_response': response.raw_response if self.config.save_raw_responses else None,
                'metagame_analysis': response.metagame_analysis if self.config.use_metagame_analysis else None,
                'timestamp': datetime.now().isoformat(),
                'prompt_hash': response.prompt_hash
            }

            jsonl_path = os.path.join(self.config.output_dir, 'raw', 'model_runs.jsonl')
            with open(jsonl_path, 'a', encoding='utf-8') as jf:
                jf.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            
            result = out_entry
            return provider.provider_name, pokemon, task_type, result
        
        except Exception as e:
            logger.error(f"Error making prediction for {provider.provider_name} - {pokemon} - {task_type.value}: {e}")
            logger.debug(traceback.format_exc())
            return provider.provider_name, pokemon, task_type, {
                'error': str(e),
                'provider': provider.provider_name,
                'pokemon': pokemon,
                'task': task_type.value
            }
    
    def _run_single_threaded(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run benchmark in single-threaded mode"""
        results = {provider_name: [] for provider_name in self.providers}
        test_pokemon = self._get_test_pokemon()
        
        total_tasks = len(self.providers) * len(test_pokemon) * len(self.config.tasks)
        
        with tqdm(total=total_tasks, desc="Running benchmark") as pbar:
            for provider_name, provider in self.providers.items():
                for pokemon in test_pokemon:
                    for task_name in self.config.tasks:
                        task_type = TaskType(task_name)
                        _, _, _, result = self._make_single_prediction(provider, pokemon, task_type)
                        results[provider_name].append(result)
                        pbar.update(1)
        
        return results
    
    def _run_parallel(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run benchmark in parallel mode"""
        results = {provider_name: [] for provider_name in self.providers}
        test_pokemon = self._get_test_pokemon()
        
        # Create all tasks
        tasks = []
        for provider_name, provider in self.providers.items():
            for pokemon in test_pokemon:
                for task_name in self.config.tasks:
                    task_type = TaskType(task_name)
                    tasks.append((provider, pokemon, task_type))
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=min(8, len(self.providers) * 2)) as executor:
            future_to_task = {
                executor.submit(self._make_single_prediction, *task): task
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Running benchmark") as pbar:
                for future in as_completed(future_to_task):
                    provider_name, pokemon, task_type, result = future.result()
                    results[provider_name].append(result)
                    pbar.update(1)
        
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run the complete benchmark"""
        logger.info("Starting benchmark...")
        logger.info(f"Providers: {list(self.providers.keys())}")
        logger.info(f"Tasks: {self.config.tasks}")
        logger.info(f"Pokemon to test: {len(self._get_test_pokemon())}")
        
        start_time = time.time()
        
        # Run predictions
        if self.config.parallel and len(self.providers) > 1:
            results = self._run_parallel()
        else:
            results = self._run_single_threaded()
        
        end_time = time.time()
        
        # Process and save results
        processed_results = self._process_results(results)
        processed_results['benchmark_info'] = {
            'total_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'dataset_path': self.config.dataset_path,
                'tasks': self.config.tasks,
                'test_pokemon_count': len(self._get_test_pokemon()),
                'providers': [p.provider_name for p in self.providers.values()]
            }
        }
        
        # Save results
        self._save_results(processed_results)
        
        logger.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
        return processed_results
    
    def _process_results(self, raw_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Process raw results and calculate aggregated metrics"""
        processed = {
            'raw_results': raw_results,
            'summary': {},
            'comparative_analysis': {}
        }
        
        # Calculate aggregated metrics for each provider
        for provider_name, provider_results in raw_results.items():
            # Filter out error results
            valid_results = [r for r in provider_results if 'error' not in r]
            
            if not valid_results:
                processed['summary'][provider_name] = {'error': 'No valid results'}
                continue
            
            # Group by task type
            by_task = {}
            for result in valid_results:
                task = result['task']
                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(result['metrics'])
            
            # Aggregate metrics by task
            task_metrics = {}
            for task, metrics_list in by_task.items():
                task_metrics[task] = aggregate_metrics(metrics_list)
            
            # Calculate overall score by task aggregation
            task_scores = {}
            for task in self.config.tasks:
                if task in by_task:
                    # Get primary metric for this task
                    if task == 'moves':
                        task_score = task_metrics[task].get('top_3_accuracy', {}).get('mean', 0.0)
                    else:
                        task_score = task_metrics[task].get('top_5_accuracy', {}).get('mean', 0.0)
                    task_scores[task] = task_score
                else:
                    task_scores[task] = 0.0
            
            # Overall score is simple average of task scores (equal weighting)
            overall_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
            
            # Calculate costs
            total_cost = sum(r.get('cost', 0) or 0 for r in valid_results)
            total_tokens = sum(r.get('tokens_used', 0) or 0 for r in valid_results)
            
            processed['summary'][provider_name] = {
                'task_metrics': task_metrics,
                'task_scores': task_scores,
                'overall_score': overall_score,
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'valid_predictions': len(valid_results),
                'failed_predictions': len(provider_results) - len(valid_results)
            }
        
        # Comparative analysis
        if len(processed['summary']) > 1:
            processed['comparative_analysis'] = self._compare_providers(processed['summary'])
        
        return processed
    
    def _compare_providers(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between providers"""
        comparison = {
            'ranking': {},
            'statistical_significance': {},
            'cost_analysis': {}
        }
        
        # Rank providers by overall score
        valid_providers = {k: v for k, v in summary.items() if 'error' not in v}
        if len(valid_providers) < 2:
            return comparison
        
        # Overall ranking
        ranked_providers = sorted(
            valid_providers.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        comparison['ranking']['overall'] = [
            {
                'provider': name,
                'score': data['overall_score'],
                'cost': data['total_cost']
            }
            for name, data in ranked_providers
        ]
        
        # Task-specific rankings
        for task in self.config.tasks:
            task_scores = []
            for provider_name, data in valid_providers.items():
                if task in data['task_metrics']:
                    score = data['task_metrics'][task].get('top_5_accuracy', {}).get('mean', 0)
                    task_scores.append((provider_name, score))
            
            task_scores.sort(key=lambda x: x[1], reverse=True)
            comparison['ranking'][f'{task}_top_5_accuracy'] = [
                {'provider': name, 'score': score} for name, score in task_scores
            ]
        
        # Cost analysis
        costs = [(name, data['total_cost']) for name, data in valid_providers.items()]
        costs.sort(key=lambda x: x[1])
        comparison['cost_analysis'] = {
            'cheapest_to_most_expensive': costs,
            'cost_per_prediction': [
                {
                    'provider': name,
                    'cost_per_prediction': data['total_cost'] / max(data['valid_predictions'], 1)
                }
                for name, data in valid_providers.items()
            ]
        }
        
        return comparison
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_path = os.path.join(self.config.output_dir, f'benchmark_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary as CSV for easy analysis
        summary_data = []
        for provider_name, summary in results['summary'].items():
            if 'error' in summary:
                continue
            
            row = {
                'provider': provider_name,
                'overall_score': summary['overall_score'],
                'total_cost': summary['total_cost'],
                'total_tokens': summary['total_tokens'],
                'valid_predictions': summary['valid_predictions'],
                'failed_predictions': summary['failed_predictions']
            }
            
            # Add task-specific metrics
            for task in self.config.tasks:
                if task in summary['task_metrics']:
                    task_data = summary['task_metrics'][task]
                    for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_10_accuracy', 'mrr']:
                        if metric in task_data:
                            row[f'{task}_{metric}_mean'] = task_data[metric]['mean']
                            row[f'{task}_{metric}_std'] = task_data[metric]['std']
            
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.config.output_dir, f'benchmark_summary_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        if summary_data:
            logger.info(f"Summary saved to {csv_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ADV OU Pokemon Benchmark')
    parser.add_argument('--config', default='benchmark_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create default configuration file and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.create_config:
        save_default_config(args.config)
        return
    
    try:
        config = load_config(args.config)
        runner = BenchmarkRunner(config)
        results = runner.run()
        
        # Print summary
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        if 'comparative_analysis' in results and 'ranking' in results['comparative_analysis']:
            print("\nOverall Ranking:")
            for i, entry in enumerate(results['comparative_analysis']['ranking']['overall'], 1):
                print(f"{i}. {entry['provider']}: {entry['score']:.3f} (${entry['cost']:.4f})")
        
        print(f"\nDetailed results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
