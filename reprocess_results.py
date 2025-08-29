#!/usr/bin/env python3

import json
import sys
from metrics import calculate_all_metrics, aggregate_metrics
from benchmark import BenchmarkRunner
from config import load_config

def reprocess_benchmark_results(results_path: str):
    """Reprocess existing benchmark results with corrected metrics"""
    
    # Load existing results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Reprocessing results from: {results_path}")
    print("="*50)
    
    # Recalculate metrics for raw results
    for provider_name, provider_results in results['raw_results'].items():
        print(f"\nReprocessing {provider_name}...")
        
        for result in provider_results:
            if 'predictions' in result and 'ground_truth' in result:
                # Recalculate metrics with fixed functions
                old_metrics = result.get('metrics', {})
                new_metrics = calculate_all_metrics(result['predictions'], result['ground_truth'])
                result['metrics'] = new_metrics
                
                # Show comparison for debugging
                if 'top_5_accuracy' in old_metrics and 'top_5_accuracy' in new_metrics:
                    old_acc = old_metrics['top_5_accuracy']
                    new_acc = new_metrics['top_5_accuracy']
                    if abs(old_acc - new_acc) > 0.001:  # Significant difference
                        print(f"  {result['task']} top_5_accuracy: {old_acc:.3f} -> {new_acc:.3f}")
    
    # Reprocess summary using BenchmarkRunner logic
    config = load_config('benchmark_config.yaml')
    runner = BenchmarkRunner(config)
    processed_results = runner._process_results(results['raw_results'])
    
    # Update results
    results['summary'] = processed_results['summary']
    results['comparative_analysis'] = processed_results['comparative_analysis']
    
    # Save corrected results
    corrected_path = results_path.replace('.json', '_corrected.json')
    with open(corrected_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCorrected results saved to: {corrected_path}")
    
    # Print new rankings
    if 'comparative_analysis' in results and 'ranking' in results['comparative_analysis']:
        print("\n" + "="*50)
        print("NEW CORRECTED RANKING:")
        print("="*50)
        for i, entry in enumerate(results['comparative_analysis']['ranking']['overall'], 1):
            print(f"{i}. {entry['provider']}: {entry['score']:.4f}")
            
        # Show task breakdown for top models
        print("\nTask Score Breakdown (Top 3):")
        print("-" * 40)
        for i, entry in enumerate(results['comparative_analysis']['ranking']['overall'][:3], 1):
            provider = entry['provider']
            if provider in results['summary'] and 'task_scores' in results['summary'][provider]:
                task_scores = results['summary'][provider]['task_scores']
                print(f"{i}. {provider}:")
                for task, score in task_scores.items():
                    print(f"   {task}: {score:.3f}")
                print()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python reprocess_results.py <results_file.json>")
        sys.exit(1)
    
    reprocess_benchmark_results(sys.argv[1])