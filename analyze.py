import json
import os
import hashlib
from typing import List, Dict, Any


def load_ground_truth(dataset_path: str) -> Dict[str, Any]:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_item(s: str) -> str:
    return s.strip().lower()


def compare_lists(predicted: List[str], actual: List[str]) -> Dict[str, Any]:
    pred_norm = [normalize_item(x) for x in predicted]
    act_norm = [normalize_item(x) for x in actual]
    matches = [p for p, p_norm in zip(predicted, pred_norm) if p_norm in act_norm]
    accuracy = len(matches) / max(len(actual), 1)
    return {
        'predicted': predicted,
        'actual': actual,
        'matches': matches,
        'accuracy': accuracy
    }


def analyze_model_runs(model_runs_path: str, dataset_path: str, target: str) -> Dict[str, Any]:
    # load dataset
    gt = load_ground_truth(dataset_path)
    if target not in gt:
        raise ValueError(f"Target not found in dataset: {target}")

    target_gt = gt[target]

    results = []
    with open(model_runs_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('target') != target:
                continue
            parsed = entry.get('parsed', {})
            model = entry.get('model')
            prompt_hash = entry.get('prompt_hash')

            counters_res = compare_lists(parsed.get('counters', []), target_gt.get('counters', []))
            partners_res = compare_lists(parsed.get('partners', []), target_gt.get('partners', []))
            moves_res = compare_lists(parsed.get('moves', []), target_gt.get('moves', []))

            overall = (counters_res['accuracy'] + partners_res['accuracy'] + moves_res['accuracy']) / 3.0

            results.append({
                'model': model,
                'prompt_hash': prompt_hash,
                'results': {
                    'counters': counters_res,
                    'partners': partners_res,
                    'moves': moves_res
                },
                'overall_accuracy': overall
            })

    return {
        'target': target,
        'analysis': results
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_runs', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()

    out = analyze_model_runs(args.model_runs, args.dataset, args.target)
    print(json.dumps(out, indent=2))
