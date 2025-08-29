import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    """Class to calculate various evaluation metrics for the benchmark"""
    
    @staticmethod
    def top_k_accuracy(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate top-k accuracy - fraction of ground truth items found in top-k predictions
        
        Args:
            predictions: Ordered list of predictions (most likely first)
            ground_truth: List of correct answers
            k: Number of top predictions to consider
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        # Check if any of top-k predictions match any ground truth items
        top_k_preds = set(predictions[:k])
        ground_truth_set = set(ground_truth)
        
        hits = len(top_k_preds.intersection(ground_truth_set))
        return min(1.0, hits / len(ground_truth_set))
    
    @staticmethod
    def mean_reciprocal_rank(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            predictions: Ordered list of predictions (most likely first)
            ground_truth: List of correct answers
            
        Returns:
            MRR score between 0 and 1
        """
        if not predictions or not ground_truth:
            return 0.0
        
        ground_truth_set = set(ground_truth)
        reciprocal_ranks = []
        
        for i, pred in enumerate(predictions):
            if pred in ground_truth_set:
                reciprocal_ranks.append(1.0 / (i + 1))
                break  # Only count the first hit
        
        if not reciprocal_ranks:
            return 0.0
        
        return reciprocal_ranks[0]  # MRR for single query
    
    @staticmethod
    def ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k (NDCG@k)
        
        Args:
            predictions: Ordered list of predictions (most likely first)
            ground_truth: List of correct answers (assumed to be in order of relevance)
            k: Number of top predictions to consider
            
        Returns:
            NDCG@k score between 0 and 1
        """
        if not predictions or not ground_truth:
            return 0.0
        
        # Create relevance mapping (ground truth items get scores based on position)
        relevance_scores = {}
        for i, item in enumerate(ground_truth):
            # Higher score for items that appear earlier in ground truth
            relevance_scores[item] = len(ground_truth) - i
        
        # Calculate DCG for predictions
        dcg = 0.0
        for i, pred in enumerate(predictions[:k]):
            if pred in relevance_scores:
                relevance = relevance_scores[pred]
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            relevance = len(ground_truth) - i
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def precision_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate precision at k
        
        Args:
            predictions: Ordered list of predictions
            ground_truth: List of correct answers
            k: Number of top predictions to consider
            
        Returns:
            Precision@k score between 0 and 1
        """
        if not predictions or not ground_truth or k == 0:
            return 0.0
        
        top_k_preds = set(predictions[:k])
        ground_truth_set = set(ground_truth)
        
        hits = len(top_k_preds.intersection(ground_truth_set))
        return hits / k
    
    @staticmethod
    def recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate recall at k
        
        Args:
            predictions: Ordered list of predictions
            ground_truth: List of correct answers
            k: Number of top predictions to consider
            
        Returns:
            Recall@k score between 0 and 1
        """
        if not predictions or not ground_truth:
            return 0.0
        
        top_k_preds = set(predictions[:k])
        ground_truth_set = set(ground_truth)
        
        hits = len(top_k_preds.intersection(ground_truth_set))
        return hits / len(ground_truth_set)

def calculate_all_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Calculate all metrics for a single prediction
    
    Args:
        predictions: Ordered list of predictions
        ground_truth: List of correct answers
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {}
    
    # Top-k accuracy for different k values
    for k in [1, 3, 5, 10]:
        metrics[f'top_{k}_accuracy'] = BenchmarkMetrics.top_k_accuracy(predictions, ground_truth, k)
        metrics[f'precision_at_{k}'] = BenchmarkMetrics.precision_at_k(predictions, ground_truth, k)
        metrics[f'recall_at_{k}'] = BenchmarkMetrics.recall_at_k(predictions, ground_truth, k)
        metrics[f'ndcg_at_{k}'] = BenchmarkMetrics.ndcg_at_k(predictions, ground_truth, k)
    
    # MRR
    metrics['mrr'] = BenchmarkMetrics.mean_reciprocal_rank(predictions, ground_truth)
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple predictions
    
    Args:
        all_metrics: List of metric dictionaries for each prediction
        
    Returns:
        Dictionary with aggregated statistics (mean, std, min, max)
    """
    if not all_metrics:
        return {}
    
    # Get all metric names
    metric_names = set()
    for metrics in all_metrics:
        metric_names.update(metrics.keys())
    
    aggregated = {}
    
    for metric_name in metric_names:
        values = [metrics.get(metric_name, 0.0) for metrics in all_metrics]
        values = [v for v in values if v is not None]  # Filter out None values
        
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        else:
            aggregated[metric_name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
    
    return aggregated

def calculate_weighted_score(metrics: Dict[str, float], task_type: str) -> float:
    """
    Calculate weighted score based on the benchmark specification
    
    Args:
        metrics: Dictionary of calculated metrics
        task_type: Type of task (counters, partners, moves)
        
    Returns:
        Weighted score between 0 and 1
    """
    # Equal weights for all tasks
    task_weights = {
        'counters': 1/3,
        'partners': 1/3,
        'moves': 1/3
    }
    
    # Use primary metrics - prioritize getting any match right over ranking
    if task_type == 'counters':
        primary_metric = metrics.get('top_5_accuracy', 0.0)
    elif task_type == 'partners':
        primary_metric = metrics.get('top_5_accuracy', 0.0)
    elif task_type == 'moves':
        primary_metric = metrics.get('top_3_accuracy', 0.0)
    else:
        primary_metric = metrics.get('top_5_accuracy', 0.0)
    
    task_weight = task_weights.get(task_type, 1.0)
    
    return primary_metric * task_weight