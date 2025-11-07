from typing import List, Set


def compute_precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute precision at rank k"""
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)

    return relevant_in_top_k / k


def compute_recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute recall at rank k"""
    if k <= 0 or not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)

    return relevant_in_top_k / len(relevant)


def compute_average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    if not relevant:
        return 0.0
    
    precisions = []
    num_relevant_found = 0
    
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / i
            precisions.append(precision_at_i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(relevant)


def compute_map(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    if not retrieved_lists:
        return 0.0
    
    ap_scores = [
        compute_average_precision(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ]
    
    return sum(ap_scores) / len(ap_scores)


def compute_reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    if not relevant:
        return 0.0
    
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    
    return 0.0


def compute_mrr(retrieved_lists: List[List[int]], relevant_sets: List[Set[int]]) -> float:
    if not retrieved_lists:
        return 0.0
    
    rr_scores = [
        compute_reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ]
    
    return sum(rr_scores) / len(rr_scores)


def compute_metrics_at_k(
    retrieved_lists: List[List[int]],
    relevant_sets: List[Set[int]],
    k_values: List[int],
) -> dict:
    metrics = {}
    
    metrics["MAP"] = compute_map(retrieved_lists, relevant_sets)
    metrics["MRR"] = compute_mrr(retrieved_lists, relevant_sets)
    
    for k in k_values:
        precisions = [
            compute_precision_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_sets)
        ]
        recalls = [
            compute_recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_sets)
        ]
        
        metrics[f"P@{k}"] = sum(precisions) / len(precisions) if precisions else 0.0
        metrics[f"R@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
    
    return metrics
