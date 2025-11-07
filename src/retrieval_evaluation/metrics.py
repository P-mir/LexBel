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
