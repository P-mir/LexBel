import pytest

from src.retrieval_evaluation.metrics import (
    compute_average_precision,
    compute_map,
    compute_mrr,
    compute_precision_at_k,
    compute_reciprocal_rank,
)


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_precision_at_k_perfect(self):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}

        assert compute_precision_at_k(retrieved, relevant, 3) == 1.0
        assert compute_precision_at_k(retrieved, relevant, 5) == 1.0

    def test_average_precision_perfect(self):
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}

        assert compute_average_precision(retrieved, relevant) == 1.0

    def test_average_precision_no_relevant(self):
        retrieved = [5, 8, 9]
        relevant = {1, 2, 3}

        assert compute_average_precision(retrieved, relevant) == 0.0

    def test_reciprocal_rank_third(self):
        retrieved = [5, 8, 1, 3]
        relevant = {1, 2, 3}

        # First relevant at position 3
        assert compute_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3, rel=1e-4)

    def test_reciprocal_rank_no_relevant(self):
        retrieved = [5, 8, 9]
        relevant = {1, 2, 3}

        assert compute_reciprocal_rank(retrieved, relevant) == 0.0

    def test_map_multiple_queries(self):
        retrieved_lists = [
            [1, 5, 3],  # AP = (1/1 + 2/3) / 2 = 0.833
            [2, 4, 6],  # AP = (1/1 + 2/2) / 2 = 1.0
        ]
        relevant_sets = [{1, 3}, {2, 4}]

        # MAP = (0.833 + 1.0) / 2 = 0.9167
        expected = ((1.0 + 2 / 3) / 2 + 1.0) / 2
        assert compute_map(retrieved_lists, relevant_sets) == pytest.approx(expected, rel=1e-4)

    def test_mrr_multiple_queries(self):
        """Test MRR across multiple queries."""
        retrieved_lists = [
            [5, 1, 3],  # RR = 1/2
            [2, 4, 6],  # RR = 1/1
        ]
        relevant_sets = [{1, 3}, {2, 4}]

        # MRR = (1/2 + 1/1) / 2 = 0.75
        assert compute_mrr(retrieved_lists, relevant_sets) == 0.75
