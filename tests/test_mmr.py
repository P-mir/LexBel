
import numpy as np
import pytest

from retrievers.mmr import mmr_select
from utils.models import RetrievalResult


class TestMMR:

    @pytest.fixture
    def sample_data(self):
        # Query embedding
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        # Candidate embeddings (3 candidates)
        candidates_emb = np.array([
            [0.9, 0.1, 0.0],  # Very similar to query
            [0.9, 0.05, 0.05],  # Very similar to query AND first candidate
            [0.0, 1.0, 0.0],  # Different from query
        ], dtype=np.float32)
        candidates_emb = candidates_emb / np.linalg.norm(candidates_emb, axis=1, keepdims=True)

        # Candidate results
        candidates = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                text=f"Text {i}",
                score=0.0,
                metadata={},
                article_id=i,
                reference=f"Art. {i}",
                code="Code Test",
            )
            for i in range(3)
        ]

        return query, candidates_emb, candidates

    def test_mmr_pure_relevance(self, sample_data):
        """Test MMR with lambda=1.0 (pure relevance)."""
        query, candidates_emb, candidates = sample_data

        results = mmr_select(
            query_embedding=query,
            candidate_embeddings=candidates_emb,
            candidates=candidates,
            top_k=2,
            lambda_param=1.0,
        )

        # Should select most relevant first (candidates 0 and 1)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_0"

    def test_mmr_balanced(self, sample_data):
        """Test MMR with lambda=0.5 (balanced)."""
        query, candidates_emb, candidates = sample_data

        results = mmr_select(
            query_embedding=query,
            candidate_embeddings=candidates_emb,
            candidates=candidates,
            top_k=2,
            lambda_param=0.5,
        )

        assert len(results) == 2
        # First should still be most relevant
        assert results[0].chunk_id == "chunk_0"
        # Second should be diverse (chunk_2, not chunk_1)
        # because chunk_1 is too similar to chunk_0
        assert results[1].chunk_id == "chunk_2"

    def test_mmr_pure_diversity(self, sample_data):
        """Test MMR with lambda=0.0 (pure diversity)."""
        query, candidates_emb, candidates = sample_data

        results = mmr_select(
            query_embedding=query,
            candidate_embeddings=candidates_emb,
            candidates=candidates,
            top_k=3,
            lambda_param=0.0,
        )

        assert len(results) == 3

    def test_mmr_empty_candidates(self):
        """Test MMR with no candidates."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = mmr_select(
            query_embedding=query,
            candidate_embeddings=np.array([]),
            candidates=[],
            top_k=5,
            lambda_param=0.5,
        )
        assert len(results) == 0

    def test_mmr_top_k_larger_than_candidates(self, sample_data):
        """Test MMR when top_k > number of candidates."""
        query, candidates_emb, candidates = sample_data

        results = mmr_select(
            query_embedding=query,
            candidate_embeddings=candidates_emb,
            candidates=candidates,
            top_k=10,
            lambda_param=0.5,
        )

        # Should return all 3 candidates
        assert len(results) == 3
