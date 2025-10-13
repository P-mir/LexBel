"""Maximal Marginal Relevance (MMR) retrieval implementation."""

import time
from typing import Any, List

import numpy as np

from utils.logging_config import setup_logger
from utils.models import RetrievalResult
from utils.types import EmbeddingMatrix, EmbeddingVector

logger = setup_logger(__name__)


def mmr_select(
    query_embedding: EmbeddingVector,
    candidate_embeddings: EmbeddingMatrix,
    candidates: List[RetrievalResult],
    top_k: int = 10,
    lambda_param: float = 0.5,
) -> List[RetrievalResult]:
    """Select documents using Maximal Marginal Relevance.

    MMR balances relevance to the query with diversity among selected documents.

    Algorithm:
    1. Start with empty selected set
    2. Iteratively select document that maximizes:
       MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
    3. Continue until k documents selected

    Args:
        query_embedding: Query embedding vector (normalized)
        candidate_embeddings: Embeddings of candidate documents (normalized)
        candidates: Candidate retrieval results
        top_k: Number of documents to select
        lambda_param: Trade-off parameter (0=max diversity, 1=max relevance)

    Returns:
        List of selected retrieval results in order of selection
    """
    if not candidates or len(candidates) == 0:
        return []

    if len(candidates) <= top_k:
        return candidates

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )

    relevance_scores = np.dot(candidate_norms, query_norm)

    selected_indices: List[int] = []
    remaining_mask = np.ones(len(candidates), dtype=bool)

    for _ in range(min(top_k, len(candidates))):
        if not remaining_mask.any():
            break

        if selected_indices:
            selected_embeddings = candidate_norms[selected_indices]
            similarities = np.dot(candidate_norms, selected_embeddings.T)
            max_similarities = np.max(similarities, axis=1)
            mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * max_similarities
        else:
            mmr_scores = relevance_scores.copy()

        mmr_scores[~remaining_mask] = -np.inf

        best_idx = int(np.argmax(mmr_scores))
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

    selected_results = [candidates[idx] for idx in selected_indices]

    logger.debug(f"MMR selected {len(selected_results)} documents with λ={lambda_param}")
    return selected_results


class MMRRetriever:
    """Retriever that uses MMR for diverse results."""

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        lambda_param: float = 0.5,
        initial_k: int = 50,
    ):
        """Initialize MMR retriever.

        Args:
            vector_store: Vector store for initial retrieval
            embedder: Embedder for query encoding
            lambda_param: MMR lambda parameter
            initial_k: Number of candidates to retrieve before MMR
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.lambda_param = lambda_param
        self.initial_k = initial_k

        logger.info(f"MMR retriever initialized with λ={lambda_param}, initial_k={initial_k}")

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using MMR.

        Args:
            query: Query string
            top_k: Number of final results to return

        Returns:
            List of diverse retrieval results
        """
        start_time = time.time()

        query_embedding = self.embedder.embed_text(query)
        embed_time = time.time() - start_time

        search_start = time.time()
        initial_candidates = self.vector_store.search(
            query_embedding, top_k=max(self.initial_k, top_k * 2), return_embeddings=True
        )
        search_time = time.time() - search_start

        if not initial_candidates:
            return []

        rerank_start = time.time()
        candidate_embeddings = np.array(
            [c.metadata["embedding"] for c in initial_candidates], dtype=np.float32
        )

        mmr_results = mmr_select(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidates=initial_candidates,
            top_k=top_k,
            lambda_param=self.lambda_param,
        )
        rerank_time = time.time() - rerank_start

        total_time = time.time() - start_time

        logger.debug(
            f"MMR retrieval: embed={embed_time * 1000:.1f}ms, "
            f"search={search_time * 1000:.1f}ms, rerank={rerank_time * 1000:.1f}ms, "
            f"total={total_time * 1000:.1f}ms"
        )

        return mmr_results
