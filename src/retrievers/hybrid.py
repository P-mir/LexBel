"""Hybrid retriever combining vector and lexical search."""

from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging_config import setup_logger
from utils.models import RetrievalResult, TextChunk

logger = setup_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector and lexical (TF-IDF) search.

    Combines:
    - Vector search: Semantic similarity via embeddings
    - Lexical search: Term matching via TF-IDF

    Scores are combined using a weighted average, configurable via alpha parameter.
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        chunks: List[TextChunk],
        alpha: float = 0.5,
        max_features: int = 10000,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: Vector store for semantic search
            embedder: Embedder for query encoding
            chunks: All text chunks for building TF-IDF index
            alpha: Weight for vector search (1-alpha for lexical)
                   0.0 = pure lexical, 1.0 = pure vector
            max_features: Maximum features for TF-IDF
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks
        self.alpha = alpha

        logger.info(f"Building TF-IDF index for {len(chunks)} chunks...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )

        chunk_texts = [chunk.original_text for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

        logger.info(f"Hybrid retriever initialized with α={alpha}")
        logger.info(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def _lexical_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Perform lexical search using TF-IDF.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            Dictionary mapping chunk_id to TF-IDF score
        """
        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Map to chunk IDs and scores
        results = {}
        for idx in top_indices:
            chunk_id = self.chunks[idx].chunk_id
            score = float(similarities[idx])
            if score > 0:  # Only include non-zero scores
                results[chunk_id] = score

        return results

    def _vector_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Perform vector search.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            Dictionary mapping chunk_id to vector similarity score
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Search
        results_list = self.vector_store.search(query_embedding, top_k=top_k)

        # Map to dictionary
        results = {r.chunk_id: r.score for r in results_list}

        return results

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of retrieval results with combined scores
        """
        # Get results from both methods (retrieve more candidates)
        candidate_k = top_k * 3

        lexical_scores = self._lexical_search(query, candidate_k)
        vector_scores = self._vector_search(query, candidate_k)

        # Combine scores
        all_chunk_ids = set(lexical_scores.keys()) | set(vector_scores.keys())

        # Normalize scores to [0, 1] range
        if lexical_scores:
            max_lexical = max(lexical_scores.values())
            if max_lexical > 0:
                lexical_scores = {k: v / max_lexical for k, v in lexical_scores.items()}

        if vector_scores:
            max_vector = max(vector_scores.values())
            if max_vector > 0:
                vector_scores = {k: v / max_vector for k, v in vector_scores.items()}

        # Compute hybrid scores
        hybrid_scores = {}
        for chunk_id in all_chunk_ids:
            lex_score = lexical_scores.get(chunk_id, 0.0)
            vec_score = vector_scores.get(chunk_id, 0.0)
            hybrid_score = self.alpha * vec_score + (1 - self.alpha) * lex_score
            hybrid_scores[chunk_id] = hybrid_score

        # Sort by hybrid score
        sorted_chunk_ids = sorted(
            hybrid_scores.keys(),
            key=lambda x: hybrid_scores[x],
            reverse=True
        )[:top_k]

        # Build results
        chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}
        results = []

        for chunk_id in sorted_chunk_ids:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk.original_text,
                    score=hybrid_scores[chunk_id],
                    metadata={
                        **chunk.metadata,
                        'vector_score': vector_scores.get(chunk_id, 0.0),
                        'lexical_score': lexical_scores.get(chunk_id, 0.0),
                        'hybrid_alpha': self.alpha,
                    },
                    article_id=chunk.article_id,
                    reference=chunk.reference,
                    code=chunk.code,
                )
                results.append(result)

        logger.debug(f"Hybrid search returned {len(results)} results")
        return results
