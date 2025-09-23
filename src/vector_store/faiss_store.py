"""FAISS-based vector store for efficient similarity search."""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from utils.logging_config import setup_logger
from utils.models import RetrievalResult, TextChunk
from utils.types import EmbeddingMatrix, EmbeddingVector

logger = setup_logger(__name__)

class FAISSVectorStore:
    """
    Uses FAISS IndexFlatIP (Inner Product) for cosine similarity
    on normalized vectors.
    """

    def __init__(self, embedding_dim: int):

        self.embedding_dim = embedding_dim

        # Use IndexFlatIP (flat == no compression/quantization inner product) for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dim)

        self.metadata: List[Dict[str, Any]] = []
        self.chunks: List[TextChunk] = []
        self.embeddings: np.ndarray = np.array([], dtype=np.float32).reshape(0, embedding_dim)

        logger.info(f"Initialized FAISS vector store with dimension {embedding_dim}")

    def add_documents(
        self,
        chunks: List[TextChunk],
        embeddings: EmbeddingMatrix,
    ) -> None:
        """Add documents with embeddings to the store.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings (must be normalized)

        Raises:
            ValueError: If chunks and embeddings don't match in length
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})"
            )

        if len(embeddings) == 0:
            logger.warning("No documents to add")
            return

        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Embeddings may not be properly normalized. Normalizing...")
            embeddings = embeddings / norms[:, np.newaxis]

        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)

        self.embeddings = np.vstack([self.embeddings, embeddings_f32]) if self.embeddings.size > 0 else embeddings_f32

        for chunk in chunks:
            self.chunks.append(chunk)
            self.metadata.append(chunk.to_dict())

        logger.info(f"Added {len(chunks)} documents. Total documents: {self.index.ntotal}")

    def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> List[RetrievalResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding (must be normalized)
            top_k: Number of results to return
            return_embeddings: If True, include embeddings in metadata

        Returns:
            List of retrieval results with scores and metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        norm = np.linalg.norm(query_embedding)
        if not np.isclose(norm, 1.0, atol=1e-5):
            logger.warning("Query embedding not normalized. Normalizing...")
            query_embedding = query_embedding / norm

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                metadata = chunk.metadata.copy()
                if return_embeddings:
                    metadata['embedding'] = self.embeddings[idx]

                result = RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.original_text,
                    score=float(dist),
                    metadata=metadata,
                    article_id=chunk.article_id,
                    reference=chunk.reference,
                    code=chunk.code,
                )
                results.append(result)

        logger.debug(f"Found {len(results)} results for query")
        return results

    def search_batch(
        self,
        query_embeddings: EmbeddingMatrix,
        top_k: int = 10,
    ) -> List[List[RetrievalResult]]:
        """Search for multiple queries in batch.

        Args:
            query_embeddings: Matrix of query embeddings (must be normalized)
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return [[] for _ in range(len(query_embeddings))]

        # Normalize queries
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / norms

        # Search
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), top_k)

        # Convert to results
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_distances, query_indices):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result = RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.original_text,
                        score=float(dist),
                        metadata=chunk.metadata,
                        article_id=chunk.article_id,
                        reference=chunk.reference,
                        code=chunk.code,
                    )
                    results.append(result)
            all_results.append(results)

        return all_results

    def persist(self, path: Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path to save to
        """
        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'chunks': self.chunks,
                'embedding_dim': self.embedding_dim,
                'embeddings': self.embeddings,
            }, f)

        logger.info(f"Vector store persisted to {path}")

    def load(self, path: Path) -> None:
        """Load the vector store from disk.

        Args:
            path: Directory path to load from

        Raises:
            FileNotFoundError: If store files don't exist
            ValueError: If embedding dimensions don't match
        """
        index_path = path / "index.faiss"
        metadata_path = path / "metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        self.metadata = data['metadata']
        self.chunks = data['chunks']
        loaded_dim = data['embedding_dim']

        self.embeddings = data.get('embeddings', np.array([], dtype=np.float32).reshape(0, loaded_dim))

        if loaded_dim != self.embedding_dim:
            raise ValueError(
                f"Loaded embedding dimension ({loaded_dim}) doesn't match "
                f"expected dimension ({self.embedding_dim})"
            )

        logger.info(f"Vector store loaded from {path}. Total documents: {self.index.ntotal}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        return {
            'total_documents': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'is_trained': self.index.is_trained,
        }
