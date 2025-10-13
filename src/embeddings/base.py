from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.types import EmbeddingMatrix, EmbeddingVector


class BaseEmbedder(ABC):
    """Abstract base class for text embedders.

    This interface defines the contract that all embedder implementations must follow.
    It supports both single text and batch embedding operations, with configurable
    parameters for different model requirements.
    """

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text string.

        Args:
            text: The text to embed

        Returns:
            A numpy array of normalized embedding values

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingMatrix:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            A numpy array of normalized embedding values, one per input text

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings.

        Returns:
            Integer dimension of the embedding vectors
        """
        pass

    def normalize_embedding(self, embedding: EmbeddingVector) -> EmbeddingVector:
        """Normalize an embedding vector to unit length.

        Args:
            embedding: The embedding vector to normalize

        Returns:
            The normalized embedding vector
        """
        return embedding / np.linalg.norm(embedding)

    def normalize_embeddings(self, embeddings: EmbeddingMatrix) -> EmbeddingMatrix:
        """Normalize multiple embedding vectors to unit length.

        Args:
            embeddings: Matrix of embedding vectors to normalize

        Returns:
            Matrix of normalized embedding vectors
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    @property
    def supports_batch(self) -> bool:
        """Whether the embedder supports efficient batching.

        Returns:
            True if the embedder supports efficient batching, False otherwise
        """
        return True


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

    pass


class ModelLoadError(EmbeddingError):
    """Exception raised when model loading fails."""

    pass


class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails."""

    pass
