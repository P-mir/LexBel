"""Mistral Cloud Embedder for fast API-based embeddings."""

import os
from typing import List

import numpy as np

from embeddings.base import BaseEmbedder, EmbeddingGenerationError, ModelLoadError
from utils.logging_config import setup_logger
from utils.types import EmbeddingMatrix, EmbeddingVector

logger = setup_logger(__name__)


class CloudEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "mistral-embed",
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        """

        Args:
            model_name: Mistral embedding model (default: mistral-embed)
            api_key: Mistral API key (if not provided, reads from MISTRAL_API_KEY env)
            batch_size: Batch size for API calls

        Raises:
            ModelLoadError: If API key is missing or Mistral client cannot be initialized
        """
        try:
            from mistralai import Mistral
        except ImportError:
            raise ModelLoadError("mistralai package required")

        self.model_name = model_name
        self.batch_size = batch_size

        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ModelLoadError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable."
            )

        try:
            self.client = Mistral(api_key=api_key)
            # Test with a small embedding to get dimension
            test_embedding = self._get_embedding_from_api("test")
            self._embedding_dim = len(test_embedding)
            logger.info(
                f"Mistral embedder initialized. Model: {model_name}, Dimension: {self._embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            raise ModelLoadError(f"Could not initialize Mistral client: {e}") from e

    def _get_embedding_from_api(self, text: str) -> List[float]:
        """Get embedding from Mistral API for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            inputs=[text],
        )
        return response.data[0].embedding

    def _get_embeddings_from_api(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Mistral API for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding lists
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            inputs=texts,
        )
        # Maintain order
        return [item.embedding for item in response.data]

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If embedding fails
        """
        try:
            embedding = self._get_embedding_from_api(text)
            vector = np.array(embedding, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}") from e

    def embed_texts(self, texts: List[str]) -> EmbeddingMatrix:
        """Embed multiple texts using batched API calls.

        Args:
            texts: List of texts to embed

        Returns:
            Matrix of embeddings

        Raises:
            EmbeddingGenerationError: If embedding fails
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                batch_embeddings = self._get_embeddings_from_api(batch)
                all_embeddings.extend(batch_embeddings)

                if (i + len(batch)) % 1000 == 0:
                    logger.info(f"Embedded {i + len(batch)}/{len(texts)} texts...")
            except Exception as e:
                logger.error(f"Failed to embed batch {i}-{i + len(batch)}: {e}")
                raise EmbeddingGenerationError(f"Batch embedding failed: {e}") from e

        # Convert to numpy array and normalize
        embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
