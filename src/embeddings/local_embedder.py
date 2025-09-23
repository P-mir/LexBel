
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base import BaseEmbedder, EmbeddingGenerationError, ModelLoadError
from utils.logging_config import setup_logger
from utils.types import EmbeddingMatrix, EmbeddingVector

logger = setup_logger(__name__)


class LocalEmbedder(BaseEmbedder):
    """Local embedder using Sentence Transformers models.

    This implementation uses the sentence-transformers library to generate
    embeddings locally without requiring API calls. Suitable for CPU execution
    with reasonable performance.

    Default model: paraphrase-multilingual-mpnet-base-v2
    - Supports French and other languages
    - 768-dimensional embeddings
    - Good balance of quality and speed on CPU
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        """Initialize the local embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_output = normalize

        try:
            logger.info(f"Loading model {model_name} on {device}...")
            self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Could not load model {model_name}: {e}") from e

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a single text string.

        Args:
            text: The text to embed

        Returns:
            A numpy array of normalized embedding values

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_output,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}") from e

    def embed_texts(self, texts: List[str]) -> EmbeddingMatrix:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            A numpy array of normalized embedding values, one per input text

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        try:
            logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_output,
                show_progress_bar=len(texts) > 100,
            )
            logger.info("Embedding completed successfully")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to embed {len(texts)} texts: {e}")
            raise EmbeddingGenerationError(f"Batch embedding generation failed: {e}") from e

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings.

        Returns:
            Integer dimension of the embedding vectors
        """
        return self._embedding_dim
