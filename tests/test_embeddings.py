
import numpy as np
import pytest

from embeddings import CloudEmbedder, LocalEmbedder


class TestLocalEmbedder:

    def test_initialization(self):
        embedder = LocalEmbedder(device="cpu")
        assert embedder.get_embedding_dim() > 0

    def test_embed_text(self):
        embedder = LocalEmbedder(device="cpu")
        text = "Ceci est un test."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_embedding_dim(),)
        assert embedding.dtype == np.float32

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embed_texts_batch(self):
        """Test batch text embedding."""
        embedder = LocalEmbedder(device="cpu", batch_size=2)
        texts = ["Premier texte.", "Deuxième texte.", "Troisième texte."]
        embeddings = embedder.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedder.get_embedding_dim())
        assert embeddings.dtype == np.float32

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty_list(self):
        """Test embedding empty list."""
        embedder = LocalEmbedder(device="cpu")
        embeddings = embedder.embed_texts([])

        assert embeddings.shape == (0, embedder.get_embedding_dim())


class TestCloudEmbedder:
    """Tests for CloudEmbedder (uses Mistral API, requires API key)."""

    @pytest.mark.skipif(
        not pytest.config.getoption("--run-cloud"),
        reason="Requires --run-cloud flag and MISTRAL_API_KEY"
    )
    def test_embed_text(self):
        """Test cloud embedding with Mistral (skipped without API key)."""
        embedder = CloudEmbedder()
        text = "Ceci est un test."
        embedding = embedder.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.get_embedding_dim(),)

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="Run tests that require cloud API"
    )
