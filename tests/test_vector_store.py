
import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils.models import TextChunk
from vector_store import FAISSVectorStore


class TestFAISSVectorStore:

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = [
            TextChunk(
                chunk_id="1_chunk_0",
                original_text="Premier article de loi.",
                article_id=1,
                reference="Art. 1",
                code="Code Test",
                book="Livre 1",
                chapter="Chapitre 1",
                section="Section 1",
                char_start=0,
                char_end=20,
                metadata={'test': 'metadata'},
            ),
            TextChunk(
                chunk_id="2_chunk_0",
                original_text="Deuxième article de loi.",
                article_id=2,
                reference="Art. 2",
                code="Code Test",
                book="Livre 1",
                chapter="Chapitre 1",
                section="Section 2",
                char_start=0,
                char_end=24,
                metadata={'test': 'metadata'},
            ),
        ]
        return chunks

    @pytest.fixture
    def sample_embeddings(self):
        # Create normalized random embeddings
        embeddings = np.random.randn(2, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def test_initialization(self):
        store = FAISSVectorStore(embedding_dim=768)
        assert store.embedding_dim == 768
        assert store.index.ntotal == 0

    def test_add_documents(self, sample_chunks, sample_embeddings):
        store = FAISSVectorStore(embedding_dim=768)
        store.add_documents(sample_chunks, sample_embeddings)

        assert store.index.ntotal == 2
        assert len(store.chunks) == 2
        assert len(store.metadata) == 2

    def test_search(self, sample_chunks, sample_embeddings):
        store = FAISSVectorStore(embedding_dim=768)
        store.add_documents(sample_chunks, sample_embeddings)

        # Search with first embedding
        query = sample_embeddings[0]
        results = store.search(query, top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == "1_chunk_0"
        assert results[0].score > 0.99  # Should match itself closely

    def test_persist_and_load(self, sample_chunks, sample_embeddings):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            store1 = FAISSVectorStore(embedding_dim=768)
            store1.add_documents(sample_chunks, sample_embeddings)
            store1.persist(tmppath)

            store2 = FAISSVectorStore(embedding_dim=768)
            store2.load(tmppath)

            assert store2.index.ntotal == 2
            assert len(store2.chunks) == 2

            # Verify search works
            query = sample_embeddings[0]
            results = store2.search(query, top_k=1)
            assert len(results) == 1

    def test_stats(self, sample_chunks, sample_embeddings):
        store = FAISSVectorStore(embedding_dim=768)
        store.add_documents(sample_chunks, sample_embeddings)

        stats = store.get_stats()
        assert stats['total_documents'] == 2
        assert stats['embedding_dimension'] == 768
        assert stats['is_trained']
