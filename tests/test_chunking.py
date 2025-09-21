
import pytest

from chunking import TextChunker
from utils.models import LegalArticle


class TestTextChunker:

    @pytest.fixture
    def sample_article(self):
        return LegalArticle(
            id=1,
            reference="Art. 1.1.1",
            article="Premier sentence. Deuxième sentence. Troisième sentence. Quatrième sentence. Cinquième sentence.",
            law_type="regional",
            code="Code Test",
            book="Livre 1",
            chapter="Chapitre 1",
            section="Section 1",
        )

    def test_chunk_short_article(self):
        article = LegalArticle(
            id=1,
            reference="Art. 1",
            article="Court article.",
            law_type="regional",
            code="Code Test",
        )

        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_article(article)

        # Short article should produce one chunk
        assert len(chunks) == 1
        assert chunks[0].original_text == "Court article."
        assert chunks[0].chunk_id == "1_chunk_0"

    def test_chunk_with_overlap(self, sample_article):
        # Small chunk size to force multiple chunks
        chunker = TextChunker(chunk_size=50, chunk_overlap=20, language="french")
        chunks = chunker.chunk_article(sample_article)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Check metadata
        for chunk in chunks:
            assert chunk.article_id == 1
            assert chunk.reference == "Art. 1.1.1"
            assert chunk.code == "Code Test"
            assert chunk.book == "Livre 1"

    def test_chunk_positions(self, sample_article):
        """Test character positions."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=20)
        chunks = chunker.chunk_article(sample_article)

        # Verify positions make sense
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start
            assert chunk.char_end <= len(sample_article.article)

    def test_batch_chunking(self):
        articles = [
            LegalArticle(
                id=i,
                reference=f"Art. {i}",
                article=f"Article numéro {i}. Contenu de l'article {i}.",
                law_type="regional",
                code="Code Test",
            )
            for i in range(3)
        ]

        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_articles(articles)

        assert len(chunks) == 3  # One chunk per article (they're short)

        # Verify chunk IDs
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        assert "0_chunk_0" in chunk_ids
        assert "1_chunk_0" in chunk_ids
        assert "2_chunk_0" in chunk_ids
