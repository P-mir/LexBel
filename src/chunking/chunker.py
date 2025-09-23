
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from utils.logging_config import setup_logger
from utils.models import LegalArticle, TextChunk

logger = setup_logger(__name__)


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        language: str = "french",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.language = language

        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        # Add French sentence tokenizer
        try:
            nltk.data.find('tokenizers/punkt/french.pickle')
        except LookupError:
            logger.info("Downloading French NLTK data...")
            nltk.download('punkt', quiet=True)

    def chunk_article(self, article: LegalArticle) -> List[TextChunk]:
        """LangChain chuncker can be limiting..."""

        text = article.article

        # Handle empty or very short text
        if len(text) < self.min_chunk_size:
            return [self._create_chunk(article, text, 0, len(text), 0)]

        # Split into sentences
        sentences = sent_tokenize(text, language=self.language)

        chunks = []
        current_chunk = []
        current_length = 0
        char_start = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                char_end = char_start + len(chunk_text)
                chunks.append(self._create_chunk(article, chunk_text, char_start, char_end, chunk_idx))
                chunk_idx += 1

                # Start new chunk with overlap
                # Find sentences to include in overlap
                overlap_chars = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_chars + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_chars += len(sent) + 1  # +1 for space
                    else:
                        break

                # Update char_start to reflect the overlap
                if overlap_sentences:
                    overlap_text = ' '.join(overlap_sentences)
                    char_start = char_end - len(overlap_text)
                    current_chunk = overlap_sentences
                    current_length = overlap_chars
                else:
                    char_start = char_end
                    current_chunk = []
                    current_length = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add final chunk if any
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            char_end = char_start + len(chunk_text)
            chunks.append(self._create_chunk(article, chunk_text, char_start, char_end, chunk_idx))

        logger.debug(f"Article {article.id} split into {len(chunks)} chunks")
        return chunks

    def _create_chunk(
        self,
        article: LegalArticle,
        text: str,
        char_start: int,
        char_end: int,
        chunk_idx: int,
    ) -> TextChunk:
        """Create a TextChunk with proper metadata.

        Args:
            article: Source article
            text: Chunk text
            char_start: Start character position
            char_end: End character position
            chunk_idx: Chunk index within article

        Returns:
            TextChunk with metadata
        """
        chunk_id = f"{article.id}_chunk_{chunk_idx}"

        return TextChunk(
            chunk_id=chunk_id,
            original_text=text,
            article_id=article.id,
            reference=article.reference,
            code=article.code,
            book=article.book,
            chapter=article.chapter,
            section=article.section,
            char_start=char_start,
            char_end=char_end,
            metadata={
                'law_type': article.law_type,
                'part': article.part,
                'act': article.act,
                'subsection': article.subsection,
                'description': article.description,
                'chunk_index': chunk_idx,
            }
        )

    def chunk_articles(self, articles: List[LegalArticle]) -> List[TextChunk]:
        """Chunk multiple articles.

        Args:
            articles: List of articles to chunk

        Returns:
            List of all chunks from all articles
        """
        all_chunks = []
        total = len(articles)

        for i, article in enumerate(articles):
            if i % 5000 == 0 and i > 0:
                logger.info(f"Chunking progress: {i}/{total} articles ({len(all_chunks)} chunks so far)")

            chunks = self.chunk_article(article)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(articles)} articles into {len(all_chunks)} chunks")
        return all_chunks

