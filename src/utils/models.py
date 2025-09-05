
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LegalArticle:
    """Represents a legal article from the CSV."""

    id: int
    reference: str
    article: str
    law_type: str
    code: str
    book: Optional[str] = None
    part: Optional[str] = None
    act: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'reference': self.reference,
            'article': self.article,
            'law_type': self.law_type,
            'code': self.code,
            'book': self.book,
            'part': self.part,
            'act': self.act,
            'chapter': self.chapter,
            'section': self.section,
            'subsection': self.subsection,
            'description': self.description,
        }


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    chunk_id: str
    original_text: str
    article_id: int
    reference: str
    code: str
    book: Optional[str]
    chapter: Optional[str]
    section: Optional[str]
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'original_text': self.original_text,
            'article_id': self.article_id,
            'reference': self.reference,
            'code': self.code,
            'book': self.book,
            'chapter': self.chapter,
            'section': self.section,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'metadata': self.metadata,
        }
