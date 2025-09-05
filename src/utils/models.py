
from dataclasses import dataclass
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
