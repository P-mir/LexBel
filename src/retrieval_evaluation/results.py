from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class QueryResult:
    """Result for a single query evaluation"""
    
    query_id: int
    query_text: str
    relevant_articles: List[int]
    retrieved_articles: List[int]
    average_precision: float
    reciprocal_rank: float
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "relevant_articles": self.relevant_articles,
            "retrieved_articles": self.retrieved_articles,
            "average_precision": round(self.average_precision, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "precision_at_k": {k: round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {k: round(v, 4) for k, v in self.recall_at_k.items()},
        }
