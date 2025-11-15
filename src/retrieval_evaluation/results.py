from dataclasses import dataclass, field
from typing import Dict, List
import json
from pathlib import Path


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


@dataclass
class EvaluationResult:
    """Complete evaluation result for a retriever configuration"""

    retriever_name: str
    config: Dict
    overall_metrics: Dict[str, float]
    query_results: List[QueryResult]
    num_queries: int
    k_values: List[int]

    def to_dict(self) -> dict:
        return {
            "retriever_name": self.retriever_name,
            "config": self.config,
            "overall_metrics": {k: round(v, 4) for k, v in self.overall_metrics.items()},
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "query_results": [qr.to_dict() for qr in self.query_results],
        }

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def print_summary(self):
        print(f"\n{'=' * 70}")
        print(f"Evaluation Results: {self.retriever_name}")
        print(f"{'=' * 70}")
        print(f"Configuration: {self.config}")
        print(f"Number of queries: {self.num_queries}")
        print("\nOverall Metrics:")
        print(f"  MAP:  {self.overall_metrics['MAP']:.4f}")
        print(f"  MRR:  {self.overall_metrics['MRR']:.4f}")
        print("\nPrecision@K:")
        for k in self.k_values:
            print(f"  P@{k:2d}: {self.overall_metrics[f'P@{k}']:.4f}")
        print("\nRecall@K:")
        for k in self.k_values:
            print(f"  R@{k:2d}: {self.overall_metrics[f'R@{k}']:.4f}")
        print(f"{'=' * 70}\n")


@dataclass
class ComparisonReport:
    results: List[EvaluationResult]
    best_by_metric: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.results:
            return

        all_metrics = self.results[0].overall_metrics.keys()

        for metric in all_metrics:
            best_result = max(self.results, key=lambda r: r.overall_metrics.get(metric, 0.0))
            self.best_by_metric[metric] = best_result.retriever_name

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "best_by_metric": self.best_by_metric,
        }

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def print_comparison_table(self):
        if not self.results:
            print("No results to compare")
            return

        print(f"\n{'=' * 100}")
        print("RETRIEVER COMPARISON")
        print(f"{'=' * 100}")

        print(f"{'Retriever':<30} {'Config':<25} {'MAP':<8} {'MRR':<8} ", end="")
        k_values = self.results[0].k_values
        for k in k_values:
            print(f"P@{k:<3} ", end="")
        for k in k_values:
            print(f"R@{k:<3} ", end="")
        print()
        print("-" * 100)

        for result in self.results:
            config_str = self._format_config(result.config)
            print(f"{result.retriever_name:<30} {config_str:<25} ", end="")
            print(f"{result.overall_metrics['MAP']:<8.4f} ", end="")
            print(f"{result.overall_metrics['MRR']:<8.4f} ", end="")

            for k in k_values:
                print(f"{result.overall_metrics[f'P@{k}']:<5.3f} ", end="")
            for k in k_values:
                print(f"{result.overall_metrics[f'R@{k}']:<5.3f} ", end="")
            print()

        print(f"{'=' * 100}")
        print("\nBest Performer by Metric:")
        for metric, retriever in self.best_by_metric.items():
            print(f"  {metric:<10}: {retriever}")
        print(f"{'=' * 100}\n")

    def _format_config(self, config: Dict) -> str:
        items = []
        for key, value in config.items():
            if isinstance(value, float):
                items.append(f"{key}={value:.2f}")
            else:
                items.append(f"{key}={value}")
        return ", ".join(items)
