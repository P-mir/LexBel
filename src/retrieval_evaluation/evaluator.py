import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from retrieval_evaluation.metrics import (
    compute_average_precision,
    compute_metrics_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)
from retrieval_evaluation.results import EvaluationResult, QueryResult
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class RetrieverEvaluator:
    """Evaluator for comparing retriever performance on test data"""

    def __init__(
        self,
        chunk_to_article_mapping: Dict[str, int],
        k_values: List[int] = [5, 10, 20],
    ):
        self.chunk_to_article_mapping = chunk_to_article_mapping
        self.k_values = sorted(k_values)
        logger.info(f"RetrieverEvaluator initialized with k_values={self.k_values}")
        logger.info(f"Loaded mapping for {len(chunk_to_article_mapping)} chunks to articles")

    @staticmethod
    def load_test_questions(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} test questions from {csv_path}")
        return df

    @staticmethod
    def load_chunk_to_article_mapping(metadata_path: Path) -> Dict[str, int]:
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_metadata = json.load(f)

        mapping = {}
        for chunk in chunks_metadata:
            chunk_id = chunk.get("chunk_id")
            article_id = chunk.get("article_id")
            if chunk_id and article_id:
                mapping[chunk_id] = article_id

        logger.info(f"Loaded mapping for {len(mapping)} chunks")
        return mapping

    def _parse_article_ids(self, article_ids_str: str) -> List[int]:
        if pd.isna(article_ids_str) or not article_ids_str:
            return []

        article_ids_str = str(article_ids_str).strip()
        if not article_ids_str:
            return []

        try:
            return [int(aid.strip()) for aid in article_ids_str.split(",")]
        except ValueError as e:
            logger.warning(f"Failed to parse article_ids: {article_ids_str} - {e}")
            return []

    def _extract_article_ids_from_results(self, retrieval_results: List[Any]) -> List[int]:
        article_ids = []
        seen = set()

        for result in retrieval_results:
            if hasattr(result, "article_id"):
                article_id = result.article_id
            elif hasattr(result, "chunk_id"):
                chunk_id = result.chunk_id
                article_id = self.chunk_to_article_mapping.get(chunk_id)
                if article_id is None:
                    logger.warning(f"No article mapping found for chunk_id: {chunk_id}")
                    continue
            else:
                logger.warning(f"Result has no chunk_id or article_id: {result}")
                continue

            if article_id not in seen:
                article_ids.append(article_id)
                seen.add(article_id)

        return article_ids

    def evaluate_retriever(
        self,
        retriever,
        test_df: pd.DataFrame,
        retriever_name: str,
        config: Dict = None,
        max_k: int = None,
    ) -> EvaluationResult:
        if config is None:
            config = {}

        if max_k is None:
            max_k = max(self.k_values)

        logger.info(f"Evaluating {retriever_name} on {len(test_df)} queries...")
        logger.info(f"Configuration: {config}")

        query_results = []
        retrieved_lists = []
        relevant_sets = []

        for idx, row in test_df.iterrows():
            query_id = int(row["id"])
            query_text = str(row["question"])

            relevant_articles = self._parse_article_ids(row["article_ids"])
            if not relevant_articles:
                logger.warning(f"Query {query_id} has no relevant articles, skipping")
                continue

            relevant_set = set(relevant_articles)

            try:
                retrieval_results = retriever.retrieve(query_text, top_k=max_k)
            except Exception as e:
                logger.error(f"Error retrieving for query {query_id}: {e}")
                continue

            retrieved_articles = self._extract_article_ids_from_results(retrieval_results)

            ap = compute_average_precision(retrieved_articles, relevant_set)
            rr = compute_reciprocal_rank(retrieved_articles, relevant_set)

            precision_at_k = {}
            recall_at_k = {}
            for k in self.k_values:
                precision_at_k[k] = compute_precision_at_k(retrieved_articles, relevant_set, k)
                recall_at_k[k] = compute_recall_at_k(retrieved_articles, relevant_set, k)

            query_result = QueryResult(
                query_id=query_id,
                query_text=query_text,
                relevant_articles=relevant_articles,
                retrieved_articles=retrieved_articles[:max_k],
                average_precision=ap,
                reciprocal_rank=rr,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
            )

            query_results.append(query_result)
            retrieved_lists.append(retrieved_articles)
            relevant_sets.append(relevant_set)

        overall_metrics = compute_metrics_at_k(retrieved_lists, relevant_sets, self.k_values)

        logger.info(f"Evaluation complete: MAP={overall_metrics['MAP']:.4f}")

        return EvaluationResult(
            retriever_name=retriever_name,
            config=config,
            overall_metrics=overall_metrics,
            query_results=query_results,
            num_queries=len(query_results),
            k_values=self.k_values,
        )
