#!/usr/bin/env python3
"""Evaluate and compare different retriever configurations.

This script evaluates MMR and BM25 (lexical) retrievers with different
configurations on the test question dataset.

Usage:
    python scripts/evaluate_retrievers.py

Results are saved to data/test/eval_results/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from embeddings.cloud_embedder import CloudEmbedder
from evaluation.evaluator import RetrieverEvaluator
from evaluation.results import ComparisonReport
from retrievers.hybrid import HybridRetriever
from retrievers.mmr import MMRRetriever
from utils.logging_config import setup_logger
from utils.models import TextChunk
from vector_store.faiss_store import FAISSVectorStore

logger = setup_logger(__name__)


def main():
    """Run retriever evaluation experiments."""
    logger.info("=" * 80)
    logger.info("Starting Retriever Evaluation")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("data")
    test_questions_path = data_dir / "test" / "questions_test.csv"
    chunks_metadata_path = data_dir / "vector_store" / "chunks_metadata.json"
    vector_store_path = data_dir / "vector_store"
    results_dir = data_dir / "test" / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    k_values = [5, 10, 20]
    
    # Alpha values for hybrid retriever (0.0 = pure lexical/BM25)
    alpha_values = [0.0, 0.5, 1.0]  # pure lexical, balanced, pure vector
    
    # Lambda values for MMR (balance relevance vs diversity)
    lambda_values = [0.7]  # more relevance

    logger.info(f"K values: {k_values}")
    logger.info(f"Alpha values for Hybrid: {alpha_values}")
    logger.info(f"Lambda values for MMR: {lambda_values}")

    # Load test data
    logger.info("\nLoading test data...")
    test_df = RetrieverEvaluator.load_test_questions(test_questions_path)
    chunk_to_article = RetrieverEvaluator.load_chunk_to_article_mapping(
        chunks_metadata_path
    )

    # Initialize evaluator
    evaluator = RetrieverEvaluator(
        chunk_to_article_mapping=chunk_to_article, k_values=k_values
    )

    # Load vector store and embedder
    logger.info("\nLoading vector store and embedder...")
    embedder = CloudEmbedder(model_name="mistral-embed")
    
    # Load vector store
    vector_store = FAISSVectorStore(embedding_dim=1024)
    vector_store.load(vector_store_path)

    # Load chunks for hybrid retriever
    logger.info("Loading chunks...")
    with open(chunks_metadata_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = [
        TextChunk(
            chunk_id=chunk["chunk_id"],
            original_text=chunk["original_text"],
            article_id=chunk["article_id"],
            reference=chunk["reference"],
            code=chunk["code"],
            book=chunk.get("book"),
            chapter=chunk.get("chapter"),
            section=chunk.get("section"),
            char_start=chunk.get("char_start", 0),
            char_end=chunk.get("char_end", len(chunk["original_text"])),
            metadata=chunk.get("metadata", {}),
        )
        for chunk in chunks_data
    ]
    logger.info(f"Loaded {len(chunks)} chunks")

    # Run evaluations
    all_results = []

    # 1. Evaluate Hybrid retrievers with different alpha values
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Hybrid Retrievers (BM25 when alpha=0.0)")
    logger.info("=" * 80)

    for alpha in alpha_values:
        logger.info(f"\n--- Hybrid Retriever (alpha={alpha}) ---")
        
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            chunks=chunks,
            alpha=alpha,
        )

        retriever_name = f"Hybrid_alpha{alpha}"
        if alpha == 0.0:
            retriever_name = "BM25_Lexical"
        elif alpha == 1.0:
            retriever_name = "VectorOnly"

        result = evaluator.evaluate_retriever(
            retriever=retriever,
            test_df=test_df,
            retriever_name=retriever_name,
            config={"alpha": alpha, "type": "hybrid"},
            max_k=max(k_values),
        )

        result.print_summary()
        result.save(results_dir / f"{retriever_name.lower()}_results.json")
        all_results.append(result)

    # 2. Evaluate MMR retrievers with different lambda values
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating MMR Retrievers")
    logger.info("=" * 80)

    for lambda_param in lambda_values:
        logger.info(f"\n--- MMR Retriever (lambda={lambda_param}) ---")
        
        retriever = MMRRetriever(
            vector_store=vector_store,
            embedder=embedder,
            lambda_param=lambda_param,
            initial_k=50,  # Retrieve 50 candidates for MMR reranking
        )

        retriever_name = f"MMR_lambda{lambda_param}"

        result = evaluator.evaluate_retriever(
            retriever=retriever,
            test_df=test_df,
            retriever_name=retriever_name,
            config={"lambda": lambda_param, "type": "mmr", "initial_k": 50},
            max_k=max(k_values),
        )

        result.print_summary()
        result.save(results_dir / f"{retriever_name.lower()}_results.json")
        all_results.append(result)

    # Generate comparison report
    logger.info("\n" + "=" * 80)
    logger.info("Generating Comparison Report")
    logger.info("=" * 80)

    comparison = ComparisonReport(results=all_results)
    comparison.print_comparison_table()
    comparison.save(results_dir / "comparison_report.json")

    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
