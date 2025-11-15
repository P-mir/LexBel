#!/usr/bin/env python3
"""Evaluate answer quality using LLM-as-a-judge on test questions.
Usage:
    python scripts/evaluate_answers.py [--limit N] [--config CONFIG_NAME]

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from dotenv import load_dotenv

from chains.conversational_qa import ConversationalQA
from embeddings.cloud_embedder import CloudEmbedder
from llm_judge import AnswerEvaluator, LLMJudge
from llm_judge.metrics import find_weak_answers, print_report_summary
from retrievers.hybrid import HybridRetriever
from retrievers.mmr import MMRRetriever
from utils.logging_config import setup_logger
from vector_store.faiss_store import FAISSVectorStore

load_dotenv()

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="run eval using LLM-as-a-judge")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test questions (for quick testing)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conversational_qa_mmr",
        help="Config name for the run",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="Lambda parameter for MMR",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="mmr",
        choices=["mmr", "hybrid", "vector"],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for hybrid retriever",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-small-latest",
        help="Mistral model name for QA generation",
    )
    return parser.parse_args()


def main():
    """Run answer quality evaluation using LLM-as-a-judge."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Starting Answer Quality Evaluation with LLM-as-a-Judge")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Retriever: {args.retriever}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Model: {args.model}")
    if args.retriever == "mmr":
        logger.info(f"MMR Lambda: {args.mmr_lambda}")
    elif args.retriever == "hybrid":
        logger.info(f"Hybrid Alpha: {args.alpha}")
    if args.limit:
        logger.info(f"Limit: {args.limit} questions")

    # paths
    data_dir = Path("data")
    test_questions_path = data_dir / "test" / "questions_test.csv"
    vector_store_path = data_dir / "vector_store"
    results_dir = Path("evals") / "end_to_end"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading test questions from {test_questions_path}")
    questions_df = pd.read_csv(test_questions_path)
    logger.info(f"Loaded {len(questions_df)} test questions")

    logger.info("Initializing embeddings and vector store...")
    embedder = CloudEmbedder()
    vector_store = FAISSVectorStore(embedding_dim=1024)
    vector_store.load(vector_store_path)

    chunks = None
    if args.retriever == "hybrid":
        import json

        chunks_metadata_path = vector_store_path / "chunks_metadata.json"
        logger.info(f"Loading chunks metadata from {chunks_metadata_path}...")
        with open(chunks_metadata_path, "r") as f:
            chunks_data = json.load(f)
        from utils.models import TextChunk

        chunks = [TextChunk(**chunk) for chunk in chunks_data]
        logger.info(f"Loaded {len(chunks)} chunks for hybrid retriever")

    if args.retriever == "mmr":
        logger.info(f"Initializing MMR retriever (lambda={args.mmr_lambda})...")
        retriever = MMRRetriever(
            vector_store=vector_store,
            embedder=embedder,
            lambda_param=args.mmr_lambda,
        )
    elif args.retriever == "hybrid":
        logger.info(f"Initializing Hybrid retriever (alpha={args.alpha})...")
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            chunks=chunks,
            alpha=args.alpha,
        )

    logger.info(f"Initializing ConversationalQA chain with {args.model}...")
    qa_chain = ConversationalQA(
        retriever=retriever,
        model_name=args.model,
    )

    logger.info("Initializing LLM Judge...")
    judge = LLMJudge(model_name="gpt-4o-mini")

    evaluator = AnswerEvaluator(
        qa_chain=qa_chain,
        judge=judge,
        config_name=args.config,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Running Evaluation...")
    logger.info("=" * 80)

    report = evaluator.evaluate_dataset(
        questions_df=questions_df,
        top_k=args.top_k,
        limit=args.limit,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"llm_judge_{args.config}_{timestamp}.json"
    output_path = results_dir / output_filename

    evaluator.save_report(report, output_path)

    print_report_summary(report)

    # Find weak answers for analysis
    weak_answers_df = find_weak_answers(report, threshold=3, limit=10)
    if not weak_answers_df.empty:
        print("\nWeak Answers (Score < 3) for Further Analysis:")
        print("-" * 80)
        print(weak_answers_df.to_string(index=False))
        print("\n")

    logger.info("=" * 80)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
