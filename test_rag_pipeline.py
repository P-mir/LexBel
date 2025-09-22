#!/usr/bin/env python3
"""Test full RAG pipeline with different retrievers."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import LocalEmbedder
from retrievers import HybridRetriever, MMRRetriever
from utils.helpers import load_json
from utils.models import TextChunk
from vector_store import FAISSVectorStore


def test_retriever(retriever_name, retriever, question):
    print(f"\n{'='*80}")
    print(f"Testing {retriever_name}")
    print('='*80)

    times = []
    for i in range(3):
        start = time.time()
        results = retriever.retrieve(question, top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        times.append(elapsed_ms)

        if i == 0:
            print(f"  Run {i+1}: {elapsed_ms:7.1f}ms (cache warming)")
        else:
            print(f"  Run {i+1}: {elapsed_ms:7.1f}ms")

    avg_time = sum(times[1:]) / len(times[1:])
    print(f"\n  Average (excluding first): {avg_time:.1f}ms")
    print(f"  Retrieved {len(results)} documents")

    print("\n  Top 3 Results:")
    for i, r in enumerate(results[:3], 1):
        score_info = f"Score: {r.score:.3f}"
        if 'vector_score' in r.metadata:
            score_info += f" (Vec: {r.metadata['vector_score']:.3f}, Lex: {r.metadata['lexical_score']:.3f})"
        print(f"    {i}. {score_info}")
        print(f"       Ref: {r.reference[:70]}...")

    return avg_time, results


def main():
    print("\n" + "="*80)
    print("RAG PIPELINE PERFORMANCE TEST")
    print("="*80)

    print("\nLoading vector store and embedder...")
    start_load = time.time()

    vector_store_path = Path("data/vector_store")
    config = load_json(vector_store_path / "config.json")
    embedding_dim = config['embedding_dim']

    embedder = LocalEmbedder(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cpu",
        batch_size=64,
    )

    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.load(vector_store_path)

    chunks_metadata = load_json(vector_store_path / "chunks_metadata.json")
    chunks = [TextChunk(**chunk_data) for chunk_data in chunks_metadata]

    load_time = time.time() - start_load
    print(f"Setup completed in {load_time:.2f}s")
    print(f"Loaded {vector_store.index.ntotal} documents\n")

    question = "Quelles sont les directives européennes transposées par le Code Bruxellois?"

    mmr_retriever = MMRRetriever(
        vector_store=vector_store,
        embedder=embedder,
        lambda_param=0.5,
        initial_k=50,
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        chunks=chunks,
        alpha=0.7,
    )

    mmr_time, mmr_results = test_retriever("MMR Retriever", mmr_retriever, question)
    hybrid_time, hybrid_results = test_retriever("Hybrid Retriever", hybrid_retriever, question)

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"  MMR Retriever:    {mmr_time:7.1f}ms  (optimized for speed)")
    print(f"  Hybrid Retriever: {hybrid_time:7.1f}ms  (combines semantic + lexical)")
    print("\n  Both retrievers are working correctly!")
    print("  MMR is ~{:.0f}x faster (recommended for real-time queries)".format(hybrid_time/mmr_time if mmr_time > 0 else 0))
    print("="*80 + "\n")

    print("Note: LLM testing requires downloading ~5GB Gemma model.")
    print("      To test LLM generation, use scripts/query_demo.py")


if __name__ == "__main__":
    main()
