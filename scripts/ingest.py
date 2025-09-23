
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking import TextChunker
from embeddings import CloudEmbedder, LocalEmbedder
from utils.helpers import save_json, update_metrics
from utils.logging_config import setup_logger
from utils.models import LegalArticle
from vector_store import FAISSVectorStore

logger = setup_logger(__name__, log_file=Path("logs/ingestion.log"))


def load_articles(csv_path: Path) -> list[LegalArticle]:

    logger.info(f"Loading articles from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} rows from CSV")

    articles = []
    records = df.to_dict('records')

    for i, row in enumerate(records):
        if i % 5000 == 0 and i > 0:
            logger.info(f"Processing article {i}/{len(records)}...")

        article = LegalArticle(
            id=int(row['id']),
            reference=str(row['reference']),
            article=str(row['article']),
            law_type=str(row['law_type']),
            code=str(row['code']),
            book=str(row['book']) if pd.notna(row['book']) else None,
            part=str(row['part']) if pd.notna(row['part']) else None,
            act=str(row['act']) if pd.notna(row['act']) else None,
            chapter=str(row['chapter']) if pd.notna(row['chapter']) else None,
            section=str(row['section']) if pd.notna(row['section']) else None,
            subsection=str(row['subsection']) if pd.notna(row['subsection']) else None,
            description=str(row['description']) if pd.notna(row['description']) else None,
        )
        articles.append(article)

    logger.info(f"Loaded {len(articles)} articles")
    return articles


def main(
    csv_path: str = "data/articles.csv",
    output_dir: str = "data/vector_store",
    embedder_type: str = "local",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    """Run the ingestion pipeline."""
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    articles = load_articles(Path(csv_path))

    logger.info(f"Chunking articles (size={chunk_size}, overlap={chunk_overlap})...")
    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language="french",
    )
    chunks = chunker.chunk_articles(articles)
    logger.info(f"Created {len(chunks)} chunks")

    # Save chunks metadata
    chunks_metadata = [chunk.to_dict() for chunk in chunks]
    save_json(chunks_metadata, output_path / "chunks_metadata.json")
    logger.info(f"Saved chunks metadata to {output_path / 'chunks_metadata.json'}")

    # Initialize embedder
    if embedder_type == "local":
        logger.info("Initializing local embedder...")
        embedder = LocalEmbedder(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device="cpu",
        )
    elif embedder_type == "cloud":
        logger.info("Initializing cloud embedder (Mistral)...")
        embedder = CloudEmbedder(model_name="mistral-embed", batch_size=100)
    elif embedder_type == "mistral":
        logger.info("Initializing Mistral embedder...")
        embedder = CloudEmbedder(model_name="mistral-embed", batch_size=100)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}. Choose 'local', 'cloud', or 'mistral'")

    embedding_dim = embedder.get_embedding_dim()
    logger.info(f"Embedding dimension: {embedding_dim}")

    logger.info("Generating embeddings...")
    chunk_texts = [chunk.original_text for chunk in chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    logger.info(f"Generated {len(embeddings)} embeddings")

    logger.info("Creating vector store...")
    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.add_documents(chunks, embeddings)

    logger.info("Persisting vector store...")
    vector_store.persist(output_path)

    config = {
        'embedder_type': embedder_type,
        'embedding_dim': embedding_dim,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'num_articles': len(articles),
        'num_chunks': len(chunks),
        'timestamp': datetime.now().isoformat(),
    }
    save_json(config, output_path / "config.json")

    # Update metrics
    elapsed_time = time.time() - start_time
    metrics_path = Path("data/metrics/metrics.json")
    update_metrics(metrics_path, {
        'ingested_docs': len(articles),
        'chunks': len(chunks),
        'last_ingestion_time_seconds': elapsed_time,
    })

    logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds")
    logger.info(f"Vector store saved to {output_path}")
    logger.info(f"Total articles: {len(articles)}, Total chunks: {len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Belgian law articles")
    parser.add_argument(
        "--csv",
        default="data/articles.csv",
        help="Path to articles CSV",
    )
    parser.add_argument(
        "--output",
        default="data/vector_store",
        help="Output directory for vector store",
    )
    parser.add_argument(
        "--embedder",
        choices=["local", "cloud", "mistral"],
        default="local",
        help="Embedder type: 'local' (CPU), 'cloud' (OpenAI), or 'mistral' (Mistral API)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters",
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        output_dir=args.output,
        embedder_type=args.embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
