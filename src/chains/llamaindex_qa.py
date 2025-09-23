
from typing import List, Optional

from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from utils.logging_config import setup_logger
from utils.models import QueryResponse, RetrievalResult, TextChunk

logger = setup_logger(__name__)


class LlamaIndexQA:
    """QA chain using LlamaIndex.

    Note: alternative orchestration using LlamaIndex.
    For full integration, it would need custom retrievers to use the FAISS store.
    """

    def __init__(
        self,
        chunks: List[TextChunk],
        llm_type: str = "cloud",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LlamaIndex QA.

        Args:
            chunks: All text chunks to index
            llm_type: "cloud" or "local"
            model_name: Model name (optional)
            api_key: API key for cloud models
        """
        self.llm_type = llm_type

        if llm_type == "cloud":
            model_name = model_name or "gpt-3.5-turbo"
            llm = OpenAI(model=model_name, temperature=0.3, api_key=api_key)
            embed_model = OpenAIEmbedding(api_key=api_key)
            logger.info(f"LlamaIndex QA initialized with cloud model: {model_name}")
        else:
            logger.warning("Local LLM not implemented for LlamaIndex")
            raise NotImplementedError("Local LLM support requires additional setup")

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )

        # Convert chunks to LlamaIndex documents
        documents = []
        for chunk in chunks:
            doc = Document(
                text=chunk.original_text,
                metadata={
                    'chunk_id': chunk.chunk_id,
                    'article_id': chunk.article_id,
                    'reference': chunk.reference,
                    'code': chunk.code,
                    'book': chunk.book or '',
                    'chapter': chunk.chapter or '',
                    'section': chunk.section or '',
                    **chunk.metadata,
                },
            )
            documents.append(doc)

        # Build index
        logger.info(f"Building llamaIndex with {len(documents)} documents...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
        )

        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )

        logger.info("llamaIndex QA initialized")

    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        logger.info(f"Querying LlamaIndex: {question[:100]}...")

        try:
            response = self.query_engine.query(question)

            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    metadata = node.node.metadata
                    result = RetrievalResult(
                        chunk_id=metadata.get('chunk_id', ''),
                        text=node.node.text,
                        score=node.score if hasattr(node, 'score') else 0.0,
                        metadata=metadata,
                        article_id=metadata.get('article_id', 0),
                        reference=metadata.get('reference', ''),
                        code=metadata.get('code', ''),
                    )
                    sources.append(result)

            query_response = QueryResponse(
                query=question,
                answer=str(response),
                sources=sources[:top_k],
                retrieval_details={
                    "num_sources": len(sources),
                    "retrieval_method": "LlamaIndex",
                    "llm_type": self.llm_type,
                },
            )

            logger.info("LlamaIndex query completed successfully")
            return query_response

        except Exception as e:
            logger.error(f"LlamaIndex query failed: {e}")
            return QueryResponse(
                query=question,
                answer=f"Erreur lors de la requête: {e}",
                sources=[],
                retrieval_details={"error": str(e)},
            )
