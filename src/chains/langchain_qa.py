
import os
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

from llm import LocalLLM as LocalLLMAdapter
from utils.logging_config import setup_logger
from utils.models import QueryResponse

logger = setup_logger(__name__)


class LocalLLMWrapper(LLM):
    """Wrapp the local model to work with LangChain.
    -->>https://python.langchain.com/docs/how_to/custom_llm/
    """

    local_llm: Any

    @property
    def _llm_type(self) -> str:
        return "local_flan_t5"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        """Call the local LLM."""
        return self.local_llm.generate(prompt, max_tokens=512, temperature=0.3)

    @property
    def _identifying_params(self):
        """Return identifying parameters."""
        return {"model_name": self.local_llm.model_name}


QA_PROMPT_TEMPLATE = """Tu es un assistant juridique expert en droit belge. Réponds à la question suivante en te basant UNIQUEMENT sur les articles de loi fournis ci-dessous.

Articles de loi pertinents:
{context}

Question: {question}

Instructions:
1. Base ta réponse uniquement sur les articles fournis
2. Cite les références exactes des articles utilisés
3. Si l'information n'est pas dans les articles fournis, indique-le clairement
4. Sois précis et concis
5. Utilise un langage juridique approprié mais compréhensible

Réponse:"""


class LangChainQA:

    def __init__(
        self,
        retriever: Any,
        llm_type: str = "cloud",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LangChain QA.

        Args:
            retriever: Retriever instance (MMR or Hybrid)
            llm_type: "cloud" for Mistral API
            model_name: Model name (optional)
            api_key: API key for Mistral (optional, reads from MISTRAL_API_KEY env var)
        """
        self.retriever = retriever
        self.llm_type = llm_type

        if llm_type == "cloud":
            model_name = model_name or "mistral-small-latest" # cheap, fast and reasonably good
            api_key = api_key or os.getenv("MISTRAL_API_KEY")

            if not api_key:
                raise ValueError(
                    "Mistral API key not found. Please set MISTRAL_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self.llm = ChatMistralAI(
                model=model_name,
                temperature=0.3,
                api_key=api_key,
            )
            logger.info(f"LangChain QA initialized with Mistral model: {model_name}")
        elif llm_type == "local":
            model_name = model_name or "Qwen/Qwen2.5-0.5B-Instruct"
            local_llm_adapter = LocalLLMAdapter(model_name=model_name, device="cpu")
            self.llm = LocalLLMWrapper(local_llm=local_llm_adapter)
            logger.info(f"LangChain QA initialized with local model: {model_name}")
        else:
            raise ValueError(f"Invalid llm_type: {llm_type}. Must be 'cloud' or 'local'")

        self.prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        """Answer a question using the RAG system.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            QueryResponse with answer and sources
        """
        logger.info(f"Retrieving documents for question: {question[:100]}...")
        sources = self.retriever.retrieve(question, top_k=top_k)

        if not sources:
            return QueryResponse(
                query=question,
                answer="Je n'ai pas trouvé d'articles pertinents pour répondre à cette question.",
                sources=[],
                retrieval_details={"num_sources": 0},
            )

        # Format context
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Article {i}] {source.reference}\n{source.text}\n"
            )
        context = "\n".join(context_parts)

        logger.info("Generating answer with LLM...")
        try:
            result = self.chain.run(context=context, question=question)
            answer = result.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Erreur lors de la génération de la réponse: {e}"

        # Build response
        response = QueryResponse(
            query=question,
            answer=answer,
            sources=sources,
            retrieval_details={
                "num_sources": len(sources),
                "retrieval_method": type(self.retriever).__name__,
                "llm_type": self.llm_type,
            },
        )

        logger.info("Query completed successfully")
        return response
