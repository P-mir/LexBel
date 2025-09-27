
import os
import time
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

from llm import LocalLLM as LocalLLMAdapter
from observability import calculate_cost, estimate_tokens, get_tracer
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
        self.model_name = model_name

    def query(
        self,
        question: str,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> QueryResponse:
        """Answer a question using the RAG system.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            trace_id: Optional trace ID for Langfuse
            session_id: Optional session ID for grouping queries

        Returns:
            QueryResponse with answer and sources
        """
        tracer = get_tracer()
        trace = None

        # Create trace if enabled
        if tracer.enabled and not trace_id:
            trace = tracer.create_trace(
                name="rag_query",
                session_id=session_id,
                metadata={
                    "retriever_type": type(self.retriever).__name__,
                    "llm_type": self.llm_type,
                    "model": self.model_name,
                    "top_k": top_k
                },
                tags=["rag", "legal", "qa"]
            )
            trace_id = trace.id if trace else None

        retrieval_start = time.time()
        logger.info(f"Retrieving documents for question: {question[:100]}...")

        # Retrieval span
        if tracer.enabled and trace_id:
            retrieval_span = tracer.client.span(
                trace_id=trace_id,
                name="retrieval",
                input={"question": question, "top_k": top_k},
                start_time=retrieval_start
            )

        sources = self.retriever.retrieve(question, top_k=top_k)
        retrieval_duration = time.time() - retrieval_start

        # Update retrieval span
        if tracer.enabled and trace_id:
            avg_score = sum(s.score for s in sources) / len(sources) if sources else 0
            retrieval_span.end(
                output={
                    "num_sources": len(sources),
                    "sources": [s.reference for s in sources],
                    "avg_similarity_score": avg_score
                },
                metadata={
                    "duration_ms": retrieval_duration * 1000,
                    "retriever": type(self.retriever).__name__
                }
            )

        if not sources:
            response = QueryResponse(
                query=question,
                answer="Je n'ai pas trouvé d'articles pertinents pour répondre à cette question.",
                sources=[],
                retrieval_details={"num_sources": 0},
            )

            if trace:
                trace.update(
                    output={"answer": response.answer, "num_sources": 0},
                    metadata={"status": "no_sources_found"}
                )
                tracer.flush()

            return response

        # Format context
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Article {i}] {source.reference}\n{source.text}\n"
            )
        context = "\n".join(context_parts)

        input_tokens = estimate_tokens(context) + estimate_tokens(question)

        logger.info("Generating answer with LLM...")
        generation_start = time.time()

        try:
            result = self.chain.run(context=context, question=question)
            answer = result.strip()
            generation_duration = time.time() - generation_start

            # Estimate output tokens and cost
            output_tokens = estimate_tokens(answer)
            cost_info = calculate_cost(self.model_name, input_tokens, output_tokens)

            # Track generation
            if tracer.enabled and trace_id:
                tracer.client.generation(
                    trace_id=trace_id,
                    name="llm_generation",
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": QA_PROMPT_TEMPLATE},
                        {"role": "user", "content": f"Context: {context[:200]}...\n\nQuestion: {question}"}
                    ],
                    output=answer,
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens,
                        "unit": "TOKENS"
                    },
                    metadata={
                        "duration_ms": generation_duration * 1000,
                        "cost_usd": cost_info["total_cost"],
                        "input_cost_usd": cost_info["input_cost"],
                        "output_cost_usd": cost_info["output_cost"]
                    },
                    start_time=generation_start,
                    end_time=time.time()
                )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Erreur lors de la génération de la réponse: {e}"
            cost_info = {"total_cost": 0}

            if tracer.enabled and trace_id:
                tracer.client.span(
                    trace_id=trace_id,
                    name="llm_error",
                    input={"question": question},
                    output={"error": str(e)},
                    metadata={"error_type": type(e).__name__},
                    level="ERROR"
                )

        # Build response
        response = QueryResponse(
            query=question,
            answer=answer,
            sources=sources,
            retrieval_details={
                "num_sources": len(sources),
                "retrieval_method": type(self.retriever).__name__,
                "llm_type": self.llm_type,
                "cost_usd": cost_info.get("total_cost", 0),
                "total_tokens": cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0)
            },
        )

        # Update trace
        if trace:
            trace.update(
                output={
                    "answer": answer,
                    "num_sources": len(sources),
                    "cost_usd": cost_info.get("total_cost", 0)
                },
                metadata={
                    "total_tokens": cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0),
                    "retrieval_duration_ms": retrieval_duration * 1000
                }
            )
            tracer.flush()

        logger.info("Query completed successfully")
        return response
