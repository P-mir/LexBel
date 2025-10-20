import os
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, trim_messages
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from observability import calculate_cost, estimate_tokens, get_tracer
from utils.logging_config import setup_logger
from utils.models import QueryResponse

logger = setup_logger(__name__)


class ConversationalQA:
    def __init__(
        self,
        retriever: Any,
        model_name: str = "mistral-small-latest",
        api_key: Optional[str] = None,
        max_history_messages: int = 10,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.max_history_messages = max_history_messages

        api_key = api_key or os.getenv("MISTRAL_API_KEY")

        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            api_key=api_key,
        )

        self.memory = MemorySaver()
        self.graph = self._build_graph()

        logger.info(f"Conversational QA initialized with {model_name}")

    def _build_graph(self):
        workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            messages = state["messages"]

            trimmed = trim_messages(
                messages,
                token_counter=len,
                max_tokens=self.max_history_messages * 2,
                strategy="last",
                start_on="human",
                include_system=True,
                allow_partial=False,
            )

            response = self.llm.invoke(trimmed)
            return {"messages": response}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        return workflow.compile(checkpointer=self.memory)

    def query(
        self,
        question: str,
        top_k: int = 5,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> QueryResponse:
        tracer = get_tracer()
        retrieval_start = time.time()

        logger.info(f"Retrieving documents for: {question[:100]}...")
        sources = self.retriever.retrieve(question, top_k=top_k)
        retrieval_duration = time.time() - retrieval_start

        if not sources:
            return QueryResponse(
                query=question,
                answer="Je n'ai pas trouvé d'articles pertinents pour répondre à cette question.",
                sources=[],
                retrieval_details={"num_sources": 0},
            )

        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Article {i}] {source.reference}\n{source.text}\n")
        context = "\n".join(context_parts)

        system_prompt = f"""Tu es un assistant juridique expert en droit belge. Réponds à la question en te basant sur les articles de loi fournis et l'historique de conversation.

Articles de loi pertinents:
{context}

Instructions:
1. Base ta réponse sur les articles fournis et enrichis avec tes connaissances juridiques si nécessaire
2. Cite les références exactes des articles utilisés
3. Si l'information n'est pas dans les articles, indique-le clairement
4. Maintiens la cohérence avec la conversation précédente
5. Sois précis et concis"""

        input_tokens = estimate_tokens(context) + estimate_tokens(question) + estimate_tokens(system_prompt)

        logger.info("Generating answer with LLM...")
        generation_start = time.time()

        try:
            config = {"configurable": {"thread_id": thread_id or session_id or "default"}}
            input_messages = [HumanMessage(content=f"{system_prompt}\n\nQuestion: {question}")]

            result = self.graph.invoke({"messages": input_messages}, config=config)
            answer = result["messages"][-1].content.strip()
            generation_duration = time.time() - generation_start

            output_tokens = estimate_tokens(answer)
            cost_info = calculate_cost(self.model_name, input_tokens, output_tokens)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Erreur lors de la génération de la réponse: {e}"
            cost_info = {"total_cost": 0, "input_tokens": 0, "output_tokens": 0}

        response = QueryResponse(
            query=question,
            answer=answer,
            sources=sources,
            retrieval_details={
                "num_sources": len(sources),
                "retrieval_method": type(self.retriever).__name__,
                "llm_type": "cloud",
                "cost_usd": cost_info.get("total_cost", 0),
                "total_tokens": cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0),
            },
        )

        logger.info("Query completed successfully")
        return response
