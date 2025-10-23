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

    def reformulate_query(
        self, query: str, chat_history: list[dict], thread_id: Optional[str] = None
    ) -> str:
        """Reformulate query using conversation context to make it standalone."""
        if not chat_history or len(chat_history) < 2:
            return query

        recent_turns = chat_history[-6:]
        history_text = "\n".join(
            [
                f"{'Utilisateur' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
                for msg in recent_turns
            ]
        )

        prompt = f"""Étant donné cet historique de conversation juridique:

{history_text}

L'utilisateur pose maintenant cette question de suivi:
"{query}"

Reformulez cette question en une question autonome et complète qui peut être comprise sans l'historique de la conversation. Gardez le même contexte juridique et la même langue (français).

Question reformulée:"""

        try:
            result = self.llm.invoke(prompt)
            reformulated = result.content.strip().strip('"').strip("'")
            logger.info(f"Query reformulated: '{query}' → '{reformulated}'")
            return reformulated
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            return query

    def query(
        self,
        question: str,
        top_k: int = 5,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_history: Optional[list[dict]] = None,
        enable_reformulation: bool = True,
    ) -> QueryResponse:
        query_start_time = time.time()
        tracer = get_tracer()
        trace = None

        # Reformulate query if we have conversation history
        original_question = question
        if enable_reformulation and chat_history:
            question = self.reformulate_query(question, chat_history, thread_id)

        if tracer.enabled:
            trace = tracer.create_trace(
                name="conversational_rag_query",
                session_id=session_id,
                metadata={
                    "thread_id": thread_id,
                    "retriever_type": type(self.retriever).__name__,
                    "model": self.model_name,
                    "top_k": top_k,
                    "conversation": True,
                    "original_query": original_question,
                    "reformulated_query": question if question != original_question else None,
                },
                tags=["rag", "conversational", "legal", "chat"],
            )

        retrieval_start = time.time()

        logger.info(f"Retrieving documents for: {question[:100]}...")

        retrieval_span = None
        if tracer.enabled and trace:
            retrieval_span = trace.start_span(
                name="retrieval", input={"question": question, "top_k": top_k}
            )

        sources = self.retriever.retrieve(question, top_k=top_k)
        retrieval_duration = time.time() - retrieval_start

        if retrieval_span:
            avg_score = sum(s.score for s in sources) / len(sources) if sources else 0
            retrieval_span.update(
                output={
                    "num_sources": len(sources),
                    "sources": [s.reference for s in sources],
                    "avg_score": avg_score,
                },
                metadata={"duration_ms": retrieval_duration * 1000},
            )
            retrieval_span.end()

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
                    metadata={"status": "no_sources"},
                )
                trace.end()
                tracer.flush()

            return response

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

        input_tokens = (
            estimate_tokens(context) + estimate_tokens(question) + estimate_tokens(system_prompt)
        )

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

            if tracer.enabled and trace:
                generation = trace.start_observation(
                    as_type="generation",
                    name="llm_generation",
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": system_prompt[:200]},
                        {"role": "user", "content": question},
                    ],
                    output=answer,
                    usage_details={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    metadata={"duration_ms": generation_duration * 1000},
                )
                generation.end()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Erreur lors de la génération de la réponse: {e}"
            cost_info = {"total_cost": 0, "input_tokens": 0, "output_tokens": 0}

            if tracer.enabled and trace:
                error_span = trace.start_span(
                    name="generation_error",
                    input={"question": question},
                    output={"error": str(e)},
                    metadata={"error_type": type(e).__name__},
                    level="ERROR",
                )
                error_span.end()

        response = QueryResponse(
            query=question,
            answer=answer,
            sources=sources,
            retrieval_details={
                "num_sources": len(sources),
                "retrieval_method": type(self.retriever).__name__,
                "llm_type": "cloud",
                "cost_usd": cost_info.get("total_cost", 0),
                "total_tokens": cost_info.get("input_tokens", 0)
                + cost_info.get("output_tokens", 0),
            },
        )

        if tracer.enabled and trace:
            trace.update(
                output={
                    "answer": answer,
                    "sources": [s.metadata.get("title", "Unknown") for s in sources],
                },
                metadata={
                    "total_duration_ms": (time.time() - query_start_time) * 1000,
                    "total_cost_usd": cost_info.get("total_cost", 0),
                    "total_tokens": cost_info.get("input_tokens", 0)
                    + cost_info.get("output_tokens", 0),
                },
            )
            trace.end()
            tracer.flush()

        logger.info("Query completed successfully")
        return response

    def generate_followup_questions(self, context: str, answer: str, n: int = 3) -> list[str]:
        try:
            prompt = f"""Basé sur cette réponse juridique, génère {n} questions de suivi pertinentes et courtes que l'utilisateur pourrait poser.

Réponse: {answer[:500]}

Génère uniquement les questions, une par ligne, sans numérotation ni tirets."""

            result = self.llm.invoke(prompt)
            questions = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
            return questions[:n]
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []
