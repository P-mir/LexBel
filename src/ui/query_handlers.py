import logging
import time

import streamlit as st

from chains import ConversationalQA
from ui.components import render_retrieval_results

logger = logging.getLogger(__name__)


def process_retrieval_query(question, retriever, top_k, retriever_type, analytics):
    start_time = time.time()
    sources = retriever.retrieve(question, top_k=top_k)
    retrieval_time_ms = (time.time() - start_time) * 1000

    analytics.log_search(
        query=question,
        retriever_type=retriever_type,
        num_results=len(sources),
        retrieval_time_ms=retrieval_time_ms,
        sources=[s.reference for s in sources],
    )

    render_retrieval_results(sources, retrieval_time_ms)


def process_conversational_query(question, retriever, top_k, analytics):
    start_time = time.time()

    qa_chain = ConversationalQA(retriever=retriever)

    response = qa_chain.query(
        question,
        top_k=top_k,
        thread_id=st.session_state.session_id,
        session_id=st.session_state.session_id,
        chat_history=st.session_state.chat_history,
        enable_reformulation=True,
    )
    total_time_ms = (time.time() - start_time) * 1000

    followup_questions = qa_chain.generate_followup_questions(
        context="", answer=response.answer, n=2
    )

    # add user message to chat history
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "timestamp": time.time()}
    )

    # add assistant response to chat history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": [s.reference for s in response.sources],
            "source_objects": [
                {
                    "reference": s.reference,
                    "text": s.text,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in response.sources
            ],
            "cost_usd": response.retrieval_details.get("cost_usd", 0),
            "tokens": response.retrieval_details.get("total_tokens", 0),
            "followup_questions": followup_questions,
            "timestamp": time.time(),
        }
    )

    analytics.log_search(
        query=question,
        retriever_type="conversational",
        num_results=len(response.sources),
        retrieval_time_ms=total_time_ms,
        sources=[s.reference for s in response.sources],
        cost_usd=response.retrieval_details.get("cost_usd", 0.0),
        tokens=response.retrieval_details.get("total_tokens", 0),
        conversation_id=st.session_state.session_id,
        turn_number=len(st.session_state.chat_history) // 2,
    )

    analytics.log_conversation_turn(
        conversation_id=st.session_state.session_id,
        cost_usd=response.retrieval_details.get("cost_usd", 0.0),
        tokens=response.retrieval_details.get("total_tokens", 0),
    )

    st.rerun()
