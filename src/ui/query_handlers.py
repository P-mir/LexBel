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

    st.session_state.chat_history.append(
        {"role": "user", "content": question, "timestamp": time.time()}
    )

    placeholder = st.empty()

    with placeholder.container():
        with st.chat_message("assistant", avatar="🔹"):
            message_placeholder = st.empty()
        full_response = ""
        sources = None
        retrieval_details = None

        for chunk, chunk_sources, chunk_details in qa_chain.query_stream(
            question,
            top_k=top_k,
            thread_id=st.session_state.session_id,
            session_id=st.session_state.session_id,
            chat_history=st.session_state.chat_history,
            enable_reformulation=False,
        ):
            if chunk:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            if chunk_sources is not None:
                sources = chunk_sources
            if chunk_details is not None:
                retrieval_details = chunk_details

        message_placeholder.markdown(full_response)

    total_time_ms = (time.time() - start_time) * 1000

    # Save to chat history FIRST, before generating followup questions
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": [s.reference for s in sources] if sources else [],
            "source_objects": [
                {
                    "reference": s.reference,
                    "text": s.text,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in sources
            ]
            if sources
            else [],
            "cost_usd": retrieval_details.get("cost_usd", 0) if retrieval_details else 0,
            "tokens": retrieval_details.get("total_tokens", 0) if retrieval_details else 0,
            "timestamp": time.time(),
        }
    )

    followup_questions = []
    if full_response and sources:
        try:
            followup_questions = qa_chain.generate_followup_questions(
                context="", answer=full_response, n=2
            )
            logger.info(
                f"🟢 Generated {len(followup_questions)} followup questions: {followup_questions}"
            )
            if followup_questions:
                st.session_state.latest_followup_questions = followup_questions
                logger.info("🟢 Stored in session_state.latest_followup_questions")
                logger.info("🔄 Triggering rerun to display followup questions")
                st.rerun()
        except Exception as e:
            logger.warning(f"Failed to generate followup questions: {e}")
            st.session_state.latest_followup_questions = []
    else:
        st.session_state.latest_followup_questions = []

    if retrieval_details and sources:
        analytics.log_search(
            query=question,
            retriever_type="conversational",
            num_results=len(sources),
            retrieval_time_ms=total_time_ms,
            sources=[s.reference for s in sources],
            cost_usd=retrieval_details.get("cost_usd", 0.0),
            tokens=retrieval_details.get("total_tokens", 0),
            conversation_id=st.session_state.session_id,
            turn_number=len(st.session_state.chat_history) // 2,
        )

        analytics.log_conversation_turn(
            conversation_id=st.session_state.session_id,
            cost_usd=retrieval_details.get("cost_usd", 0.0),
            tokens=retrieval_details.get("total_tokens", 0),
        )
