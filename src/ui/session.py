import logging

import streamlit as st

from observability import generate_session_id

logger = logging.getLogger(__name__)


def initialize_session_state(analytics):
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
        logger.info(f"New session started: {st.session_state.session_id}")
        analytics.log_conversation_start(st.session_state.session_id)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Chat history initialized")
