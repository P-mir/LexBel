import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from analytics.metrics import LexBelAnalytics  # noqa: E402
from embeddings import CloudEmbedder  # noqa: E402
from observability import get_tracer  # noqa: E402
from retrievers import HybridRetriever, MMRRetriever  # noqa: E402
from ui.components import (  # noqa: E402
    get_conversational_input,
    get_retrieval_input,
    render_advanced_params,
    render_chat_history,
    render_footer_contact,
    render_header,
    render_mode_selector,
    render_page_footer,
    render_page_navigation,
    render_retriever_settings,
    render_sidebar_logo,
    render_suggested_questions,
    render_system_stats,
)
from ui.dashboard import render_dashboard  # noqa: E402
from ui.query_handlers import (  # noqa: E402
    process_conversational_query,
    process_retrieval_query,
)
from ui.session import initialize_session_state  # noqa: E402
from ui.styling import get_custom_css  # noqa: E402
from utils.helpers import load_json  # noqa: E402
from utils.logging_config import setup_logger  # noqa: E402
from utils.models import TextChunk  # noqa: E402
from vector_store import FAISSVectorStore  # noqa: E402

logger = setup_logger(__name__)

st.set_page_config(
    page_title="LexBel - Intelligence Juridique Belge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_system(vector_store_dir: str, retriever_type: str):
    """Load and cache the RAG system."""
    vector_store_path = Path(vector_store_dir)

    config = load_json(vector_store_path / "config.json")
    embedding_dim = config["embedding_dim"]

    embedder = CloudEmbedder(model_name="mistral-embed", batch_size=100)

    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.load(vector_store_path)

    chunks_metadata = load_json(vector_store_path / "chunks_metadata.json")
    chunks = [TextChunk(**chunk_data) for chunk_data in chunks_metadata]

    if retriever_type == "mmr":
        retriever = MMRRetriever(
            vector_store=vector_store,
            embedder=embedder,
            lambda_param=st.session_state.get("mmr_lambda", 0.5),
            initial_k=50,
        )
    else:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            chunks=chunks,
            alpha=st.session_state.get("hybrid_alpha", 0.7),
        )

    return retriever, embedder, vector_store, chunks, config


@st.cache_resource
def load_analytics():
    """Load analytics tracker."""
    return LexBelAnalytics()


def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    analytics = load_analytics()

    initialize_session_state(analytics)

    # Initialize Langfuse tracer
    tracer = get_tracer()
    if tracer.enabled:
        logger.info("Langfuse tracing enabled")

    st.markdown(
        """
    <style>
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }

    section[data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] > div {
        display: block !important;
    }

    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_sidebar_logo()

        st.markdown("### ⚙️ Configuration")

        page = render_page_navigation()

        st.markdown("---")

    # Main content based on page selection
    if page == "📊 Tableau de Bord":
        render_dashboard(analytics.get_dashboard_stats())
        return

    # Search page
    render_header()
    render_suggested_questions()

    # Sidebar configuration - must come before mode-dependent UI
    with st.sidebar:
        st.markdown("### Paramètres de Recherche")

        mode = render_mode_selector()

        vector_store_dir = "data/vector_store"

    if mode == "Assistant Conversationnel":
        render_chat_history(st.session_state.chat_history)
        question = get_conversational_input()
        search_button = False  # not used for chat bot
    else:
        question, search_button = get_retrieval_input()

    # sidebar settings
    with st.sidebar:
        vector_store_dir = "data/vector_store"

        retriever_type, top_k = render_retriever_settings(mode, analytics, st.session_state)

        render_advanced_params(retriever_type)

        st.markdown("---")

    # this preloading helps reduce waiting time when searching
    loading_placeholder = st.sidebar.empty()
    stats_placeholder = st.sidebar.empty()

    try:
        with loading_placeholder:
            with st.spinner("⏳ Chargement du système..."):
                retriever, _, _, _, config = load_system(vector_store_dir, retriever_type)

        loading_placeholder.empty()

        with stats_placeholder:
            render_system_stats(config)

        with st.sidebar:
            render_footer_contact()

    except Exception as e:
        loading_placeholder.empty()
        logger.error(f"Failed to load system: {e}", exc_info=True)
        st.sidebar.error(f"❌ Erreur: {e}")
        st.error("Veuillez vérifier la configuration du système.")
        return

    should_process = (mode == "Assistant Conversationnel" and question) or (
        search_button and question
    )

    if should_process:
        token_estimate = len(question) // 4
        if token_estimate > 1000:
            st.error(
                "❌ Question trop longue. Veuillez réduire à maximum 1000 tokens (~4000 caractères)."
            )
            return

        with st.spinner("Traitement de votre requête..."):
            try:
                if mode == "Récupération Seule":
                    process_retrieval_query(question, retriever, top_k, retriever_type, analytics)
                else:  # conversational mode
                    process_conversational_query(question, retriever, top_k, analytics)

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {e}")
                logger.error(f"Query error: {e}", exc_info=True)

    render_page_footer()


if __name__ == "__main__":
    main()
