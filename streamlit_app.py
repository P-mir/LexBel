import base64
import logging
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from analytics.metrics import LexBelAnalytics  # noqa: E402
from chains import ConversationalQA, LangChainQA  # noqa: E402
from embeddings import CloudEmbedder  # noqa: E402
from observability import generate_session_id, get_tracer  # noqa: E402
from retrievers import HybridRetriever, MMRRetriever  # noqa: E402
from ui.dashboard import render_dashboard  # noqa: E402
from ui.styling import get_custom_css  # noqa: E402
from utils.helpers import load_json  # noqa: E402
from utils.logging_config import setup_logger  # noqa: E402
from utils.models import TextChunk  # noqa: E402
from vector_store import FAISSVectorStore  # noqa: E402

logger = setup_logger(__name__)

# Page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="LexBel - Intelligence Juridique Belge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_logo() -> str:
    """Load and encode logo for display."""
    logo_path = Path("assets/Logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


@st.cache_resource
def load_system(vector_store_dir: str, retriever_type: str):
    """Load and cache the RAG system."""
    vector_store_path = Path(vector_store_dir)

    # Load config
    config = load_json(vector_store_path / "config.json")
    embedding_dim = config["embedding_dim"]

    embedder = CloudEmbedder(model_name="mistral-embed", batch_size=100)

    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.load(vector_store_path)

    chunks_metadata = load_json(vector_store_path / "chunks_metadata.json")
    chunks = [TextChunk(**chunk_data) for chunk_data in chunks_metadata]

    # Initialize retriever
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
    """Main  app function"""

    st.markdown(get_custom_css(), unsafe_allow_html=True)
    analytics = load_analytics()

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()
        logger.info(f"New session started: {st.session_state.session_id}")
        analytics.log_conversation_start(st.session_state.session_id)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Chat history initialized")

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
        logo_b64 = load_logo()
        if logo_b64:
            st.markdown(
                f"""
            <div class="lexbel-logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="lexbel-logo" alt="LexBel"/>
                <div class="lexbel-tagline">Intelligence Juridique Belge</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="lexbel-logo-container">
                <h1 style="color: var(--lexbel-accent); margin: 0;">LexBel</h1>
                <div class="lexbel-tagline">Intelligence Juridique Belge</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("### ⚙️ Configuration")

        page = st.radio(
            "Navigation",
            options=["🔍 Recherche", "📊 Tableau de Bord"],
            label_visibility="collapsed",
        )

        st.markdown("---")

    # Main content based on page selection
    if page == "📊 Tableau de Bord":
        render_dashboard(analytics.get_dashboard_stats())
        return

    # Search page
    st.markdown("# Recherche Juridique")
    st.markdown("*Assistant intelligent pour le droit belge*")

    with st.expander("📝 Questions Suggérées", expanded=False):
        st.markdown("""
        - Quels sont les objectifs climatiques de la Région bruxelloise ?
        - Quelles informations doit mentionner le notaire lors d'une vente immobilière ?
        - Quelles matières sont visées par l'article 39 de la Constitution ?
        - Quels services existent pour les personnes handicapées en Wallonie ?
        """)

    # Sidebar configuration - must come before mode-dependent UI
    with st.sidebar:
        st.markdown("### Paramètres de Recherche")

        mode = st.selectbox(
            "Mode",
            options=["Assistant Conversationnel", "Récupération Seule"],
            index=0,
            help="Assistant Conversationnel: Chat avec mémoire | Récupération: Documents uniquement",
        )

        vector_store_dir = "data/vector_store"

    if mode == "Assistant Conversationnel":
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("source_objects"):
                    with st.expander(f"📚 {len(message['source_objects'])} sources"):
                        for i, src in enumerate(message["source_objects"], 1):
                            with st.container():
                                st.markdown(f"**{i}. {src['reference']}** (Score: {src['score']:.3f})")
                                with st.expander("📄 Voir l'extrait"):
                                    st.text(src['text'])
                                    if src.get('metadata'):
                                        if src['metadata'].get('article_number'):
                                            st.caption(f"Article: {src['metadata']['article_number']}")
                                        if src['metadata'].get('section'):
                                            st.caption(f"Section: {src['metadata']['section']}")
                elif message["role"] == "assistant" and message.get("sources"):
                    # Fallback for old messages without source_objects
                    with st.expander(f"📚 {len(message['sources'])} sources"):
                        for i, src in enumerate(message["sources"], 1):
                            st.caption(f"**{i}.** {src}")

                if message["role"] == "assistant" and message.get("followup_questions"):
                    st.markdown("**Questions suggérées:**")
                    cols = st.columns(len(message["followup_questions"]))
                    for i, (col, q) in enumerate(zip(cols, message["followup_questions"])):
                        with col:
                            if st.button(q, key=f"followup_{message.get('timestamp', 0)}_{i}", use_container_width=True):
                                st.session_state.selected_followup = q

        question = st.chat_input("Posez votre question sur le droit belge...")

        if "selected_followup" in st.session_state:
            question = st.session_state.selected_followup
            del st.session_state.selected_followup

        search_button = False  # Not used in conversational mode
    else:
        question = st.text_area(
            "Votre Question",
            height=120,
            placeholder="Posez votre question sur le droit belge...",
            label_visibility="collapsed",
            max_chars=1000,
        )

        if question:
            char_count = len(question)
            token_estimate = char_count // 4
            if token_estimate > 1000:
                st.error(f"⚠️ Question trop longue: ~{token_estimate} tokens (max: 1000 tokens)")
            elif token_estimate > 800:
                st.warning(f"Attention: ~{token_estimate} tokens / 1000")
            else:
                st.caption(f"📊 ~{token_estimate} tokens / 1000")

        col1, col2 = st.columns([2, 1])
        with col1:
            search_button = st.button("🔎 Rechercher", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Effacer", use_container_width=True)

        if clear_button:
            st.rerun()

    # Continue sidebar configuration
    with st.sidebar:
        vector_store_dir = "data/vector_store"

        retriever_type = st.selectbox(
            "Type de Récupérateur",
            options=["mmr", "hybrid"],
            index=0,
            help="MMR: Diversité maximale | Hybrid: Sémantique + Lexical",
        ).lower()

        top_k = st.slider(
            "Nombre de Sources",
            min_value=1,
            max_value=20,
            value=5,
            help="Nombre d'articles juridiques à récupérer",
        )

        if mode == "Assistant Conversationnel":
            st.markdown("---")
            if st.button("🔄 Nouvelle Conversation", use_container_width=True):
                if len(st.session_state.chat_history) > 0:
                    analytics.end_conversation(st.session_state.session_id)
                st.session_state.session_id = generate_session_id()
                st.session_state.chat_history = []
                analytics.log_conversation_start(st.session_state.session_id)
                st.rerun()
                analytics.log_conversation_start(st.session_state.session_id)
                st.rerun()

        st.markdown("### Paramètres Avancés")

        if retriever_type == "mmr":
            mmr_lambda = st.slider(
                "MMR Lambda (λ)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0.0 = diversité max | 1.0 = pertinence max",
            )
            st.session_state["mmr_lambda"] = mmr_lambda
        else:
            hybrid_alpha = st.slider(
                "Hybrid Alpha (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="0.0 = lexical pur | 1.0 = vectoriel pur",
            )
            st.session_state["hybrid_alpha"] = hybrid_alpha

        st.markdown("---")

    # Load RAG system in background (after UI is displayed)
    loading_placeholder = st.sidebar.empty()
    stats_placeholder = st.sidebar.empty()

    try:
        with loading_placeholder:
            with st.spinner("⏳ Chargement du système..."):
                retriever, _, _, _, config = load_system(vector_store_dir, retriever_type)

        loading_placeholder.empty()

        with stats_placeholder:
            st.markdown(
                f"""
            <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin-bottom: 0.25rem;">Articles</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{config.get("num_articles", "N/A"):,}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin-bottom: 0.25rem;">Chunks</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{config.get("num_chunks", "N/A"):,}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with st.sidebar:
            st.markdown("---")
            st.markdown(
                """
            <div style="text-align: center; font-size: 0.75rem; margin-top: 2rem;">
                <a href="mailto:ptrk.miro@gmail.com" style="color: white; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
                    📧 ptrk.miro@gmail.com
                </a>
            </div>
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        loading_placeholder.empty()
        logger.error(f"Failed to load system: {e}", exc_info=True)
        st.sidebar.error(f"❌ Erreur: {e}")
        st.error("Veuillez vérifier la configuration du système.")
        return

    should_process = (mode == "Assistant Conversationnel" and question) or (search_button and question)

    if should_process:
        token_estimate = len(question) // 4
        if token_estimate > 1000:
            st.error(
                "❌ Question trop longue. Veuillez réduire à maximum 1000 tokens (~4000 caractères)."
            )
            return

        with st.spinner("Traitement de votre requête..."):
            start_time = time.time()

            try:
                if mode == "Récupération Seule":
                    sources = retriever.retrieve(question, top_k=top_k)
                    retrieval_time_ms = (time.time() - start_time) * 1000

                    # Log to analytics
                    analytics.log_search(
                        query=question,
                        retriever_type=retriever_type,
                        num_results=len(sources),
                        retrieval_time_ms=retrieval_time_ms,
                        sources=[s.reference for s in sources],
                    )

                    st.markdown("## 📚 Documents Récupérés")
                    st.success(
                        f"**{len(sources)} articles** trouvés en **{retrieval_time_ms:.0f}ms**"
                    )

                    for i, source in enumerate(sources, 1):
                        with st.expander(
                            f"📄 **Source {i}** - {source.reference} (Score: {source.score:.3f})"
                        ):
                            st.markdown("**Texte:**")
                            st.markdown(f"*{source.text}*")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Article ID:** `{source.article_id}`")
                                st.markdown(f"**Code:** `{source.code}`")
                                if source.metadata.get("book"):
                                    st.markdown(f"**Livre:** {source.metadata['book']}")
                            with col2:
                                if source.metadata.get("chapter"):
                                    st.markdown(f"**Chapitre:** {source.metadata['chapter']}")
                                if source.metadata.get("section"):
                                    st.markdown(f"**Section:** {source.metadata['section']}")

                            if "vector_score" in source.metadata:
                                st.markdown("**Scores Détaillés:**")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Vectoriel", f"{source.metadata['vector_score']:.3f}")
                                with c2:
                                    st.metric("Lexical", f"{source.metadata['lexical_score']:.3f}")

                else:  # Assistant Conversationnel
                    qa_chain = ConversationalQA(
                        retriever=retriever,
                    )

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
                        context="",
                        answer=response.answer,
                        n=2
                    )

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "timestamp": time.time()
                    })

                    st.session_state.chat_history.append({
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
                        "timestamp": time.time()
                    })

                    analytics.log_search(
                        query=question,
                        retriever_type=retriever_type,
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

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {e}")
                logger.error(f"Query error: {e}", exc_info=True)

    # Footer
    st.markdown(
        """
    <div class="lexbel-footer">
        <strong>LexBel</strong> - Intelligence Juridique Belge<br/>
        Système RAG pour le droit belge<br/>
        <a href="https://github.com/P-mir/LexBel" target="_blank" style="color: var(--lexbel-gray); font-size: 1.0rem; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
             GitHub Repository
        </a>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
