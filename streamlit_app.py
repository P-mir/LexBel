
import base64
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables FIRST before any other imports
load_dotenv()

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.metrics import LexBelAnalytics
from chains import LangChainQA
from embeddings import CloudEmbedder, LocalEmbedder
from retrievers import HybridRetriever, MMRRetriever
from ui.dashboard import render_dashboard
from ui.styling import get_custom_css
from utils.helpers import load_json, update_metrics
from utils.logging_config import setup_logger
from utils.models import TextChunk
from vector_store import FAISSVectorStore

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
    embedding_dim = config['embedding_dim']
    embedder_type = config.get('embedder_type', 'local')

    # Initialize embedder based on config
    if embedder_type == "mistral":
        embedder = CloudEmbedder(model_name="mistral-embed", batch_size=100)
    else:
        embedder = LocalEmbedder(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device="cpu",
        )

    vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
    vector_store.load(vector_store_path)

    chunks_metadata = load_json(vector_store_path / "chunks_metadata.json")
    chunks = [TextChunk(**chunk_data) for chunk_data in chunks_metadata]

    # Initialize retriever
    if retriever_type == "mmr":
        retriever = MMRRetriever(
            vector_store=vector_store,
            embedder=embedder,
            lambda_param=st.session_state.get('mmr_lambda', 0.5),
            initial_k=50,
        )
    else:
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            chunks=chunks,
            alpha=st.session_state.get('hybrid_alpha', 0.7),
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

    st.markdown("""
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
    """, unsafe_allow_html=True)

    with st.sidebar:
        logo_b64 = load_logo()
        if logo_b64:
            st.markdown(f"""
            <div class="lexbel-logo-container">
                <img src="data:image/png;base64,{logo_b64}" class="lexbel-logo" alt="LexBel"/>
                <div class="lexbel-tagline">Intelligence Juridique Belge</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="lexbel-logo-container">
                <h1 style="color: var(--lexbel-accent); margin: 0;">LexBel</h1>
                <div class="lexbel-tagline">Intelligence Juridique Belge</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Configuration")

        page = st.radio(
            "Navigation",
            options=["🔍 Recherche", "📊 Tableau de Bord"],
            label_visibility="collapsed"
        )

        st.markdown("---")

    # Main content based on page selection
    if page == "📊 Tableau de Bord":
        render_dashboard(analytics.get_dashboard_stats())
        return

    # Search page
    st.markdown("# Recherche Juridique")
    st.markdown("*Assistant intelligent pour le droit belge*")

    with st.sidebar:
        st.markdown("### Paramètres de Recherche")

        mode = st.selectbox(
            "Mode",
            options=["RAG Complet", "Récupération Seule"],
            index=0,
            help="RAG Complet: Recherche + Réponse IA | Récupération: Documents uniquement",
        )

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
            st.session_state['mmr_lambda'] = mmr_lambda
        else:
            hybrid_alpha = st.slider(
                "Hybrid Alpha (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="0.0 = lexical pur | 1.0 = vectoriel pur",
            )
            st.session_state['hybrid_alpha'] = hybrid_alpha

        st.markdown("---")

    # Load RAG system
    try:
        retriever, _, _, _, config = load_system(
            vector_store_dir, retriever_type
        )

        with st.sidebar:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin-bottom: 0.25rem;">Articles</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{config.get('num_articles', 'N/A'):,}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin-bottom: 0.25rem;">Chunks</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">{config.get('num_chunks', 'N/A'):,}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sidebar footer with contact
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; font-size: 0.75rem; margin-top: 2rem;">
                <a href="mailto:ptrk.miro@gmail.com" style="color: white; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
                    📧 ptrk.miro@gmail.com
                </a>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.sidebar.error(f"❌ Erreur: {e}")
        st.error("Veuillez vérifier la configuration du système.")
        return

    # Query interface
    with st.expander("📝 Questions Suggérées", expanded=False):
        st.markdown("""
        - Quelles sont les directives européennes transposées par le Code Bruxellois de l'Air ?
        - Quels sont les objectifs du présent Code en matière d'énergie ?
        - Qu'est-ce qu'un pouvoir public selon le Code ?
        - Quelle est la définition de l'efficacité énergétique ?
        - Comment est définie la biomasse ?
        """)

    question = st.text_area(
        "Votre Question",
        height=120,
        placeholder="Posez votre question sur le droit belge...",
        label_visibility="collapsed",
        max_chars=1000
    )

    if question:
        char_count = len(question)
        # rough estimation: 1 token ≈ 4 characters for French
        token_estimate = char_count // 4
        if token_estimate > 1000:
            st.error(f"⚠️ Question trop longue: ~{token_estimate} tokens (max: 1000 tokens)")
        elif token_estimate > 800:
            st.warning(f"Attention: ~{token_estimate} tokens / 1000")
        else:
            st.caption(f"📊 ~{token_estimate} tokens / 1000")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_button = st.button("🔎 Rechercher", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Effacer", use_container_width=True)

    if clear_button:
        st.rerun()

    if search_button and question:
        # Validate token limit
        token_estimate = len(question) // 4
        if token_estimate > 1000:
            st.error("❌ Question trop longue. Veuillez réduire à maximum 1000 tokens (~4000 caractères).")
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
                        sources=[s.reference for s in sources]
                    )

                    st.markdown("## 📚 Documents Récupérés")
                    st.success(f"**{len(sources)} articles** trouvés en **{retrieval_time_ms:.0f}ms**")

                    for i, source in enumerate(sources, 1):
                        with st.expander(f"📄 **Source {i}** - {source.reference} (Score: {source.score:.3f})"):
                            st.markdown("**Texte:**")
                            st.markdown(f"*{source.text}*")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Article ID:** `{source.article_id}`")
                                st.markdown(f"**Code:** `{source.code}`")
                                if source.metadata.get('book'):
                                    st.markdown(f"**Livre:** {source.metadata['book']}")
                            with col2:
                                if source.metadata.get('chapter'):
                                    st.markdown(f"**Chapitre:** {source.metadata['chapter']}")
                                if source.metadata.get('section'):
                                    st.markdown(f"**Section:** {source.metadata['section']}")

                            if 'vector_score' in source.metadata:
                                st.markdown("**Scores Détaillés:**")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("Vectoriel", f"{source.metadata['vector_score']:.3f}")
                                with c2:
                                    st.metric("Lexical", f"{source.metadata['lexical_score']:.3f}")

                else:  # RAG Complet
                    qa_chain = LangChainQA(
                        retriever=retriever,
                        llm_type="cloud",
                    )

                    response = qa_chain.query(question, top_k=top_k)
                    total_time_ms = (time.time() - start_time) * 1000

                    # Log to analytics
                    analytics.log_search(
                        query=question,
                        retriever_type=retriever_type,
                        num_results=len(response.sources),
                        retrieval_time_ms=total_time_ms,
                        sources=[s.reference for s in response.sources]
                    )

                    st.markdown("## 💡 Réponse")
                    st.markdown(f"""
                    <div class="dashboard-card">
                        {response.answer}
                    </div>
                    """, unsafe_allow_html=True)

                    col_time, col_sources = st.columns(2)
                    with col_time:
                        st.metric("Temps Total", f"{total_time_ms:.0f}ms")
                    with col_sources:
                        st.metric("Sources Utilisées", len(response.sources))

                    st.markdown("## 📚 Sources")

                    for i, source in enumerate(response.sources, 1):
                        with st.expander(f"📄 **Source {i}** - {source.reference} (Score: {source.score:.3f})"):
                            st.markdown("**Texte:**")
                            st.markdown(f"*{source.text}*")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Article ID:** `{source.article_id}`")
                                st.markdown(f"**Code:** `{source.code}`")
                                if source.metadata.get('book'):
                                    st.markdown(f"**Livre:** {source.metadata['book']}")
                            with col2:
                                if source.metadata.get('chapter'):
                                    st.markdown(f"**Chapitre:** {source.metadata['chapter']}")
                                if source.metadata.get('section'):
                                    st.markdown(f"**Section:** {source.metadata['section']}")

                            if 'vector_score' in source.metadata:
                                st.markdown("**Scores:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Vectoriel", f"{source.metadata['vector_score']:.3f}")
                                with col2:
                                    st.metric("Lexical", f"{source.metadata['lexical_score']:.3f}")
                                with col3:
                                    st.metric("Hybrid", f"{source.score:.3f}")

                with st.expander("🔧 Détails Techniques"):
                    if mode == "Récupération Seule":
                        details = {
                            "Temps de récupération (ms)": f"{retrieval_time_ms:.2f}",
                            "Nombre de sources": len(sources),
                            "Type de récupérateur": retriever_type.upper(),
                            "Embedder": "Local (sentence-transformers)",
                        }
                    else:
                        details = {
                            "Temps total (ms)": f"{total_time_ms:.2f}",
                            "Nombre de sources": len(response.sources),
                            "Type de récupérateur": retriever_type.upper(),
                            "Embedder": "Local (sentence-transformers)",
                            "LLM": "Mistral API (mistral-small-latest)",
                        }
                    st.json(details)

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {e}")
                logger.error(f"Query error: {e}", exc_info=True)

    # Footer
    st.markdown("""
    <div class="lexbel-footer">
        <strong>LexBel</strong> - Intelligence Juridique Belge<br/>
        Système RAG avancé pour le droit belge<br/>
        <a href="mailto:ptrk.miro@gmail.com" style="color: var(--lexbel-gray); font-size: 0.75rem; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
            📧 ptrk.miro@gmail.com
        </a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
