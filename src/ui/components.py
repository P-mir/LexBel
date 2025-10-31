import streamlit as st


def render_sidebar_logo():
    st.markdown(
        """
        <div class="lexbel-logo-container">
            <h1 style="color: var(--lexbel-accent); margin: 0;">LexBel</h1>
            <div class="lexbel-tagline">Intelligence Juridique Belge</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_navigation():
    return st.radio(
        "Navigation",
        options=["🔍 Recherche", "📊 Tableau de Bord"],
        label_visibility="collapsed",
    )


def render_header():
    st.markdown("# Recherche Juridique")
    st.markdown("*Assistant intelligent pour le droit belge*")


def render_suggested_questions():
    with st.expander("📝 Questions Suggérées", expanded=False):
        st.markdown("""
        - Quels sont les objectifs climatiques de la Région bruxelloise ?
        - Quelles informations doit mentionner le notaire lors d'une vente immobilière ?
        - Quelles matières sont visées par l'article 39 de la Constitution ?
        - Quels services existent pour les personnes handicapées en Wallonie ?
        """)


def render_mode_selector():
    return st.selectbox(
        "Mode",
        options=["Assistant Conversationnel", "Récupération Seule"],
        index=0,
        help="Assistant Conversationnel: Chat avec mémoire | Récupération: Documents uniquement",
    )


def render_chat_history(chat_history):
    for message in chat_history:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🔹"):
                st.markdown(message["content"])

                if message.get("source_objects"):
                    with st.expander(f"📚 {len(message['source_objects'])} sources"):
                        for i, src in enumerate(message["source_objects"], 1):
                            with st.container():
                                st.markdown(
                                    f"**{i}. {src['reference']}** (Score: {src['score']:.3f})"
                                )
                                with st.expander("📄 Voir l'extrait"):
                                    st.text(src["text"])
                                    if src.get("metadata"):
                                        if src["metadata"].get("article_number"):
                                            st.caption(
                                                f"Article: {src['metadata']['article_number']}"
                                            )
                                        if src["metadata"].get("section"):
                                            st.caption(f"Section: {src['metadata']['section']}")
                elif message.get("sources"):
                    # old format compatibility
                    with st.expander(f"📚 {len(message['sources'])} sources"):
                        for i, src in enumerate(message["sources"], 1):
                            st.caption(f"**{i}.** {src}")

                if message.get("followup_questions"):
                    st.markdown("**Questions suggérées:**")
                    cols = st.columns(len(message["followup_questions"]))
                    for i, (col, q) in enumerate(zip(cols, message["followup_questions"])):
                        with col:
                            if st.button(
                                q,
                                key=f"followup_{message.get('timestamp', 0)}_{i}",
                                use_container_width=True,
                            ):
                                st.session_state.selected_followup = q
        else:
            with st.chat_message(message["role"], avatar="🟢"):
                st.markdown(message["content"])


def get_conversational_input():
    import logging

    logger = logging.getLogger(__name__)

    # check for followup question first (set by button click in previous render)
    if "selected_followup" in st.session_state:
        question = st.session_state.selected_followup
        logger.info(f"🔵 Found selected_followup: {question}")
        del st.session_state.selected_followup
        st.session_state.latest_followup_questions = []
        return question

    question = st.chat_input("Posez votre question sur le droit belge...")
    if question:
        logger.info(f"🟢 Got question from chat_input: {question}")
    return question


def render_followup_questions():
    import logging

    logger = logging.getLogger(__name__)

    logger.info(
        f"🔵 render_followup_questions called. Session state has: {st.session_state.get('latest_followup_questions', 'NONE')}"
    )

    if st.session_state.get("latest_followup_questions"):
        logger.info(
            f"🟢 Rendering {len(st.session_state.latest_followup_questions)} followup questions"
        )
        st.markdown("**Questions suggérées:**")
        cols = st.columns(len(st.session_state.latest_followup_questions))
        for i, (col, q) in enumerate(zip(cols, st.session_state.latest_followup_questions)):
            with col:
                button_key = f"followup_{hash(q)}_{i}"
                if st.button(q, key=button_key, use_container_width=True):
                    logger.info(f"🟡 Button clicked! Setting selected_followup: {q}")
                    st.session_state.selected_followup = q
                    st.session_state.latest_followup_questions = []
                    st.rerun()


def get_retrieval_input():
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

    return question, search_button


def render_retriever_settings(mode, analytics, session_state):
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
        if st.button("🔄 Nouvelle Conversation", use_container_width=True):
            if len(session_state.chat_history) > 0:
                analytics.end_conversation(session_state.session_id)
            session_state.session_id = session_state.session_id  # will be regenerated
            session_state.chat_history = []
            analytics.log_conversation_start(session_state.session_id)
            st.rerun()

    return retriever_type, top_k


def render_advanced_params(retriever_type):
    st.markdown(
        '<h3 style="margin-top: 0.3rem; margin-bottom: 0.5rem;">Paramètres Avancés</h3>',
        unsafe_allow_html=True,
    )

    if retriever_type == "mmr":
        mmr_lambda = st.slider(
            "MMR Lambda (λ)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
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


def render_system_stats(config):
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-around; margin: 0.3rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8); margin-bottom: 0.15rem;">Articles</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: white;">{config.get("num_articles", "N/A"):,}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8); margin-bottom: 0.15rem;">Chunks</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: white;">{config.get("num_chunks", "N/A"):,}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_contact():
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 0.5rem;">
            <a href="mailto:ptrk.miro@gmail.com" style="color: white; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
                📧 ptrk.miro@gmail.com
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retrieval_results(sources, retrieval_time_ms):
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


def render_page_footer():
    st.markdown(
        """
        <div style="text-align: center; font-size: 1.0rem; margin-top: 0.5rem; padding: 0.5rem 0; background-color: transparent;">
            <strong style="color: white;">LexBel</strong> <span style="color: rgba(255,255,255,0.7);">- Intelligence Juridique Belge</span><br/>
            <span style="color: rgba(255,255,255,0.6); font-size: 1.0rem;">Système RAG pour le droit belge</span><br/>
            <a href="https://github.com/P-mir/LexBel" target="_blank" style="color: rgba(255,255,255,0.6); font-size: 1.0rem; text-decoration: none; opacity: 0.7; transition: opacity 0.3s;">
                 GitHub Repository
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
