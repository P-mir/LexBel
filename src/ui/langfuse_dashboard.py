import os

import streamlit as st


def render_langfuse_dashboard(stats: dict):
    """Render Langfuse monitoring dashboard.

    Args:
        stats: Dictionary with Langfuse metrics
    """
    st.markdown("# Langfuse Monitoring")

    if not stats.get("enabled", False):
        st.warning("""
        ⚠️ **Langfuse not configured**""")
        return

    # Key metrics overview
    st.markdown("## Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Queries",
            f"{stats.get('total_queries', 0):,}",
            help="Total number of queries processed"
        )

    with col2:
        cost = stats.get('total_cost', 0)
        st.metric(
            "Total Cost",
            f"${cost:.4f}",
            help="Cumulative API cost"
        )

    with col3:
        tokens = stats.get('total_tokens', 0)
        st.metric(
            "Total Tokens",
            f"{tokens:,}",
            help="Sum of input + output tokens"
        )

    with col4:
        error_rate = stats.get('error_rate', 0)
        st.metric(
            "Error Rate",
            f"{error_rate:.1f}%",
            delta=f"{-error_rate:.1f}%" if error_rate < 5 else None,
            help="Percentage of failed queries"
        )

    st.markdown("---")

    # Latency
    st.markdown("## Latency Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average",
            f"{stats.get('avg_latency_ms', 0):.0f}ms",
            help="Mean response time"
        )

    with col2:
        st.metric(
            "P50 (Median)",
            f"{stats.get('p50_latency_ms', 0):.0f}ms",
            help="50th percentile - half of requests faster"
        )

    with col3:
        st.metric(
            "P95",
            f"{stats.get('p95_latency_ms', 0):.0f}ms",
            help="95th percentile - 95% of requests faster"
        )

    with col4:
        st.metric(
            "P99",
            f"{stats.get('p99_latency_ms', 0):.0f}ms",
            help="99th percentile - 99% of requests faster"
        )

    # Retrieval
    st.markdown("## 🎯 Retrieval Metrics")

    col1, col2 = st.columns(2)

    with col1:
        avg_sim = stats.get('avg_similarity_score', 0)
        st.metric(
            "Avg Similarity Score",
            f"{avg_sim:.3f}",
            help="Average cosine similarity of retrieved documents"
        )

    with col2:
        st.metric(
            "Avg Docs Retrieved",
            f"{stats.get('avg_docs_retrieved', 5):.1f}",
            help="Average number of documents per query"
        )

    st.markdown("---")

    st.markdown("## 🌐 Langfuse Dashboard")

    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    st.markdown(f"""
    For detailed analytics -->> Langfuse dashboard:

    [{langfuse_host}]({langfuse_host})

    """)

