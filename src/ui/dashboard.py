"""Dashboard UI components"""

from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_dashboard(stats: Dict):

    st.markdown("## 📊 Tableau de Bord")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total des Recherches",
            value=f"{stats['total_queries']:,}",
            delta=f"+{stats['queries_today']} aujourd'hui"
        )

    with col2:
        st.metric(
            label="Temps Moyen (ms)",
            value=f"{stats['avg_retrieval_ms']:.0f}",
            delta="Optimisé" if stats['avg_retrieval_ms'] < 100 else None
        )

    with col3:
        retriever_primary = max(
            stats['retriever_usage'].items(),
            key=lambda x: x[1],
            default=("N/A", 0)
        )[0] if stats['retriever_usage'] else "N/A"
        st.metric(
            label="Récupérateur Principal",
            value=retriever_primary.upper()
        )

    with col4:
        system_uptime = _calculate_uptime(stats['system_info'].get('first_started'))
        st.metric(
            label="Disponibilité",
            value=system_uptime
        )

    st.markdown("---")

    # Two column layout for charts
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📈 Tendance des Recherches (7 jours)")
        _render_weekly_trend_chart(stats['weekly_stats'])

    with col_right:
        st.markdown("### 🎯 Utilisation des Récupérateurs")
        _render_retriever_pie_chart(stats['retriever_usage'])

    st.markdown("---")

    # Popular queries and recent searches
    col_pop, col_recent = st.columns(2)

    with col_pop:
        st.markdown("### 🔥 Requêtes Populaires")
        _render_popular_queries(stats['popular_queries'])

    with col_recent:
        st.markdown("### 🕒 Recherches Récentes")
        _render_recent_searches(stats['recent_searches'])

def _calculate_uptime(first_started: str) -> str:
    """Calculate system uptime."""
    if not first_started:
        return "N/A"

    try:
        start = datetime.fromisoformat(first_started)
        delta = datetime.now() - start

        if delta.days > 0:
            return f"{delta.days}j"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours}h"
        else:
            minutes = delta.seconds // 60
            return f"{minutes}m"
    except:
        return "N/A"

def _render_weekly_trend_chart(weekly_stats: List[Dict]):
    """Render weekly trend line chart."""
    if not weekly_stats:
        st.info("Pas encore de données disponibles")
        return

    df = pd.DataFrame(weekly_stats)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['queries'],
        mode='lines+markers',
        name='Requêtes',
        line=dict(color='#1a365d', width=3),
        marker=dict(size=8, color='#d4af37'),
        fill='tozeroy',
        fillcolor='rgba(26, 54, 93, 0.1)'
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            title=None,
            tickformat='%d/%m'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            title='Nombre de Requêtes'
        ),
        hovermode='x unified',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def _render_retriever_pie_chart(retriever_usage: Dict):
    """Render retriever usage pie chart."""
    if not retriever_usage:
        st.info("Pas encore de données disponibles")
        return

    labels = [k.upper() for k in retriever_usage.keys()]
    values = list(retriever_usage.values())

    colors = ['#1a365d', '#2c5282', '#d4af37', '#2f855a']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)]),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>%{value} recherches<br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def _render_popular_queries(popular_queries: List[tuple]):
    """Render popular queries list."""
    if not popular_queries:
        st.info("Aucune requête enregistrée pour le moment")
        return

    for i, (query, count) in enumerate(popular_queries, 1):
        # Create colored badge for ranking
        if i == 1:
            badge_color = "#d4af37"  # Gold
            medal = "🥇"
        elif i == 2:
            badge_color = "#c0c0c0"  # Silver
            medal = "🥈"
        elif i == 3:
            badge_color = "#cd7f32"  # Bronze
            medal = "🥉"
        else:
            badge_color = "#718096"  # Gray
            medal = f"{i}."

        # Use single-line HTML to avoid parsing issues
        html_content = f'<div style="background: white; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {badge_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><div style="display: flex; justify-content: space-between; align-items: center;"><span style="color: #1a365d; font-weight: 500;"><span style="background: {badge_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.5rem; font-size: 0.75rem; font-weight: 700;">{medal}</span>{query}</span><span style="background: #edf2f7; color: #1a365d; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.875rem; font-weight: 600;">{count}×</span></div></div>'

        st.markdown(html_content, unsafe_allow_html=True)

def _render_recent_searches(recent_searches: List[Dict]):
    """Render recent searches table."""
    if not recent_searches:
        st.info("Aucune recherche récente")
        return

    for search in recent_searches[:10]:
        timestamp = datetime.fromisoformat(search['timestamp'])
        time_str = timestamp.strftime('%H:%M:%S')

        retrieval_color = "#2f855a" if search['retrieval_time_ms'] < 100 else "#d97706"

        # Truncate long queries
        display_query = search['query'][:60] + "..." if len(search['query']) > 60 else search['query']

        # Use single-line HTML to avoid parsing issues
        html_content = f'<div style="background: white; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #2c5282; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"><div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.25rem;"><span style="color: #1a365d; font-weight: 500; flex: 1;">{display_query}</span><span style="color: #718096; font-size: 0.75rem; margin-left: 1rem;">{time_str}</span></div><div style="display: flex; gap: 0.5rem; font-size: 0.75rem;"><span style="background: #edf2f7; color: #2c5282; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">{search["retriever_type"].upper()}</span><span style="background: {retrieval_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">{search["retrieval_time_ms"]:.0f}ms</span><span style="background: #edf2f7; color: #718096; padding: 0.2rem 0.5rem; border-radius: 4px;">{search["num_results"]} docs</span></div></div>'

        st.markdown(html_content, unsafe_allow_html=True)
