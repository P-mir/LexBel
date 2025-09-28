
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class LexBelAnalytics:
    """Trac and analyse system usage metrics"""

    def __init__(self, metrics_file: Path = Path("data/metrics/analytics.json")):
        self.metrics_file = metrics_file
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_metrics()

    def _load_metrics(self) -> Dict:
        """Load existing metrics or create new."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'searches': [],
            'performance': [],
            'daily_stats': {},
            'popular_queries': {},
            'system_info': {
                'first_started': datetime.now().isoformat(),
                'total_queries': 0,
                'total_users': 0
            }
        }

    def _save_metrics(self):
        """Save metrics to disk."""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def log_search(
        self,
        query: str,
        retriever_type: str,
        num_results: int,
        retrieval_time_ms: float,
        sources: List[str],
        user_id: Optional[str] = None,
        cost_usd: float = 0.0,
        tokens: int = 0
    ):
        """Log a search query."""
        search_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'retriever_type': retriever_type,
            'num_results': num_results,
            'retrieval_time_ms': retrieval_time_ms,
            'sources': sources,
            'user_id': user_id or 'anonymous',
            'cost_usd': cost_usd,
            'tokens': tokens
        }

        self.data['searches'].append(search_record)

        # Update popular queries
        if query in self.data['popular_queries']:
            self.data['popular_queries'][query] += 1
        else:
            self.data['popular_queries'][query] = 1

        # Update daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.data['daily_stats']:
            self.data['daily_stats'][today] = {
                'queries': 0,
                'avg_retrieval_ms': 0,
                'total_retrieval_ms': 0,
                'total_cost_usd': 0.0,
                'total_tokens': 0
            }

        stats = self.data['daily_stats'][today]
        stats['queries'] += 1
        stats['total_retrieval_ms'] += retrieval_time_ms
        stats['avg_retrieval_ms'] = stats['total_retrieval_ms'] / stats['queries']
        stats['total_cost_usd'] += cost_usd
        stats['total_tokens'] += tokens

        # Update system info
        self.data['system_info']['total_queries'] += 1

        self._save_metrics()

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Log performance metric."""
        perf_record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_ms': duration_ms,
            'metadata': metadata or {}
        }

        self.data['performance'].append(perf_record)
        self._save_metrics()

    def get_dashboard_stats(self) -> Dict:
        """Get statistics for dashboard display."""
        # Total queries
        total_queries = len(self.data['searches'])

        # Average retrieval time
        if self.data['searches']:
            avg_retrieval = sum(s['retrieval_time_ms'] for s in self.data['searches']) / len(self.data['searches'])
        else:
            avg_retrieval = 0

        # Queries today
        today = datetime.now().strftime('%Y-%m-%d')
        queries_today = self.data['daily_stats'].get(today, {}).get('queries', 0)

        # Most popular queries (top 10)
        popular = sorted(
            self.data['popular_queries'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Recent searches
        recent = self.data['searches'][-20:]
        recent.reverse()

        # Retriever use
        retriever_counts = Counter(s['retriever_type'] for s in self.data['searches'])

        # Performance over time (last 7 days)
        weekly_stats = self._get_weekly_stats()

        return {
            'total_queries': total_queries,
            'queries_today': queries_today,
            'avg_retrieval_ms': round(avg_retrieval, 1),
            'popular_queries': popular,
            'recent_searches': recent,
            'retriever_usage': dict(retriever_counts),
            'weekly_stats': weekly_stats,
            'system_info': self.data['system_info']
        }

    def _get_weekly_stats(self) -> List[Dict]:
        """Get stats for last 7 days."""
        from datetime import timedelta

        stats = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            day_stats = self.data['daily_stats'].get(date, {
                'queries': 0,
                'avg_retrieval_ms': 0
            })
            stats.append({
                'date': date,
                'queries': day_stats.get('queries', 0),
                'avg_retrieval_ms': day_stats.get('avg_retrieval_ms', 0)
            })

        stats.reverse()
        return stats

    def get_performance_report(self) -> pd.DataFrame:
        """Get performance report as DataFrame."""
        if not self.data['performance']:
            return pd.DataFrame()

        df = pd.DataFrame(self.data['performance'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        # Fillter searches
        self.data['searches'] = [
            s for s in self.data['searches']
            if s['timestamp'] > cutoff_iso
        ]

        self.data['performance'] = [
            p for p in self.data['performance']
            if p['timestamp'] > cutoff_iso
        ]

        self._save_metrics()
