import os
from typing import Dict, List

import numpy as np

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class LangfuseAnalytics:
    """Fetch and analyze Langfuse metrics."""

    def __init__(self):
        self.enabled = False

        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

        if secret_key and public_key:
            try:
                from langfuse import Langfuse

                self.client = Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                self.enabled = True
                logger.info("Langfuse analytics initialized")
            except Exception as e:
                logger.warning(f"Failed to init Langfuse analytics: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse analytics not configured")

    def get_dashboard_stats(self, days: int = 7) -> Dict:
        """Get aggregated stats for dashboard.

        Args:
            days: Number of days to look back

        Returns:
            Dict with aggregated metrics
        """
        if not self.enabled:
            return self._get_mock_stats()

        try:
            # Fetch traces from last N days
            # Note: Langfuse API might not support all these methods yet
            # This is a placeholder for actual API calls
            stats = {
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "error_rate": 0.0,
                "avg_similarity_score": 0.0,
                "total_tokens": 0,
                "queries_per_day": [],
                "cost_per_day": [],
                "latencies": [],
                "enabled": True,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to fetch Langfuse stats: {e}")
            return self._get_mock_stats()

    def _get_mock_stats(self) -> Dict:
        """Return mock stats when Langfuse not available."""
        return {
            "total_queries": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "error_rate": 0.0,
            "avg_similarity_score": 0.0,
            "total_tokens": 0,
            "queries_per_day": [],
            "cost_per_day": [],
            "latencies": [],
            "enabled": False,
        }

    def calculate_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }
