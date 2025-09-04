import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_json(data: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_metrics(metrics_file: Path, updates: Dict[str, Any]) -> None:
    """Update metrics file.

    Args:
        metrics_file: Path to metrics file
        updates: Dictionary of metrics to update
    """
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metrics or start fresh
    if metrics_file.exists():
        metrics = load_json(metrics_file)
    else:
        metrics = {
            'last_updated': None,
            'ingested_docs': 0,
            'chunks': 0,
            'queries_served': 0,
            'total_retrieval_ms': 0,
            'avg_retrieval_ms': 0,
        }

    for key, value in updates.items():
        if key in metrics and isinstance(metrics[key], (int, float)) and isinstance(value, (int, float)):
            metrics[key] += value
        else:
            metrics[key] = value

    # Update average if applicable
    if 'queries_served' in metrics and metrics['queries_served'] > 0:
        metrics['avg_retrieval_ms'] = metrics['total_retrieval_ms'] / metrics['queries_served']

    metrics['last_updated'] = datetime.now().isoformat()

    save_json(metrics, metrics_file)

