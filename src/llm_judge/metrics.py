from llm_judge.evaluator import EvaluationReport
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


def aggregate_results(report: EvaluationReport) -> dict:
    """Get summary eval statistics."""
    return {
        "config_name": report.config_name,
        "timestamp": report.timestamp,
        "total_questions": report.total_questions,
        "avg_relevance": round(report.avg_relevance, 2),
        "avg_groundedness": round(report.avg_groundedness, 2),
        "avg_response_time": round(report.avg_response_time, 2),
        "relevance_distribution": report.relevance_distribution,
        "groundedness_distribution": report.groundedness_distribution,
        "qa_chain": report.metadata.get("qa_chain"),
        "judge_model": report.metadata.get("judge_model"),
    }


def print_report_summary(report: EvaluationReport) -> None:
    """Print evaluation report summary."""
    print("\n" + "=" * 80)
    print(f"EVALUATION REPORT: {report.config_name}")
    print("=" * 80)
    print(f"Total Questions: {report.total_questions}")
    print(f"Average Relevance: {report.avg_relevance:.2f}/5.0")
    print(f"Average Groundedness: {report.avg_groundedness:.2f}/5.0")
    print(f"Average Response Time: {report.avg_response_time:.2f}s")
    print("=" * 80 + "\n")
