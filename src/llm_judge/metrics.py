import pandas as pd

from llm_judge.evaluator import EvaluationReport
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


def aggregate_results(report: EvaluationReport) -> dict:
    """summary eval statistics"""
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
    print("\n" + "=" * 80)
    print(f"EVALUATION REPORT: {report.config_name}")
    print("=" * 80)
    print(f"Total Questions: {report.total_questions}")
    print(f"Average Relevance: {report.avg_relevance:.2f}/5.0")
    print(f"Average Groundedness: {report.avg_groundedness:.2f}/5.0")
    print(f"Average Response Time: {report.avg_response_time:.2f}s")
    print("\nRelevance Distribution:")
    for score in range(1, 6):
        count = report.relevance_distribution.get(score, 0)
        pct = (count / report.total_questions) * 100
        print(f"  Score {score}: {count:3d} ({pct:5.1f}%)")
    print("\nGroundedness Distribution:")
    for score in range(1, 6):
        count = report.groundedness_distribution.get(score, 0)
        pct = (count / report.total_questions) * 100
        print(f"  Score {score}: {count:3d} ({pct:5.1f}%)")
    print("=" * 80 + "\n")


def find_weak_answers(
    report: EvaluationReport,
    threshold: int = 3,
    limit: int = 20,
) -> pd.DataFrame:
    """Find answers with low scores for further analysis.

    Args:
        report: EvaluationReport to analyze
        threshold: Score threshold (answers below this are considered weak)
        limit: Maximum number of weak answers to return

    Returns:
        DataFrame with weak answers
    """
    weak_answers = []

    for eval_data in report.evaluations:
        if eval_data["relevance_score"] < threshold or eval_data["groundedness_score"] < threshold:
            weak_answers.append(
                {
                    "Question ID": eval_data["question_id"],
                    "Question": eval_data["question"][:100] + "...",
                    "Relevance": eval_data["relevance_score"],
                    "Groundedness": eval_data["groundedness_score"],
                    "Rel Reasoning": eval_data["relevance_reasoning"][:80] + "...",
                    "Ground Reasoning": eval_data["groundedness_reasoning"][:80] + "...",
                }
            )

    df = pd.DataFrame(weak_answers)

    if not df.empty:
        df = df.sort_values(["Relevance", "Groundedness"]).head(limit)

    return df
