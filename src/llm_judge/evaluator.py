"""Evaluator for running answer quality evaluation on test datasets."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from llm_judge.judge import JudgmentScores, LLMJudge
from utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class AnswerEvaluation:
    """Single answer evaluation result."""

    question_id: str
    question: str
    answer: str
    retrieved_context: str
    relevance_score: int
    groundedness_score: int
    relevance_reasoning: str
    groundedness_reasoning: str
    response_time: float
    timestamp: str


@dataclass
class EvaluationReport:
    """Complete evaluation report with aggregated metrics."""

    config_name: str
    timestamp: str
    total_questions: int
    avg_relevance: float
    avg_groundedness: float
    avg_response_time: float
    relevance_distribution: dict[int, int]
    groundedness_distribution: dict[int, int]
    evaluations: list[dict]
    metadata: dict


class AnswerEvaluator:
    """Evaluator for answer quality using LLM-as-a-judge."""

    def __init__(
        self,
        qa_chain: Any,
        judge: LLMJudge,
        config_name: str = "default",
    ):
        self.qa_chain = qa_chain
        self.judge = judge
        self.config_name = config_name
        logger.info(f"AnswerEvaluator initialized with config: {config_name}")

    def evaluate_question(
        self,
        question_id: str,
        question: str,
        top_k: int = 5,
    ) -> AnswerEvaluation:
        """Evaluate a single question."""

        start_time = time.time()
        response = self.qa_chain.query(
            question=question,
            top_k=top_k,
            enable_reformulation=False,
        )
        response_time = time.time() - start_time

        retrieved_context = "\n\n".join(
            [
                f"[Article {src.metadata.get('article_id', 'Unknown')}]\n{src.text}"
                for src in response.sources
            ]
        )

        judgment: JudgmentScores = self.judge.judge_answer(
            question=question,
            answer=response.answer,
            retrieved_context=retrieved_context,
        )

        return AnswerEvaluation(
            question_id=question_id,
            question=question,
            answer=response.answer,
            retrieved_context=retrieved_context,
            relevance_score=judgment.relevance,
            groundedness_score=judgment.groundedness,
            relevance_reasoning=judgment.relevance_reasoning,
            groundedness_reasoning=judgment.groundedness_reasoning,
            response_time=response_time,
            timestamp=datetime.now().isoformat(),
        )

    def evaluate_dataset(
        self,
        questions_df: pd.DataFrame,
        top_k: int = 10,
        limit: Optional[int] = None,
    ) -> EvaluationReport:
        """Evaluate multiple questions from dataset."""

        if limit:
            questions_df = questions_df.head(limit)
            logger.info(f"Limiting evaluation to {limit} questions")

        evaluations: list[AnswerEvaluation] = []

        logger.info(f"Starting evaluation on {len(questions_df)} questions")

        for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
            question_id = str(row.get("id", idx))
            question = row["question"]

            try:
                eval_result = self.evaluate_question(
                    question_id=question_id,
                    question=question,
                    top_k=top_k,
                )
                evaluations.append(eval_result)
                logger.info(
                    f"Q{question_id}: Rel={eval_result.relevance_score}, "
                    f"Ground={eval_result.groundedness_score}"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate question {question_id}: {e}")
                continue

        return self._create_report(evaluations)

    def _create_report(self, evaluations: list[AnswerEvaluation]) -> EvaluationReport:
        """Create evaluation report with aggregated metrics."""
        if not evaluations:
            raise ValueError("No evaluations to report")

        relevance_scores = [e.relevance_score for e in evaluations]
        groundedness_scores = [e.groundedness_score for e in evaluations]
        response_times = [e.response_time for e in evaluations]

        relevance_dist = {i: relevance_scores.count(i) for i in range(1, 6)}
        groundedness_dist = {i: groundedness_scores.count(i) for i in range(1, 6)}

        report = EvaluationReport(
            config_name=self.config_name,
            timestamp=datetime.now().isoformat(),
            total_questions=len(evaluations),
            avg_relevance=sum(relevance_scores) / len(relevance_scores),
            avg_groundedness=sum(groundedness_scores) / len(groundedness_scores),
            avg_response_time=sum(response_times) / len(response_times),
            relevance_distribution=relevance_dist,
            groundedness_distribution=groundedness_dist,
            evaluations=[asdict(e) for e in evaluations],
            metadata={
                "qa_chain": type(self.qa_chain).__name__,
                "judge_model": self.judge.model_name,
            },
        )

        logger.info(
            f"Evaluation complete: Avg Relevance={report.avg_relevance:.2f}, "
            f"Avg Groundedness={report.avg_groundedness:.2f}"
        )

        return report

    def save_report(self, report: EvaluationReport, output_path: Path) -> None:
        """Save evaluation report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to {output_path}")

    @staticmethod
    def load_report(report_path: Path) -> EvaluationReport:
        """Load evaluation report from JSON file."""
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationReport(**data)
