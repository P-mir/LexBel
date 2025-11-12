"""Evaluator for running answer quality evaluation on test datasets."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
