import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class JudgmentScores(BaseModel):
    """Structured output for LLM judge evaluation."""

    relevance: int = Field(
        ge=1,
        le=5,
        description="Relevance score (1-5)",
    )
    groundedness: int = Field(
        ge=1,
        le=5,
        description="Groundedness score (1-5)",
    )
    relevance_reasoning: str = Field(description="Brief explanation for relevance")
    groundedness_reasoning: str = Field(description="Brief explanation for groundedness")


class LLMJudge:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        logger.info(f"LLMJudge initialized with model: {model_name}")

    def judge_answer(
        self,
        question: str,
        answer: str,
        retrieved_context: str,
    ) -> JudgmentScores:
        """Evaluate answer quality."""

        system_prompt = """Tu es un expert juridique évaluant la qualité des réponses sur le droit belge.

Évalue deux aspects:
1. Relevance (1-5): La réponse répond-elle à la question?
2. Groundedness (1-5): La réponse est-elle basée sur les articles fournis?"""

        user_prompt = f"""Question: {question}

Articles fournis:
{retrieved_context}

Réponse:
{answer}

Évalue cette réponse."""

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=JudgmentScores,
            temperature=self.temperature,
        )

        return completion.choices[0].message.parsed
