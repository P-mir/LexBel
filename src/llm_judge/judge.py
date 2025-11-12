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


JUDGE_SYSTEM_PROMPT = """Tu es un expert juridique chargé d'évaluer la qualité des réponses générées par un assistant juridique sur le droit belge.

RÈGLE IMPORTANTE: Tu dois évaluer uniquement en te basant sur les articles de loi fournis dans le contexte. N'utilise pas de connaissances juridiques externes.

Ta tâche est d'évaluer deux aspects critiques:

1. **Relevance (Pertinence)**: Dans quelle mesure la réponse répond-elle à la question juridique posée?
   - 1: Hors sujet ou n'aborde pas la question
   - 2: Aborde partiellement, éléments clés manquants
   - 3: Acceptable, mais manque de précision ou de profondeur
   - 4: Pertinente et répond bien à la question
   - 5: Parfaitement pertinente, complète et ciblée

2. **Groundedness (Ancrage aux sources)**: Dans quelle mesure la réponse est-elle fondée sur les articles fournis?
   - 1: Aucune base dans les articles fournis (hallucination totale)
   - 2: Faible utilisation des articles ou ajouts non vérifiables
   - 3: Appui modéré sur les articles
   - 4: Bien ancrée dans les articles fournis
   - 5: Entièrement basée sur les articles avec citations précises

Pour chaque score, fournis une justification courte (1-2 phrases) expliquant ton évaluation.

Sois objectif, factuel et rigoureux dans ton évaluation."""


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

        user_prompt = f"""Question posée par l'utilisateur:
{question}

Articles de loi fournis au système:
{retrieved_context}

Réponse générée par le système:
{answer}

Évalue cette réponse selon les deux critères (relevance et groundedness) en fournissant un score de 1 à 5 et une brève justification pour chaque score."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=JudgmentScores,
                temperature=self.temperature,
            )

            judgment = completion.choices[0].message.parsed
            logger.debug(
                f"Judgment: Relevance={judgment.relevance}, Groundedness={judgment.groundedness}"
            )
            return judgment

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            raise

    def batch_judge(
        self,
        evaluations: list[dict[str, str]],
    ) -> list[JudgmentScores]:
        """Evaluate multiple answers in batch."""
        results = []
        for i, eval_item in enumerate(evaluations, 1):
            logger.info(f"Judging answer {i}/{len(evaluations)}")
            try:
                judgment = self.judge_answer(
                    question=eval_item["question"],
                    answer=eval_item["answer"],
                    retrieved_context=eval_item["retrieved_context"],
                )
                results.append(judgment)
            except Exception as e:
                logger.error(f"Failed to judge answer {i}: {e}")
                results.append(
                    JudgmentScores(
                        relevance=0,
                        groundedness=0,
                        relevance_reasoning=f"Evaluation failed: {str(e)}",
                        groundedness_reasoning=f"Evaluation failed: {str(e)}",
                    )
                )

        return results
