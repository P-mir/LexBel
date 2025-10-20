import os
from typing import Any, Optional

from langchain_mistralai import ChatMistralAI

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ConversationalQA:
    def __init__(
        self,
        retriever: Any,
        model_name: str = "mistral-small-latest",
        api_key: Optional[str] = None,
    ):
        self.retriever = retriever
        self.model_name = model_name

        api_key = api_key or os.getenv("MISTRAL_API_KEY")

        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            api_key=api_key,
        )

        logger.info(f"Conversational QA initialized with {model_name}")
