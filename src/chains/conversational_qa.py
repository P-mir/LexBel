import os
from typing import Any, Optional

from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ConversationalQA:
    def __init__(
        self,
        retriever: Any,
        model_name: str = "mistral-small-latest",
        api_key: Optional[str] = None,
        max_history_messages: int = 10,
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.max_history_messages = max_history_messages

        api_key = api_key or os.getenv("MISTRAL_API_KEY")

        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            api_key=api_key,
        )

        self.memory = MemorySaver()

        logger.info(f"Conversational QA initialized with {model_name}")
