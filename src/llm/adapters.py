import os
from abc import ABC, abstractmethod
from typing import Optional

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass


class CloudLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        api_key: Optional[str] = None,
    ):
        """Initialize cloud LLM with Mistral API.

        Args:
            model_name: Mistral model name
            api_key: Mistral API key
        """
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai package required for CloudLLM. Install with: pip install mistralai"
            )

        self.model_name = model_name
        api_key = api_key or os.getenv("MISTRAL_API_KEY")

        if not api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable or pass api_key parameter."
            )

        self.client = Mistral(api_key=api_key)
        logger.info(f"CloudLLM initialized with Mistral model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using Mistral API."""
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Mistral API generation failed: {e}")
            raise


class LocalLLM(BaseLLM):
    """Local LLM for CPU inference using HuggingFace transformers.

    Uses Qwen 2.5 instruction-tuned model for efficient CPU inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        max_length: int = 2048,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required for LocalLLM. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading local LLM: {model_name} on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.model.to(device)
        self.model.eval()

        logger.info(f"Local LLM loaded successfully: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using local model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text
        """
        try:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            raise
