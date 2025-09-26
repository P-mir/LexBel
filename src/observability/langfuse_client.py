import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional

from langfuse import Langfuse

from utils.logging_config import setup_logger

logger = setup_logger(__name__)


class LangfuseTracer:
    """Centralized Langfuse tracing."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = False
        self.client = None

        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if secret_key and public_key:
            try:
                self.client = Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=host
                )
                self.enabled = True
                logger.info(f"Langfuse initialized successfully (host: {host})")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse not configured (missing API keys)")

        self._initialized = True

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ):
        """Create a new trace."""
        if not self.enabled:
            return None

        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id or "anonymous",
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or []
            )
            return trace
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None

    def create_span(
        self,
        trace_id: str,
        name: str,
        input_data: Any = None,
        metadata: Optional[Dict] = None,
        start_time: Optional[float] = None
    ):
        """Create a span within a trace."""
        if not self.enabled:
            return None

        try:
            span = self.client.span(
                trace_id=trace_id,
                name=name,
                input=input_data,
                metadata=metadata or {},
                start_time=start_time
            )
            return span
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return None

    def create_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input_data: Any,
        output_data: Any = None,
        metadata: Optional[Dict] = None,
        usage: Optional[Dict] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Track LLM generation with token usage."""
        if not self.enabled:
            return None

        try:
            generation = self.client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=input_data,
                output=output_data,
                metadata=metadata or {},
                usage=usage,
                start_time=start_time,
                end_time=end_time
            )
            return generation
        except Exception as e:
            logger.error(f"Failed to create generation: {e}")
            return None

    def flush(self):
        """Flush pending events to Langfuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


_tracer = None

def get_tracer() -> LangfuseTracer:
    """Get singleton tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


def generate_session_id() -> str:
    """Generate anonymous session ID."""
    return f"session_{uuid.uuid4().hex[:16]}"


def anonymize_user_id(identifier: str) -> str:
    """Create anonymous but consistent user ID from identifier."""
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


@contextmanager
def trace_context(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    tags: Optional[List[str]] = None
):
    """Context manager for tracing."""
    tracer = get_tracer()

    if not tracer.enabled:
        yield None
        return

    trace = tracer.create_trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags
    )

    try:
        yield trace
    finally:
        if trace:
            tracer.flush()


def observe_retrieval(func):
    """Decorator for retrieval operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()

        if not tracer.enabled:
            return func(*args, **kwargs)

        start = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            # Log retrieval metrics
            num_results = len(result) if isinstance(result, list) else 0
            logger.debug(f"Retrieval completed: {num_results} results in {duration_ms:.2f}ms")

            return result
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    return wrapper


def observe_llm(func):
    """Decorator for LLM generation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()

        if not tracer.enabled:
            return func(*args, **kwargs)

        start = time.time()

        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            logger.debug(f"LLM generation completed in {duration_ms:.2f}ms")

            return result
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    return wrapper
