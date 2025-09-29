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
        """Create a new trace by starting a span."""
        if not self.enabled:
            return None

        try:
            # In Langfuse 3.x, start_span creates a trace automatically
            trace_metadata = metadata or {}
            if session_id:
                trace_metadata['session_id'] = session_id
            if user_id:
                trace_metadata['user_id'] = user_id
            
            span = self.client.start_span(
                name=name,
                metadata=trace_metadata
            )
            # Return the span object which contains trace_id
            return span
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
        """Create a span within a trace - DEPRECATED, use trace.start_span() instead."""
        # This method is kept for backward compatibility but should not be used
        # Instead, use the returned span object from create_trace()
        logger.warning("create_span() is deprecated. Use span.start_span() instead.")
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
        """Track LLM generation - DEPRECATED, use span.start_generation() instead."""
        # This method is kept for backward compatibility but should not be used
        # Instead, use the span object from create_trace()
        logger.warning("create_generation() is deprecated. Use span.start_generation() instead.")
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
