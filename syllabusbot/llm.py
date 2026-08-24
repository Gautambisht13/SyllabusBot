"""Chat model factory.

Both providers are wired up; `LLM_PROVIDER` picks one. Temperature defaults to
0.0 — answering from a syllabus is an extraction task, and sampling only adds
opportunities to drift off the retrieved text.
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

from syllabusbot.config import Settings, get_settings
from syllabusbot.errors import MissingCredentialsError, MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = ["MissingCredentialsError", "MissingDependencyError", "build_llm"]


def build_llm(settings: Settings | None = None, *, streaming: bool = False) -> BaseChatModel:
    settings = settings or get_settings()

    if settings.llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingCredentialsError(
                "OPENAI_API_KEY is not set. Add it to .env, or set "
                "LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY."
            )
        try:
            from langchain_openai import ChatOpenAI  # lazy import
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "LLM_PROVIDER=openai needs the OpenAI integration:\n"
                "    pip install langchain-openai"
            ) from exc

        logger.info("Chat model: OpenAI %s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=streaming,
            timeout=60,
            max_retries=2,
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingCredentialsError(
            "ANTHROPIC_API_KEY is not set. Add it to .env, or set "
            "LLM_PROVIDER=openai with OPENAI_API_KEY."
        )
    try:
        from langchain_anthropic import ChatAnthropic  # lazy import
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "LLM_PROVIDER=anthropic needs the Anthropic integration:\n"
            "    pip install langchain-anthropic"
        ) from exc

    logger.info("Chat model: Anthropic %s", settings.anthropic_model)
    return ChatAnthropic(
        model=settings.anthropic_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )
