"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at an OpenAI-compatible API.

    Base URL is configurable via OPENROUTER_BASE_URL so the same code can
    target OpenRouter (default) or FPT AI Marketplace, etc.
    """
    kwargs = {
        "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
        "openai_api_base": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    }
    # Optional latency knobs (no-op if env vars unset → backward compatible).
    if (max_tokens := os.getenv("OPENROUTER_MAX_TOKENS")):
        kwargs["max_tokens"] = int(max_tokens)
    if (temperature := os.getenv("OPENROUTER_TEMPERATURE")):
        kwargs["temperature"] = float(temperature)
    return ChatOpenAI(**kwargs)