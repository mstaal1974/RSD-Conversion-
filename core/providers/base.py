"""Abstract LLM provider protocol — swap between OpenAI and Anthropic without touching callers."""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every provider must satisfy."""

    @property
    def name(self) -> str: ...

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int = 1500,
    ) -> str: ...


def with_fallback(primary: LLMProvider, fallback: LLMProvider):
    """Return a thin wrapper that tries primary then fallback on rate-limit / server errors."""

    class FallbackProvider:
        name = f"{primary.name}→{fallback.name}"

        def chat_completion(self, messages, model, temperature, max_tokens=1500):
            import time, logging

            log = logging.getLogger(__name__)
            for attempt, provider in enumerate([primary, fallback], 1):
                try:
                    return provider.chat_completion(messages, model, temperature, max_tokens)
                except Exception as exc:
                    err = str(exc).lower()
                    retryable = any(k in err for k in ("rate_limit", "ratelimit", "529", "503", "502", "overloaded"))
                    if retryable and attempt == 1:
                        log.warning("Primary provider rate-limited, waiting 20 s then falling back.")
                        time.sleep(20)
                        continue
                    raise

    return FallbackProvider()
