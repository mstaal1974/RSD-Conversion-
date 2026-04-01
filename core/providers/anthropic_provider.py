"""Anthropic provider — wraps the anthropic SDK."""
from __future__ import annotations
import time
import logging

log = logging.getLogger(__name__)

_DEFAULT_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._name = "anthropic"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def default_models() -> list[str]:
        return _DEFAULT_MODELS

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int = 1500,
    ) -> str:
        import anthropic

        # Separate system message — Anthropic uses a dedicated param
        system: str | None = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
        )
        if system:
            kwargs["system"] = system

        for attempt in range(3):
            try:
                resp = self._client.messages.create(**kwargs)
                return resp.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 10
                log.warning("Anthropic rate limit, retrying in %s s (attempt %s/3)", wait, attempt + 1)
                time.sleep(wait)
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt * 5
                    log.warning("Anthropic server error %s, retrying in %s s", exc.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Anthropic: exhausted 3 retry attempts")
