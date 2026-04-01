"""OpenAI provider — wraps the openai SDK."""
from __future__ import annotations
import time
import logging

log = logging.getLogger(__name__)

_DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]


class OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._name = "openai"

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
        import openai

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except openai.RateLimitError:
                wait = 2 ** attempt * 10
                log.warning("OpenAI rate limit hit, retrying in %s s (attempt %s/3)", wait, attempt + 1)
                time.sleep(wait)
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt * 5
                    log.warning("OpenAI server error %s, retrying in %s s", exc.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("OpenAI: exhausted 3 retry attempts")
