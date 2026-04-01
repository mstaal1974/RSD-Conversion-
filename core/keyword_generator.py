"""Generate semicolon-separated search keywords for a skill statement."""
from __future__ import annotations
from core.providers.base import LLMProvider

_SYSTEM = """\
You are a VET search indexing specialist.
Given a BART skill statement and the performance criteria it covers,
output 5–10 semicolon-separated lowercase keyword phrases useful for full-text search.
No preamble, no explanation — only the semicolon-separated list.
"""


def generate_keywords(
    provider: LLMProvider,
    model: str,
    skill_statement: str,
    pcs_text: str,
    temperature: float = 0.1,
) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Skill statement:\n{skill_statement}\n\n"
                f"Performance criteria:\n{pcs_text}"
            ),
        },
    ]
    return provider.chat_completion(messages, model, temperature, max_tokens=200).strip()
