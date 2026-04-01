"""
BART skill-statement generator.

BART = Behaviour · Action · Result · Timeframe/Context

Each element gets one sentence (30-60 words) that describes:
  - the method/action the learner applies  (has_method_phrase)
  - the observable outcome/result          (has_outcome_phrase)

QA runs after every generation and auto-rewrites up to max_fixes times.
"""
from __future__ import annotations
import re
from core.providers.base import LLMProvider

_SYSTEM = """\
You are an expert Australian VET curriculum writer.
Generate a single BART-framework skill statement for the given training package element.

Rules:
1. Exactly one sentence — no colons, semicolons, or bullet points.
2. 30–60 words.
3. Include a method phrase (how the learner does it).
4. Include an outcome phrase (what is produced or achieved).
5. Use plain English; avoid jargon and passive voice where possible.
6. Do NOT start with "I" or the learner's name.
7. Output ONLY the skill statement — no preamble, no explanation.
"""

_REWRITE_SYSTEM = """\
You are an expert Australian VET curriculum writer revising a BART skill statement.
Fix the specific issue described below, keeping everything else the same.
Output ONLY the corrected skill statement — no preamble, no explanation.
"""

_METHOD_HINTS = [
    r"\busing\b", r"\bby\b", r"\bthrough\b", r"\bapplying\b",
    r"\bfollowing\b", r"\butilising\b", r"\bemploying\b", r"\bvia\b",
]
_OUTCOME_HINTS = [
    r"\bto\b", r"\benabling\b", r"\bensuring\b", r"\bso that\b",
    r"\bresulting in\b", r"\bproducing\b", r"\bachieving\b",
    r"\bdemonstrat\b", r"\bdeliver\b",
]


def _qa(text: str) -> dict:
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    word_count = len(text.split())
    one_sentence = len(sentences) == 1 or (len(sentences) == 2 and sentences[-1] == "")
    has_method = any(re.search(p, text, re.IGNORECASE) for p in _METHOD_HINTS)
    has_outcome = any(re.search(p, text, re.IGNORECASE) for p in _OUTCOME_HINTS)
    passes = one_sentence and 30 <= word_count <= 60 and has_method and has_outcome
    return {
        "one_sentence": one_sentence,
        "word_count": word_count,
        "has_method_phrase": has_method,
        "has_outcome_phrase": has_outcome,
        "passes": passes,
    }


def _build_issue_description(qa: dict) -> str:
    issues = []
    if not qa["one_sentence"]:
        issues.append("The statement must be exactly one sentence.")
    if qa["word_count"] < 30:
        issues.append(f"Too short ({qa['word_count']} words). Aim for 30–60 words.")
    if qa["word_count"] > 60:
        issues.append(f"Too long ({qa['word_count']} words). Shorten to 30–60 words.")
    if not qa["has_method_phrase"]:
        issues.append("Add a method phrase (e.g. 'using …', 'by applying …', 'through …').")
    if not qa["has_outcome_phrase"]:
        issues.append("Add an outcome phrase (e.g. 'to ensure …', 'enabling …', 'resulting in …').")
    return " ".join(issues)


def generate_skill_statement(
    provider: LLMProvider,
    model: str,
    unit_code: str,
    unit_title: str,
    element_title: str,
    pcs_text: str,
    max_fixes: int = 1,
    temperature: float = 0.2,
) -> tuple[str, dict, str]:
    """
    Returns:
        skill_statement  — final text
        qa               — QA dict (passes bool + details + rewrite_count)
        prompt           — the initial user prompt sent to the LLM
    """
    prompt = (
        f"Unit code: {unit_code}\n"
        f"Unit title: {unit_title}\n"
        f"Element title: {element_title}\n"
        f"Performance criteria:\n{pcs_text}\n\n"
        "Write one BART skill statement for this element."
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]

    skill = provider.chat_completion(messages, model, temperature).strip()
    qa = _qa(skill)
    rewrite_count = 0

    for _ in range(max_fixes):
        if qa["passes"]:
            break
        issue = _build_issue_description(qa)
        fix_messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original statement:\n{skill}\n\n"
                    f"Issue to fix:\n{issue}\n\n"
                    "Rewrite:"
                ),
            },
        ]
        skill = provider.chat_completion(fix_messages, model, temperature).strip()
        qa = _qa(skill)
        rewrite_count += 1

    qa["rewrite_count"] = rewrite_count
    return skill, qa, prompt
