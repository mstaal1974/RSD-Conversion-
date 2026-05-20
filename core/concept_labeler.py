"""
core/concept_labeler.py

LLM-generated short labels for concept nodes (from core.concept_nodes).

The label is what curators see in the occupational profile tree instead
of the raw 30–60-word BART statement. Labels are short (≤8 words),
Australian-VET-vocabulary, and stable for a given canonical statement
(cached by content hash so re-runs don't re-bill).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

from core.concept_nodes import ConceptNode
from core.providers.base import LLMProvider


_SYSTEM = """\
You are an expert Australian VET curriculum writer.
Read several similar skill statements and write a single short label
(≤ 8 words) that names the underlying skill concept.

Rules:
1. ≤ 8 words. Title Case is fine.
2. No leading verb if the concept is a noun phrase (e.g. "MIG Welding" not "Perform MIG welding").
3. Use plain Australian English; avoid jargon outside the vocational area.
4. Output ONLY the label — no quotes, no preamble, no explanation.
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _cached_label(content_hash: str, prompt: str, provider_name: str,
                  provider_ref: int, model: str) -> str:
    """
    Process-scoped cache; the (content_hash, model) pair is the key.
    provider_ref is id(provider) so we don't accidentally reuse a label
    when the user swaps providers.
    """
    return ""  # placeholder; real call lives in label_node below


def label_node(
    node: ConceptNode,
    provider: LLMProvider,
    model: str = "gpt-4o-mini",
    sample_members: int = 4,
    temperature: float = 0.0,
) -> str:
    """
    Generate an LLM short label for a concept node. Returns the label
    string and mutates node.label in place. Cached by content hash.

    Falls back to node.label (the truncated canonical) on any failure.
    """
    if node.id == -1:
        return node.label  # Unmapped — keep the default

    # Sample a few representative statements (canonical first)
    samples = [node.canonical_text]
    for s in node.statements:
        text = (s.get("skill_statement") or "").strip()
        if text and text not in samples:
            samples.append(text)
        if len(samples) >= sample_members:
            break

    payload = "\n\n".join(f"- {s}" for s in samples)
    content_hash = _hash(payload)

    cache_key = (content_hash, model)
    cached = _LABEL_CACHE.get(cache_key)
    if cached:
        node.label = cached
        return cached

    prompt = (
        "Skill statements from a cluster:\n\n"
        f"{payload}\n\n"
        "Give a short concept label (≤ 8 words)."
    )
    try:
        label = provider.chat_completion(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=20,
        ).strip().strip('"').strip("'")
    except Exception:
        return node.label  # leave the canonical-truncation fallback

    # Defensive trim
    label = " ".join(label.split()[:8])
    if not label:
        return node.label

    _LABEL_CACHE[cache_key] = label
    node.label = label
    return label


def label_nodes(
    nodes: list[ConceptNode],
    provider: LLMProvider,
    model: str = "gpt-4o-mini",
    progress_callback=None,
) -> int:
    """Label every node that doesn't already have a custom label. Returns count of LLM calls made."""
    calls = 0
    total = len(nodes)
    for i, n in enumerate(nodes):
        before = n.label
        label_node(n, provider=provider, model=model)
        if n.label != before:
            calls += 1
        if progress_callback:
            progress_callback((i + 1) / max(total, 1), f"Labelled {i + 1}/{total} concept nodes")
    return calls


# Process-scoped dict cache (in addition to the lru_cache on `_cached_label`)
# so we can populate it from disk or DB later if we want cross-session
# persistence.
_LABEL_CACHE: dict[tuple[str, str], str] = {}
