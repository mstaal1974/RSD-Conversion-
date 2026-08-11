"""
core/laiser_align.py

Thin, optional wrapper around the LAiSER skill-extraction library
(https://github.com/LAiSER-Software/extract-module, BSD-3-Clause).

LAiSER is used here as an *independent* second aligner: it extracts skill
concepts from text and aligns them to multiple taxonomies (ESCO, O*NET, OSN)
via its own models + FAISS indexes. We feed it our existing RSD skill
statements to (a) cross-validate our in-house MiniLM ESCO alignment and
(b) enrich each statement with multi-taxonomy IDs.

LAiSER is a heavy, API-backed dependency (torch, transformers,
sentence-transformers, faiss-cpu, google-genai) and is *not* in the core
requirements. Install it only where this feature runs:

    pip install -r requirements-laiser.txt      # or: pip install laiser

The LLM step is API-backed (OpenAI or Gemini); set the matching key
(OPENAI_API_KEY or GEMINI_API_KEY). The import is guarded so the rest of the
app never breaks when LAiSER isn't installed.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

# Normalised output column names this module always emits, regardless of any
# future churn in LAiSER's own column labels.
COLUMNS = [
    "id", "type", "raw_concept", "taxonomy_concept",
    "taxonomy_description", "taxonomy_source", "source_url", "score",
]

# Map LAiSER's documented output columns → our normalised names.
_LAISER_COLMAP = {
    "Research ID": "id",
    "Type": "type",
    "Raw Concept": "raw_concept",
    "Taxonomy Concept": "taxonomy_concept",
    "Taxonomy Description": "taxonomy_description",
    "Taxonomy Source": "taxonomy_source",
    "Source Url": "source_url",
    "Correlation Coefficient": "score",
}


def is_available() -> bool:
    """True if the `laiser` package is importable in this environment."""
    try:
        import laiser  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_backend(backend: str | None, api_key: str | None) -> tuple[str, str, str]:
    """Return (backend, model_id, api_key), inferring from env when unset.

    Prefers an explicit backend; otherwise picks OpenAI if OPENAI_API_KEY is
    present, else Gemini if a Google key is present. model_id defaults are
    overridable via LAISER_MODEL_ID because LAiSER is still evolving.
    """
    backend = (backend or os.getenv("LAISER_BACKEND", "")).strip().lower()
    if not backend:
        if os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            backend = "gemini"
        else:
            raise RuntimeError(
                "No LLM key found. Set OPENAI_API_KEY or GEMINI_API_KEY, or "
                "pass backend/api_key explicitly."
            )

    if backend == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        model_id = os.getenv("LAISER_MODEL_ID", "gpt-4o-mini")
    elif backend == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        model_id = os.getenv("LAISER_MODEL_ID", "gemini-2.0-flash")
    else:
        key = api_key or ""
        model_id = os.getenv("LAISER_MODEL_ID", backend)

    if not key:
        raise RuntimeError(f"No API key available for LAiSER backend '{backend}'.")
    return backend, model_id, key


def align_statements(
    statements: Sequence[dict],
    *,
    id_col: str = "id",
    text_col: str = "skill_statement",
    allowed_sources: Iterable[str] = ("esco",),
    concepts: Iterable[str] = ("skills",),
    backend: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    similarity_threshold: float = 0.2,
    top_k: int = 5,
    use_gpu: bool = False,
):
    """Run LAiSER over a list of statement dicts; return a normalised DataFrame.

    Each input dict must carry `id_col` and `text_col`. The returned DataFrame
    has the columns in `COLUMNS`: one row per aligned taxonomy concept, so a
    single statement can yield several rows (e.g. its ESCO, O*NET and OSN hits).

    Raises RuntimeError with an actionable message if LAiSER isn't installed or
    no LLM key is configured.
    """
    if not is_available():
        raise RuntimeError(
            "The `laiser` package is not installed. Install it where this "
            "feature runs: pip install -r requirements-laiser.txt"
        )

    import pandas as pd
    from laiser.skill_extractor_refactored import SkillExtractorRefactored

    backend, model_id, api_key = _resolve_backend(backend, api_key)

    df_in = pd.DataFrame(list(statements))
    if id_col not in df_in.columns or text_col not in df_in.columns:
        raise ValueError(f"statements must contain '{id_col}' and '{text_col}'.")

    extractor = SkillExtractorRefactored(
        model_id=model_id,
        backend=backend,
        api_key=api_key,
        use_gpu=use_gpu,
    )
    raw = extractor.extract_concepts(
        data=df_in,
        id_column=id_col,
        text_columns=[text_col],
        input_type="skill_statement",
        allowed_sources=list(allowed_sources),
        concepts=list(concepts),
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        return_edges=False,
    )
    return _normalise(raw, id_col)


def _normalise(raw, id_col: str):
    """Rename LAiSER's columns to our stable schema; tolerate minor variants."""
    import pandas as pd

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLUMNS)

    df = raw.rename(columns=_LAISER_COLMAP).copy()
    # If LAiSER used the caller's id column name, fold it into 'id'.
    if "id" not in df.columns and id_col in df.columns:
        df["id"] = df[id_col]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "score" else 0.0
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df[COLUMNS]
