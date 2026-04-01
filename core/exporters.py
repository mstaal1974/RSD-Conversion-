"""Convert skill-record DataFrames to the output CSV formats.

Handles both old column names (rsd_skill_records schema) and
new column names (skill_records schema) so downloads always work.
"""
from __future__ import annotations
import pandas as pd


def _get(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    """Return the first column that exists in df, or a series of defaults."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), dtype=str)


def to_rsd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSD template format for downstream tools.

    Accepts both schema variants:
      - keywords  (old)
      - keywords_semicolon  (new)
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "Unit Code", "Unit Title", "Element Title",
            "Skill Statement", "Keywords", "QA Pass",
        ])

    out = pd.DataFrame()
    out["Unit Code"]       = _get(df, "unit_code")
    out["Unit Title"]      = _get(df, "unit_title")
    out["Element Title"]   = _get(df, "element_title")
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Keywords"]        = _get(df, "keywords_semicolon", "keywords")
    out["QA Pass"]         = _get(df, "qa_passes")
    return out


def to_traceability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full audit trail per element.

    Accepts both schema variants for all columns.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "Unit Code", "Unit Title", "Element",
            "Performance Criteria", "Skill Statement",
            "Keywords", "Prompt", "Model", "Temperature",
            "QA: One Sentence", "QA: Word Count",
            "QA: Has Method", "QA: Has Outcome",
            "QA: Passes", "Rewrites", "Error",
        ])

    out = pd.DataFrame()
    out["Unit Code"]          = _get(df, "unit_code")
    out["Unit Title"]         = _get(df, "unit_title")
    out["Element"]            = _get(df, "element_title")
    out["Performance Criteria"] = _get(df, "pcs_text")
    out["Skill Statement"]    = _get(df, "skill_statement")
    out["Keywords"]           = _get(df, "keywords_semicolon", "keywords")
    out["Prompt"]             = _get(df, "bart_prompt")
    out["Model"]              = _get(df, "bart_model")
    out["Temperature"]        = _get(df, "bart_temperature")
    out["QA: One Sentence"]   = _get(df, "qa_one_sentence")
    out["QA: Word Count"]     = _get(df, "qa_word_count")
    out["QA: Has Method"]     = _get(df, "qa_has_method")
    out["QA: Has Outcome"]    = _get(df, "qa_has_outcome")
    out["QA: Passes"]         = _get(df, "qa_passes")
    out["Rewrites"]           = _get(df, "rewrite_count")
    out["Error"]              = _get(df, "error_message")
    return out
