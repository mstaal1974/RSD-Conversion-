"""Convert skill-record DataFrames to the output CSV formats."""
from __future__ import annotations
import pandas as pd


def to_rsd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSD template format expected by downstream tools.
    Columns: unit_code, unit_title, element_title, skill_statement, keywords, qa_passes
    """
    cols = {
        "unit_code": "Unit Code",
        "unit_title": "Unit Title",
        "element_title": "Element Title",
        "skill_statement": "Skill Statement",
        "keywords": "Keywords",
        "qa_passes": "QA Pass",
    }
    out = pd.DataFrame()
    for src, dst in cols.items():
        out[dst] = df.get(src, pd.Series([""] * len(df), dtype=str))
    return out


def to_traceability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traceability format — full audit trail per element.
    """
    cols = {
        "unit_code": "Unit Code",
        "unit_title": "Unit Title",
        "element_title": "Element",
        "pcs_text": "Performance Criteria",
        "skill_statement": "Skill Statement",
        "bart_prompt": "Prompt",
        "bart_model": "Model",
        "bart_temperature": "Temperature",
        "qa_one_sentence": "QA: One Sentence",
        "qa_word_count": "QA: Word Count",
        "qa_has_method": "QA: Has Method",
        "qa_has_outcome": "QA: Has Outcome",
        "qa_passes": "QA: Passes",
        "rewrite_count": "Rewrites",
        "error_message": "Error",
    }
    out = pd.DataFrame()
    for src, dst in cols.items():
        out[dst] = df.get(src, pd.Series([""] * len(df), dtype=str))
    return out
