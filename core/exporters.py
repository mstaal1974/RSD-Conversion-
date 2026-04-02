"""
Convert skill-record DataFrames to output CSV formats.

Matches the example_output_from_Analysis.xlsx format:
  RSD Name    = element title
  Unit Code   = BSBAUD411.1, BSBAUD411.2 ... (unit_code + element number within unit)
  Unit Title  = UoC title from the training package data
  Element Title = element title (same as RSD Name)
  TP Code     = training package code (e.g. BSB)
  TP Title    = training package title (e.g. Business Services Training Package)
  QA Pass     = boolean
"""
from __future__ import annotations
import pandas as pd


def _get(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    """Return first matching column, or a series of defaults."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), dtype=str)


def _build_unit_codes(df: pd.DataFrame) -> pd.Series:
    """
    Build BSBAUD411.1, BSBAUD411.2 ... codes.
    Groups rows by unit_code and numbers elements sequentially within each unit.
    """
    unit_codes = _get(df, "unit_code").astype(str).str.strip()
    result = [""] * len(df)
    counters: dict[str, int] = {}
    for i, code in enumerate(unit_codes):
        if not code or code in ("", "nan", "None"):
            code = "UNIT"
        counters[code] = counters.get(code, 0) + 1
        result[i] = f"{code}.{counters[code]}"
    return pd.Series(result, dtype=str)


# ── RSD review format (matches example output) ────────────────────────────────

_RSD_COLUMNS = [
    "RSD Name",
    "Unit Code",
    "Unit Title",
    "Element Title",
    "Skill Statement",
    "Keywords",
    "TP Code ",      # trailing space matches example file header exactly
    "TP Title ",     # trailing space matches example file header exactly
    "QA Pass",
]


def to_rsd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSD format matching example_output_from_Analysis.xlsx exactly.

    RSD Name      = element title
    Unit Code     = BSBAUD411.1 (unit_code.element_number)
    Unit Title    = UoC title
    Element Title = element title (same as RSD Name)
    TP Code       = e.g. BSB
    TP Title      = e.g. Business Services Training Package
    QA Pass       = boolean
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_RSD_COLUMNS)

    out = pd.DataFrame()
    out["RSD Name"]      = _get(df, "element_title")
    out["Unit Code"]     = _build_unit_codes(df)
    out["Unit Title"]    = _get(df, "unit_title")
    out["Element Title"] = _get(df, "element_title")
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Keywords"]      = _get(df, "keywords_semicolon", "keywords")
    out["TP Code "]      = _get(df, "tp_code")
    out["TP Title "]     = _get(df, "tp_title")
    out["QA Pass"]       = _get(df, "qa_passes")
    return out


# ── OSMT batch import ─────────────────────────────────────────────────────────

_OSMT_COLUMNS = [
    "RSD Name", "Authors", "Skill Statement", "Categories", "Keywords",
    "Standards", "Certifications", "Occupation Major Groups",
    "Occupation Minor Groups", "Broad Occupations", "Detailed Occupations",
    "O*NET Job Codes", "Employers", "Alignment Name", "Alignment URL",
    "Alignment Framework", "Alignment 2 Name", "Alignment 2 URL",
    "Alignment 2 Framework",
]


def to_osmt_rows(df: pd.DataFrame, author: str = "") -> pd.DataFrame:
    """
    OSMT batch import format.
    RSD Name = element title (not unit code) to match OSMT naming convention.
    Standards = unit code dot element number for traceability.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_OSMT_COLUMNS)

    out = pd.DataFrame(index=range(len(df)))
    out["RSD Name"]        = _get(df, "element_title")
    out["Authors"]         = author
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Categories"]      = _get(df, "unit_title")
    out["Keywords"]        = _get(df, "keywords_semicolon", "keywords")
    out["Standards"]       = _build_unit_codes(df)   # BSBAUD411.1 etc.

    for col in [
        "Certifications", "Occupation Major Groups", "Occupation Minor Groups",
        "Broad Occupations", "Detailed Occupations", "O*NET Job Codes",
        "Employers", "Alignment Name", "Alignment URL", "Alignment Framework",
        "Alignment 2 Name", "Alignment 2 URL", "Alignment 2 Framework",
    ]:
        out[col] = ""

    return out[_OSMT_COLUMNS]


# ── Traceability ──────────────────────────────────────────────────────────────

def to_traceability(df: pd.DataFrame) -> pd.DataFrame:
    """Full audit trail per element."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "RSD Name", "Unit Code", "Unit Title", "Element",
            "Performance Criteria", "Skill Statement", "Keywords",
            "TP Code", "TP Title",
            "Prompt", "Model", "Temperature",
            "QA: One Sentence", "QA: Word Count",
            "QA: Has Method", "QA: Has Outcome",
            "QA: Passes", "Rewrites", "Error",
        ])

    out = pd.DataFrame()
    out["RSD Name"]             = _get(df, "element_title")
    out["Unit Code"]            = _build_unit_codes(df)
    out["Unit Title"]           = _get(df, "unit_title")
    out["Element"]              = _get(df, "element_title")
    out["Performance Criteria"] = _get(df, "pcs_text")
    out["Skill Statement"]      = _get(df, "skill_statement")
    out["Keywords"]             = _get(df, "keywords_semicolon", "keywords")
    out["TP Code"]              = _get(df, "tp_code")
    out["TP Title"]             = _get(df, "tp_title")
    out["Prompt"]               = _get(df, "bart_prompt")
    out["Model"]                = _get(df, "bart_model")
    out["Temperature"]          = _get(df, "bart_temperature")
    out["QA: One Sentence"]     = _get(df, "qa_one_sentence")
    out["QA: Word Count"]       = _get(df, "qa_word_count")
    out["QA: Has Method"]       = _get(df, "qa_has_method")
    out["QA: Has Outcome"]      = _get(df, "qa_has_outcome")
    out["QA: Passes"]           = _get(df, "qa_passes")
    out["Rewrites"]             = _get(df, "rewrite_count")
    out["Error"]                = _get(df, "error_message")
    return out
