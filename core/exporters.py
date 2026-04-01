"""Convert skill-record DataFrames to the output CSV formats.

Handles both old column names (rsd_skill_records schema) and
new column names (skill_records schema) so downloads always work.

Formats:
  - to_rsd_rows()       — internal RSD review format
  - to_traceability()   — full audit trail
  - to_osmt_rows()      — OSMT batch import format (RSD Name = BSBAUD411.1 etc.)
"""
from __future__ import annotations
import pandas as pd


def _get(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    """Return the first column that exists in df, or a series of defaults."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), dtype=str)


def _build_rsd_names(df: pd.DataFrame) -> pd.Series:
    """
    Build RSD Name codes: BSBAUD411.1, BSBAUD411.2 … per unit.

    Groups rows by unit_code (preserving order) and numbers
    elements sequentially within each unit:
        BSBAUD411.1  ← first element of BSBAUD411
        BSBAUD411.2  ← second element of BSBAUD411
        BSBWHS201.1  ← first element of BSBWHS201
    """
    unit_codes = _get(df, "unit_code").astype(str).str.strip()
    names = [""] * len(df)
    counters: dict[str, int] = {}

    for i, code in enumerate(unit_codes):
        if not code or code == "nan":
            code = "UNIT"
        counters[code] = counters.get(code, 0) + 1
        names[i] = f"{code}.{counters[code]}"

    return pd.Series(names, dtype=str)


# ── OSMT batch import ─────────────────────────────────────────────────────────

_OSMT_COLUMNS = [
    "RSD Name",
    "Authors",
    "Skill Statement",
    "Categories",
    "Keywords",
    "Standards",
    "Certifications",
    "Occupation Major Groups",
    "Occupation Minor Groups",
    "Broad Occupations",
    "Detailed Occupations",
    "O*NET Job Codes",
    "Employers",
    "Alignment Name",
    "Alignment URL",
    "Alignment Framework",
    "Alignment 2 Name",
    "Alignment 2 URL",
    "Alignment 2 Framework",
]


def to_osmt_rows(df: pd.DataFrame, author: str = "") -> pd.DataFrame:
    """
    OSMT batch import format.

    RSD Name  = {unit_code}.{element_number_within_unit}
                e.g. BSBAUD411.1, BSBAUD411.2, BSBWHS201.1 ...

    Categories is populated with the unit title.
    Standards  is populated with the unit code for traceability.
    All other OSMT columns are blank — ready for manual completion.

    Args:
        df:     skill records DataFrame (from DB or session state)
        author: optional author string for the Authors column
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_OSMT_COLUMNS)

    out = pd.DataFrame(index=range(len(df)))
    out["RSD Name"]        = _build_rsd_names(df)
    out["Authors"]         = author
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Categories"]      = _get(df, "unit_title")
    out["Keywords"]        = _get(df, "keywords_semicolon", "keywords")
    out["Standards"]       = _get(df, "unit_code")

    for col in [
        "Certifications",
        "Occupation Major Groups",
        "Occupation Minor Groups",
        "Broad Occupations",
        "Detailed Occupations",
        "O*NET Job Codes",
        "Employers",
        "Alignment Name",
        "Alignment URL",
        "Alignment Framework",
        "Alignment 2 Name",
        "Alignment 2 URL",
        "Alignment 2 Framework",
    ]:
        out[col] = ""

    return out[_OSMT_COLUMNS]


# ── Internal RSD review format ────────────────────────────────────────────────

def to_rsd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Internal RSD review format — includes element title and QA status."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "RSD Name", "Unit Code", "Unit Title", "Element Title",
            "Skill Statement", "Keywords", "QA Pass",
        ])

    out = pd.DataFrame()
    out["RSD Name"]        = _build_rsd_names(df)
    out["Unit Code"]       = _get(df, "unit_code")
    out["Unit Title"]      = _get(df, "unit_title")
    out["Element Title"]   = _get(df, "element_title")
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Keywords"]        = _get(df, "keywords_semicolon", "keywords")
    out["QA Pass"]         = _get(df, "qa_passes")
    return out


# ── Traceability ──────────────────────────────────────────────────────────────

def to_traceability(df: pd.DataFrame) -> pd.DataFrame:
    """Full audit trail — prompt, QA checks, rewrite count, errors."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "RSD Name", "Unit Code", "Unit Title", "Element",
            "Performance Criteria", "Skill Statement", "Keywords",
            "Prompt", "Model", "Temperature",
            "QA: One Sentence", "QA: Word Count",
            "QA: Has Method", "QA: Has Outcome",
            "QA: Passes", "Rewrites", "Error",
        ])

    out = pd.DataFrame()
    out["RSD Name"]             = _build_rsd_names(df)
    out["Unit Code"]            = _get(df, "unit_code")
    out["Unit Title"]           = _get(df, "unit_title")
    out["Element"]              = _get(df, "element_title")
    out["Performance Criteria"] = _get(df, "pcs_text")
    out["Skill Statement"]      = _get(df, "skill_statement")
    out["Keywords"]             = _get(df, "keywords_semicolon", "keywords")
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
