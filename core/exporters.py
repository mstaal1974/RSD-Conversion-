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
import numpy as np
import pandas as pd


def _get(df: pd.DataFrame, *candidates: str, default: str = "") -> pd.Series:
    """Return first matching column, or a series of defaults."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), dtype=str)


def _str_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Like _get but normalised: NaN → '', stripped, str dtype."""
    return _get(df, *candidates).fillna("").astype(str).str.strip()


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

    Auto-populated when the source dataframe carries:
      esco_skill_title / esco_skill_uri   → Alignment {Name, URL, Framework}
      anzsco_code / anzsco_title          → Alignment 2 + Occupation Major / Minor Groups
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_OSMT_COLUMNS)

    out = pd.DataFrame(index=range(len(df)))
    out["RSD Name"]        = _get(df, "element_title")
    out["Authors"]         = author
    out["Skill Statement"] = _get(df, "skill_statement")
    out["Categories"]      = _get(df, "unit_title")
    out["Keywords"]        = _get(df, "keywords_semicolon", "keywords")
    out["Standards"]       = _build_standards(df)

    out["Certifications"]          = ""
    out["Occupation Major Groups"] = _anzsco_major(df)
    out["Occupation Minor Groups"] = _anzsco_minor(df)
    out["Broad Occupations"]       = _anzsco_broad(df)
    out["Detailed Occupations"]    = _anzsco_detailed(df)
    out["O*NET Job Codes"]         = ""
    out["Employers"]               = ""

    # Primary alignment → ESCO skill
    esco_title = _str_col(df, "esco_skill_title")
    esco_uri   = _str_col(df, "esco_skill_uri")
    out["Alignment Name"]      = esco_title
    out["Alignment URL"]       = esco_uri
    out["Alignment Framework"] = np.where(esco_uri == "", "", "ESCO v1.2.1")

    # Secondary alignment → ANZSCO occupation (primary on the UOC)
    anzsco_title = _str_col(df, "anzsco_title")
    anzsco_code  = _str_col(df, "anzsco_code")
    out["Alignment 2 Name"]      = anzsco_title
    out["Alignment 2 URL"]       = np.where(
        anzsco_code == "", "",
        "https://www.abs.gov.au/ausstats/abs@.nsf/mf/1220.0?code=" + anzsco_code,
    )
    out["Alignment 2 Framework"] = np.where(anzsco_code == "", "", "ANZSCO 2022")

    return out[_OSMT_COLUMNS]


_TGA_BASE = "https://training.gov.au/Training/Details/"


def _build_standards(df: pd.DataFrame) -> pd.Series:
    """
    Combine the BSBAUD411.1 reference with a TGA URL for the parent unit.
    Format: "BSBAUD411.1 (https://training.gov.au/Training/Details/BSBAUD411)"
    """
    refs = _build_unit_codes(df)
    units = _get(df, "unit_code").astype(str).str.strip()
    urls = units.where(units.eq(""), _TGA_BASE + units)
    combined = refs + urls.where(urls.eq(""), " (" + urls + ")")
    return combined


def _anzsco_part(df: pd.DataFrame, prefix_len: int) -> pd.Series:
    """Return the n-digit prefix of anzsco_code as ANZSCO group string, blank if absent."""
    code = _str_col(df, "anzsco_code")
    return code.str.slice(0, prefix_len).where(code != "", "")


def _anzsco_major(df: pd.DataFrame) -> pd.Series:
    return _anzsco_part(df, 1)


def _anzsco_minor(df: pd.DataFrame) -> pd.Series:
    return _anzsco_part(df, 2)


def _anzsco_broad(df: pd.DataFrame) -> pd.Series:
    return _anzsco_part(df, 3)


def _anzsco_detailed(df: pd.DataFrame) -> pd.Series:
    return _anzsco_part(df, 4)


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
