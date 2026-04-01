"""
Extractor for training.gov.au 'blob' CSV format.

In this format a single column contains a block of text that includes
both Element titles and numbered Performance Criteria, e.g.::

    1 Prepare to work safely
    Performance criteria
    1.1 Identify hazards in the workplace and report them
    1.2 Follow WHS procedures at all times

    2 Complete work tasks
    Performance criteria
    2.1 Use correct tools
"""
from __future__ import annotations
import re
import pandas as pd


# Matches "1 Element title" or "1. Element title" but NOT "1.1 PC text"
_ELEM_RE = re.compile(r"^(\d+)\.?\s+(?![\d.])(.*)", re.MULTILINE)
# Matches "1.1 Some PC text" or "1.1. Some PC text"
_PC_RE = re.compile(r"^(\d+\.\d+)\.?\s+(.*)", re.MULTILINE)


def _candidate_blob_col(df: pd.DataFrame) -> str | None:
    """Return the name of the most likely blob column, or None."""
    best_col, best_score = None, 0.0
    for col in df.columns:
        sample = "\n".join(df[col].dropna().head(10).astype(str))
        n_pcs = len(_PC_RE.findall(sample))
        n_elems = len(_ELEM_RE.findall(sample))
        score = n_pcs * 0.7 + n_elems * 0.3
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score >= 2 else None


def _parse_blob(text: str) -> list[tuple[str, str, list[str]]]:
    """Return list of (element_num, element_title, [pc_text, ...])."""
    lines = text.splitlines()
    elements: list[tuple[str, str, list[str]]] = []
    current_elem: tuple[str, str] | None = None
    current_pcs: list[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("performance criteria"):
            continue
        pc_match = _PC_RE.match(line)
        if pc_match:
            if current_elem is not None:
                current_pcs.append(f"{pc_match.group(1)} {pc_match.group(2).strip()}")
            continue
        elem_match = _ELEM_RE.match(line)
        if elem_match:
            if current_elem is not None:
                elements.append((current_elem[0], current_elem[1], current_pcs))
            current_elem = (elem_match.group(1), elem_match.group(2).strip())
            current_pcs = []
    if current_elem is not None:
        elements.append((current_elem[0], current_elem[1], current_pcs))
    return elements


class BlobExtractor:
    name = "training_gov_blob"
    version = "1.1.0"

    def score(self, df: pd.DataFrame) -> float:
        col = _candidate_blob_col(df)
        if col is None:
            return 0.0
        sample = "\n".join(df[col].dropna().head(10).astype(str))
        n_pcs = len(_PC_RE.findall(sample))
        return min(1.0, n_pcs / 10)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        blob_col = _candidate_blob_col(df)
        if blob_col is None:
            raise ValueError("BlobExtractor: could not identify a blob column")

        unit_code_col = _find_col(df, ["unit_code", "code", "aqf_code"])
        unit_title_col = _find_col(df, ["unit_title", "title", "unit_name"])

        records = []
        for _, row in df.iterrows():
            blob = str(row.get(blob_col, "") or "")
            unit_code = str(row[unit_code_col]).strip() if unit_code_col else ""
            unit_title = str(row[unit_title_col]).strip() if unit_title_col else ""

            for _enum, elem_title, pcs in _parse_blob(blob):
                if not elem_title or not pcs:
                    continue
                records.append(
                    dict(
                        unit_code=unit_code,
                        unit_title=unit_title,
                        element_title=elem_title,
                        pcs_text="\n".join(pcs),
                    )
                )
        return pd.DataFrame(records, columns=["unit_code", "unit_title", "element_title", "pcs_text"])


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None
