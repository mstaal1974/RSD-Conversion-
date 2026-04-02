"""
Extractor for training.gov.au 'blob' CSV/XLSX format.

Handles both classic column names (unit_code, unit_title) and
training.gov.au export column names (UoC Code, UoC Title, TP Code, TP Title).
"""
from __future__ import annotations
import re
import pandas as pd

_ELEM_RE = re.compile(r"^(\d+)\.?\s+(?![\d.])(.*)", re.MULTILINE)
_PC_RE   = re.compile(r"^(\d+\.\d+)\.?\s+(.*)", re.MULTILINE)


def _candidate_blob_col(df: pd.DataFrame) -> str | None:
    best_col, best_score = None, 0.0
    for col in df.columns:
        sample = "\n".join(df[col].dropna().head(10).astype(str))
        n_pcs   = len(_PC_RE.findall(sample))
        n_elems = len(_ELEM_RE.findall(sample))
        score   = n_pcs * 0.7 + n_elems * 0.3
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score >= 2 else None


def _parse_blob(text: str) -> list[tuple[str, str, list[str]]]:
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


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive column lookup supporting spaces, underscores, dots."""
    normalised = {c.lower().replace(" ", "_").replace(".", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace(".", "")
        if key in normalised:
            return normalised[key]
    return None


class BlobExtractor:
    name    = "training_gov_blob"
    version = "1.2.0"

    def score(self, df: pd.DataFrame) -> float:
        col = _candidate_blob_col(df)
        if col is None:
            return 0.0
        sample = "\n".join(df[col].dropna().head(10).astype(str))
        n_pcs  = len(_PC_RE.findall(sample))
        return min(1.0, n_pcs / 10)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        blob_col = _candidate_blob_col(df)
        if blob_col is None:
            raise ValueError("BlobExtractor: could not identify a blob column")

        unit_code_col  = _find_col(df, ["uoc_code", "uoccode", "unit_code", "code", "aqf_code"])
        unit_title_col = _find_col(df, ["uoc_title", "uoctitle", "unit_title", "title", "unit_name"])
        tp_code_col    = _find_col(df, ["tp_code", "tpcode", "training_package_code"])
        tp_title_col   = _find_col(df, ["tp_title", "tptitle", "training_package_title", "training_package"])

        records = []
        for _, row in df.iterrows():
            blob       = str(row.get(blob_col, "") or "")
            unit_code  = str(row[unit_code_col]).strip()  if unit_code_col  else ""
            unit_title = str(row[unit_title_col]).strip() if unit_title_col else ""
            tp_code    = str(row[tp_code_col]).strip()    if tp_code_col    else ""
            tp_title   = str(row[tp_title_col]).strip()   if tp_title_col   else ""

            for _enum, elem_title, pcs in _parse_blob(blob):
                if not elem_title or not pcs:
                    continue
                records.append(dict(
                    unit_code=unit_code,
                    unit_title=unit_title,
                    element_title=elem_title,
                    pcs_text="\n".join(pcs),
                    tp_code=tp_code,
                    tp_title=tp_title,
                ))

        return pd.DataFrame(records, columns=[
            "unit_code", "unit_title", "element_title", "pcs_text",
            "tp_code", "tp_title",
        ])
