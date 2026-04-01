"""
Extractor for 'row per performance criteria' CSV format.

Each row contains one PC, with explicit Element and PC columns, e.g.::

    unit_code | unit_title | element_number | element_title | pc_number | pc_text
    BSBWHS201 | Contribute… | 1              | Prepare…      | 1.1       | Identify hazards…
"""
from __future__ import annotations
import re
import pandas as pd

_PC_TOKEN_RE = re.compile(r"^\d+\.\d+")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


class RowPerPCExtractor:
    name = "row_per_pc"
    version = "1.1.0"

    def score(self, df: pd.DataFrame) -> float:
        """High score when we find element and PC columns side-by-side."""
        has_elem_title = _find_col(df, ["element_title", "element", "element_name"]) is not None
        has_pc_col = _find_col(df, ["performance_criteria", "pc_text", "pc", "criteria"]) is not None
        # Also check for a column whose values start with PC tokens
        pc_token_col = any(
            df[c].dropna().head(20).astype(str).apply(lambda v: bool(_PC_TOKEN_RE.match(v))).mean() > 0.5
            for c in df.columns
        )
        score = 0.0
        if has_elem_title:
            score += 0.5
        if has_pc_col or pc_token_col:
            score += 0.5
        return score

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = _find_col(df, ["unit_code", "code", "aqf_code"])
        unit_title_col = _find_col(df, ["unit_title", "title", "unit_name"])
        elem_title_col = _find_col(df, ["element_title", "element", "element_name"])
        pc_col = _find_col(df, ["performance_criteria", "pc_text", "pc", "criteria", "pc_description"])

        if elem_title_col is None:
            raise ValueError("RowPerPCExtractor: cannot find element title column")

        # If no explicit PC col, look for a column where values look like PC tokens
        if pc_col is None:
            for c in df.columns:
                vals = df[c].dropna().head(20).astype(str)
                if vals.apply(lambda v: bool(_PC_TOKEN_RE.match(v))).mean() > 0.5:
                    pc_col = c
                    break

        group_keys = []
        if unit_code_col:
            group_keys.append(unit_code_col)
        group_keys.append(elem_title_col)

        records = []
        for keys, grp in df.groupby(group_keys, sort=False):
            if isinstance(keys, str):
                keys = (keys,)
            unit_code = str(keys[0]) if unit_code_col else ""
            elem_title = str(keys[-1])
            unit_title = ""
            if unit_title_col:
                unit_title = str(grp[unit_title_col].iloc[0]) if len(grp) else ""

            pcs: list[str] = []
            if pc_col:
                pcs = grp[pc_col].dropna().astype(str).tolist()
            else:
                # Fall back: collect all non-key text columns
                for c in grp.columns:
                    if c in group_keys or c == unit_title_col:
                        continue
                    vals = grp[c].dropna().astype(str).tolist()
                    pcs.extend(v for v in vals if v.strip())

            if not pcs:
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
