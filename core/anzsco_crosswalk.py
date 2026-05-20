"""
core/anzsco_crosswalk.py

ISCO-08 ↔ ANZSCO correspondence loader. The crosswalk file isn't shipped
with the repo (licensed by ABS) — drop the user-prepared CSV at
data/anzsco/isco_to_anzsco.csv (or set $ANZSCO_CROSSWALK_PATH).

Expected CSV schema (header row required):
    isco_code,anzsco_code,anzsco_title,match_quality

  • isco_code      — ISCO-08 unit group, 4-digit string (e.g. "7212")
  • anzsco_code    — ANZSCO 6-digit (or 4-digit minor group) string
  • anzsco_title   — human-readable ANZSCO title
  • match_quality  — one of: "exact", "partial", "broader", "narrower"

See data/anzsco/README.md for how to prepare the file from the ABS release.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd


def _crosswalk_path() -> Path:
    env = os.getenv("ANZSCO_CROSSWALK_PATH", "").strip()
    if env:
        return Path(env).resolve()
    return Path("data/anzsco/isco_to_anzsco.csv").resolve()


def is_available() -> bool:
    return _crosswalk_path().exists()


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    path = _crosswalk_path()
    if not path.exists():
        raise FileNotFoundError(
            f"ANZSCO crosswalk not found at {path}. "
            "See data/anzsco/README.md."
        )
    df = pd.read_csv(
        path,
        dtype={"isco_code": str, "anzsco_code": str},
        comment="#",
        skip_blank_lines=True,
    )
    df["isco_code"] = df["isco_code"].str.strip()
    df["anzsco_code"] = df["anzsco_code"].str.strip()
    df["match_quality"] = df.get("match_quality", "exact").fillna("exact").str.lower()
    return df


# Quality weights — lets callers fold mapping uncertainty into their score.
QUALITY_WEIGHT = {"exact": 1.00, "partial": 0.70, "broader": 0.60, "narrower": 0.50}


def isco_to_anzsco(isco_code: str | int) -> list[dict]:
    """All ANZSCO occupations that correspond to a given ISCO-08 code."""
    code = str(isco_code).strip()
    df = _load()
    matches = df[df["isco_code"] == code]
    return [
        {
            "anzsco_code": row.anzsco_code,
            "anzsco_title": row.anzsco_title,
            "quality": row.match_quality,
            "weight": QUALITY_WEIGHT.get(row.match_quality, 0.5),
        }
        for row in matches.itertuples(index=False)
    ]


def anzsco_to_isco(anzsco_code: str) -> list[dict]:
    """All ISCO-08 codes that map to a given ANZSCO occupation."""
    code = str(anzsco_code).strip()
    df = _load()
    matches = df[df["anzsco_code"] == code]
    return [
        {"isco_code": row.isco_code, "quality": row.match_quality}
        for row in matches.itertuples(index=False)
    ]


def title_for_anzsco(anzsco_code: str) -> str:
    """Canonical ANZSCO title for a code, or '' if unknown."""
    code = str(anzsco_code).strip()
    df = _load()
    m = df[df["anzsco_code"] == code]
    return "" if m.empty else str(m.iloc[0]["anzsco_title"])


def search_titles(query: str, limit: int = 30) -> list[dict]:
    """Case-insensitive substring search over ANZSCO titles.

    Returns distinct (code, title) pairs ranked exact > startswith > substring.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    df = _load()
    titles = df[["anzsco_code", "anzsco_title"]].drop_duplicates()
    lowered = titles["anzsco_title"].str.lower()
    mask = lowered.str.contains(q, regex=False, na=False)
    hits = titles[mask].copy()
    if hits.empty:
        return []
    hits["_t"] = hits["anzsco_title"].str.lower()
    hits["_rank"] = hits["_t"].apply(
        lambda t: 0 if t == q else (1 if t.startswith(q) else 2)
    )
    hits = hits.sort_values(["_rank", "anzsco_title"]).head(limit)
    return [
        {"anzsco_code": r.anzsco_code, "anzsco_title": r.anzsco_title}
        for r in hits.itertuples(index=False)
    ]
