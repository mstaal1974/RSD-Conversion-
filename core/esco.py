"""
core/esco.py

ESCO REST API client for matching skill statements to ESCO skills
and associating them with ESCO occupations.

ESCO API base: https://ec.europa.eu/esco/api
No API key required — public EU API with fair-use rate limiting.

Workflow per skill statement:
  1. /search?text=...&type=skill  → top N ESCO skill matches + scores
  2. /resource/related?uri=...&relation=isEssentialForOccupation → occupations
  3. /resource/related?uri=...&relation=isOptionalForOccupation  → optional occs
  4. Aggregate, deduplicate, return structured results
"""
from __future__ import annotations

import time
import json
import urllib.parse
import urllib.request
from typing import Optional
import pandas as pd

ESCO_BASE = "https://ec.europa.eu/esco/api"
DEFAULT_LANGUAGE = "en"
REQUEST_DELAY = 0.3   # seconds between calls — respect fair-use policy
TIMEOUT = 15          # seconds per request


def _get(url: str, retries: int = 3) -> dict:
    """GET JSON from ESCO API with retry on transient errors."""
    headers = {
        "Accept": "application/json",
        "Accept-Language": DEFAULT_LANGUAGE,
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited
                time.sleep(2 ** attempt)
            elif e.code >= 500:
                time.sleep(1)
            else:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return {}


def search_esco_skills(
    text: str,
    limit: int = 5,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """
    Search ESCO for skills matching the given text.

    Returns list of:
      {uri, title, score, description}
    """
    params = urllib.parse.urlencode({
        "text":     text[:500],   # truncate very long statements
        "type":     "skill",
        "language": language,
        "limit":    limit,
    })
    url = f"{ESCO_BASE}/search?{params}"
    time.sleep(REQUEST_DELAY)

    try:
        data = _get(url)
    except Exception:
        return []

    results = data.get("_embedded", {}).get("results", [])
    out = []
    for r in results:
        out.append({
            "uri":         r.get("uri", ""),
            "title":       r.get("title", ""),
            "score":       round(float(r.get("score", 0)), 4),
            "description": r.get("description", {}).get("en", {}).get("literal", "") if isinstance(r.get("description"), dict) else "",
        })
    return out


def get_occupations_for_skill(
    skill_uri: str,
    relation: str = "isEssentialForOccupation",
    limit: int = 10,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """
    Get ESCO occupations associated with a skill URI.

    relation: 'isEssentialForOccupation' or 'isOptionalForOccupation'

    Returns list of:
      {uri, title, isco_group}
    """
    params = urllib.parse.urlencode({
        "uri":      skill_uri,
        "relation": relation,
        "language": language,
        "limit":    limit,
    })
    url = f"{ESCO_BASE}/resource/related?{params}"
    time.sleep(REQUEST_DELAY)

    try:
        data = _get(url)
    except Exception:
        return []

    embedded = data.get("_embedded", {})
    # The key varies — try known keys
    occupations = (
        embedded.get("isEssentialForOccupation") or
        embedded.get("isOptionalForOccupation") or
        embedded.get("occupation") or
        []
    )

    out = []
    for occ in occupations:
        out.append({
            "uri":        occ.get("uri", ""),
            "title":      occ.get("title", ""),
            "isco_group": occ.get("iscoGroup", {}).get("uri", "") if isinstance(occ.get("iscoGroup"), dict) else "",
        })
    return out


def match_statement_to_esco(
    skill_statement: str,
    top_n_skills: int = 3,
    top_n_occupations: int = 5,
    min_score: float = 0.0,
) -> dict:
    """
    Full pipeline for one skill statement:
      1. Search ESCO for matching skills
      2. For top match, fetch essential + optional occupations
      3. Return structured result

    Returns:
    {
        esco_skills: [{uri, title, score, description}, ...],
        top_skill_uri: str,
        top_skill_title: str,
        top_skill_score: float,
        essential_occupations: [{uri, title, isco_group}, ...],
        optional_occupations: [{uri, title, isco_group}, ...],
        all_occupation_titles: str,   # semicolon-joined for DB storage
        all_occupation_uris: str,     # semicolon-joined for DB storage
    }
    """
    skills = search_esco_skills(skill_statement, limit=top_n_skills)

    if not skills:
        return _empty_result()

    # Filter by minimum score
    skills = [s for s in skills if s["score"] >= min_score]
    if not skills:
        return _empty_result()

    top = skills[0]

    # Fetch occupations for top match
    essential = get_occupations_for_skill(
        top["uri"],
        relation="isEssentialForOccupation",
        limit=top_n_occupations,
    )
    optional = get_occupations_for_skill(
        top["uri"],
        relation="isOptionalForOccupation",
        limit=top_n_occupations,
    )

    all_occs = essential + optional
    # Deduplicate by URI
    seen = set()
    deduped = []
    for occ in all_occs:
        if occ["uri"] not in seen:
            seen.add(occ["uri"])
            deduped.append(occ)

    return {
        "esco_skills":             skills,
        "top_skill_uri":           top["uri"],
        "top_skill_title":         top["title"],
        "top_skill_score":         top["score"],
        "top_skill_description":   top["description"],
        "essential_occupations":   essential,
        "optional_occupations":    optional,
        "all_occupation_titles":   "; ".join(o["title"] for o in deduped),
        "all_occupation_uris":     "; ".join(o["uri"] for o in deduped),
    }


def _empty_result() -> dict:
    return {
        "esco_skills":             [],
        "top_skill_uri":           "",
        "top_skill_title":         "",
        "top_skill_score":         0.0,
        "top_skill_description":   "",
        "essential_occupations":   [],
        "optional_occupations":    [],
        "all_occupation_titles":   "",
        "all_occupation_uris":     "",
    }


def batch_match(
    df: pd.DataFrame,
    top_n_skills: int = 3,
    top_n_occupations: int = 5,
    min_score: float = 0.0,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run ESCO matching for all rows in df.

    Adds columns:
      esco_skill_uri, esco_skill_title, esco_skill_score,
      esco_occupation_titles, esco_occupation_uris

    Returns annotated DataFrame.
    """
    results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        stmt = str(row.get("skill_statement", "") or "")
        if not stmt.strip():
            results.append(_empty_result())
        else:
            match = match_statement_to_esco(
                stmt,
                top_n_skills=top_n_skills,
                top_n_occupations=top_n_occupations,
                min_score=min_score,
            )
            results.append(match)

        if progress_callback and i % 5 == 0:
            progress_callback(i / total, f"Matched {i}/{total} statements…")

    df_out = df.copy().reset_index(drop=True)
    df_out["esco_skill_uri"]         = [r["top_skill_uri"]         for r in results]
    df_out["esco_skill_title"]       = [r["top_skill_title"]       for r in results]
    df_out["esco_skill_score"]       = [r["top_skill_score"]       for r in results]
    df_out["esco_occupation_titles"] = [r["all_occupation_titles"] for r in results]
    df_out["esco_occupation_uris"]   = [r["all_occupation_uris"]   for r in results]

    if progress_callback:
        progress_callback(1.0, "ESCO matching complete ✅")

    return df_out
