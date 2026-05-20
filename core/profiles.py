"""
core/profiles.py

Build per-UOC and per-qualification occupational profiles by chaining
the local ESCO matcher's outputs through the ISCO ↔ ANZSCO crosswalk.

For every skill statement in rsd_skill_records we already have:
  esco_skill_uri / esco_skill_title / esco_skill_score
  esco_occupation_uris (pipe-separated, up to ~8 per statement)

The pipeline here:
  1. Pull all matched statements for a UOC (or a qualification's UOCs)
  2. For each ESCO occupation URI, look up its iscoGroup via esco_local
  3. Map ISCO → ANZSCO via anzsco_crosswalk
  4. Aggregate scores: score for an ANZSCO occupation = sum over
     statements of (esco_skill_score × isco→anzsco quality weight),
     deduped by (statement, anzsco_code) so a statement only votes
     once per ANZSCO target.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from core import anzsco_crosswalk, esco_local


def _statements_for_uoc(engine: Engine, uoc_code: str) -> pd.DataFrame:
    sql = text("""
        SELECT id, unit_code, element_title, skill_statement,
               esco_skill_uri, esco_skill_title, esco_skill_score,
               esco_occupation_uris, esco_occupation_titles
          FROM rsd_skill_records
         WHERE unit_code = :uc
           AND esco_skill_uri IS NOT NULL AND esco_skill_uri <> ''
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"uc": uoc_code}).mappings().all()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _statements_for_qual(engine: Engine, qual_code: str) -> pd.DataFrame:
    sql = text("""
        SELECT r.id, r.unit_code, r.element_title, r.skill_statement,
               r.esco_skill_uri, r.esco_skill_title, r.esco_skill_score,
               r.esco_occupation_uris, r.esco_occupation_titles,
               qu.is_core
          FROM rsd_skill_records r
          JOIN qual_uoc_membership qu ON qu.uoc_code = r.unit_code
         WHERE qu.qual_code = :qc
           AND r.esco_skill_uri IS NOT NULL AND r.esco_skill_uri <> ''
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"qc": qual_code}).mappings().all()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _aggregate(df: pd.DataFrame, matcher) -> dict:
    """
    Walk every statement and tally ANZSCO occupations.

    Returns:
      anzsco_rows : list[dict] — ranked ANZSCO occupations with score + evidence
      coverage    : {n_statements, n_with_anzsco, esco_only}
    """
    if df.empty:
        return {"anzsco_rows": [], "coverage": {"n_statements": 0, "n_with_anzsco": 0}}

    # (statement_id, anzsco_code) → accumulated score + evidence
    evidence: dict[tuple[int, str], dict] = {}
    # anzsco_code → aggregate
    agg: dict[str, dict] = defaultdict(lambda: {
        "anzsco_code": "",
        "anzsco_title": "",
        "score": 0.0,
        "n_statements": 0,
        "supporting_esco_occupations": set(),
        "supporting_skill_statements": set(),
    })

    n_with_anzsco = 0
    for row in df.itertuples(index=False):
        stmt_id = row.id
        occ_uris = [u.strip() for u in (row.esco_occupation_uris or "").split("|") if u.strip()]
        if not occ_uris:
            continue

        statement_hit_anzsco = False
        for occ_uri in occ_uris:
            meta = matcher.occupation_meta(occ_uri)
            if not meta or not meta.get("isco_group"):
                continue
            isco = meta["isco_group"]
            for anz in anzsco_crosswalk.isco_to_anzsco(isco):
                code = anz["anzsco_code"]
                key = (stmt_id, code)
                vote = float(row.esco_skill_score) * anz["weight"]
                if key in evidence:
                    # Same statement already voted for this ANZSCO via another
                    # ESCO occupation — keep the highest-weighted vote only.
                    if vote <= evidence[key]["vote"]:
                        continue
                    agg[code]["score"] -= evidence[key]["vote"]
                evidence[key] = {"vote": vote, "via_esco_uri": occ_uri,
                                 "via_esco_title": meta["title"]}
                bucket = agg[code]
                bucket["anzsco_code"] = code
                bucket["anzsco_title"] = anz["anzsco_title"]
                bucket["score"] += vote
                bucket["supporting_esco_occupations"].add(meta["title"])
                bucket["supporting_skill_statements"].add(stmt_id)
                statement_hit_anzsco = True

        if statement_hit_anzsco:
            n_with_anzsco += 1

    # Materialise n_statements + finalise sets → lists
    out_rows = []
    for code, bucket in agg.items():
        bucket["n_statements"] = len(bucket["supporting_skill_statements"])
        bucket["supporting_esco_occupations"] = sorted(bucket["supporting_esco_occupations"])
        bucket["supporting_skill_statements"] = sorted(bucket["supporting_skill_statements"])
        out_rows.append(bucket)
    out_rows.sort(key=lambda r: r["score"], reverse=True)

    return {
        "anzsco_rows": out_rows,
        "coverage": {
            "n_statements": len(df),
            "n_with_anzsco": n_with_anzsco,
        },
    }


def uoc_profile(engine: Engine, uoc_code: str) -> dict:
    matcher = esco_local.get_matcher()
    df = _statements_for_uoc(engine, uoc_code)
    return {"uoc_code": uoc_code, "statements": df, **_aggregate(df, matcher)}


def qualification_profile(engine: Engine, qual_code: str) -> dict:
    matcher = esco_local.get_matcher()
    df = _statements_for_qual(engine, qual_code)
    return {"qual_code": qual_code, "statements": df, **_aggregate(df, matcher)}
