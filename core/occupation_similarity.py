"""
core/occupation_similarity.py

Per-ANZSCO skill-centroid embeddings + dim reduction for the occupation
similarity map, plus pairwise skill-set gap analysis.

The chain:
  ANZSCO → ISCO (via anzsco_crosswalk)
         → ESCO occupations (via esco_local.occupations_for_isco)
         → ESCO skills (via the inverted relations index)
         → MiniLM embeddings (already loaded as matcher.embeddings)

Each ANZSCO is then represented as a weighted centroid of its ESCO skill
embeddings (essential = 1.0, optional = 0.5), L2-normalised so cosine
distance is meaningful.
"""
from __future__ import annotations

import functools
from collections import defaultdict

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from core import anzsco_crosswalk, esco_local


# ── ESCO occupation → skill index ─────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _esco_occ_to_skills() -> dict[str, list[tuple[int, str]]]:
    """occupation_uri → [(skill_row_idx, relation), ...]

    skill_row_idx indexes into matcher.embeddings.
    Cached because building it walks ~200k relation rows.
    """
    matcher = esco_local.get_matcher()
    skill_idx_by_uri = {
        uri: i for i, uri in enumerate(matcher.skills_df["conceptUri"].tolist())
    }
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in matcher.relations_df.itertuples(index=False):
        idx = skill_idx_by_uri.get(getattr(r, "skillUri", None))
        if idx is None:
            continue
        out[getattr(r, "occupationUri")].append((idx, getattr(r, "relationType", "optional")))
    return dict(out)


# ── ANZSCO → skill set / embedding ────────────────────────────────────────────

def skills_for_anzsco(anzsco_code: str) -> dict:
    """Union of ESCO skills behind an ANZSCO occupation.

    Each skill counted once; if any ESCO occupation marks it essential,
    it's essential here too.
    """
    matcher = esco_local.get_matcher()
    skill_uris = matcher.skills_df["conceptUri"].tolist()
    occ_to_skills = _esco_occ_to_skills()

    isco_links = anzsco_crosswalk.anzsco_to_isco(anzsco_code)
    if not isco_links:
        return {"skill_indices": [], "skill_uris": [], "relations": []}

    seen: dict[int, str] = {}
    for link in isco_links:
        for occ in matcher.occupations_for_isco(link["isco_code"]):
            for idx, rel in occ_to_skills.get(occ["uri"], []):
                if idx not in seen or rel == "essential":
                    seen[idx] = rel

    idxs = list(seen.keys())
    return {
        "skill_indices": idxs,
        "skill_uris": [skill_uris[i] for i in idxs],
        "relations": [seen[i] for i in idxs],
    }


def anzsco_embedding(anzsco_code: str, *, weighted: bool = True) -> np.ndarray | None:
    """L2-normalised skill-centroid for an ANZSCO occupation.

    Returns None when no ESCO skills are reachable from the ANZSCO.
    """
    matcher = esco_local.get_matcher()
    info = skills_for_anzsco(anzsco_code)
    if not info["skill_indices"]:
        return None

    idxs = np.array(info["skill_indices"], dtype=np.int64)
    vecs = matcher.embeddings[idxs]

    if weighted:
        w = np.array([1.0 if r == "essential" else 0.5 for r in info["relations"]],
                     dtype=np.float32)
        cent = (vecs * w[:, None]).sum(axis=0) / w.sum()
    else:
        cent = vecs.mean(axis=0)

    n = float(np.linalg.norm(cent))
    return (cent / n).astype(np.float32) if n > 0 else cent.astype(np.float32)


def build_embedding_matrix(anzsco_codes: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack centroids for a list of ANZSCO codes.

    Drops codes that don't resolve to any skills. Returns (matrix, kept_codes, kept_titles).
    """
    kept_vecs, kept_codes, kept_titles = [], [], []
    for code in anzsco_codes:
        v = anzsco_embedding(code)
        if v is not None:
            kept_vecs.append(v)
            kept_codes.append(code)
            kept_titles.append(anzsco_crosswalk.title_for_anzsco(code))
    if not kept_vecs:
        return np.zeros((0, 384), dtype=np.float32), [], []
    return np.vstack(kept_vecs).astype(np.float32), kept_codes, kept_titles


# ── Scope: which ANZSCOs the corpus actually touches ──────────────────────────

def corpus_anzsco_codes(engine: Engine) -> list[str]:
    """ANZSCO codes reachable from any ESCO-matched statement in rsd_skill_records."""
    matcher = esco_local.get_matcher()
    sql = text("""
        SELECT DISTINCT esco_occupation_uris
          FROM rsd_skill_records
         WHERE esco_skill_uri IS NOT NULL
           AND esco_occupation_uris IS NOT NULL
           AND esco_occupation_uris <> ''
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).all()

    isco_seen: set[str] = set()
    for (uris,) in rows:
        for uri in (uris or "").split("|"):
            uri = uri.strip()
            if not uri:
                continue
            meta = matcher.occupation_meta(uri)
            if meta and meta.get("isco_group"):
                isco_seen.add(meta["isco_group"])

    anzsco_seen: set[str] = set()
    for isco in isco_seen:
        for anz in anzsco_crosswalk.isco_to_anzsco(isco):
            anzsco_seen.add(anz["anzsco_code"])
    return sorted(anzsco_seen)


# ── Dim reduction ─────────────────────────────────────────────────────────────

def reduce_dims(matrix: np.ndarray, *, n_components: int = 2,
                random_state: int = 42) -> np.ndarray:
    """UMAP projection (cosine metric, since centroids are unit-normed)."""
    if matrix.shape[0] < n_components + 1:
        return np.zeros((matrix.shape[0], n_components), dtype=np.float32)
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        n_neighbors=min(15, matrix.shape[0] - 1),
        min_dist=0.1,
    )
    return reducer.fit_transform(matrix).astype(np.float32)


# ── Gap analysis ──────────────────────────────────────────────────────────────

def gap_analysis(anzsco_a: str, anzsco_b: str) -> dict:
    """A∖B, B∖A, A∩B over the ESCO skill sets of two ANZSCO occupations.

    Each skill carries its (essential / optional) relation under each side.
    """
    matcher = esco_local.get_matcher()
    a = skills_for_anzsco(anzsco_a)
    b = skills_for_anzsco(anzsco_b)

    a_map = dict(zip(a["skill_uris"], a["relations"]))
    b_map = dict(zip(b["skill_uris"], b["relations"]))
    title_by_uri = dict(zip(matcher.skills_df["conceptUri"], matcher.skills_df["preferredLabel"]))

    a_set, b_set = set(a_map), set(b_map)

    def rows(uris: set[str]) -> list[dict]:
        return sorted(
            [
                {
                    "skill_uri": u,
                    "skill_title": title_by_uri.get(u, ""),
                    "relation_a": a_map.get(u),
                    "relation_b": b_map.get(u),
                }
                for u in uris
            ],
            key=lambda r: (r["relation_a"] != "essential" and r["relation_b"] != "essential",
                           r["skill_title"]),
        )

    union = a_set | b_set
    return {
        "a_code": anzsco_a,
        "a_title": anzsco_crosswalk.title_for_anzsco(anzsco_a),
        "b_code": anzsco_b,
        "b_title": anzsco_crosswalk.title_for_anzsco(anzsco_b),
        "a_only": rows(a_set - b_set),
        "b_only": rows(b_set - a_set),
        "shared": rows(a_set & b_set),
        "a_skill_count": len(a_set),
        "b_skill_count": len(b_set),
        "jaccard": (len(a_set & b_set) / len(union)) if union else 0.0,
    }
