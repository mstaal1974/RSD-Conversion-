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


# ── Empirical basis: per-ANZSCO centroid of corpus statement embeddings ──────

import pandas as pd  # local import — top of file already has numpy


def build_corpus_index(engine: Engine) -> dict:
    """
    Pull every ESCO-matched skill statement in the corpus, encode the
    statement text with the same MiniLM model the matcher uses, then
    build a per-ANZSCO weighting that mirrors the contribution score
    used elsewhere (esco_skill_score × ISCO↔ANZSCO weight, best-of when
    a statement has multiple ESCO occupations behind it).

    The encode pass is the expensive step (~tens of seconds for a
    30k-statement corpus). Cache the return value at the page level via
    @st.cache_resource.

    Returns:
      {
        "embeddings":      np.ndarray (n_stmts, 384) L2-normed,
        "df":              DataFrame of the statements
                           (id, unit_code, skill_statement, esco_skill_score),
        "anzsco_to_stmts": dict[anzsco_code, list[(row_idx, weight)]],
      }
    """
    matcher = esco_local.get_matcher()
    sql = text("""
        SELECT id, unit_code, skill_statement, esco_skill_score, esco_occupation_uris
          FROM rsd_skill_records
         WHERE esco_skill_uri IS NOT NULL
           AND skill_statement IS NOT NULL AND skill_statement <> ''
           AND esco_occupation_uris IS NOT NULL AND esco_occupation_uris <> ''
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    df = df.reset_index(drop=True)

    empty = {
        "embeddings": np.zeros((0, 384), dtype=np.float32),
        "df": df.drop(columns=["esco_occupation_uris"], errors="ignore"),
        "anzsco_to_stmts": {},
    }
    if df.empty:
        return empty

    texts = df["skill_statement"].fillna("").tolist()
    embeddings = matcher.model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Walk statements → ESCO occupation URIs → ISCO → ANZSCO; per (anzsco,
    # statement) keep the best (esco_score × crosswalk_weight).
    anzsco_to_stmts: dict[str, dict[int, float]] = defaultdict(dict)
    occ_uris_col = df["esco_occupation_uris"].fillna("").tolist()
    scores_col = df["esco_skill_score"].fillna(0.0).astype(float).tolist()

    for i, (uris, esco_score) in enumerate(zip(occ_uris_col, scores_col)):
        if not uris or esco_score <= 0.0:
            continue
        for uri in uris.split("|"):
            uri = uri.strip()
            if not uri:
                continue
            meta = matcher.occupation_meta(uri)
            if not meta or not meta.get("isco_group"):
                continue
            for anz in anzsco_crosswalk.isco_to_anzsco(meta["isco_group"]):
                w = esco_score * anz["weight"]
                code = anz["anzsco_code"]
                if w > anzsco_to_stmts[code].get(i, 0.0):
                    anzsco_to_stmts[code][i] = w

    return {
        "embeddings": embeddings,
        "df": df.drop(columns=["esco_occupation_uris"], errors="ignore"),
        "anzsco_to_stmts": {c: list(s.items()) for c, s in anzsco_to_stmts.items()},
    }


def build_empirical_matrix(
    index: dict, *, min_statements: int = 3,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """
    Per-ANZSCO weighted centroid of corpus skill-statement embeddings.

    Only includes ANZSCO codes backed by at least `min_statements`
    statements in the corpus.

    Returns (matrix, codes, titles, n_statements_per_code) — last list
    aligned with codes, for hover labels.
    """
    embeddings = index["embeddings"]
    anzsco_to_stmts = index["anzsco_to_stmts"]
    if not anzsco_to_stmts or embeddings.shape[0] == 0:
        return np.zeros((0, 384), dtype=np.float32), [], [], []

    kept_vecs, kept_codes, kept_titles, kept_n = [], [], [], []
    for code in sorted(anzsco_to_stmts):
        stmts = anzsco_to_stmts[code]
        if len(stmts) < min_statements:
            continue
        idxs = np.fromiter((s[0] for s in stmts), dtype=np.int64, count=len(stmts))
        weights = np.fromiter((s[1] for s in stmts), dtype=np.float32, count=len(stmts))
        vecs = embeddings[idxs]
        if weights.sum() > 0:
            cent = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
        else:
            cent = vecs.mean(axis=0)
        n = float(np.linalg.norm(cent))
        cent = (cent / n).astype(np.float32) if n > 0 else cent.astype(np.float32)
        kept_vecs.append(cent)
        kept_codes.append(code)
        kept_titles.append(anzsco_crosswalk.title_for_anzsco(code))
        kept_n.append(len(stmts))

    if not kept_vecs:
        return np.zeros((0, 384), dtype=np.float32), [], [], []
    return np.vstack(kept_vecs), kept_codes, kept_titles, kept_n


# ── Dim reduction ─────────────────────────────────────────────────────────────

def reduce_dims(matrix: np.ndarray, *, n_components: int = 2,
                random_state: int = 42) -> np.ndarray:
    """2D/3D projection. Tries UMAP (cosine metric); falls back to
    scikit-learn TSNE if UMAP's numba backend can't initialise — common
    on sandboxed deployments where numba can't locate its source files
    for cache addressing.

    The input is expected to be L2-normalised so euclidean rank in TSNE
    matches cosine rank (||a - b||² = 2 - 2·cos(a,b) for unit vectors).
    """
    if matrix.shape[0] < n_components + 1:
        return np.zeros((matrix.shape[0], n_components), dtype=np.float32)

    # Give numba a writable cache dir before UMAP triggers its JIT.
    import os, tempfile
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "numba_cache"),
    )

    try:
        import umap
        if not hasattr(umap, "UMAP"):
            # Either umap-learn failed to re-export (partial init from a
            # previous numba error left in sys.modules), or a different
            # PyPI package named "umap" is shadowing umap-learn.
            raise ImportError("umap module has no UMAP class")
        reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            random_state=random_state,
            n_neighbors=min(15, matrix.shape[0] - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(matrix).astype(np.float32)
    except (RuntimeError, ImportError, AttributeError, OSError):
        from sklearn.manifold import TSNE
        n = matrix.shape[0]
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
            perplexity=min(30.0, max(5.0, n / 3.0)),
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
