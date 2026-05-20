"""
core/concept_nodes.py

Data-driven concept backbone for occupational profiles.

Where the ESCO backbone uses an external taxonomy to group verbose
Australian skill statements, the concept-node backbone clusters the
statements directly with HDBSCAN on MiniLM embeddings and uses the
clusters themselves as the grouping layer.

Each concept node carries:
  • id              — stable per-run cluster id (-1 = noise/unmapped)
  • canonical_text  — the member closest to the cluster centroid
  • label           — short human-readable name (LLM-generated, optional)
  • size            — number of member statements
  • member_ids      — rsd_skill_records.id values in the cluster
  • member_indices  — positional indices into the input dataframe
  • modal_esco_uri  — most common esco_skill_uri among members
  • modal_esco_title
  • modal_isco_group — most common ISCO group (4-digit)
  • mean_esco_score — mean of esco_skill_score for members

The "Unmapped" bucket (id = -1) is HDBSCAN noise — statements whose
embeddings didn't cluster confidently. It's exposed honestly rather
than force-grouped under a weak nearest concept.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import esco_local
from core.semantic import cluster_statements


@dataclass
class ConceptNode:
    id: int
    canonical_text: str
    size: int
    member_ids: list           # rsd_skill_records.id
    member_indices: list[int]  # positional indices
    label: str = ""
    modal_esco_uri: str = ""
    modal_esco_title: str = ""
    modal_isco_group: str = ""
    mean_esco_score: float = 0.0
    statements: list[dict] = field(default_factory=list)


def _modal(values: list[str]) -> str:
    cleaned = [v for v in values if v]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def build_concept_nodes(
    df: pd.DataFrame,
    *,
    min_cluster_size: int = 4,
    embedding_model=None,
) -> tuple[list[ConceptNode], np.ndarray]:
    """
    Cluster statements into concept nodes.

    df must contain at least:
      id, skill_statement, esco_skill_uri, esco_skill_title, esco_skill_score

    Returns (nodes, labels) where:
      • nodes is a list including the noise node (id=-1) if any
      • labels is the per-row cluster id, aligned with df.reset_index()

    The embedding model defaults to the bundled MiniLM via esco_local;
    pass any sentence-transformers-compatible model to override.
    """
    if df is None or df.empty:
        return [], np.array([], dtype=int)

    df = df.reset_index(drop=True)
    texts = df["skill_statement"].fillna("").tolist()

    model = embedding_model or esco_local.get_matcher().model
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype(np.float32)

    # HDBSCAN via the project's shared wrapper — falls back to DBSCAN
    # if hdbscan isn't installed.
    labels = cluster_statements(
        embeddings,
        min_cluster_size=max(2, min_cluster_size),
        algorithm="hdbscan",
    )

    nodes: list[ConceptNode] = []
    for cid in sorted(set(int(x) for x in labels)):
        member_mask = labels == cid
        indices = np.where(member_mask)[0]
        member_embs = embeddings[indices]

        if cid != -1:
            centroid = member_embs.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) or 1.0)
            sims = member_embs @ centroid
            canonical_local = int(np.argmax(sims))
        else:
            canonical_local = 0  # noise — first member by document order

        canonical_idx = int(indices[canonical_local])
        canonical_text = texts[canonical_idx]

        members_df = df.iloc[indices]
        scores = pd.to_numeric(members_df.get("esco_skill_score"), errors="coerce").fillna(0.0)

        node = ConceptNode(
            id=cid,
            canonical_text=canonical_text,
            size=int(len(indices)),
            member_ids=members_df["id"].tolist() if "id" in members_df.columns else indices.tolist(),
            member_indices=indices.tolist(),
            modal_esco_uri=_modal(members_df.get("esco_skill_uri", pd.Series(dtype=str)).fillna("").tolist()),
            modal_esco_title=_modal(members_df.get("esco_skill_title", pd.Series(dtype=str)).fillna("").tolist()),
            mean_esco_score=float(scores.mean()) if len(scores) else 0.0,
            statements=members_df.to_dict(orient="records"),
        )

        # Modal ISCO from the modal ESCO occupation's metadata, if available
        if node.modal_esco_uri:
            try:
                matcher = esco_local.get_matcher()
                # The statement's esco_skill_uri points to an ESCO *skill*.
                # The ISCO group comes from the most-frequent essential occupation
                # for that skill. Walk one hop.
                ess_uris = []
                for occ in (matcher._occ_by_skill.get(node.modal_esco_uri, [])):
                    if occ.get("relation") == "essential":
                        ess_uris.append(occ["uri"])
                if ess_uris:
                    isco_codes = []
                    for u in ess_uris:
                        meta = matcher.occupation_meta(u)
                        if meta and meta.get("isco_group"):
                            isco_codes.append(meta["isco_group"])
                    if isco_codes:
                        node.modal_isco_group = Counter(isco_codes).most_common(1)[0][0]
            except Exception:
                pass

        # Default label = truncated canonical when no LLM has been run yet
        node.label = _fallback_label(canonical_text, cid)
        nodes.append(node)

    return nodes, labels


def _fallback_label(canonical: str, cid: int) -> str:
    """Truncated canonical statement as a stand-in for the LLM label."""
    if cid == -1:
        return "Unmapped statements"
    words = canonical.split()
    return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
