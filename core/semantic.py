"""
core/semantic.py

Semantic analysis of skill statements using OpenAI embeddings.

Steps:
1. Generate embeddings for all skill statements
2. Compute cosine similarity matrix
3. Cluster using DBSCAN (density-based — no need to specify cluster count)
4. Within each cluster, identify a canonical statement (closest to centroid)
5. Flag near-duplicates (similarity > threshold)
6. Return cluster assignments, similarity scores, and merge suggestions
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def get_embeddings(
    texts: list[str],
    provider,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> np.ndarray:
    """
    Generate embeddings using OpenAI's embedding API.
    Returns numpy array of shape (n_texts, embedding_dim).
    """
    from openai import OpenAI

    # Access the underlying OpenAI client
    if hasattr(provider, '_client'):
        client = provider._client
    else:
        raise ValueError("Provider must be OpenAIProvider with _client attribute")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean texts — empty strings cause errors
        batch = [t if t and t.strip() else "empty" for t in batch]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalise rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalised = embeddings / norms
    return normalised @ normalised.T


def cluster_statements(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 2,
) -> np.ndarray:
    """
    Cluster skill statements using DBSCAN on cosine distance.

    similarity_threshold: statements with similarity >= this are considered
                          in the same neighbourhood (0.85 = very similar)
    Returns array of cluster labels (-1 = noise/singleton)
    """
    from sklearn.cluster import DBSCAN

    # Convert similarity to distance
    sim_matrix = cosine_similarity_matrix(embeddings)
    distance_matrix = 1.0 - sim_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)

    eps = 1.0 - similarity_threshold  # distance threshold
    db = DBSCAN(
        eps=eps,
        min_samples=min_cluster_size,
        metric="precomputed",
    )
    labels = db.fit_predict(distance_matrix)
    return labels


def find_canonical(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
) -> dict[int, dict]:
    """
    For each cluster, find the canonical statement (closest to centroid).

    Returns dict: {cluster_id: {canonical_idx, canonical_text, size, members}}
    """
    clusters = {}
    unique_labels = set(labels) - {-1}

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]
        cluster_embeddings = embeddings[indices]

        # Centroid
        centroid = cluster_embeddings.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) or 1)

        # Find member closest to centroid
        sims_to_centroid = []
        for emb in cluster_embeddings:
            emb_norm = emb / (np.linalg.norm(emb) or 1)
            sims_to_centroid.append(float(centroid_norm @ emb_norm))

        best_local_idx = int(np.argmax(sims_to_centroid))
        canonical_idx = int(indices[best_local_idx])

        clusters[int(label)] = {
            "canonical_idx":  canonical_idx,
            "canonical_text": texts[canonical_idx],
            "size":           int(mask.sum()),
            "member_indices": indices.tolist(),
            "member_texts":   [texts[i] for i in indices],
        }

    return clusters


def find_near_duplicates(
    sim_matrix: np.ndarray,
    texts: list[str],
    threshold: float = 0.92,
) -> list[dict]:
    """
    Find pairs of statements with similarity above threshold.
    Returns list of {idx_a, idx_b, similarity, text_a, text_b}
    """
    pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                pairs.append({
                    "idx_a":      i,
                    "idx_b":      j,
                    "similarity": round(sim, 4),
                    "text_a":     texts[i],
                    "text_b":     texts[j],
                })
    # Sort by similarity descending
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs


def analyse_statements(
    df: pd.DataFrame,
    provider,
    embedding_model: str = "text-embedding-3-small",
    cluster_threshold: float = 0.85,
    duplicate_threshold: float = 0.92,
    min_cluster_size: int = 2,
    progress_callback=None,
) -> dict:
    """
    Full analysis pipeline.

    Returns:
        embeddings       — numpy array
        sim_matrix       — cosine similarity matrix
        labels           — DBSCAN cluster labels
        clusters         — canonical statement per cluster
        near_duplicates  — high-similarity pairs
        df_annotated     — original df with cluster_id and is_canonical columns
    """
    texts = df["skill_statement"].fillna("").tolist()

    if progress_callback:
        progress_callback(0.1, "Generating embeddings…")

    embeddings = get_embeddings(texts, provider, model=embedding_model)

    if progress_callback:
        progress_callback(0.4, "Computing similarity matrix…")

    sim_matrix = cosine_similarity_matrix(embeddings)

    if progress_callback:
        progress_callback(0.6, "Clustering statements…")

    labels = cluster_statements(embeddings, cluster_threshold, min_cluster_size)

    if progress_callback:
        progress_callback(0.75, "Finding canonical statements…")

    clusters = find_canonical(embeddings, labels, texts)

    if progress_callback:
        progress_callback(0.85, "Finding near-duplicates…")

    near_duplicates = find_near_duplicates(sim_matrix, texts, duplicate_threshold)

    # Annotate dataframe
    df_out = df.copy().reset_index(drop=True)
    df_out["cluster_id"] = labels
    canonical_indices = {v["canonical_idx"] for v in clusters.values()}
    df_out["is_canonical"] = df_out.index.isin(canonical_indices)
    df_out["is_singleton"] = labels == -1

    if progress_callback:
        progress_callback(1.0, "Done")

    return {
        "embeddings":      embeddings,
        "sim_matrix":      sim_matrix,
        "labels":          labels,
        "clusters":        clusters,
        "near_duplicates": near_duplicates,
        "df_annotated":    df_out,
    }
