"""
pages/16_🗺️_Occupation_Similarity_Map.py

Project ANZSCO occupations into 2D/3D by ESCO-skill centroid similarity,
then drill into pairwise gap analysis (A∖B, B∖A, A∩B).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from core import anzsco_crosswalk, esco_local
from core import occupation_similarity as sim

load_dotenv()

st.set_page_config(page_title="Occupation Similarity Map", layout="wide")
st.title("🗺️ Occupation Similarity Map")
st.caption(
    "Each ANZSCO occupation sits at the centroid of its ESCO essential/optional "
    "skill embeddings, projected to 2D/3D via UMAP. Nearby points share more skills. "
    "Switch to the Gap analysis tab to compare any pair."
)


# ── Connection ────────────────────────────────────────────────────────────────
def _secret(k, d=""):
    try:
        return st.secrets.get(k, os.getenv(k, d)) or d
    except Exception:
        return os.getenv(k, d) or d


DB_URL = _secret("DATABASE_URL")
if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()


@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)


# ── Preconditions ────────────────────────────────────────────────────────────
if not esco_local.is_available():
    st.error("ESCO CSVs not found.")
    st.stop()
if not anzsco_crosswalk.is_available():
    st.error("ANZSCO crosswalk not found.")
    st.stop()

with st.spinner("Loading ESCO matcher…"):
    esco_local.get_matcher()

engine = get_engine(DB_URL)

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Map")
    basis = st.radio(
        "Similarity basis",
        ["ESCO skills (taxonomic)", "Corpus statements (empirical)"],
        help=(
            "**ESCO**: occupation centroid = average of MiniLM embeddings of "
            "its ESCO essential/optional skills. Taxonomy-anchored, works for "
            "every ANZSCO regardless of corpus coverage.\n\n"
            "**Corpus**: occupation centroid = weighted average of MiniLM "
            "embeddings of your actual `rsd_skill_records.skill_statement` texts "
            "that map to it (weights = esco_score × crosswalk_weight). Reflects "
            "how your TPs actually teach the occupation — only covers ANZSCOs "
            "with corpus statements behind them."
        ),
    )
    is_empirical = basis.startswith("Corpus")

    if not is_empirical:
        scope = st.radio(
            "Occupations to plot",
            ["Corpus only", "All ANZSCO"],
            help=(
                "**Corpus only**: ANZSCO occupations reachable from your "
                "ESCO-matched statements (fast).\n\n"
                "**All ANZSCO**: every ANZSCO in the crosswalk (~1.1k — first "
                "build takes 30-60s, then cached)."
            ),
        )
        min_stmts = 1
    else:
        scope = "Corpus only"
        min_stmts = st.slider(
            "Min supporting statements per occupation",
            1, 20, 3,
            help="Hide ANZSCOs backed by fewer corpus statements than this — "
                 "noisy centroids from 1-2 statements aren't trustworthy.",
        )

    n_dim = st.radio("Dimensions", [2, 3], horizontal=True, index=0)
    point_size = st.slider("Point size", 4, 16, 8)


# ── Build embedding matrix ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_matrix_esco(scope_label: str, corpus_codes_tuple: tuple):
    if scope_label == "Corpus only":
        codes = list(corpus_codes_tuple)
    else:
        df = anzsco_crosswalk._load()
        codes = sorted(df["anzsco_code"].drop_duplicates().tolist())
    matrix, kept_codes, kept_titles = sim.build_embedding_matrix(codes)
    # Pad with empty n_statements for shape compatibility downstream
    return matrix, kept_codes, kept_titles, [0] * len(kept_codes)


@st.cache_resource(show_spinner=False)
def _get_corpus_index(_engine):
    """Encode every ESCO-matched skill statement once per session.
    @st.cache_resource doesn't hash args, so the engine identity is fine."""
    return sim.build_corpus_index(_engine)


@st.cache_data(show_spinner=False)
def _build_matrix_empirical(min_statements: int, _index_id: int):
    """min_statements is part of the cache key; _index_id pins us to a
    specific corpus index (id() of the cached dict)."""
    index = _get_corpus_index(engine)
    return sim.build_empirical_matrix(index, min_statements=min_statements)


@st.cache_data(show_spinner=False)
def _project(matrix_bytes: bytes, shape: tuple, n_dim: int):
    m = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(shape)
    return sim.reduce_dims(m, n_components=n_dim)


if is_empirical:
    with st.spinner("Encoding corpus skill statements (one-time per session)…"):
        index = _get_corpus_index(engine)
    with st.spinner(f"Building empirical centroids (min {min_stmts} statements)…"):
        matrix, codes, titles, n_stmts_per_code = _build_matrix_empirical(min_stmts, id(index))
else:
    corpus_codes_tuple: tuple = ()
    if scope == "Corpus only":
        with st.spinner("Finding ANZSCO occupations touched by your corpus…"):
            corpus_codes_tuple = tuple(sim.corpus_anzsco_codes(engine))
    with st.spinner(f"Building ESCO skill-centroid embeddings ({scope})…"):
        matrix, codes, titles, n_stmts_per_code = _build_matrix_esco(scope, corpus_codes_tuple)

if len(codes) < 3:
    if is_empirical:
        st.warning(
            f"Only {len(codes)} ANZSCO occupations have ≥{min_stmts} corpus "
            "statements behind them — not enough to build a map. Lower the "
            "Min-supporting-statements slider or match more skill statements."
        )
    else:
        st.warning(
            f"Only {len(codes)} occupations have skills behind them — not enough "
            "to build a map. Match more skill statements (ESCO Alignment page) "
            "or switch to 'All ANZSCO'."
        )
    st.stop()

with st.spinner(f"Projecting {len(codes)} occupations to {n_dim}D via UMAP…"):
    coords = _project(matrix.tobytes(), matrix.shape, n_dim)


# ── ANZSCO major-group colour ─────────────────────────────────────────────────
MAJOR_NAMES = {
    "1": "1 Managers",
    "2": "2 Professionals",
    "3": "3 Technicians & Trades",
    "4": "4 Community & Personal Service",
    "5": "5 Clerical & Admin",
    "6": "6 Sales Workers",
    "7": "7 Machinery Operators & Drivers",
    "8": "8 Labourers",
}

df_plot = pd.DataFrame({
    "code": codes,
    "title": titles,
    "major_group": [c[:1] for c in codes],
    "n_statements": n_stmts_per_code,
    **{f"x{i}": coords[:, i] for i in range(n_dim)},
})
df_plot["major_group_label"] = df_plot["major_group"].map(MAJOR_NAMES).fillna(df_plot["major_group"])

basis_label = "empirical (corpus statements)" if is_empirical else "ESCO skills (taxonomic)"

tab_map, tab_gap = st.tabs(["🗺️ Map", "🆚 Gap analysis"])


# ── Map tab ───────────────────────────────────────────────────────────────────
with tab_map:
    hover = {"code": True, "title": True, "major_group_label": False,
             "n_statements": is_empirical,
             **{f"x{i}": False for i in range(n_dim)}}
    chart_title = f"{len(codes)} ANZSCO occupations · basis: {basis_label}"
    size_col = "n_statements" if is_empirical else None
    if n_dim == 2:
        fig = px.scatter(
            df_plot, x="x0", y="x1",
            color="major_group_label",
            size=size_col,
            size_max=point_size * 2 if size_col else None,
            hover_data=hover,
            title=chart_title,
            height=720,
        )
    else:
        fig = px.scatter_3d(
            df_plot, x="x0", y="x1", z="x2",
            color="major_group_label",
            size=size_col,
            size_max=point_size * 2 if size_col else None,
            hover_data=hover,
            title=chart_title,
            height=720,
        )
    if size_col is None:
        fig.update_traces(marker=dict(size=point_size, opacity=0.85))
    else:
        fig.update_traces(marker=dict(opacity=0.85))
    fig.update_layout(legend_title_text="ANZSCO major group")
    st.plotly_chart(fig, use_container_width=True)

    if is_empirical:
        st.caption(
            "**Reading the empirical map**: each point is the weighted centroid of your "
            "actual `skill_statement` embeddings that map to that ANZSCO (weight = "
            "esco_score × crosswalk_weight). Point size = number of supporting statements. "
            "Closeness here means *your TPs teach these occupations with similar language* — "
            "not necessarily that ESCO defines them as similar."
        )
    else:
        st.caption(
            "**Reading the ESCO map**: UMAP layout — axes themselves are not meaningful, "
            "but neighbours are. Each point is the centroid of an ANZSCO's ESCO essential/"
            "optional skill embeddings. Trades cluster together, healthcare elsewhere, etc."
        )

    with st.expander("Find an occupation on the map"):
        q = st.text_input("Search", placeholder="e.g. plumber, nurse, accountant…",
                          key="map_search")
        if q:
            ql = q.lower()
            hits = df_plot[df_plot["title"].str.lower().str.contains(ql, na=False)]
            st.dataframe(
                hits[["code", "title", "major_group_label"]].rename(
                    columns={"major_group_label": "major group"}
                ),
                use_container_width=True, hide_index=True, height=min(400, 40 + len(hits) * 36),
            )


# ── Gap analysis tab ──────────────────────────────────────────────────────────
with tab_gap:
    st.subheader("Pairwise gap analysis")

    label_options = [f"{c} — {t}" for c, t in zip(codes, titles)]
    col_a, col_b = st.columns(2)
    with col_a:
        sel_a = st.selectbox("Occupation A", label_options, index=0, key="gap_a")
    with col_b:
        default_b = min(1, len(label_options) - 1)
        sel_b = st.selectbox("Occupation B", label_options, index=default_b, key="gap_b")

    code_a = codes[label_options.index(sel_a)]
    code_b = codes[label_options.index(sel_b)]

    if code_a == code_b:
        st.info("Pick two different occupations.")
        st.stop()

    gap = sim.gap_analysis(code_a, code_b)

    # Cosine similarity between the two centroids
    ia, ib = codes.index(code_a), codes.index(code_b)
    cosine = float(np.dot(matrix[ia], matrix[ib]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"A ({gap['a_code']}) skills", gap["a_skill_count"])
    m2.metric(f"B ({gap['b_code']}) skills", gap["b_skill_count"])
    m3.metric("Shared (∩)", len(gap["shared"]))
    m4.metric("Centroid cosine", f"{cosine:.3f}")

    st.caption(
        f"**{gap['a_title']}** ↔ **{gap['b_title']}**  ·  "
        f"Jaccard skill overlap: **{gap['jaccard']:.1%}**"
    )

    c1, c2, c3 = st.columns(3)

    def _render(col, header: str, items: list, rel_col: str):
        col.markdown(f"**{header}** · {len(items)} skills")
        if not items:
            col.caption("—")
            return
        df = pd.DataFrame(items)
        # Sort essential first
        df["_ess"] = (df[rel_col] == "essential").astype(int)
        df = df.sort_values(["_ess", "skill_title"], ascending=[False, True])
        df = df[["skill_title", rel_col]].rename(columns={rel_col: "relation"})
        col.dataframe(df, use_container_width=True, hide_index=True, height=500)

    _render(c1, f"Only in A ({gap['a_code']})", gap["a_only"], "relation_a")
    _render(c2, f"Only in B ({gap['b_code']})", gap["b_only"], "relation_b")
    _render(c3, "Shared", gap["shared"], "relation_a")

    st.caption(
        "**Interpretation**: skills in *Only in B* are the gap someone in occupation "
        "A would need to close to qualify for B (and vice versa). Essential skills "
        "rank higher in each list. Use the Skills by Occupation page to find which "
        "UOCs in your corpus teach any given gap skill."
    )

    # Downloadables
    st.divider()
    dl_cols = st.columns(3)
    for col, (label, rows) in zip(
        dl_cols,
        [("a_only", gap["a_only"]), ("b_only", gap["b_only"]), ("shared", gap["shared"])],
    ):
        if rows:
            col.download_button(
                f"Download {label}.csv",
                pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                file_name=f"gap_{code_a}_vs_{code_b}_{label}.csv",
                mime="text/csv",
            )
