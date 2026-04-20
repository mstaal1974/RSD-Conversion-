"""
pages/12_🔬_Skill_Intelligence.py

Three advanced analytical views:

  View A — UMAP Semantic Cluster Map
            Each dot = one skill statement, coloured by Training Package.
            "Colour bleeding" reveals cross-package skill overlap.
            Click a cluster to see its centroid and name a micro-credential.

  View B — Cross-Package Similarity Heatmap (RPL Engine)
            X axis = UOCs from one TP, Y axis = UOCs from another.
            Cell intensity = average cosine similarity between skill statements.
            High-intensity cells = RPL candidates.

  View C — Skill DNA Parallel Coordinates
            Each line = one UOC drawn across four dimensions:
              1. ESCO semantic match score
              2. Confidence of occupation linkage (AQF proxy)
              3. Action verb frequency (Bloom's taxonomy level)
              4. Occupation breadth (number of distinct occupations)
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
st.set_page_config(page_title="Skill Intelligence", layout="wide")

# ── Dark-theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; letter-spacing:-0.03em; }

.section-badge {
    display: inline-block;
    background: linear-gradient(135deg,#0d2137,#1a3a5c);
    border: 1px solid #1e4d7b;
    border-radius: 6px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: #7eb8f7;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.insight-box {
    background: linear-gradient(135deg,#0a1628,#0d2137);
    border-left: 3px solid #3f8fd4;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #90b8d8;
    line-height: 1.6;
}
.metric-chip {
    display:inline-block;
    background:#0d2137;
    border:1px solid #1e4d7b;
    border-radius:20px;
    padding:3px 12px;
    font-size:0.75rem;
    color:#7eb8f7;
    margin:3px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Skill Intelligence")
st.caption("Three analytical lenses — semantic clustering, RPL similarity, and skill DNA profiling.")

# ── DB ─────────────────────────────────────────────────────────────────────────
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

engine = get_engine(DB_URL)

# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading skill statements…")
def load_skills(tp_codes: tuple, max_rows: int) -> pd.DataFrame:
    tp_filter = ""
    params: dict = {"lim": max_rows}
    if tp_codes:
        tp_filter = "AND tp_code = ANY(:tps)"
        params["tps"] = list(tp_codes)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT unit_code, unit_title, tp_code, element_title,
                   skill_statement, keywords,
                   esco_skill_score, esco_skill_title,
                   esco_occupation_titles
            FROM rsd_skill_records
            WHERE skill_statement IS NOT NULL AND skill_statement != ''
            {tp_filter}
            ORDER BY tp_code, unit_code
            LIMIT :lim
        """), params).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=300, show_spinner="Loading occupation links…")
def load_occ_links() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT uoc_code, anzsco_code, anzsco_title,
                   confidence, mapping_source, occupation_titles
            FROM uoc_occupation_links
            WHERE is_primary = TRUE
              AND anzsco_code IS NOT NULL AND anzsco_code != ''
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=600, show_spinner="Available training packages…")
def load_tp_codes() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT tp_code FROM rsd_skill_records
            WHERE tp_code IS NOT NULL AND tp_code != ''
            ORDER BY tp_code
        """)).fetchall()
    return [r[0] for r in rows]


# ── Sidebar ────────────────────────────────────────────────────────────────────
all_tps = load_tp_codes()

with st.sidebar:
    st.header("Global filters")
    sel_tps = st.multiselect("Training packages", all_tps,
                             default=all_tps,
                             help="Applies to View A and C. View B queries DB directly.")
    max_stmts = st.slider("Max skill statements to load", 500, 10000, 3000, 500)
    st.divider()
    st.caption("Each view has its own additional controls below the chart.")

df_skills = load_skills(tuple(sel_tps), max_stmts)

if df_skills.empty:
    st.warning("No skill statements found. Check your TP filter or upload data first.")
    st.stop()

df_occ = load_occ_links()

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW A — UMAP SEMANTIC CLUSTER MAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-badge">View A</div>', unsafe_allow_html=True)
st.subheader("Semantic Cluster Map")
st.markdown("""
<div class="insight-box">
Each dot is one skill statement projected into 2D semantic space using TF-IDF + UMAP.
<b>Colour bleeding</b> — where dots from different training packages overlap — reveals
identical skill requirements described in different "dialects." These overlaps are
your <b>micro-credential opportunities</b> and <b>cross-credit candidates</b>.
</div>
""", unsafe_allow_html=True)

col_a1, col_a2, col_a3 = st.columns([1, 1, 1])
with col_a1:
    n_clusters   = st.slider("Clusters", 5, 30, 12, key="a_clusters")
with col_a2:
    umap_dims    = st.radio("Dimensions", [2, 3], horizontal=True, key="a_dims")
with col_a3:
    colour_by    = st.radio("Colour by", ["Training Package", "Cluster"], horizontal=True, key="a_colour")

run_umap = st.button("▶ Run UMAP analysis", type="primary", key="a_run")

if run_umap or "umap_df" in st.session_state:
    if run_umap:
        with st.spinner("Building TF-IDF embeddings and running UMAP…"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import normalize

                texts = df_skills["skill_statement"].fillna("").tolist()

                vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                     stop_words="english", min_df=2)
                X = vec.fit_transform(texts)
                X_norm = normalize(X)

                # UMAP
                import os as _os
                _os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
                _os.environ["NUMBA_DISABLE_JIT"] = "0"
                import numba
                numba.config.CACHE_DIR = "/tmp/numba_cache"
                import umap as umap_lib
                n_comp = umap_dims

                # Wrap in plain function — st.cache_data can't serialise
                # Numba JIT functions used internally by UMAP
                def _run_umap(X, n_components):
                    r = umap_lib.UMAP(n_components=n_components,
                                      n_neighbors=15, min_dist=0.1,
                                      random_state=42, metric="cosine")
                    arr = X.toarray() if hasattr(X, "toarray") else X
                    return r.fit_transform(arr)

                embedding = _run_umap(X_norm, n_comp)

                # KMeans clusters
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(embedding)

                umap_df = df_skills[["unit_code", "unit_title", "tp_code",
                                     "element_title", "skill_statement"]].copy()
                umap_df["cluster"] = labels.astype(str)
                umap_df["x"] = embedding[:, 0]
                umap_df["y"] = embedding[:, 1]
                if n_comp == 3:
                    umap_df["z"] = embedding[:, 2]
                else:
                    umap_df["z"] = 0.0

                # Centroid keyword extraction per cluster
                centroids = {}
                feature_names = vec.get_feature_names_out()
                for cl in range(n_clusters):
                    mask = labels == cl
                    if mask.sum() == 0:
                        continue
                    cluster_vec = X_norm[mask].mean(axis=0)
                    if hasattr(cluster_vec, "A1"):
                        cluster_vec = np.asarray(cluster_vec).flatten()
                    top_idx = cluster_vec.argsort()[-6:][::-1]
                    top_words = [feature_names[i] for i in top_idx]
                    centroids[str(cl)] = ", ".join(top_words)

                st.session_state["umap_df"]    = umap_df
                st.session_state["centroids"]  = centroids
                st.session_state["umap_dims"]  = n_comp
            except ImportError:
                st.error("Install umap-learn: `pip install umap-learn`")
                st.stop()
            except Exception as e:
                st.error(f"UMAP failed: {e}")
                st.stop()

    umap_df   = st.session_state.get("umap_df", pd.DataFrame())
    centroids = st.session_state.get("centroids", {})
    stored_dims = st.session_state.get("umap_dims", 2)

    if not umap_df.empty:
        color_col  = "tp_code" if colour_by == "Training Package" else "cluster"
        hover_data = ["unit_code", "element_title", "skill_statement", "cluster"]

        if stored_dims == 3 and "z" in umap_df.columns:
            fig_umap = px.scatter_3d(
                umap_df, x="x", y="y", z="z",
                color=color_col,
                hover_data=hover_data,
                opacity=0.75,
                size_max=6,
                height=650,
                title="3D Semantic Skill Space",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_umap.update_traces(marker=dict(size=3))
        else:
            fig_umap = px.scatter(
                umap_df, x="x", y="y",
                color=color_col,
                hover_data=hover_data,
                opacity=0.75,
                height=580,
                title="2D Semantic Skill Space — UMAP projection",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_umap.update_traces(marker=dict(size=5))

        fig_umap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#080e1a",
            font_color="#cfd8dc",
            legend=dict(bgcolor="rgba(10,22,40,0.9)", bordercolor="#1a2a4a",
                        font_size=10),
        )
        st.plotly_chart(fig_umap, use_container_width=True)

        # ── Centroid explorer ─────────────────────────────────────────────────
        st.markdown("**Cluster centroids — top defining terms per cluster**")
        st.caption("Use these to name micro-credentials or identify shared competency themes.")

        centroid_rows = [
            {"Cluster": k, "Defining terms": v, "Skill count": int((umap_df["cluster"] == k).sum())}
            for k, v in sorted(centroids.items(), key=lambda x: int(x[0]))
        ]
        centroid_df = pd.DataFrame(centroid_rows)

        # TP breakdown per cluster
        tp_breakdown = (
            umap_df.groupby(["cluster", "tp_code"])
            .size().reset_index(name="count")
            .sort_values(["cluster", "count"], ascending=[True, False])
        )

        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.dataframe(centroid_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Cluster": st.column_config.TextColumn(width="small"),
                             "Skill count": st.column_config.NumberColumn(width="small"),
                         })
        with col_c2:
            sel_cluster = st.selectbox("Explore cluster",
                                       sorted(centroids.keys(), key=int),
                                       key="a_sel_cluster")
            if sel_cluster:
                cl_tps = tp_breakdown[tp_breakdown["cluster"] == sel_cluster]
                st.markdown(f"**TPs in cluster {sel_cluster}:**")
                for _, row in cl_tps.iterrows():
                    pct = row["count"] / int((umap_df["cluster"] == sel_cluster).sum()) * 100
                    st.markdown(
                        f'<span class="metric-chip">{row["tp_code"]}: {row["count"]} ({pct:.0f}%)</span>',
                        unsafe_allow_html=True
                    )
                overlap = cl_tps[cl_tps["tp_code"].isin(sel_tps)]["tp_code"].nunique()
                if overlap > 1:
                    st.success(f"⚡ {overlap} TPs overlap here — micro-credential candidate!")

        # Download
        st.download_button(
            "⬇ Download cluster assignments CSV",
            umap_df[["unit_code", "unit_title", "tp_code",
                     "element_title", "skill_statement", "cluster"]]
            .to_csv(index=False).encode(),
            "skill_clusters.csv", "text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW B — CROSS-PACKAGE SIMILARITY HEATMAP (RPL ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-badge">View B</div>', unsafe_allow_html=True)
st.subheader("Cross-Package Similarity Heatmap — RPL Engine")
st.markdown("""
<div class="insight-box">
Each cell = average cosine similarity between the skill statements of two UOCs.
<b>High-intensity cells</b> mean a student who completed Unit A has already met
most requirements of Unit B — a direct <b>Recognition of Prior Learning</b> signal.
Diagonal clusters reveal internal redundancy within a training package.
</div>
""", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    tp_x = st.selectbox("Package X (columns)", sel_tps or all_tps[:1], key="b_tpx")
with col_b2:
    tp_y = st.selectbox("Package Y (rows)", sel_tps or all_tps[:1],
                        index=min(1, len(sel_tps or all_tps) - 1), key="b_tpy")
with col_b3:
    rpl_threshold = st.slider("Highlight RPL threshold", 0.3, 0.95, 0.55, 0.05, key="b_thresh")

run_heatmap = st.button("▶ Build similarity heatmap", type="primary", key="b_run")

if run_heatmap:
    with st.spinner("Computing cosine similarity matrix…"):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

                    # Query DB directly — not limited by sidebar filter
            @st.cache_data(ttl=300, show_spinner=False)
            def load_tp_skills(tp):
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT unit_code, unit_title, tp_code, skill_statement
                        FROM rsd_skill_records
                        WHERE tp_code = :tp
                          AND skill_statement IS NOT NULL
                          AND skill_statement != ''
                    """), {"tp": tp}).mappings().all()
                return pd.DataFrame([dict(r) for r in rows])

            df_x = load_tp_skills(tp_x)
            df_y = load_tp_skills(tp_y)

            if df_x.empty or df_y.empty:
                st.warning(f"One or both packages have no skill statements. "
                           f"{tp_x}: {len(df_x)} rows, {tp_y}: {len(df_y)} rows.")
            else:
                # Aggregate skill statements per UOC
                uoc_x = (df_x.groupby(["unit_code", "unit_title"])["skill_statement"]
                         .apply(lambda s: " ".join(s.dropna()))
                         .reset_index())
                uoc_y = (df_y.groupby(["unit_code", "unit_title"])["skill_statement"]
                         .apply(lambda s: " ".join(s.dropna()))
                         .reset_index())

                # Cap at 60 UOCs per side for readability
                uoc_x = uoc_x.head(60)
                uoc_y = uoc_y.head(60)

                all_docs = uoc_x["skill_statement"].tolist() + uoc_y["skill_statement"].tolist()
                vec = TfidfVectorizer(ngram_range=(1, 2), max_features=6000,
                                      stop_words="english")
                M = vec.fit_transform(all_docs)
                nx_ = len(uoc_x)
                sim = cos_sim(M[:nx_], M[nx_:])

                labels_x = [f"{r['unit_code']}" for _, r in uoc_x.iterrows()]
                labels_y = [f"{r['unit_code']}" for _, r in uoc_y.iterrows()]

                fig_heat = go.Figure(go.Heatmap(
                    z=sim,
                    x=labels_x,
                    y=labels_y,
                    colorscale=[
                        [0.0,  "#080e1a"],
                        [0.3,  "#0d2137"],
                        [0.55, "#1565c0"],
                        [0.75, "#2196f3"],
                        [0.9,  "#42a5f5"],
                        [1.0,  "#e3f2fd"],
                    ],
                    zmin=0, zmax=1,
                    hovertemplate=(
                        "<b>%{x}</b> ↔ <b>%{y}</b><br>"
                        "Similarity: %{z:.3f}<extra></extra>"
                    ),
                ))

                # Overlay RPL threshold line annotation
                fig_heat.update_layout(
                    height=max(500, len(labels_y) * 14 + 100),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#080e1a",
                    font=dict(color="#cfd8dc", size=9),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                    yaxis=dict(tickfont=dict(size=8)),
                    title=(f"Cosine Similarity: {tp_x} (cols) × {tp_y} (rows) — "
                           f"{len(labels_x)} × {len(labels_y)} UOCs"),
                    coloraxis_colorbar=dict(
                        title="Similarity",
                        tickfont=dict(color="#cfd8dc"),
                    ),
                )

                st.plotly_chart(fig_heat, use_container_width=True)

                # RPL candidates table
                rpl_pairs = []
                for i, rx in enumerate(labels_x):
                    for j, ry in enumerate(labels_y):
                        if sim[i, j] >= rpl_threshold and rx != ry:
                            rpl_pairs.append({
                                f"{tp_x} Unit": rx,
                                f"{tp_y} Unit": ry,
                                "Similarity": round(float(sim[i, j]), 3),
                            })

                rpl_df = pd.DataFrame(rpl_pairs).sort_values("Similarity", ascending=False)

                st.markdown(f"**RPL candidates ≥ {rpl_threshold}** — "
                            f"{len(rpl_df)} pairs found")
                if not rpl_df.empty:
                    st.dataframe(rpl_df.head(50), use_container_width=True,
                                 hide_index=True,
                                 column_config={
                                     "Similarity": st.column_config.ProgressColumn(
                                         "Similarity", min_value=0, max_value=1,
                                         format="%.3f"),
                                 })
                    st.download_button(
                        "⬇ Download RPL candidates CSV",
                        rpl_df.to_csv(index=False).encode(),
                        f"rpl_{tp_x}_{tp_y}.csv", "text/csv",
                    )
                else:
                    st.info(f"No pairs above {rpl_threshold} similarity. "
                            "Try lowering the threshold.")

        except Exception as e:
            st.error(f"Heatmap failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIEW C — SKILL DNA PARALLEL COORDINATES
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-badge">View C</div>', unsafe_allow_html=True)
st.subheader("Skill DNA — Parallel Coordinates")
st.markdown("""
<div class="insight-box">
Each line is one Unit of Competency drawn across four dimensions of its skill profile.
Lines that run <b>parallel and close together</b> represent UOCs with identical DNA —
prime candidates for credit transfer or bundling into a micro-credential.
</div>
""", unsafe_allow_html=True)

col_d1, col_d2 = st.columns(2)
with col_d1:
    dna_tps = st.multiselect("Packages to compare", sel_tps or all_tps,
                             default=(sel_tps or all_tps)[:4], key="c_tps")
with col_d2:
    dna_max = st.slider("Max UOCs per package", 10, 100, 30, key="c_max")

run_dna = st.button("▶ Build Skill DNA chart", type="primary", key="c_run")

# Bloom's action verbs by level
BLOOMS = {
    "remember":   ["identify", "recall", "list", "name", "define", "state", "recognise"],
    "understand": ["describe", "explain", "interpret", "classify", "summarise", "compare"],
    "apply":      ["apply", "use", "implement", "demonstrate", "perform", "execute", "operate"],
    "analyse":    ["analyse", "examine", "differentiate", "inspect", "test", "investigate"],
    "evaluate":   ["evaluate", "assess", "judge", "justify", "critique", "review", "recommend"],
    "create":     ["design", "develop", "create", "construct", "plan", "formulate", "produce"],
}

def bloom_level(statements: list[str]) -> float:
    """Return 1–6 Bloom's level score (higher = more complex)."""
    level_scores = []
    for stmt in statements:
        words = re.findall(r'\b\w+\b', stmt.lower())
        for level, (score, verbs) in enumerate(
            [(1, BLOOMS["remember"]), (2, BLOOMS["understand"]),
             (3, BLOOMS["apply"]),   (4, BLOOMS["analyse"]),
             (5, BLOOMS["evaluate"]),(6, BLOOMS["create"])], start=1
        ):
            if any(v in words for v in verbs):
                level_scores.append(level)
    return float(np.mean(level_scores)) if level_scores else 2.0

if run_dna and dna_tps:
    with st.spinner("Computing Skill DNA dimensions…"):
        try:
            df_dna_src = df_skills[df_skills["tp_code"].isin(dna_tps)].copy()

            uoc_groups = (
                df_dna_src.groupby(["unit_code", "unit_title", "tp_code"])
                .agg(
                    statements=("skill_statement", list),
                    esco_score=("esco_skill_score", "mean"),
                    occ_titles=("esco_occupation_titles", lambda x:
                                len(set("|".join(x.dropna()).split("|")))),
                )
                .reset_index()
            )

            # Cap per TP
            uoc_groups = (
                uoc_groups.groupby("tp_code", group_keys=False)
                .apply(lambda g: g.head(dna_max))
                .reset_index(drop=True)
            )

            # Dim 1: ESCO semantic match score (0–1)
            uoc_groups["dim_esco"] = pd.to_numeric(
                uoc_groups["esco_score"], errors="coerce").fillna(0.0)

            # Dim 2: Occupation linkage confidence from uoc_occupation_links
            if not df_occ.empty:
                conf_map = df_occ.groupby("uoc_code")["confidence"].mean().to_dict()
                uoc_groups["dim_conf"] = uoc_groups["unit_code"].map(conf_map).fillna(0.0)
            else:
                uoc_groups["dim_conf"] = 0.0

            # Dim 3: Bloom's taxonomy level (1–6)
            uoc_groups["dim_blooms"] = uoc_groups["statements"].apply(bloom_level)

            # Dim 4: Occupation breadth (unique occupations)
            uoc_groups["dim_occ_breadth"] = pd.to_numeric(
                uoc_groups["occ_titles"], errors="coerce").fillna(0.0)
            # Normalise breadth to 0–1
            max_occ = uoc_groups["dim_occ_breadth"].max() or 1
            uoc_groups["dim_occ_norm"] = uoc_groups["dim_occ_breadth"] / max_occ

            # Colour map per TP
            tp_color_map = {
                tp: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
                for i, tp in enumerate(dna_tps)
            }
            uoc_groups["color_val"] = pd.Categorical(
                uoc_groups["tp_code"]).codes.astype(float)

            dims = [
                dict(label="ESCO Semantic Score",
                     values=uoc_groups["dim_esco"],
                     range=[0, 1]),
                dict(label="Occupation Confidence",
                     values=uoc_groups["dim_conf"],
                     range=[0, 1]),
                dict(label="Bloom's Level (1=recall→6=create)",
                     values=uoc_groups["dim_blooms"],
                     range=[1, 6],
                     tickvals=[1, 2, 3, 4, 5, 6],
                     ticktext=["Recall", "Understand", "Apply",
                               "Analyse", "Evaluate", "Create"]),
                dict(label="Occupation Breadth (normalised)",
                     values=uoc_groups["dim_occ_norm"],
                     range=[0, 1]),
            ]

            fig_dna = go.Figure(go.Parcoords(
                line=dict(
                    color=uoc_groups["color_val"],
                    colorscale=[[i / max(len(dna_tps) - 1, 1), c]
                                for i, c in enumerate(
                                    [tp_color_map[tp] for tp in dna_tps])],
                    showscale=False,
                    cmin=0,
                    cmax=len(dna_tps) - 1,
                    colorbar=dict(thickness=0),
                ),
                dimensions=dims,
                unselected=dict(line=dict(opacity=0.08)),
            ))

            fig_dna.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#080e1a",
                font=dict(color="#cfd8dc", family="DM Mono, monospace", size=11),
                title="Skill DNA — drag axes to filter, drag lines to select",
            )
            st.plotly_chart(fig_dna, use_container_width=True)

            # Legend
            st.markdown("**Training package legend:**")
            legend_html = " ".join(
                f'<span class="metric-chip" style="border-color:{tp_color_map[tp]};'
                f'color:{tp_color_map[tp]}">{tp}</span>'
                for tp in dna_tps
            )
            st.markdown(legend_html, unsafe_allow_html=True)

            st.markdown("""
<div class="insight-box">
<b>How to read this chart:</b> Drag along any axis to filter UOCs. Lines that run 
parallel across all four axes have identical skill profiles regardless of which training 
package they belong to — these are your strongest credit transfer and micro-credential candidates.
<br><br>
<b>Dimension guide:</b><br>
• <b>ESCO Semantic Score</b> — how well the skill maps to international standards (0 = weak, 1 = strong)<br>
• <b>Occupation Confidence</b> — strength of ANZSCO occupation linkage (higher = more certain)<br>
• <b>Bloom's Level</b> — cognitive complexity of skills (Recall → Create)<br>
• <b>Occupation Breadth</b> — how many distinct occupations use this unit (higher = more transferable)
</div>
""", unsafe_allow_html=True)

            # Download DNA data
            dna_export = uoc_groups[[
                "unit_code", "unit_title", "tp_code",
                "dim_esco", "dim_conf", "dim_blooms", "dim_occ_breadth"
            ]].rename(columns={
                "dim_esco":        "ESCO_Score",
                "dim_conf":        "Occupation_Confidence",
                "dim_blooms":      "Blooms_Level",
                "dim_occ_breadth": "Occupation_Breadth",
            })
            st.download_button(
                "⬇ Download Skill DNA CSV",
                dna_export.to_csv(index=False).encode(),
                "skill_dna.csv", "text/csv",
            )

        except Exception as e:
            st.error(f"Skill DNA failed: {e}")
