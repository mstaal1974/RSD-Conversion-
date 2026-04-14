"""
pages/2_🔍_Semantic_Analysis.py

Semantic analysis of skill statements with rich visualisations:
  - Embedding-based similarity (OpenAI text-embedding-3-small OR sentence-transformers)
  - DBSCAN clustering
  - Interactive cluster browser with canonical selection
  - Near-duplicate detection table
  - Similarity heatmap (Altair)
  - Cluster size distribution chart
  - Score distribution histogram
  - Three export formats
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="Semantic Analysis", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px; text-align: center;
  }
  .metric-card .value {
    font-family: 'DM Mono', monospace; font-size: 2.2rem;
    font-weight: 500; color: #38bdf8; line-height: 1;
  }
  .metric-card .label {
    font-size: 0.75rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px;
  }
  .metric-card .delta {
    font-family: 'DM Mono', monospace; font-size: 0.85rem;
    color: #4ade80; margin-top: 4px;
  }
  .cluster-card {
    background: #0f172a; border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 12px;
  }
  .cluster-card.canonical { border-left-color: #4ade80; background: #0a1f0f; }
  .cluster-card .unit-badge {
    display: inline-block; background: #1e3a5f; color: #7dd3fc;
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 2px 8px; border-radius: 4px; margin-right: 8px;
  }
  .cluster-card .stmt {
    font-size: 0.9rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;
  }
  .dupe-row {
    background: #1a0a0a; border: 1px solid #7f1d1d;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
  }
  .sim-badge {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.8rem; padding: 3px 10px; border-radius: 20px; margin-bottom: 8px;
  }
  .sim-high  { background: #450a0a; color: #fca5a5; }
  .sim-med   { background: #431407; color: #fdba74; }
  .sim-low   { background: #1c1917; color: #d6d3d1; }
  .section-header {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: #475569;
    border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin: 32px 0 16px 0;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    border: none; color: white; font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em; font-size: 0.85rem;
    padding: 10px 24px; border-radius: 8px;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    transform: translateY(-1px);
  }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Semantic Analysis")
st.caption("Cluster, deduplicate and align skill statements using embedding-based similarity")

def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

DB_URL     = _secret("DATABASE_URL")
OPENAI_KEY = _secret("OPENAI_API_KEY")

if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

with st.sidebar:
    st.markdown("### ⚙️ Embedding model")
    embedding_backend = st.radio(
        "Backend",
        ["OpenAI (API — most accurate)", "Sentence-Transformers (local — free)"],
        index=0,
    )
    use_openai = embedding_backend.startswith("OpenAI")
    if use_openai:
        embedding_model = st.selectbox(
            "OpenAI model",
            ["text-embedding-3-small", "text-embedding-3-large"],
        )
        if not OPENAI_KEY:
            st.warning("OPENAI_API_KEY not set — switch to Sentence-Transformers.")
    else:
        st.info("Uses `all-MiniLM-L6-v2` — 384-dim, CPU-friendly, no API key needed.")
        embedding_model = "all-MiniLM-L6-v2"

    st.divider()
    st.markdown("### 🎛️ Clustering")
    cluster_threshold   = st.slider("Cluster similarity threshold", 0.70, 0.98, 0.85, 0.01)
    duplicate_threshold = st.slider("Near-duplicate threshold", 0.85, 0.99, 0.92, 0.01)
    min_cluster_size    = st.slider("Min cluster size", 2, 5, 2)

    st.divider()
    st.markdown("### 🔎 Filter source")

    with engine.connect() as conn:
        runs = conn.execute(text(
            "SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC"
        )).mappings().all()
    runs = [dict(r) for r in runs]

    run_options = ["All runs"] + [f"{str(r['id'])[:8]}… {r['source_filename']}" for r in runs]
    run_ids     = [None] + [str(r["id"]) for r in runs]
    sel_run_lbl = st.selectbox("Source run", run_options)
    sel_run_id  = run_ids[run_options.index(sel_run_lbl)]
    unit_filter = st.text_input("Unit code prefix", placeholder="e.g. BSB")

    st.divider()
    run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)

@st.cache_data(ttl=60)
def load_statements(run_id, unit_prefix):
    with engine.connect() as conn:
        if run_id:
            res = conn.execute(
                text("SELECT * FROM rsd_skill_records WHERE run_id=:rid ORDER BY unit_code, row_index"),
                {"rid": run_id},
            ).mappings().all()
        else:
            res = conn.execute(
                text("SELECT * FROM rsd_skill_records ORDER BY unit_code, row_index")
            ).mappings().all()
    df = pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()
    if not df.empty and unit_prefix:
        df = df[df["unit_code"].str.upper().str.startswith(unit_prefix.upper())]
    return df

df_raw = load_statements(sel_run_id, unit_filter)

if df_raw.empty:
    st.info("No skill statements found. Run some batches first.")
    st.stop()

n = len(df_raw)
avg_tokens = 50
est_cost   = round(n * avg_tokens * 0.00000002, 4)

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.info(f"**{n}** statements loaded")
if use_openai:
    col_info2.info(f"Est. cost: **~${est_cost}** (OpenAI)")
else:
    col_info2.info("Cost: **Free** (local model)")
col_info3.info(f"Est. time: **~{round(n * 0.05 / 60, 1)} min**")

def get_embeddings_openai(texts, model, api_key, batch_size=100):
    from openai import OpenAI
    client  = OpenAI(api_key=api_key)
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t and t.strip() else "empty" for t in texts[i:i+batch_size]]
        resp  = client.embeddings.create(input=batch, model=model)
        all_emb.extend([item.embedding for item in resp.data])
    return np.array(all_emb, dtype=np.float32)

def get_embeddings_local(texts, model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error(
            "sentence-transformers is not installed. "
            "Switch to OpenAI backend or add `sentence-transformers>=3.0,<4.0` "
            "to requirements.txt and redeploy."
        )
        st.stop()
    model = SentenceTransformer(model_name)
    clean = [t if t and t.strip() else "empty" for t in texts]
    return model.encode(clean, normalize_embeddings=True, show_progress_bar=False)

def cosine_sim_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    n = embeddings / norms
    return n @ n.T

def run_dbscan(embeddings, threshold, min_samples):
    from sklearn.cluster import DBSCAN
    dist   = np.clip(1.0 - cosine_sim_matrix(embeddings), 0, 2)
    labels = DBSCAN(eps=1.0-threshold, min_samples=min_samples,
                    metric="precomputed").fit_predict(dist)
    return labels

def find_canonical(embeddings, labels, texts):
    clusters = {}
    for label in set(labels) - {-1}:
        mask     = labels == label
        idx      = np.where(mask)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1)
        sims     = [(float((embeddings[i] / (np.linalg.norm(embeddings[i]) or 1)) @ centroid), int(i))
                    for i in idx]
        best = max(sims, key=lambda x: x[0])
        clusters[int(label)] = {
            "canonical_idx":  best[1],
            "canonical_text": texts[best[1]],
            "size":           int(mask.sum()),
            "member_indices": idx.tolist(),
        }
    return clusters

def find_near_dupes(sim_matrix, threshold, df):
    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                pairs.append({
                    "idx_a": i, "idx_b": j,
                    "similarity":   round(s, 4),
                    "unit_a":       df.iloc[i].get("unit_code",""),
                    "unit_title_a": df.iloc[i].get("unit_title",""),
                    "unit_b":       df.iloc[j].get("unit_code",""),
                    "unit_title_b": df.iloc[j].get("unit_title",""),
                    "text_a":  df.iloc[i].get("skill_statement",""),
                    "text_b":  df.iloc[j].get("skill_statement",""),
                    "elem_a":  df.iloc[i].get("element_title",""),
                    "elem_b":  df.iloc[j].get("element_title",""),
                })
    return sorted(pairs, key=lambda x: x["similarity"], reverse=True)

if run_btn:
    if n > 3000:
        st.warning(f"Large dataset ({n} statements). Consider filtering by unit code prefix.")

    progress = st.progress(0)
    status   = st.empty()

    try:
        texts = df_raw["skill_statement"].fillna("").tolist()
        status.write("⏳ Generating embeddings…")
        progress.progress(0.1)

        if use_openai:
            if not OPENAI_KEY:
                st.error("OpenAI API key not configured.")
                st.stop()
            embeddings = get_embeddings_openai(texts, embedding_model, OPENAI_KEY)
        else:
            embeddings = get_embeddings_local(texts, embedding_model)

        progress.progress(0.45)
        status.write("⏳ Computing similarity matrix…")
        sim_matrix = cosine_sim_matrix(embeddings)

        progress.progress(0.6)
        status.write("⏳ Clustering…")
        labels = run_dbscan(embeddings, cluster_threshold, min_cluster_size)

        progress.progress(0.75)
        status.write("⏳ Finding canonical statements…")
        clusters = find_canonical(embeddings, labels, texts)

        progress.progress(0.88)
        status.write("⏳ Finding near-duplicates…")
        near_dupes = find_near_dupes(sim_matrix, duplicate_threshold, df_raw)

        df_ann             = df_raw.copy().reset_index(drop=True)
        df_ann["cluster_id"]   = labels
        canonical_set          = {v["canonical_idx"] for v in clusters.values()}
        df_ann["is_canonical"] = df_ann.index.isin(canonical_set)
        df_ann["is_singleton"] = labels == -1

        st.session_state.update({
            "sa_results":    True,
            "sa_df_ann":     df_ann,
            "sa_sim_matrix": sim_matrix,
            "sa_clusters":   clusters,
            "sa_near_dupes": near_dupes,
            "sa_labels":     labels,
            "sa_embeddings": embeddings,
            "sa_overrides":  {},
        })

        progress.progress(1.0)
        status.write("✅ Analysis complete")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if not st.session_state.get("sa_results"):
    st.info("Configure settings in the sidebar and click **▶ Run analysis**.")
    st.stop()

df_ann     = st.session_state["sa_df_ann"]
sim_matrix = st.session_state["sa_sim_matrix"]
clusters   = st.session_state["sa_clusters"]
near_dupes = st.session_state["sa_near_dupes"]
labels     = st.session_state["sa_labels"]

n_total       = len(df_ann)
n_clustered   = int((labels >= 0).sum())
n_singletons  = int((labels == -1).sum())
n_clusters    = len(clusters)
n_dupes       = len(near_dupes)
n_reduction   = n_clustered - n_clusters
pct_reduction = round(100 * n_reduction / max(n_total, 1))

st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
cols    = st.columns(6)
metrics = [
    ("Total statements", n_total,      None),
    ("Clusters found",   n_clusters,   None),
    ("Clustered",        n_clustered,  f"{round(100*n_clustered/max(n_total,1))}%"),
    ("Singletons",       n_singletons, None),
    ("Near-dupes",       n_dupes,      None),
    ("Potential saving", n_reduction,  f"−{pct_reduction}%"),
]
for col, (label, value, delta) in zip(cols, metrics):
    with col:
        delta_html = f'<div class="delta">{delta}</div>' if delta else ""
        st.markdown(
            f'<div class="metric-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div>{delta_html}</div>',
            unsafe_allow_html=True,
        )
st.markdown(
    f"<br>Keeping one canonical per cluster reduces "
    f"**{n_total} → {n_singletons + n_clusters}** statements.",
    unsafe_allow_html=True,
)

# ── Cluster size distribution ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster size distribution</div>', unsafe_allow_html=True)

if clusters:
    sizes = pd.DataFrame([
        {"Cluster": f"C{k}", "Size": v["size"],
         "Canonical": v["canonical_text"][:60]+"…"}
        for k, v in sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)
    ])
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        raw_sizes = [v["size"] for v in clusters.values()]
        bin_order = ["2-3","4-5","6-10","11-20","21-50","51+"]
        def size_bin(s):
            if s<=3: return "2-3"
            if s<=5: return "4-5"
            if s<=10: return "6-10"
            if s<=20: return "11-20"
            if s<=50: return "21-50"
            return "51+"
        bin_counts = {b: 0 for b in bin_order}
        for s in raw_sizes:
            bin_counts[size_bin(s)] += 1
        hist_data = pd.DataFrame([
            {"Range": b, "Clusters": bin_counts[b]}
            for b in bin_order if bin_counts[b] > 0
        ])
        chart = alt.Chart(hist_data).mark_bar(
            color="#38bdf8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
        ).encode(
            x=alt.X("Range:N", sort=bin_order, title="Cluster size",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Clusters:Q", title="Number of clusters"),
            tooltip=["Range","Clusters"],
        ).properties(height=220, title="Cluster size distribution")
        st.altair_chart(chart, use_container_width=True)
        median_size = int(pd.Series(raw_sizes).median())
        st.caption(f"Median: {median_size} · Largest: {max(raw_sizes)} statements")
    with col_table:
        st.dataframe(sizes.head(15), use_container_width=True,
            column_config={
                "Cluster":   st.column_config.TextColumn("ID", width="small"),
                "Size":      st.column_config.NumberColumn("Size"),
                "Canonical": st.column_config.TextColumn("Canonical statement"),
            }, hide_index=True)

# ── Similarity score distribution ─────────────────────────────────────────────
st.markdown('<div class="section-header">Pairwise similarity distribution</div>', unsafe_allow_html=True)
n_show    = min(500, len(df_ann))
sim_slice = sim_matrix[:n_show, :n_show]
upper     = sim_slice[np.triu_indices(n_show, k=1)]
hist_vals, hist_bins = np.histogram(upper, bins=40, range=(0, 1))
hist_df = pd.DataFrame({
    "similarity": [float((hist_bins[i]+hist_bins[i+1])/2) for i in range(len(hist_vals))],
    "count":      [int(v) for v in hist_vals],
})
hist_df = hist_df[hist_df["count"] > 0]
score_chart  = alt.Chart(hist_df).mark_bar(
    color="#7c3aed", cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
).encode(
    x=alt.X("similarity:Q", bin=False, title="Cosine similarity",
            scale=alt.Scale(domain=[0,1])),
    y=alt.Y("count:Q", title="Pair count"),
    tooltip=[alt.Tooltip("similarity:Q", format=".2f"), alt.Tooltip("count:Q")],
).properties(height=180,
    title=f"Distribution of pairwise similarities (first {n_show} statements)")
cluster_line = alt.Chart(pd.DataFrame({"x":[cluster_threshold]})).mark_rule(
    color="#38bdf8", strokeDash=[4,2], size=1.5).encode(x="x:Q")
dupe_line    = alt.Chart(pd.DataFrame({"x":[duplicate_threshold]})).mark_rule(
    color="#f43f5e", strokeDash=[4,2], size=1.5).encode(x="x:Q")
st.altair_chart((score_chart + cluster_line + dupe_line).resolve_scale(x="shared"),
                use_container_width=True)
st.caption(f"🔵 Cluster threshold ({cluster_threshold})  🔴 Near-duplicate threshold ({duplicate_threshold})")

# ── Similarity heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Similarity heatmap</div>', unsafe_allow_html=True)

heatmap_tab1, heatmap_tab2 = st.tabs([
    "📦 Unit-level (overview — all data)",
    "🔬 Statement-level (drill-down)",
])

with heatmap_tab1:
    st.caption(
        "Average pairwise similarity between all statements in each unit. "
        "Top 40 selected by **highest average semantic similarity** to all other units."
    )

    unit_codes_all = df_ann["unit_code"].fillna("(blank)").values
    unique_units   = sorted(set(unit_codes_all))

    if len(unique_units) <= 1:
        st.info("Need at least 2 different unit codes to show unit-level heatmap.")
    else:
        # ── KEY CHANGE: select top 40 by average semantic similarity ──────────
        # Build per-unit avg similarity to all other units
        all_units_list = sorted(set(unit_codes_all))
        unit_idx_map   = {u: np.where(unit_codes_all == u)[0]
                          for u in all_units_list}

        # Compute avg outward similarity for each unit
        unit_avg_sim = {}
        for u, idxs in unit_idx_map.items():
            # avg similarity of this unit's statements to ALL OTHER statements
            other_idxs = np.array(
                [i for i in range(len(df_ann)) if i not in set(idxs.tolist())]
            )
            if len(other_idxs) == 0:
                unit_avg_sim[u] = 0.0
                continue
            submat = sim_matrix[np.ix_(idxs, other_idxs)]
            unit_avg_sim[u] = float(submat.mean())

        # Top 40 by avg outward similarity
        top_units = sorted(
            unit_avg_sim.keys(),
            key=lambda u: unit_avg_sim[u],
            reverse=True,
        )[:40]
        top_units_sorted = sorted(top_units)

        # Compute avg similarity per unit pair
        unit_rows = []
        for ua in top_units_sorted:
            for ub in top_units_sorted:
                idxs_a = unit_idx_map[ua]
                idxs_b = unit_idx_map[ub]
                submat = sim_matrix[np.ix_(idxs_a, idxs_b)]
                unit_rows.append({
                    "Unit A":         ua,
                    "Unit B":         ub,
                    "Avg similarity": round(float(submat.mean()), 3),
                })

        unit_heat_df = pd.DataFrame(unit_rows)
        label_map    = {u: u[-7:] if len(u) > 7 else u for u in top_units_sorted}
        unit_heat_df["A_label"] = unit_heat_df["Unit A"].map(label_map)
        unit_heat_df["B_label"] = unit_heat_df["Unit B"].map(label_map)

        unit_heatmap = alt.Chart(unit_heat_df).mark_rect().encode(
            x=alt.X("B_label:N", sort=None,
                    axis=alt.Axis(labelAngle=-45, title="Training package →")),
            y=alt.Y("A_label:N", sort=None,
                    axis=alt.Axis(title="↑ Training package")),
            color=alt.Color(
                "Avg similarity:Q",
                scale=alt.Scale(scheme="viridis", domain=[0.3, 1.0]),
                legend=alt.Legend(title="Avg similarity"),
            ),
            tooltip=[
                alt.Tooltip("Unit A:N"),
                alt.Tooltip("Unit B:N"),
                alt.Tooltip("Avg similarity:Q", format=".3f"),
            ],
        ).properties(
            width=600, height=600,
            title=f"Unit-level similarity — top {len(top_units_sorted)} units "
                  f"by semantic similarity",
        ).interactive(False)
        st.altair_chart(unit_heatmap, use_container_width=True)
        st.caption(
            f"Showing top {len(top_units_sorted)} units by average outward semantic similarity. "
            "Diagonal = 1.0. High off-diagonal = semantically overlapping skills."
        )

        # Show which units were selected and their avg sim score
        with st.expander("Units included and their avg semantic similarity"):
            sim_rank_df = pd.DataFrame([
                {"Unit": u, "Avg outward similarity": round(unit_avg_sim[u], 4)}
                for u in top_units
            ]).sort_values("Avg outward similarity", ascending=False)
            st.dataframe(sim_rank_df, use_container_width=True, hide_index=True,
                column_config={
                    "Avg outward similarity": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.4f"),
                })

        cross = unit_heat_df[
            (unit_heat_df["Unit A"] != unit_heat_df["Unit B"]) &
            (unit_heat_df["Avg similarity"] >= cluster_threshold)
        ].sort_values("Avg similarity", ascending=False).head(20)

        if len(cross):
            with st.expander(
                f"⚡ {len(cross)} high-overlap unit pairs (avg sim ≥ {cluster_threshold})"
            ):
                st.dataframe(cross[["Unit A","Unit B","Avg similarity"]],
                             use_container_width=True, hide_index=True)

with heatmap_tab2:
    st.caption("Select two units to compare their statements head-to-head.")
    col_ua, col_ub = st.columns(2)
    with col_ua:
        unit_a_sel = st.selectbox(
            "Unit A",
            top_units_sorted if len(unique_units) > 1 else unique_units,
            key="heat_ua",
        )
    with col_ub:
        remaining  = [u for u in (top_units_sorted if len(unique_units) > 1 else unique_units)
                      if u != unit_a_sel]
        unit_b_sel = st.selectbox("Unit B", remaining, key="heat_ub") if remaining else unit_a_sel

    idxs_a2 = np.where(unit_codes_all == unit_a_sel)[0]
    idxs_b2 = np.where(unit_codes_all == unit_b_sel)[0]

    if len(idxs_a2) == 0 or len(idxs_b2) == 0:
        st.info("No statements found for one of the selected units.")
    else:
        max_drill = 30
        idxs_a2   = idxs_a2[:max_drill]
        idxs_b2   = idxs_b2[:max_drill]
        submat    = sim_matrix[np.ix_(idxs_a2, idxs_b2)]

        drill_rows = []
        for ii, ia in enumerate(idxs_a2):
            for jj, ib in enumerate(idxs_b2):
                drill_rows.append({
                    "A":   df_ann.iloc[ia].get("element_title","")[:35],
                    "B":   df_ann.iloc[ib].get("element_title","")[:35],
                    "sim": round(float(submat[ii, jj]), 3),
                })

        drill_df = pd.DataFrame(drill_rows)
        a_labels = sorted(drill_df["A"].unique().tolist())
        b_labels = sorted(drill_df["B"].unique().tolist())

        drill_chart = alt.Chart(drill_df).mark_rect().encode(
            x=alt.X("B:N", sort=b_labels,
                    axis=alt.Axis(labelAngle=-45, title=f"{unit_b_sel} elements →")),
            y=alt.Y("A:N", sort=a_labels,
                    axis=alt.Axis(title=f"↑ {unit_a_sel} elements")),
            color=alt.Color("sim:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                legend=alt.Legend(title="Similarity")),
            tooltip=[
                alt.Tooltip("A:N", title=unit_a_sel),
                alt.Tooltip("B:N", title=unit_b_sel),
                alt.Tooltip("sim:Q", format=".3f", title="Similarity"),
            ],
        ).properties(
            width=600, height=500,
            title=f"{unit_a_sel} vs {unit_b_sel} — element-level similarity",
        ).interactive(False)
        st.altair_chart(drill_chart, use_container_width=True)

        hot_pairs = drill_df[drill_df["sim"] >= cluster_threshold].sort_values(
            "sim", ascending=False)
        if len(hot_pairs):
            st.markdown(
                f"**{len(hot_pairs)} element pairs above threshold {cluster_threshold}:**")
            st.dataframe(hot_pairs, use_container_width=True, hide_index=True,
                column_config={
                    "A":   st.column_config.TextColumn(unit_a_sel),
                    "B":   st.column_config.TextColumn(unit_b_sel),
                    "sim": st.column_config.NumberColumn("Similarity", format="%.3f"),
                })
        else:
            st.info(f"No element pairs above threshold {cluster_threshold}.")

# ── Cluster browser ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster browser</div>', unsafe_allow_html=True)
st.caption("Each cluster groups semantically similar statements. "
           "⭐ = canonical (closest to cluster centre).")

if not clusters:
    st.info("No clusters found — try lowering the similarity threshold.")
else:
    overrides = st.session_state.get("sa_overrides", {})
    cluster_rows = []
    for cid, info in clusters.items():
        members     = df_ann[df_ann.index.isin(info["member_indices"])]
        unit_codes  = members["unit_code"].dropna().unique().tolist()
        tp_prefixes = sorted(set(c[:3] for c in unit_codes if c))
        cross_tp    = len(tp_prefixes) > 1
        cluster_rows.append({
            "id": cid, "size": info["size"],
            "canonical": info["canonical_text"],
            "units": ", ".join(unit_codes[:4]) + (" …" if len(unit_codes) > 4 else ""),
            "tp": ", ".join(tp_prefixes),
            "cross_tp": cross_tp,
            "info": info,
        })
    cluster_rows.sort(key=lambda x: x["size"], reverse=True)

    cb_tab1, cb_tab2 = st.tabs(["📋 Summary table", "🔎 Detailed browser"])

    with cb_tab1:
        summary_df = pd.DataFrame([{
            "Cluster":   f"C{r['id']}",
            "Size":      r["size"],
            "Cross-TP":  "⚡" if r["cross_tp"] else "",
            "Packages":  r["tp"],
            "Units":     r["units"],
            "Canonical": r["canonical"][:90]+"…",
        } for r in cluster_rows])

        f1, f2, f3 = st.columns(3)
        with f1:
            search_term = st.text_input("🔍 Search canonical", key="cb_search")
        with f2:
            tp_options = sorted(set(r["tp"] for r in cluster_rows if r["tp"]))
            tp_filter2 = st.selectbox("Filter by TP", ["All"]+tp_options, key="cb_tp")
        with f3:
            cross_only = st.checkbox("⚡ Cross-TP only", key="cb_cross")

        filtered = summary_df.copy()
        if search_term:
            filtered = filtered[filtered["Canonical"].str.contains(
                search_term, case=False, na=False)]
        if tp_filter2 != "All":
            filtered = filtered[filtered["Packages"].str.contains(tp_filter2, na=False)]
        if cross_only:
            filtered = filtered[filtered["Cross-TP"] == "⚡"]

        st.caption(f"Showing {len(filtered)} of {len(summary_df)} clusters")
        st.dataframe(filtered, use_container_width=True,
            column_config={
                "Cluster":   st.column_config.TextColumn("ID", width="small"),
                "Size":      st.column_config.NumberColumn("Size", width="small"),
                "Cross-TP":  st.column_config.TextColumn("⚡", width="small"),
                "Packages":  st.column_config.TextColumn("Packages", width="small"),
                "Units":     st.column_config.TextColumn("Units"),
                "Canonical": st.column_config.TextColumn("Canonical statement", width="large"),
            }, hide_index=True)

        cross_tp_clusters = [r for r in cluster_rows if r["cross_tp"]]
        if cross_tp_clusters:
            st.markdown(
                f"**{len(cross_tp_clusters)} cross-TP clusters** — "
                "prime candidates for a shared canonical statement.")

    with cb_tab2:
        f1d, f2d, f3d = st.columns(3)
        with f1d: search_d = st.text_input("🔍 Search", key="cb_search_d")
        with f2d: tp_d = st.selectbox("Filter by TP", ["All"]+tp_options, key="cb_tp_d")
        with f3d: sort_by = st.selectbox("Sort by",
            ["Size ↓","Cross-TP first","Alphabetical"], key="cb_sort")

        filtered_rows = cluster_rows.copy()
        if search_d:
            filtered_rows = [r for r in filtered_rows
                             if search_d.lower() in r["canonical"].lower()
                             or search_d.lower() in r["units"].lower()]
        if tp_d != "All":
            filtered_rows = [r for r in filtered_rows if tp_d in r["tp"]]
        if sort_by == "Cross-TP first":
            filtered_rows.sort(key=lambda x: (not x["cross_tp"], -x["size"]))
        elif sort_by == "Alphabetical":
            filtered_rows.sort(key=lambda x: x["canonical"][:30])

        per_page    = 8
        total_pages = max(1, (len(filtered_rows)+per_page-1)//per_page)
        page_col, info_col = st.columns([2, 3])
        with page_col:
            page = st.number_input("Page", 1, total_pages, 1, key="cluster_page")
        with info_col:
            st.caption(f"Page {page}/{total_pages} — {len(filtered_rows)} clusters")

        for row in filtered_rows[(page-1)*per_page : page*per_page]:
            cluster_id   = row["id"]
            info         = row["info"]
            members      = df_ann[df_ann.index.isin(info["member_indices"])]
            canon_idx    = overrides.get(cluster_id, info["canonical_idx"])
            cross_badge  = " ⚡ CROSS-TP" if row["cross_tp"] else ""

            with st.expander(
                f"**C{cluster_id}** {cross_badge} — {info['size']} stmts  |  "
                f"⭐ {info['canonical_text'][:65]}…"
            ):
                st.markdown(
                    f"<div style='display:flex;gap:16px;margin-bottom:12px'>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
                    f"color:#38bdf8'>Packages: {row['tp']}</span>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
                    f"color:#94a3b8'>Units: {row['units']}</span></div>",
                    unsafe_allow_html=True,
                )
                for _, mem in members.iterrows():
                    is_canon   = mem.name == canon_idx
                    card_class = "cluster-card canonical" if is_canon else "cluster-card"
                    badge      = "⭐ CANONICAL &nbsp;" if is_canon else "○ &nbsp;"
                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<span class="unit-badge">{mem.get("unit_code","")}</span>'
                        f'<span style="font-size:0.75rem;color:#64748b">'
                        f'{mem.get("unit_title","")}</span><br>'
                        f'<strong style="color:#94a3b8;font-size:0.8rem">{badge}'
                        f'<span style="font-style:italic">'
                        f'{mem.get("element_title","")}</span></strong>'
                        f'<div class="stmt">{mem.get("skill_statement","")}</div></div>',
                        unsafe_allow_html=True,
                    )
                opts = {
                    f"[{df_ann.loc[i,'unit_code']}] "
                    f"{df_ann.loc[i,'skill_statement'][:70]}…": i
                    for i in info["member_indices"]
                }
                opt_values  = list(opts.values())
                safe_index  = opt_values.index(canon_idx) if canon_idx in opt_values else 0
                chosen_label = st.selectbox("Set canonical", list(opts.keys()),
                                            index=safe_index,
                                            key=f"canon_sel_{cluster_id}")
                chosen_idx = opts[chosen_label]
                if chosen_idx != info["canonical_idx"]:
                    overrides[cluster_id] = chosen_idx
                    st.session_state["sa_overrides"] = overrides
                    st.success("Override saved ✅")

# ── Near-duplicate pairs ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Near-duplicate pairs</div>', unsafe_allow_html=True)
st.caption(
    f"Statement pairs with cosine similarity ≥ **{duplicate_threshold}** — "
    "review and consolidate these"
)

if not near_dupes:
    st.success(f"✅ No near-duplicates at threshold {duplicate_threshold}")
else:
    st.markdown(f"**{n_dupes} pairs found** — sorted by similarity descending")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        dupe_score_min = st.slider(
            "Min similarity to show",
            duplicate_threshold, 1.0, duplicate_threshold, 0.01, key="dupe_score")
    with col_f2:
        # Default cross_unit_only to True so same-unit pairs are hidden by default
        cross_unit_only = st.checkbox(
            "Cross-unit only",
            value=True,
            help="Only show duplicates from different unit codes",
            key="cross_unit",
        )

    shown_dupes = [d for d in near_dupes if d["similarity"] >= dupe_score_min]
    if cross_unit_only:
        shown_dupes = [d for d in shown_dupes if d["unit_a"] != d["unit_b"]]

    st.caption(f"Showing {len(shown_dupes)} pairs")

    for pair in shown_dupes[:50]:
        sim = pair["similarity"]
        badge_cls, badge_lbl = (
            ("sim-high", "NEAR-IDENTICAL") if sim >= 0.97 else
            ("sim-med",  "VERY SIMILAR")   if sim >= 0.94 else
            ("sim-low",  "SIMILAR")
        )
        cross = " ⚡ CROSS-UNIT" if pair["unit_a"] != pair["unit_b"] else ""
        st.markdown(
            f'<div class="dupe-row">'
            f'<span class="sim-badge {badge_cls}">{badge_lbl} {sim:.3f}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#64748b">{cross}</span>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;'
            f'gap:16px;margin-top:10px">'
            f'<div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#7dd3fc">{pair["unit_a"]}</div>'
            f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">'
            f'{pair["unit_title_a"]}</div>'
            f'<div style="font-size:0.72rem;color:#475569;margin-top:2px;'
            f'font-style:italic">Element: {pair["elem_a"]}</div>'
            f'<div style="font-size:0.88rem;color:#e2e8f0;margin-top:6px">'
            f'{pair["text_a"]}</div></div>'
            f'<div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#7dd3fc">{pair["unit_b"]}</div>'
            f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">'
            f'{pair["unit_title_b"]}</div>'
            f'<div style="font-size:0.72rem;color:#475569;margin-top:2px;'
            f'font-style:italic">Element: {pair["elem_b"]}</div>'
            f'<div style="font-size:0.88rem;color:#e2e8f0;margin-top:6px">'
            f'{pair["text_b"]}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    if len(shown_dupes) > 50:
        st.caption(
            f"… and {len(shown_dupes)-50} more. Export cluster analysis CSV for full list.")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

overrides      = st.session_state.get("sa_overrides", {})
final_canonical = set()
for cid, info in clusters.items():
    final_canonical.add(overrides.get(cid, info["canonical_idx"]))

singleton_mask = df_ann["cluster_id"] == -1
export_df      = df_ann[df_ann.index.isin(final_canonical) | singleton_mask].copy()

col_exp1, col_exp2 = st.columns([3, 2])
with col_exp1:
    st.markdown(
        f"Export contains **{len(export_df)} statements** "
        f"({n_total - len(export_df)} removed as non-canonical duplicates)."
    )
with col_exp2:
    st.markdown(
        f"Reduction: **{n_total}** → **{len(export_df)}** "
        f"(−{round(100*(n_total-len(export_df))/max(n_total,1))}%)"
    )

with st.expander("Preview export"):
    st.dataframe(
        export_df[["unit_code","unit_title","element_title",
                   "skill_statement","cluster_id","is_canonical"]].head(20),
        use_container_width=True, hide_index=True,
    )

from core.exporters import to_rsd_rows, to_traceability, to_osmt_rows

c1, c2, c3 = st.columns(3)
c1.download_button(
    "⬇ Deduplicated OSMT CSV",
    to_osmt_rows(export_df).to_csv(index=False).encode(),
    "deduplicated_osmt.csv", "text/csv", use_container_width=True,
)
c2.download_button(
    "⬇ Deduplicated RSD CSV",
    to_rsd_rows(export_df).to_csv(index=False).encode(),
    "deduplicated_rsd.csv", "text/csv", use_container_width=True,
)
c3.download_button(
    "⬇ Cluster analysis CSV",
    df_ann[[c for c in [
        "unit_code","unit_title","element_title","skill_statement",
        "cluster_id","is_canonical","is_singleton",
    ] if c in df_ann.columns]].to_csv(index=False).encode(),
    "cluster_analysis.csv", "text/csv", use_container_width=True,
)
st.caption(
    "**OSMT** — ready for RSD import  |  "
    "**RSD CSV** — matches example output format  |  "
    "**Cluster analysis** — every statement with cluster_id"
)
