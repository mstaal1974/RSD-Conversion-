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

# ── Custom styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .value {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: #38bdf8;
    line-height: 1;
  }
  .metric-card .label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
  }
  .metric-card .delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #4ade80;
    margin-top: 4px;
  }

  .cluster-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .cluster-card.canonical {
    border-left-color: #4ade80;
    background: #0a1f0f;
  }
  .cluster-card .unit-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 8px;
  }
  .cluster-card .stmt {
    font-size: 0.9rem;
    color: #e2e8f0;
    margin-top: 8px;
    line-height: 1.6;
  }
  .dupe-row {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }
  .sim-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
  }
  .sim-high  { background: #450a0a; color: #fca5a5; }
  .sim-med   { background: #431407; color: #fdba74; }
  .sim-low   { background: #1c1917; color: #d6d3d1; }

  .section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin: 32px 0 16px 0;
  }

  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    border: none;
    color: white;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    font-size: 0.85rem;
    padding: 10px 24px;
    border-radius: 8px;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    transform: translateY(-1px);
  }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Semantic Analysis")
st.caption("Cluster, deduplicate and align skill statements using embedding-based similarity")

# ── DB + provider connections ─────────────────────────────────────────────────
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Embedding model")

    embedding_backend = st.radio(
        "Backend",
        ["OpenAI (API — most accurate)", "Sentence-Transformers (local — free)"],
        index=0,
        help=(
            "OpenAI text-embedding-3-small: best accuracy, costs ~$0.002/1000 tokens.\n\n"
            "all-MiniLM-L6-v2: free, runs on CPU, slightly less accurate for domain-specific text."
        ),
    )
    use_openai = embedding_backend.startswith("OpenAI")

    if use_openai:
        embedding_model = st.selectbox(
            "OpenAI model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            help="3-small: fast, cheap, very good. 3-large: slower, costlier, marginally better.",
        )
        if not OPENAI_KEY:
            st.warning("OPENAI_API_KEY not set — switch to Sentence-Transformers.")
    else:
        st.info("Uses `all-MiniLM-L6-v2` — 384-dim, CPU-friendly, no API key needed.")
        embedding_model = "all-MiniLM-L6-v2"

    st.divider()
    st.markdown("### 🎛️ Clustering")

    cluster_threshold = st.slider(
        "Cluster similarity threshold",
        0.70, 0.98, 0.85, 0.01,
        help="Statements ≥ this similarity are grouped. ↑ = tighter groups.",
    )
    duplicate_threshold = st.slider(
        "Near-duplicate threshold",
        0.85, 0.99, 0.92, 0.01,
        help="Pairs ≥ this are flagged as near-duplicates.",
    )
    min_cluster_size = st.slider("Min cluster size", 2, 5, 2)

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

# ── Load data ─────────────────────────────────────────────────────────────────
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

# Estimate API cost
n = len(df_raw)
avg_tokens = 50
est_cost = round(n * avg_tokens * 0.00000002, 4)  # $0.02/1M tokens for 3-small

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.info(f"**{n}** statements loaded")
if use_openai:
    col_info2.info(f"Est. cost: **~${est_cost}** (OpenAI)")
else:
    col_info2.info("Cost: **Free** (local model)")
col_info3.info(f"Est. time: **~{round(n * 0.05 / 60, 1)} min**")

# ── Run analysis ──────────────────────────────────────────────────────────────
def get_embeddings_openai(texts, model, api_key, batch_size=100):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t and t.strip() else "empty" for t in texts[i:i+batch_size]]
        resp = client.embeddings.create(input=batch, model=model)
        all_emb.extend([item.embedding for item in resp.data])
    return np.array(all_emb, dtype=np.float32)


def get_embeddings_local(texts, model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error("sentence-transformers not installed. Add it to requirements.txt.")
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
    dist = np.clip(1.0 - cosine_sim_matrix(embeddings), 0, 2)
    labels = DBSCAN(eps=1.0-threshold, min_samples=min_samples, metric="precomputed").fit_predict(dist)
    return labels


def find_canonical(embeddings, labels, texts):
    clusters = {}
    for label in set(labels) - {-1}:
        mask = labels == label
        idx  = np.where(mask)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1)
        sims = [(float((embeddings[i] / (np.linalg.norm(embeddings[i]) or 1)) @ centroid), int(i)) for i in idx]
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
                    "similarity": round(s, 4),
                    "unit_a": df.iloc[i].get("unit_code",""),
                    "unit_b": df.iloc[j].get("unit_code",""),
                    "text_a": df.iloc[i].get("skill_statement",""),
                    "text_b": df.iloc[j].get("skill_statement",""),
                    "elem_a": df.iloc[i].get("element_title",""),
                    "elem_b": df.iloc[j].get("element_title",""),
                })
    return sorted(pairs, key=lambda x: x["similarity"], reverse=True)


if run_btn:
    if n > 3000:
        st.warning(f"Large dataset ({n} statements). Consider filtering by unit code prefix to speed up analysis.")

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

        df_ann = df_raw.copy().reset_index(drop=True)
        df_ann["cluster_id"]   = labels
        canonical_set          = {v["canonical_idx"] for v in clusters.values()}
        df_ann["is_canonical"] = df_ann.index.isin(canonical_set)
        df_ann["is_singleton"] = labels == -1

        st.session_state.update({
            "sa_results":      True,
            "sa_df_ann":       df_ann,
            "sa_sim_matrix":   sim_matrix,
            "sa_clusters":     clusters,
            "sa_near_dupes":   near_dupes,
            "sa_labels":       labels,
            "sa_embeddings":   embeddings,
            "sa_overrides":    {},
        })

        progress.progress(1.0)
        status.write("✅ Analysis complete")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ── Check results ─────────────────────────────────────────────────────────────
if not st.session_state.get("sa_results"):
    st.info("Configure settings in the sidebar and click **▶ Run analysis**.")
    st.stop()

df_ann     = st.session_state["sa_df_ann"]
sim_matrix = st.session_state["sa_sim_matrix"]
clusters   = st.session_state["sa_clusters"]
near_dupes = st.session_state["sa_near_dupes"]
labels     = st.session_state["sa_labels"]

n_total      = len(df_ann)
n_clustered  = int((labels >= 0).sum())
n_singletons = int((labels == -1).sum())
n_clusters   = len(clusters)
n_dupes      = len(near_dupes)
n_reduction  = n_clustered - n_clusters
pct_reduction = round(100 * n_reduction / max(n_total, 1))

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

cols = st.columns(6)
metrics = [
    ("Total statements", n_total, None),
    ("Clusters found",   n_clusters, None),
    ("Clustered",        n_clustered, f"{round(100*n_clustered/max(n_total,1))}%"),
    ("Singletons",       n_singletons, None),
    ("Near-dupes",       n_dupes, None),
    ("Potential saving", n_reduction, f"−{pct_reduction}%"),
]
for col, (label, value, delta) in zip(cols, metrics):
    with col:
        delta_html = f'<div class="delta">{delta}</div>' if delta else ""
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="value">{value}</div>'
            f'<div class="label">{label}</div>'
            f'{delta_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f"<br>Keeping one canonical per cluster reduces **{n_total} → {n_singletons + n_clusters}** statements.",
    unsafe_allow_html=True,
)

# ── Cluster size distribution ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster size distribution</div>', unsafe_allow_html=True)

if clusters:
    sizes = pd.DataFrame([
        {"Cluster": f"C{k}", "Size": v["size"], "Canonical": v["canonical_text"][:60]+"…"}
        for k, v in sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)
    ])

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        raw_sizes = [v["size"] for v in clusters.values()]
        bin_order = ["2-3", "4-5", "6-10", "11-20", "21-50", "51+"]
        def size_bin(s):
            if s <= 3:  return "2-3"
            if s <= 5:  return "4-5"
            if s <= 10: return "6-10"
            if s <= 20: return "11-20"
            if s <= 50: return "21-50"
            return "51+"
        bin_counts = {b: 0 for b in bin_order}
        for s in raw_sizes:
            bin_counts[size_bin(s)] += 1
        hist_data = pd.DataFrame([
            {"Range": b, "Clusters": bin_counts[b]}
            for b in bin_order if bin_counts[b] > 0
        ])
        chart = alt.Chart(hist_data).mark_bar(
            color="#38bdf8",
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        ).encode(
            x=alt.X("Range:N", sort=bin_order, title="Cluster size (statements)",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Clusters:Q", title="Number of clusters"),
            tooltip=["Range", "Clusters"],
        ).properties(height=220, title="Cluster size distribution")
        st.altair_chart(chart, use_container_width=True)
        median_size = int(pd.Series(raw_sizes).median())
        st.caption(f"Median: {median_size} · Largest cluster: {max(raw_sizes)} statements")

    with col_table:
        st.dataframe(
            sizes.head(15),
            use_container_width=True,
            column_config={
                "Cluster":   st.column_config.TextColumn("ID", width="small"),
                "Size":      st.column_config.NumberColumn("Size"),
                "Canonical": st.column_config.TextColumn("Canonical statement"),
            },
            hide_index=True,
        )

# ── Similarity score distribution ────────────────────────────────────────────
st.markdown('<div class="section-header">Pairwise similarity distribution</div>', unsafe_allow_html=True)

# Sample upper triangle — avoid OOM on large matrices
n_show = min(500, len(df_ann))
sim_slice = sim_matrix[:n_show, :n_show]
upper = sim_slice[np.triu_indices(n_show, k=1)]

hist_vals, hist_bins = np.histogram(upper, bins=40, range=(0, 1))
hist_df = pd.DataFrame({
    "similarity": [float((hist_bins[i]+hist_bins[i+1])/2) for i in range(len(hist_vals))],
    "count":      [int(v) for v in hist_vals],
})
# Drop empty bins to avoid infinite extent warnings
hist_df = hist_df[hist_df["count"] > 0]
score_chart = alt.Chart(hist_df).mark_bar(
    color="#7c3aed",
    cornerRadiusTopLeft=2,
    cornerRadiusTopRight=2,
).encode(
    x=alt.X("similarity:Q", bin=False, title="Cosine similarity", scale=alt.Scale(domain=[0,1])),
    y=alt.Y("count:Q", title="Pair count"),
    tooltip=[
        alt.Tooltip("similarity:Q", format=".2f"),
        alt.Tooltip("count:Q"),
    ],
).properties(
    height=180,
    title=f"Distribution of pairwise similarities (first {n_show} statements)",
)
# Add threshold lines
cluster_line = alt.Chart(pd.DataFrame({"x": [cluster_threshold]})).mark_rule(
    color="#38bdf8", strokeDash=[4,2], size=1.5
).encode(x="x:Q")
dupe_line = alt.Chart(pd.DataFrame({"x": [duplicate_threshold]})).mark_rule(
    color="#f43f5e", strokeDash=[4,2], size=1.5
).encode(x="x:Q")

combined = (score_chart + cluster_line + dupe_line).resolve_scale(x="shared")
st.altair_chart(combined, use_container_width=True)
st.caption(f"🔵 Cluster threshold ({cluster_threshold})  🔴 Near-duplicate threshold ({duplicate_threshold})")

# ── Similarity heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Similarity heatmap</div>', unsafe_allow_html=True)

heatmap_n = st.slider("Statements to show in heatmap", 20, min(100, len(df_ann)), min(50, len(df_ann)), 5)
sim_h = sim_matrix[:heatmap_n, :heatmap_n]

rows_heat = []
for i in range(heatmap_n):
    for j in range(heatmap_n):
        uc = df_ann.iloc[i].get("unit_code","")
        rows_heat.append({
            "x": f"{j}",
            "y": f"{i}",
            "sim": float(sim_h[i, j]),
            "unit_x": df_ann.iloc[j].get("unit_code",""),
            "unit_y": uc,
            "elem_y": df_ann.iloc[i].get("element_title","")[:40],
        })

heat_df = pd.DataFrame(rows_heat)
heatmap = alt.Chart(heat_df).mark_rect().encode(
    x=alt.X("x:N", axis=alt.Axis(labels=False, ticks=False, title="Statement index →")),
    y=alt.Y("y:N", sort=None, axis=alt.Axis(labels=False, ticks=False, title="↑ Statement index")),
    color=alt.Color(
        "sim:Q",
        scale=alt.Scale(scheme="viridis"),
        legend=alt.Legend(title="Similarity"),
    ),
    tooltip=[
        alt.Tooltip("unit_y:N", title="Unit"),
        alt.Tooltip("elem_y:N", title="Element"),
        alt.Tooltip("unit_x:N", title="Compared with"),
        alt.Tooltip("sim:Q", format=".3f", title="Similarity"),
    ],
).properties(
    width=600,
    height=600,
    title=f"Cosine similarity heatmap — first {heatmap_n} statements (diagonal = 1.0)",
).interactive(False)
st.altair_chart(heatmap, use_container_width=True)
st.caption(
    "Bright diagonal = each statement is identical to itself. "
    "Bright off-diagonal clusters reveal groups of similar statements."
)

# ── Cluster browser ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster browser</div>', unsafe_allow_html=True)
st.caption("Each cluster groups semantically similar statements. ⭐ = canonical (closest to cluster centre). Override the canonical below each cluster.")

if not clusters:
    st.info("No clusters found — try lowering the similarity threshold.")
else:
    overrides = st.session_state.get("sa_overrides", {})

    # Sort largest → smallest
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)

    # Pagination
    per_page = 10
    total_pages = max(1, (len(sorted_clusters) + per_page - 1) // per_page)
    page = st.number_input("Page", 1, total_pages, 1, key="cluster_page")
    page_clusters = sorted_clusters[(page-1)*per_page : page*per_page]

    for cluster_id, info in page_clusters:
        members = df_ann[df_ann.index.isin(info["member_indices"])]
        canon_idx = overrides.get(cluster_id, info["canonical_idx"])

        with st.expander(
            f"**Cluster {cluster_id}** — {info['size']} statements  |  "
            f"⭐ {info['canonical_text'][:70]}…"
        ):
            for _, mem in members.iterrows():
                is_canon = mem.name == canon_idx
                unit  = mem.get("unit_code", "")
                elem  = mem.get("element_title", "")
                stmt  = mem.get("skill_statement", "")
                card_class = "cluster-card canonical" if is_canon else "cluster-card"
                badge = "⭐ CANONICAL &nbsp;" if is_canon else "○ &nbsp;"
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<span class="unit-badge">{unit}</span>'
                    f'<strong style="color:#94a3b8;font-size:0.8rem">{badge}{elem}</strong>'
                    f'<div class="stmt">{stmt}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Override selector
            opts = {
                f"[{df_ann.loc[i,'unit_code']}] {df_ann.loc[i,'skill_statement'][:70]}…": i
                for i in info["member_indices"]
            }
            opt_values = list(opts.values())
            safe_index = opt_values.index(canon_idx) if canon_idx in opt_values else 0
            chosen_label = st.selectbox(
                "Set canonical",
                list(opts.keys()),
                index=safe_index,
                key=f"canon_sel_{cluster_id}",
            )
            chosen_idx = opts[chosen_label]
            if chosen_idx != info["canonical_idx"]:
                overrides[cluster_id] = chosen_idx
                st.session_state["sa_overrides"] = overrides
                st.success("Override saved ✅")

    st.caption(f"Showing page {page} of {total_pages} — {len(sorted_clusters)} clusters total")

# ── Near-duplicate pairs ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Near-duplicate pairs</div>', unsafe_allow_html=True)
st.caption(f"Statement pairs with cosine similarity ≥ **{duplicate_threshold}** — review and consolidate these")

if not near_dupes:
    st.success(f"✅ No near-duplicates at threshold {duplicate_threshold}")
else:
    st.markdown(f"**{n_dupes} pairs found** — sorted by similarity descending")

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        dupe_score_min = st.slider("Min similarity to show", duplicate_threshold, 1.0, duplicate_threshold, 0.01, key="dupe_score")
    with col_f2:
        cross_unit_only = st.checkbox("Cross-unit only", help="Show only duplicates from different unit codes")

    shown_dupes = [d for d in near_dupes if d["similarity"] >= dupe_score_min]
    if cross_unit_only:
        shown_dupes = [d for d in shown_dupes if d["unit_a"] != d["unit_b"]]

    st.caption(f"Showing {len(shown_dupes)} pairs")

    for i, pair in enumerate(shown_dupes[:50]):
        sim = pair["similarity"]
        if sim >= 0.97:
            badge_cls, badge_lbl = "sim-high", "NEAR-IDENTICAL"
        elif sim >= 0.94:
            badge_cls, badge_lbl = "sim-med", "VERY SIMILAR"
        else:
            badge_cls, badge_lbl = "sim-low", "SIMILAR"

        cross = " ⚡ CROSS-UNIT" if pair["unit_a"] != pair["unit_b"] else ""

        st.markdown(
            f'<div class="dupe-row">'
            f'<span class="sim-badge {badge_cls}">{badge_lbl} {sim:.3f}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#64748b">{cross}</span>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px">'
            f'<div>'
            f'  <div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#7dd3fc">'
            f'    {pair["unit_a"]} / {pair["elem_a"]}</div>'
            f'  <div style="font-size:0.88rem;color:#e2e8f0;margin-top:4px">{pair["text_a"]}</div>'
            f'</div>'
            f'<div>'
            f'  <div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#7dd3fc">'
            f'    {pair["unit_b"]} / {pair["elem_b"]}</div>'
            f'  <div style="font-size:0.88rem;color:#e2e8f0;margin-top:4px">{pair["text_b"]}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(shown_dupes) > 50:
        st.caption(f"… and {len(shown_dupes)-50} more. Export the cluster analysis CSV for full list.")

# ── Build export dataframe ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

overrides = st.session_state.get("sa_overrides", {})
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
    st.markdown(f"Reduction: **{n_total}** → **{len(export_df)}** (−{round(100*(n_total-len(export_df))/max(n_total,1))}%)")

# Preview
with st.expander("Preview export"):
    st.dataframe(
        export_df[["unit_code","unit_title","element_title","skill_statement","cluster_id","is_canonical"]].head(20),
        use_container_width=True, hide_index=True,
    )

from core.exporters import to_rsd_rows, to_traceability, to_osmt_rows

c1, c2, c3 = st.columns(3)
c1.download_button(
    "⬇ Deduplicated OSMT CSV",
    to_osmt_rows(export_df).to_csv(index=False).encode(),
    "deduplicated_osmt.csv", "text/csv",
    use_container_width=True,
)
c2.download_button(
    "⬇ Deduplicated RSD CSV",
    to_rsd_rows(export_df).to_csv(index=False).encode(),
    "deduplicated_rsd.csv", "text/csv",
    use_container_width=True,
)
c3.download_button(
    "⬇ Cluster analysis CSV",
    df_ann[[c for c in [
        "unit_code","unit_title","element_title","skill_statement",
        "cluster_id","is_canonical","is_singleton",
    ] if c in df_ann.columns]].to_csv(index=False).encode(),
    "cluster_analysis.csv", "text/csv",
    use_container_width=True,
)
st.caption(
    "**OSMT** — ready for RSD import  |  "
    "**RSD CSV** — matches example output format  |  "
    "**Cluster analysis** — every statement with cluster_id for review in Excel"
)
