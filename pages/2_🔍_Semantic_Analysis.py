"""
pages/2_🔍_Semantic_Analysis.py

Scalable semantic analysis — handles 50,000+ statements.

Key changes vs original:
  - Never builds full N×N similarity matrix (OOM at >5k statements)
  - TF-IDF embeddings (no API cost, fast, sparse)
  - MiniBatchKMeans clustering (scales to 100k+)
  - LSH-based near-duplicate detection (approximate, O(n) not O(n²))
  - Unit-level heatmap uses sampled/aggregated similarity
  - OpenAI embeddings optional — used only on filtered subsets
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
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
  .metric-card{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;
    border-radius:12px;padding:20px 24px;text-align:center}
  .metric-card .value{font-family:'DM Mono',monospace;font-size:2.2rem;font-weight:500;
    color:#38bdf8;line-height:1}
  .metric-card .label{font-size:0.75rem;color:#94a3b8;text-transform:uppercase;
    letter-spacing:0.1em;margin-top:6px}
  .metric-card .delta{font-family:'DM Mono',monospace;font-size:0.85rem;
    color:#4ade80;margin-top:4px}
  .cluster-card{background:#0f172a;border:1px solid #1e3a5f;border-left:3px solid #38bdf8;
    border-radius:8px;padding:16px 20px;margin-bottom:12px}
  .cluster-card.canonical{border-left-color:#4ade80;background:#0a1f0f}
  .cluster-card .unit-badge{display:inline-block;background:#1e3a5f;color:#7dd3fc;
    font-family:'DM Mono',monospace;font-size:0.72rem;padding:2px 8px;
    border-radius:4px;margin-right:8px}
  .cluster-card .stmt{font-size:0.9rem;color:#e2e8f0;margin-top:8px;line-height:1.6}
  .dupe-row{background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;
    padding:14px 18px;margin-bottom:10px}
  .sim-badge{display:inline-block;font-family:'DM Mono',monospace;font-size:0.8rem;
    padding:3px 10px;border-radius:20px;margin-bottom:8px}
  .sim-high{background:#450a0a;color:#fca5a5}
  .sim-med{background:#431407;color:#fdba74}
  .sim-low{background:#1c1917;color:#d6d3d1}
  .section-header{font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.15em;
    text-transform:uppercase;color:#475569;border-bottom:1px solid #1e293b;
    padding-bottom:8px;margin:32px 0 16px 0}
  .stButton>button[kind="primary"]{background:linear-gradient(135deg,#0ea5e9,#0284c7);
    border:none;color:white;font-family:'DM Mono',monospace;letter-spacing:0.05em;
    font-size:0.85rem;padding:10px 24px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Semantic Analysis")
st.caption("Scalable cluster, deduplicate and alignment — handles 50,000+ statements")

def _secret(k, d=""):
    try: return st.secrets.get(k, os.getenv(k, d)) or d
    except: return os.getenv(k, d) or d

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
        ["TF-IDF (fast — no API cost)",
         "OpenAI (API — most accurate)",
         "Sentence-Transformers (local — free)"],
        index=0,
        help=(
            "TF-IDF: instant, free, handles 50k+ statements. Best for large runs.\n\n"
            "OpenAI: most accurate but costs money and only practical on filtered subsets.\n\n"
            "Sentence-Transformers: free but slow on CPU for large datasets."
        ),
    )
    use_tfidf  = embedding_backend.startswith("TF-IDF")
    use_openai = embedding_backend.startswith("OpenAI")

    if use_openai:
        embedding_model = st.selectbox(
            "OpenAI model",
            ["text-embedding-3-small","text-embedding-3-large"],
        )
        if not OPENAI_KEY:
            st.warning("OPENAI_API_KEY not set.")
        st.warning("OpenAI mode — filter to <2000 statements for best results.")
    elif use_tfidf:
        st.info("TF-IDF mode — handles all 47k+ statements efficiently.")
        embedding_model = "tfidf"
    else:
        embedding_model = "all-MiniLM-L6-v2"
        st.warning("Sentence-Transformers — filter to <5000 statements.")

    st.divider()
    st.markdown("### 🎛️ Clustering")
    cluster_threshold   = st.slider("Cluster similarity threshold", 0.50, 0.98, 0.75, 0.01)
    duplicate_threshold = st.slider("Near-duplicate threshold", 0.80, 0.99, 0.92, 0.01)
    n_clusters_kmeans   = st.slider("Number of clusters (K-Means)", 10, 200, 50, 5,
        help="MiniBatchKMeans — scales to 50k+ statements")

    st.divider()
    st.markdown("### 🔎 Filter source")

    try:
        with engine.connect() as conn:
            runs = conn.execute(text(
                "SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC"
            )).mappings().all()
        runs = [dict(r) for r in runs]
    except Exception:
        runs = []

    run_options = ["All runs"] + [f"{str(r['id'])[:8]}… {r['source_filename']}" for r in runs]
    run_ids     = [None] + [str(r["id"]) for r in runs]
    sel_run_lbl = st.selectbox("Source run", run_options)
    sel_run_id  = run_ids[run_options.index(sel_run_lbl)]
    unit_filter = st.text_input("Unit code prefix", placeholder="e.g. BSB")
    tp_filter   = st.text_input("TP code prefix", placeholder="e.g. MSL")

    st.divider()
    run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_statements(run_id, unit_prefix, tp_prefix):
    filters = ["skill_statement IS NOT NULL", "skill_statement != ''"]
    params: dict = {}
    if run_id:
        filters.append("run_id = :rid")
        params["rid"] = run_id
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM rsd_skill_records WHERE {' AND '.join(filters)} "
                 f"ORDER BY unit_code, row_index"),
            params,
        ).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    if not df.empty and unit_prefix:
        df = df[df["unit_code"].fillna("").str.upper().str.startswith(unit_prefix.upper())]
    if not df.empty and tp_prefix:
        df = df[df["tp_code"].fillna("").str.upper().str.startswith(tp_prefix.upper())]
    return df

df_raw = load_statements(sel_run_id, unit_filter, tp_filter)

if df_raw.empty:
    st.info("No skill statements found. Run some batches first.")
    st.stop()

n = len(df_raw)
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.info(f"**{n:,}** statements loaded")
col_info2.info(f"**{df_raw['unit_code'].nunique():,}** UOCs")
col_info3.info(f"**{df_raw['tp_code'].nunique() if 'tp_code' in df_raw else '—'}** TPs")

if not use_tfidf and n > 5000:
    st.warning(
        f"⚠️ {n:,} statements is large for {embedding_backend}. "
        "Consider using **TF-IDF** mode or filtering by unit/TP prefix. "
        "TF-IDF handles all 47k+ statements in seconds."
    )

# ── Embedding functions ───────────────────────────────────────────────────────
def get_embeddings_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec   = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                            max_features=20000, min_df=2, sublinear_tf=True)
    mat   = vec.fit_transform(texts)
    dense = normalize(mat).toarray().astype(np.float32)
    return dense, vec

def get_embeddings_openai(texts, model, api_key, batch_size=100):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t and t.strip() else "empty" for t in texts[i:i+batch_size]]
        resp  = client.embeddings.create(input=batch, model=model)
        all_emb.extend([item.embedding for item in resp.data])
    return np.array(all_emb, dtype=np.float32), None

def get_embeddings_local(texts, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        st.error("sentence-transformers not installed. Use TF-IDF or OpenAI backend.")
        st.stop()
    model = SentenceTransformer(model_name)
    clean = [t if t and t.strip() else "empty" for t in texts]
    emb   = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32), None

# ── Scalable clustering — MiniBatchKMeans ────────────────────────────────────
def run_kmeans(embeddings, n_clust):
    from sklearn.cluster import MiniBatchKMeans
    k  = min(n_clust, len(embeddings)//3, len(embeddings)-1)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3,
                         batch_size=min(2048, len(embeddings)))
    labels = km.fit_predict(embeddings)
    return labels, km

def find_canonical_kmeans(embeddings, labels, texts, km):
    """Find the statement closest to each cluster centroid."""
    clusters = {}
    centroids = km.cluster_centers_  # already unit vectors (normalized)
    for label in sorted(set(labels)):
        mask = labels == label
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        c = centroids[label]
        c = c / (np.linalg.norm(c) or 1)
        sims = embeddings[idx] @ c
        best = idx[int(np.argmax(sims))]
        clusters[int(label)] = {
            "canonical_idx":  int(best),
            "canonical_text": texts[best],
            "size":           int(mask.sum()),
            "member_indices": idx.tolist(),
        }
    return clusters

# ── Scalable near-duplicate detection — LSH banding ─────────────────────────
def find_near_dupes_lsh(embeddings, df, threshold, max_pairs=2000):
    """
    Approximate near-duplicate detection using random projection LSH.
    O(n * n_bands) not O(n²) — handles 50k+ statements.
    """
    n, d    = embeddings.shape
    n_bits  = 128
    n_bands = 16
    rows_per_band = n_bits // n_bands

    # Random projection binary hash
    rng      = np.random.RandomState(42)
    planes   = rng.randn(d, n_bits).astype(np.float32)
    proj     = (embeddings @ planes) > 0  # n × n_bits bool

    # Band buckets
    candidates = set()
    for b in range(n_bands):
        start = b * rows_per_band
        end   = start + rows_per_band
        band  = proj[:, start:end]
        buckets: dict[tuple, list[int]] = {}
        for i, row in enumerate(band):
            key = tuple(row.tolist())
            buckets.setdefault(key, []).append(i)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for ai in range(len(bucket)):
                    for bi in range(ai+1, len(bucket)):
                        candidates.add((min(bucket[ai], bucket[bi]),
                                        max(bucket[ai], bucket[bi])))

    # Verify candidates with exact cosine similarity
    pairs = []
    cand_list = list(candidates)
    if len(cand_list) > 200_000:
        # Too many candidates — sample
        rng2 = np.random.RandomState(0)
        cand_list = [cand_list[i]
                     for i in rng2.choice(len(cand_list), 200_000, replace=False)]

    for i, j in cand_list:
        s = float(embeddings[i] @ embeddings[j])
        if s >= threshold:
            ui = df.iloc[i].get("unit_code","")
            uj = df.iloc[j].get("unit_code","")
            pairs.append({
                "idx_a": i, "idx_b": j,
                "similarity":   round(s, 4),
                "unit_a":       ui,
                "unit_title_a": df.iloc[i].get("unit_title",""),
                "unit_b":       uj,
                "unit_title_b": df.iloc[j].get("unit_title",""),
                "text_a":  df.iloc[i].get("skill_statement",""),
                "text_b":  df.iloc[j].get("skill_statement",""),
                "elem_a":  df.iloc[i].get("element_title",""),
                "elem_b":  df.iloc[j].get("element_title",""),
            })
        if len(pairs) >= max_pairs:
            break

    return sorted(pairs, key=lambda x: x["similarity"], reverse=True)

# ── Unit-level heatmap — chunked, no full matrix ─────────────────────────────
def compute_unit_heatmap(embeddings, unit_codes_arr, top_units):
    """
    Compute average pairwise similarity per unit pair using chunked dot products.
    Never builds full N×N matrix.
    """
    unit_rows_out = []
    unit_idx_map  = {u: np.where(unit_codes_arr == u)[0] for u in top_units}

    for ua in top_units:
        idxs_a = unit_idx_map[ua]
        emb_a  = embeddings[idxs_a]          # (na, d)
        for ub in top_units:
            idxs_b = unit_idx_map[ub]
            emb_b  = embeddings[idxs_b]       # (nb, d)
            # Chunked dot product: mean of emb_a @ emb_b.T
            avg = float((emb_a @ emb_b.T).mean())
            unit_rows_out.append({
                "Unit A": ua, "Unit B": ub,
                "Avg similarity": round(avg, 3),
            })
    return pd.DataFrame(unit_rows_out)

# ── Main run ──────────────────────────────────────────────────────────────────
if run_btn:
    progress = st.progress(0)
    status   = st.empty()

    try:
        texts = df_raw["skill_statement"].fillna("").tolist()

        status.write(f"⏳ Generating embeddings for {n:,} statements…")
        progress.progress(0.05)

        if use_tfidf:
            embeddings, tfidf_vec = get_embeddings_tfidf(texts)
        elif use_openai:
            if not OPENAI_KEY:
                st.error("OpenAI API key not configured.")
                st.stop()
            embeddings, tfidf_vec = get_embeddings_openai(texts, embedding_model, OPENAI_KEY)
        else:
            embeddings, tfidf_vec = get_embeddings_local(texts, embedding_model)

        progress.progress(0.30)
        status.write(f"⏳ Clustering {n:,} statements with MiniBatchKMeans…")

        labels, km = run_kmeans(embeddings, n_clusters_kmeans)

        progress.progress(0.55)
        status.write("⏳ Finding canonical statements…")

        clusters = find_canonical_kmeans(embeddings, labels, texts, km)

        progress.progress(0.70)
        status.write(f"⏳ Finding near-duplicates (LSH approximate, {n:,} statements)…")

        near_dupes = find_near_dupes_lsh(
            embeddings, df_raw.reset_index(drop=True),
            threshold=duplicate_threshold, max_pairs=5000,
        )

        progress.progress(0.88)
        status.write("⏳ Computing unit similarity heatmap…")

        df_ann             = df_raw.copy().reset_index(drop=True)
        df_ann["cluster_id"]   = labels
        canonical_set          = {v["canonical_idx"] for v in clusters.values()}
        df_ann["is_canonical"] = df_ann.index.isin(canonical_set)
        df_ann["is_singleton"] = False  # KMeans assigns every point

        # Unit heatmap — top 40 by avg outward similarity
        unit_codes_arr = df_ann["unit_code"].fillna("(blank)").values
        all_units      = list(set(unit_codes_arr))
        unit_idx_map   = {u: np.where(unit_codes_arr == u)[0] for u in all_units}

        # Avg outward similarity per unit (chunked)
        unit_avg_sim = {}
        for u, idxs in unit_idx_map.items():
            emb_u     = embeddings[idxs]
            other_idx = np.array([i for i in range(n) if i not in set(idxs.tolist())])
            if len(other_idx) == 0:
                unit_avg_sim[u] = 0.0
                continue
            # Sample other_idx if very large
            if len(other_idx) > 2000:
                other_idx = other_idx[
                    np.random.RandomState(42).choice(
                        len(other_idx), 2000, replace=False)]
            emb_other = embeddings[other_idx]
            unit_avg_sim[u] = float((emb_u @ emb_other.T).mean())

        top40 = sorted(unit_avg_sim, key=lambda u: unit_avg_sim[u], reverse=True)[:40]
        unit_heat_df = compute_unit_heatmap(embeddings, unit_codes_arr, top40)

        st.session_state.update({
            "sa_results":      True,
            "sa_df_ann":       df_ann,
            "sa_embeddings":   embeddings,
            "sa_clusters":     clusters,
            "sa_near_dupes":   near_dupes,
            "sa_labels":       labels,
            "sa_overrides":    {},
            "sa_unit_heat_df": unit_heat_df,
            "sa_unit_avg_sim": unit_avg_sim,
            "sa_top40":        top40,
            "sa_unit_codes_arr": unit_codes_arr,
        })

        progress.progress(1.0)
        status.success(
            f"✅ Analysis complete — {len(clusters)} clusters, "
            f"{len(near_dupes)} near-duplicate pairs"
        )

    except Exception as e:
        import traceback
        st.error(f"Analysis failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

if not st.session_state.get("sa_results"):
    st.info("Configure settings in the sidebar and click **▶ Run analysis**.")
    st.stop()

df_ann         = st.session_state["sa_df_ann"]
embeddings     = st.session_state["sa_embeddings"]
clusters       = st.session_state["sa_clusters"]
near_dupes     = st.session_state["sa_near_dupes"]
labels         = st.session_state["sa_labels"]
unit_heat_df   = st.session_state["sa_unit_heat_df"]
unit_avg_sim   = st.session_state["sa_unit_avg_sim"]
top40          = st.session_state["sa_top40"]
unit_codes_arr = st.session_state["sa_unit_codes_arr"]

n_total       = len(df_ann)
n_clusters_n  = len(clusters)
n_dupes       = len(near_dupes)
n_reduction   = n_total - n_clusters_n
pct_reduction = round(100 * n_reduction / max(n_total, 1))

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
cols    = st.columns(5)
metrics = [
    ("Total statements",  n_total,        None),
    ("Clusters found",    n_clusters_n,   None),
    ("Near-dupes found",  n_dupes,        None),
    ("Potential saving",  n_reduction,    f"−{pct_reduction}%"),
    ("UOCs analysed",     df_ann["unit_code"].nunique(), None),
]
for col, (label, value, delta) in zip(cols, metrics):
    with col:
        delta_html = f'<div class="delta">{delta}</div>' if delta else ""
        st.markdown(
            f'<div class="metric-card"><div class="value">{value:,}</div>'
            f'<div class="label">{label}</div>{delta_html}</div>',
            unsafe_allow_html=True,
        )

# ── Cluster size distribution ─────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster size distribution</div>', unsafe_allow_html=True)
raw_sizes = [v["size"] for v in clusters.values()]
bin_order = ["1","2-3","4-5","6-10","11-20","21-50","51+"]
def size_bin(s):
    if s==1: return "1"
    if s<=3: return "2-3"
    if s<=5: return "4-5"
    if s<=10: return "6-10"
    if s<=20: return "11-20"
    if s<=50: return "21-50"
    return "51+"
bin_counts = {b:0 for b in bin_order}
for s in raw_sizes: bin_counts[size_bin(s)] += 1
hist_data = pd.DataFrame([
    {"Range":b,"Clusters":bin_counts[b]} for b in bin_order if bin_counts[b]>0])
chart = alt.Chart(hist_data).mark_bar(
    color="#38bdf8",cornerRadiusTopLeft=4,cornerRadiusTopRight=4,
).encode(
    x=alt.X("Range:N",sort=bin_order,title="Cluster size",
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Clusters:Q",title="Number of clusters"),
    tooltip=["Range","Clusters"],
).properties(height=220,title="Cluster size distribution")
st.altair_chart(chart, use_container_width=True)
st.caption(
    f"Median cluster size: {int(pd.Series(raw_sizes).median())} · "
    f"Largest: {max(raw_sizes)} statements · "
    f"Method: MiniBatchKMeans (scalable to 100k+)"
)

# ── Similarity heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Similarity heatmap</div>', unsafe_allow_html=True)

heatmap_tab1, heatmap_tab2 = st.tabs([
    "📦 Unit-level (overview)",
    "🔬 Statement-level (drill-down)",
])

with heatmap_tab1:
    st.caption(
        "Top 40 units by **average semantic similarity to all other units**. "
        "Computed without building a full N×N matrix."
    )
    top40_sorted = sorted(top40)
    label_map    = {u: u[-7:] if len(u) > 7 else u for u in top40_sorted}
    heat_df_plot = unit_heat_df.copy()
    heat_df_plot["A_label"] = heat_df_plot["Unit A"].map(label_map)
    heat_df_plot["B_label"] = heat_df_plot["Unit B"].map(label_map)

    unit_heatmap = alt.Chart(heat_df_plot).mark_rect().encode(
        x=alt.X("B_label:N", sort=None,
                axis=alt.Axis(labelAngle=-45, title="Unit →")),
        y=alt.Y("A_label:N", sort=None,
                axis=alt.Axis(title="↑ Unit")),
        color=alt.Color("Avg similarity:Q",
            scale=alt.Scale(scheme="viridis", domain=[0.3,1.0]),
            legend=alt.Legend(title="Avg similarity")),
        tooltip=[
            alt.Tooltip("Unit A:N"),
            alt.Tooltip("Unit B:N"),
            alt.Tooltip("Avg similarity:Q", format=".3f"),
        ],
    ).properties(
        width=600, height=600,
        title=f"Unit-level similarity — top {len(top40_sorted)} units by semantic similarity",
    )
    st.altair_chart(unit_heatmap, use_container_width=True)

    with st.expander("Unit similarity rankings"):
        sim_rank = pd.DataFrame([
            {"Unit": u, "Avg outward similarity": round(unit_avg_sim[u], 4)}
            for u in top40
        ]).sort_values("Avg outward similarity", ascending=False)
        st.dataframe(sim_rank, use_container_width=True, hide_index=True,
            column_config={
                "Avg outward similarity": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.4f"),
            })

    cross = heat_df_plot[
        (heat_df_plot["Unit A"] != heat_df_plot["Unit B"]) &
        (heat_df_plot["Avg similarity"] >= cluster_threshold)
    ].sort_values("Avg similarity", ascending=False).head(20)
    if len(cross):
        with st.expander(
            f"⚡ {len(cross)} high-overlap unit pairs (avg sim ≥ {cluster_threshold})"
        ):
            st.dataframe(cross[["Unit A","Unit B","Avg similarity"]],
                         use_container_width=True, hide_index=True)

with heatmap_tab2:
    st.caption("Select two units to compare statements head-to-head.")
    col_ua, col_ub = st.columns(2)
    with col_ua:
        unit_a_sel = st.selectbox("Unit A", top40_sorted, key="heat_ua")
    with col_ub:
        remaining  = [u for u in top40_sorted if u != unit_a_sel]
        unit_b_sel = st.selectbox("Unit B", remaining, key="heat_ub") \
                     if remaining else unit_a_sel

    idxs_a2 = np.where(unit_codes_arr == unit_a_sel)[0][:30]
    idxs_b2 = np.where(unit_codes_arr == unit_b_sel)[0][:30]

    if len(idxs_a2) == 0 or len(idxs_b2) == 0:
        st.info("No statements found.")
    else:
        submat = embeddings[idxs_a2] @ embeddings[idxs_b2].T
        drill_rows = []
        for ii, ia in enumerate(idxs_a2):
            for jj, ib in enumerate(idxs_b2):
                drill_rows.append({
                    "A":   df_ann.iloc[ia].get("element_title","")[:35],
                    "B":   df_ann.iloc[ib].get("element_title","")[:35],
                    "sim": round(float(submat[ii,jj]), 3),
                })
        drill_df = pd.DataFrame(drill_rows)
        a_labels = sorted(drill_df["A"].unique())
        b_labels = sorted(drill_df["B"].unique())
        drill_chart = alt.Chart(drill_df).mark_rect().encode(
            x=alt.X("B:N", sort=b_labels,
                    axis=alt.Axis(labelAngle=-45, title=f"{unit_b_sel} →")),
            y=alt.Y("A:N", sort=a_labels,
                    axis=alt.Axis(title=f"↑ {unit_a_sel}")),
            color=alt.Color("sim:Q",
                scale=alt.Scale(scheme="viridis", domain=[0,1]),
                legend=alt.Legend(title="Similarity")),
            tooltip=[
                alt.Tooltip("A:N", title=unit_a_sel),
                alt.Tooltip("B:N", title=unit_b_sel),
                alt.Tooltip("sim:Q", format=".3f"),
            ],
        ).properties(width=600, height=500,
            title=f"{unit_a_sel} vs {unit_b_sel} — element similarity")
        st.altair_chart(drill_chart, use_container_width=True)

        hot = drill_df[drill_df["sim"] >= cluster_threshold].sort_values(
            "sim", ascending=False)
        if len(hot):
            st.markdown(f"**{len(hot)} pairs above {cluster_threshold}:**")
            st.dataframe(hot, use_container_width=True, hide_index=True,
                column_config={
                    "A": st.column_config.TextColumn(unit_a_sel),
                    "B": st.column_config.TextColumn(unit_b_sel),
                    "sim": st.column_config.NumberColumn("Similarity", format="%.3f"),
                })

# ── Cluster browser ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Cluster browser</div>', unsafe_allow_html=True)

if not clusters:
    st.info("No clusters found.")
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
            "units": ", ".join(unit_codes[:4]) + (" …" if len(unit_codes)>4 else ""),
            "tp": ", ".join(tp_prefixes),
            "cross_tp": cross_tp, "info": info,
        })
    cluster_rows.sort(key=lambda x: x["size"], reverse=True)

    cb_tab1, cb_tab2 = st.tabs(["📋 Summary table", "🔎 Detailed browser"])
    tp_options = sorted(set(r["tp"] for r in cluster_rows if r["tp"]))

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
        with f1: search_term = st.text_input("🔍 Search", key="cb_search")
        with f2: tp_f = st.selectbox("Filter TP", ["All"]+tp_options, key="cb_tp")
        with f3: cross_only = st.checkbox("⚡ Cross-TP only", key="cb_cross")

        filtered = summary_df.copy()
        if search_term:
            filtered = filtered[filtered["Canonical"].str.contains(
                search_term, case=False, na=False)]
        if tp_f != "All":
            filtered = filtered[filtered["Packages"].str.contains(tp_f, na=False)]
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

    with cb_tab2:
        f1d, f2d, f3d = st.columns(3)
        with f1d: search_d = st.text_input("🔍 Search", key="cb_search_d")
        with f2d: tp_d = st.selectbox("Filter TP", ["All"]+tp_options, key="cb_tp_d")
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
        pc, ic = st.columns([2,3])
        with pc: page = st.number_input("Page", 1, total_pages, 1, key="cluster_page")
        with ic: st.caption(f"Page {page}/{total_pages} — {len(filtered_rows)} clusters")

        for row in filtered_rows[(page-1)*per_page : page*per_page]:
            cid  = row["id"]
            info = row["info"]
            members   = df_ann[df_ann.index.isin(info["member_indices"])]
            canon_idx = overrides.get(cid, info["canonical_idx"])
            cross_b   = " ⚡ CROSS-TP" if row["cross_tp"] else ""

            with st.expander(
                f"**C{cid}** {cross_b} — {info['size']} stmts  |  "
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
                        f'<div class="stmt">{mem.get("skill_statement","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                opts = {
                    f"[{df_ann.loc[i,'unit_code']}] "
                    f"{df_ann.loc[i,'skill_statement'][:70]}…": i
                    for i in info["member_indices"]
                }
                opt_values  = list(opts.values())
                safe_index  = opt_values.index(canon_idx) if canon_idx in opt_values else 0
                chosen_lbl  = st.selectbox("Set canonical", list(opts.keys()),
                                           index=safe_index, key=f"canon_sel_{cid}")
                chosen_idx  = opts[chosen_lbl]
                if chosen_idx != info["canonical_idx"]:
                    overrides[cid] = chosen_idx
                    st.session_state["sa_overrides"] = overrides
                    st.success("Override saved ✅")

# ── Near-duplicate pairs ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Near-duplicate pairs</div>', unsafe_allow_html=True)
st.caption(
    f"Approximate near-duplicate detection using LSH banding — "
    f"handles {n_total:,} statements without building a full similarity matrix."
)

if not near_dupes:
    st.success(f"✅ No near-duplicates found at threshold {duplicate_threshold}")
else:
    st.markdown(f"**{n_dupes} pairs found** (approximate — LSH may miss some pairs)")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        dupe_score_min = st.slider("Min similarity to show",
            duplicate_threshold, 1.0, duplicate_threshold, 0.01, key="dupe_score")
    with col_f2:
        cross_unit_only = st.checkbox("Cross-unit only", value=True,
            help="Only show duplicates from different unit codes")

    shown = [d for d in near_dupes if d["similarity"] >= dupe_score_min]
    if cross_unit_only:
        shown = [d for d in shown if d["unit_a"] != d["unit_b"]]

    st.caption(f"Showing {len(shown)} pairs")

    for pair in shown[:50]:
        sim = pair["similarity"]
        badge_cls, badge_lbl = (
            ("sim-high","NEAR-IDENTICAL") if sim>=0.97 else
            ("sim-med", "VERY SIMILAR")   if sim>=0.94 else
            ("sim-low", "SIMILAR")
        )
        cross = " ⚡ CROSS-UNIT" if pair["unit_a"] != pair["unit_b"] else ""
        st.markdown(
            f'<div class="dupe-row">'
            f'<span class="sim-badge {badge_cls}">{badge_lbl} {sim:.3f}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#64748b">{cross}</span>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px">'
            f'<div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#7dd3fc">{pair["unit_a"]}</div>'
            f'<div style="font-size:0.75rem;color:#94a3b8">{pair["unit_title_a"]}</div>'
            f'<div style="font-size:0.72rem;color:#475569;font-style:italic">'
            f'Element: {pair["elem_a"]}</div>'
            f'<div style="font-size:0.88rem;color:#e2e8f0;margin-top:6px">'
            f'{pair["text_a"]}</div></div>'
            f'<div>'
            f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;'
            f'color:#7dd3fc">{pair["unit_b"]}</div>'
            f'<div style="font-size:0.75rem;color:#94a3b8">{pair["unit_title_b"]}</div>'
            f'<div style="font-size:0.72rem;color:#475569;font-style:italic">'
            f'Element: {pair["elem_b"]}</div>'
            f'<div style="font-size:0.88rem;color:#e2e8f0;margin-top:6px">'
            f'{pair["text_b"]}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    if len(shown) > 50:
        st.caption(f"… and {len(shown)-50} more. Export CSV for full list.")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

overrides       = st.session_state.get("sa_overrides", {})
final_canonical = {overrides.get(cid, info["canonical_idx"])
                   for cid, info in clusters.items()}
export_df = df_ann[df_ann.index.isin(final_canonical)].copy()

st.markdown(
    f"Export: **{len(export_df):,} canonical statements** "
    f"(reduced from {n_total:,} — "
    f"−{round(100*(n_total-len(export_df))/max(n_total,1))}%)"
)

from core.exporters import to_rsd_rows, to_osmt_rows

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
    df_ann[[c for c in ["unit_code","unit_title","element_title",
            "skill_statement","cluster_id","is_canonical"]
            if c in df_ann.columns]].to_csv(index=False).encode(),
    "cluster_analysis.csv", "text/csv", use_container_width=True,
)
