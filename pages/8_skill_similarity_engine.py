"""
pages/7_🧠_Skill_Similarity_Engine.py

Multidimensional Skill Similarity Engine

Layers:
  1. Semantic similarity   — TF-IDF embeddings + cosine similarity
  2. Structural similarity — verb/action patterns
  3. Network relationships — NetworkX graph of connected elements
  4. System-level metrics  — redundancy, transferability, uniqueness

Visualisations:
  A. Skill Embedding Map (2D/3D UMAP)
  B. Skill Network Graph
  C. Redundancy Surface (3D)
  D. Skill Cluster Explorer
  E. Skill DNA Fingerprint
  F. Redundancy Index Dashboard
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()
st.set_page_config(page_title="Skill Similarity Engine", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
  .eng-card{background:linear-gradient(135deg,#0a0f1e,#0d1b2a);border:1px solid #1a237e;
    border-radius:12px;padding:20px 24px;margin-bottom:12px}
  .eng-val{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;color:#82b1ff}
  .eng-lbl{font-size:0.68rem;color:#5c6bc0;text-transform:uppercase;letter-spacing:0.12em;margin-top:4px}
  .cluster-pill{display:inline-block;padding:2px 10px;border-radius:12px;
    font-size:0.72rem;font-family:'IBM Plex Mono',monospace;margin:2px}
  .dna-bar{height:18px;border-radius:3px;margin-bottom:2px}
  .section-hdr{font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.18em;
    text-transform:uppercase;color:#37474f;border-bottom:1px solid #1a237e;
    padding-bottom:6px;margin:24px 0 14px 0}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Skill Similarity Engine")
st.caption("Multidimensional analysis of skill statements across all Units of Competency")

# ── DB connection ─────────────────────────────────────────────────────────────
def _secret(k, d=""):
    try: return st.secrets.get(k, os.getenv(k, d)) or d
    except: return os.getenv(k, d) or d

DB_URL = _secret("DATABASE_URL")
if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

# ── Load skill data ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading skill statements…")
def load_skills() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                s.id, s.unit_code, s.unit_title, s.tp_code,
                s.element_title, s.skill_statement, s.keywords,
                o.anzsco_code, o.anzsco_title, o.anzsco_major_group, o.confidence
            FROM rsd_skill_records s
            LEFT JOIN uoc_occupation_links o
                ON o.uoc_code = s.unit_code
                AND o.is_primary = TRUE AND o.valid_to IS NULL
            WHERE s.skill_statement IS NOT NULL AND s.skill_statement != ''
            ORDER BY s.unit_code, s.element_title
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])

df_raw = load_skills()

if df_raw.empty:
    st.warning("No skill statements found. Run the RSD generator first.")
    st.stop()

st.success(f"Loaded **{len(df_raw):,}** skill statements from **{df_raw['unit_code'].nunique()}** UOCs")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Engine settings")

    n_clusters = st.slider("Number of skill clusters", 5, 50, 15)
    sim_threshold = st.slider("Similarity threshold (network edges)", 0.1, 0.9, 0.35, 0.05)
    max_stmts = st.slider("Max statements to analyse", 100, min(5000, len(df_raw)), min(2000, len(df_raw)), 100)
    dim_method = st.selectbox("Dimensionality reduction", ["UMAP", "t-SNE", "PCA"])
    n_dims = st.radio("Embedding dimensions", [2, 3], horizontal=True)

    st.divider()
    tp_filter = st.multiselect("Filter by TP", sorted(df_raw["tp_code"].dropna().unique()))

df = df_raw.sample(min(max_stmts, len(df_raw)), random_state=42).copy()
if tp_filter:
    df = df[df["tp_code"].isin(tp_filter)]
    if df.empty:
        st.warning("No data for selected TPs.")
        st.stop()

# ── Core computation ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing embeddings…", ttl=600)
def compute_everything(statements: list[str], n_clust: int, thresh: float,
                        ndims: int, method: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    # ── Layer 1: TF-IDF semantic embeddings ──────────────────────────────────
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                          max_features=8000, min_df=1)
    tfidf = vec.fit_transform(statements)

    # Reduce to 100-dim semantic space
    svd = TruncatedSVD(n_components=min(100, tfidf.shape[1]-1, tfidf.shape[0]-1),
                       random_state=42)
    embeddings = normalize(svd.fit_transform(tfidf))

    # ── Layer 2: Verb/action extraction ──────────────────────────────────────
    action_verbs = [
        "analyse","apply","assess","calculate","check","clean","collect",
        "communicate","complete","conduct","confirm","control","coordinate",
        "create","define","demonstrate","design","develop","document",
        "ensure","establish","evaluate","identify","implement","inspect",
        "install","interpret","maintain","manage","measure","monitor",
        "operate","perform","plan","prepare","process","record","report",
        "review","select","test","use","verify",
    ]
    def extract_verbs(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w in action_verbs]

    verb_lists = [extract_verbs(s) for s in statements]
    primary_verbs = [v[0] if v else "other" for v in verb_lists]

    # ── Layer 3: Dimensionality reduction for visualisation ──────────────────
    n_out = min(ndims, embeddings.shape[0]-1, embeddings.shape[1]-1)
    if method == "UMAP":
        try:
            import umap.umap_ as umap
            reducer = umap.UMAP(n_components=n_out, random_state=42,
                                n_neighbors=min(15, len(statements)-1))
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            pca = PCA(n_components=n_out, random_state=42)
            coords = pca.fit_transform(embeddings)
    elif method == "t-SNE":
        from sklearn.manifold import TSNE
        perp = min(30, len(statements)//4, embeddings.shape[0]-1)
        tsne = TSNE(n_components=n_out, perplexity=max(5,perp),
                    random_state=42, max_iter=500)
        coords = tsne.fit_transform(embeddings)
    else:
        pca = PCA(n_components=n_out, random_state=42)
        coords = pca.fit_transform(embeddings)

    # ── Clustering ────────────────────────────────────────────────────────────
    k = min(n_clust, len(statements)//3, len(statements)-1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(embeddings)

    # Cluster labels — top TF-IDF terms per cluster
    feature_names = vec.get_feature_names_out()
    cluster_labels = {}
    for c in range(k):
        idx = np.where(clusters == c)[0]
        if len(idx) == 0:
            cluster_labels[c] = f"Cluster {c}"
            continue
        centroid = embeddings[idx].mean(axis=0)
        # Map back through SVD
        tfidf_approx = svd.inverse_transform(centroid.reshape(1,-1))
        top_idx = tfidf_approx[0].argsort()[-4:][::-1]
        top_terms = [feature_names[i] for i in top_idx
                     if i < len(feature_names)]
        cluster_labels[c] = " · ".join(top_terms[:3]) or f"Cluster {c}"

    # ── Layer 4: System metrics ───────────────────────────────────────────────
    # Cosine similarity matrix (sampled for performance)
    sample_n = min(500, len(statements))
    sim_matrix = cosine_similarity(embeddings[:sample_n])

    # Redundancy: % pairs with sim > threshold
    upper = sim_matrix[np.triu_indices(sample_n, k=1)]
    redundancy_pct = float((upper > thresh).mean() * 100)

    # Uniqueness score per statement
    mean_sims = sim_matrix.mean(axis=1)  # only for sample
    uniqueness = 1 - mean_sims

    # Cluster frequency (transferability proxy)
    cluster_counts = pd.Series(clusters).value_counts()

    return {
        "embeddings": embeddings,
        "coords":     coords,
        "clusters":   clusters,
        "cluster_labels": cluster_labels,
        "primary_verbs":  primary_verbs,
        "sim_matrix": sim_matrix,
        "redundancy_pct": redundancy_pct,
        "uniqueness": uniqueness,
        "cluster_counts": cluster_counts,
        "k": k,
    }

with st.spinner("Running multidimensional analysis…"):
    res = compute_everything(
        df["skill_statement"].tolist(),
        n_clusters, sim_threshold, n_dims, dim_method,
    )

df["cluster"]      = res["clusters"]
df["cluster_label"]= df["cluster"].map(res["cluster_labels"])
df["primary_verb"] = res["primary_verbs"]
df["uniqueness"]   = np.pad(res["uniqueness"],
                            (0, max(0, len(df) - len(res["uniqueness"]))),
                            constant_values=0.5)[:len(df)]

coords = res["coords"]
if coords.shape[1] >= 1: df["x"] = coords[:, 0]
if coords.shape[1] >= 2: df["y"] = coords[:, 1]
if coords.shape[1] >= 3: df["z"] = coords[:, 2]

# ── Top-level metrics ─────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">System overview</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
def eng_card(col, val, lbl):
    col.markdown(f'<div class="eng-card"><div class="eng-val">{val}</div>'
                 f'<div class="eng-lbl">{lbl}</div></div>', unsafe_allow_html=True)

eng_card(m1, f"{len(df):,}", "Statements analysed")
eng_card(m2, f"{res['k']}", "Skill clusters")
eng_card(m3, f"{res['redundancy_pct']:.1f}%", "Redundancy rate")
eng_card(m4, f"{df['primary_verb'].nunique()}", "Distinct action verbs")
eng_card(m5, f"{df['unit_code'].nunique()}", "UOCs covered")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌌 Skill Map",
    "🌐 Network Graph",
    "🔥 Redundancy Surface",
    "🧩 Cluster Explorer",
    "🧬 Skill DNA",
    "📉 Redundancy Index",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SKILL EMBEDDING MAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-hdr">Skill Embedding Map</div>', unsafe_allow_html=True)
    st.caption(f"Each dot = one skill statement. Proximity = semantic similarity. Method: {dim_method}")

    colour_by = st.radio("Colour by", ["Cluster", "UOC", "Verb", "TP", "Uniqueness"],
                         horizontal=True, key="map_colour")

    colour_col = {
        "Cluster":    "cluster_label",
        "UOC":        "unit_code",
        "Verb":       "primary_verb",
        "TP":         "tp_code",
        "Uniqueness": "uniqueness",
    }[colour_by]

    hover = ["unit_code", "element_title", "skill_statement", "cluster_label", "primary_verb"]
    hover = [c for c in hover if c in df.columns]

    if n_dims == 3 and "z" in df.columns:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z",
            color=colour_col,
            hover_data=hover,
            color_continuous_scale="Blues" if colour_by == "Uniqueness" else None,
            opacity=0.75,
            height=650,
            title=f"Skill Embedding Map — {dim_method} (3D)",
        )
        fig.update_traces(marker_size=3)
    else:
        fig = px.scatter(
            df, x="x", y="y",
            color=colour_col,
            hover_data=hover,
            color_continuous_scale="Blues" if colour_by == "Uniqueness" else None,
            opacity=0.75,
            height=550,
            title=f"Skill Embedding Map — {dim_method} (2D)",
        )
        fig.update_traces(marker_size=5)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,1)",
        font_color="#cfd8dc",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary table
    with st.expander("Cluster summary"):
        clust_summary = (
            df.groupby(["cluster", "cluster_label"])
            .agg(
                size=("skill_statement", "count"),
                uocs=("unit_code", "nunique"),
                tps=("tp_code", "nunique"),
                top_verb=("primary_verb", lambda x: x.value_counts().index[0] if len(x) else "—"),
                avg_uniqueness=("uniqueness", "mean"),
            )
            .reset_index()
            .sort_values("size", ascending=False)
        )
        clust_summary["avg_uniqueness"] = clust_summary["avg_uniqueness"].round(3)
        st.dataframe(clust_summary, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — NETWORK GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-hdr">Skill Network Graph</div>', unsafe_allow_html=True)
    st.caption(f"Nodes = skill statements. Edges = cosine similarity > {sim_threshold}. "
               "Central nodes = core competencies repeated across UOCs.")

    max_nodes = st.slider("Max nodes to display", 50, min(500, len(df)), 150, key="net_nodes")

    @st.cache_data(show_spinner="Building network…", ttl=300)
    def build_network(stmts, embs, thresh, mn, clust, clabels):
        import networkx as nx
        from sklearn.metrics.pairwise import cosine_similarity

        n = min(mn, len(stmts))
        sub_emb = embs[:n]
        sim = cosine_similarity(sub_emb)

        G = nx.Graph()
        for i in range(n):
            G.add_node(i,
                       label=stmts[i][:60],
                       cluster=int(clust[i]),
                       cluster_label=clabels[int(clust[i])],
            )
        edge_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if sim[i,j] > thresh:
                    G.add_edge(i, j, weight=float(sim[i,j]))
                    edge_count += 1
                    if edge_count > 5000:
                        break
            if edge_count > 5000:
                break

        # Layout
        pos = nx.spring_layout(G, k=1.5/np.sqrt(max(n,1)), seed=42)
        degree = dict(G.degree())
        centrality = nx.degree_centrality(G)
        return G, pos, degree, centrality, sim

    G, pos, degree, centrality, sim_net = build_network(
        df["skill_statement"].tolist(),
        res["embeddings"],
        sim_threshold, max_nodes,
        res["clusters"], res["cluster_labels"],
    )

    # Build plotly network
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [5 + degree.get(n,0)*2 for n in G.nodes()]
    node_color = [res["clusters"][n] for n in G.nodes()]
    node_text = [df["skill_statement"].iloc[n][:80] + "…"
                 if n < len(df) else "" for n in G.nodes()]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="#1a237e"),
        hoverinfo="none", name="edges",
    ))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=node_size, color=node_color,
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Cluster")),
        text=node_text, hoverinfo="text", name="skills",
    ))
    fig_net.update_layout(
        height=600, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,1)",
        font_color="#cfd8dc",
        title=f"Skill Network — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # Top central nodes (core competencies)
    st.markdown('<div class="section-hdr">Core competencies (highest centrality)</div>',
                unsafe_allow_html=True)
    top_central = sorted(centrality.items(), key=lambda x: -x[1])[:10]
    for node_id, cent in top_central:
        if node_id < len(df):
            row = df.iloc[node_id]
            st.markdown(
                f"<div class='eng-card' style='padding:10px 14px'>"
                f"<span style='font-family:monospace;font-size:0.72rem;color:#82b1ff'>"
                f"centrality {cent:.3f}</span> &nbsp;"
                f"<code style='font-size:0.75rem;color:#5c6bc0'>{row['unit_code']}</code> &nbsp;"
                f"<span style='font-size:0.85rem;color:#cfd8dc'>{row['skill_statement'][:120]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — REDUNDANCY SURFACE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">Redundancy Surface (3D)</div>', unsafe_allow_html=True)
    st.caption("X = similarity to nearest neighbour · Y = frequency in cluster · "
               "Z = cluster density. High peaks = redundant skill areas.")

    @st.cache_data(show_spinner="Computing redundancy surface…", ttl=300)
    def compute_surface(embs, clust, k):
        from sklearn.metrics.pairwise import cosine_similarity
        n = min(500, embs.shape[0])
        sim = cosine_similarity(embs[:n])
        np.fill_diagonal(sim, 0)
        max_sim = sim.max(axis=1)  # nearest neighbour similarity

        clust_sub = clust[:n]
        counts = pd.Series(clust_sub).value_counts()
        freq = np.array([counts.get(c, 0) for c in clust_sub], dtype=float)
        freq = freq / freq.max()

        # Cluster density = mean intra-cluster similarity
        density = np.zeros(n)
        for c in range(k):
            idx = np.where(clust_sub == c)[0]
            if len(idx) < 2:
                continue
            sub = sim[np.ix_(idx, idx)]
            d = sub.mean()
            density[idx] = d

        return max_sim, freq, density

    ms, freq, dens = compute_surface(res["embeddings"], res["clusters"], res["k"])
    n_surf = min(len(ms), len(df))

    surf_df = pd.DataFrame({
        "similarity":   ms[:n_surf],
        "frequency":    freq[:n_surf],
        "density":      dens[:n_surf],
        "cluster":      [str(c) for c in res["clusters"][:n_surf]],
        "unit_code":    df["unit_code"].values[:n_surf],
        "skill":        df["skill_statement"].str[:80].values[:n_surf],
    })

    fig_surf = px.scatter_3d(
        surf_df, x="similarity", y="frequency", z="density",
        color="cluster",
        hover_data=["unit_code", "skill"],
        opacity=0.7, height=620,
        title="Redundancy Surface — peaks indicate high duplication zones",
        labels={
            "similarity": "Sim to nearest neighbour",
            "frequency":  "Cluster frequency",
            "density":    "Cluster density",
        },
    )
    fig_surf.update_traces(marker_size=3)
    fig_surf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cfd8dc",
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    # Redundancy hotspots
    hotspots = surf_df.nlargest(10, "density")
    st.markdown('<div class="section-hdr">Top redundancy hotspots</div>', unsafe_allow_html=True)
    st.dataframe(hotspots[["unit_code","cluster","similarity","frequency","density","skill"]],
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTER EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-hdr">Skill Cluster Explorer</div>', unsafe_allow_html=True)
    st.caption("Each cluster = a 'meta-skill'. Select a cluster to explore which UOCs use it.")

    cluster_options = {
        f"Cluster {c} — {lbl} ({(df['cluster']==c).sum()} skills)": c
        for c, lbl in sorted(res["cluster_labels"].items(),
                              key=lambda x: -(df["cluster"]==x[0]).sum())
    }
    sel_cluster_label = st.selectbox("Select meta-skill cluster", list(cluster_options.keys()))
    sel_cluster = cluster_options[sel_cluster_label]

    clust_df = df[df["cluster"] == sel_cluster].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Skill statements", len(clust_df))
    with c2:
        st.metric("UOCs", clust_df["unit_code"].nunique())
    with c3:
        st.metric("Training packages", clust_df["tp_code"].nunique())

    # Verb distribution
    verb_counts = clust_df["primary_verb"].value_counts().head(8)
    fig_verbs = px.bar(
        verb_counts.reset_index(),
        x="primary_verb", y="count",
        color="count", color_continuous_scale="Blues",
        title="Action verb distribution in this cluster",
        height=280,
    )
    fig_verbs.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(13,27,42,1)",
                             font_color="#cfd8dc", showlegend=False)
    st.plotly_chart(fig_verbs, use_container_width=True)

    # UOC breakdown
    st.markdown('<div class="section-hdr">UOCs using this meta-skill</div>',
                unsafe_allow_html=True)
    uoc_breakdown = (
        clust_df.groupby(["unit_code","unit_title","tp_code"])
        .agg(skill_count=("skill_statement","count"))
        .reset_index()
        .sort_values("skill_count", ascending=False)
    )
    st.dataframe(uoc_breakdown, use_container_width=True, hide_index=True)

    # Skill variations
    st.markdown('<div class="section-hdr">Skill statement variations</div>',
                unsafe_allow_html=True)
    for _, row in clust_df.head(20).iterrows():
        st.markdown(
            f"<div class='eng-card' style='padding:8px 12px;margin-bottom:6px'>"
            f"<code style='color:#5c6bc0;font-size:0.72rem'>{row['unit_code']}</code>"
            f" &nbsp; <span style='color:#37474f;font-size:0.7rem'>"
            f"{row.get('element_title','')}</span><br>"
            f"<span style='color:#cfd8dc;font-size:0.85rem'>{row['skill_statement']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    if len(clust_df) > 20:
        st.caption(f"… and {len(clust_df)-20} more statements in this cluster")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SKILL DNA FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">Skill DNA Fingerprint</div>', unsafe_allow_html=True)
    st.caption("Each UOC represented as a vector of cluster memberships — "
               "like a genomic fingerprint. Similar UOCs have similar DNA profiles.")

    # Build DNA vectors: UOC × cluster proportions
    dna = (
        df.groupby(["unit_code","cluster"])
        .size()
        .unstack(fill_value=0)
    )
    dna_norm = dna.div(dna.sum(axis=1), axis=0)

    # Select UOC to inspect
    uoc_options = dna_norm.index.tolist()
    sel_uoc = st.selectbox("Select UOC for DNA profile", uoc_options, key="dna_uoc")

    col_dna1, col_dna2 = st.columns([2, 3])

    with col_dna1:
        uoc_row = dna_norm.loc[sel_uoc]
        uoc_meta = df[df["unit_code"] == sel_uoc].iloc[0]

        st.markdown(f"**{sel_uoc}** — {uoc_meta.get('unit_title','')}")
        st.caption(f"TP: {uoc_meta.get('tp_code','')}")

        # DNA bar chart
        dna_df = uoc_row[uoc_row > 0].reset_index()
        dna_df.columns = ["cluster", "proportion"]
        dna_df["cluster_label"] = dna_df["cluster"].map(res["cluster_labels"])
        dna_df = dna_df.sort_values("proportion", ascending=False)

        fig_dna = px.bar(
            dna_df, x="proportion", y="cluster_label",
            orientation="h",
            color="proportion",
            color_continuous_scale="Blues",
            height=max(250, len(dna_df)*28),
            title=f"Skill DNA — {sel_uoc}",
        )
        fig_dna.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,1)",
            font_color="#cfd8dc", showlegend=False,
            yaxis_title="", xaxis_title="Proportion",
        )
        st.plotly_chart(fig_dna, use_container_width=True)

    with col_dna2:
        # Most similar UOCs (cosine similarity on DNA vectors)
        from sklearn.metrics.pairwise import cosine_similarity as cs
        dna_arr = dna_norm.values
        idx = dna_norm.index.tolist().index(sel_uoc)
        sims = cs(dna_arr[idx:idx+1], dna_arr)[0]
        sim_series = pd.Series(sims, index=dna_norm.index).sort_values(ascending=False)
        top_similar = sim_series.iloc[1:11]  # exclude self

        st.markdown('<div class="section-hdr">Most similar UOCs (DNA distance)</div>',
                    unsafe_allow_html=True)

        sim_df = top_similar.reset_index()
        sim_df.columns = ["unit_code", "dna_similarity"]
        sim_df = sim_df.merge(
            df[["unit_code","unit_title","tp_code"]].drop_duplicates("unit_code"),
            on="unit_code", how="left",
        )
        sim_df["dna_similarity"] = sim_df["dna_similarity"].round(3)

        st.dataframe(
            sim_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "unit_code":      st.column_config.TextColumn("Unit", width="small"),
                "unit_title":     st.column_config.TextColumn("Title"),
                "tp_code":        st.column_config.TextColumn("TP", width="small"),
                "dna_similarity": st.column_config.NumberColumn("DNA similarity", format="%.3f"),
            }
        )

        # Heatmap of top 15 UOCs DNA comparison
        top15 = [sel_uoc] + top_similar.head(9).index.tolist()
        dna_sub = dna_norm.loc[[u for u in top15 if u in dna_norm.index]]
        sim_heat = cs(dna_sub.values)

        fig_heat = px.imshow(
            sim_heat,
            x=dna_sub.index.tolist(),
            y=dna_sub.index.tolist(),
            color_continuous_scale="Blues",
            title="DNA similarity heatmap — top 10 similar UOCs",
            height=350,
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cfd8dc",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — REDUNDANCY INDEX DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-hdr">Redundancy Index Dashboard</div>',
                unsafe_allow_html=True)
    st.caption("System-level policy insights — where duplication is concentrated.")

    # Overall metrics
    total_stmts = len(df)
    unique_clusters = df["cluster"].nunique()
    sim_arr = res["sim_matrix"]
    upper_tri = sim_arr[np.triu_indices(sim_arr.shape[0], k=1)]
    high_sim_pairs = (upper_tri > sim_threshold).sum()
    total_pairs = len(upper_tri)
    redundancy_rate = high_sim_pairs / max(total_pairs, 1) * 100
    avg_uniqueness = float(df["uniqueness"].mean())

    # Transferability: UOCs that appear in multiple clusters
    uoc_cluster_counts = df.groupby("unit_code")["cluster"].nunique()
    highly_transferable = (uoc_cluster_counts >= 3).sum()
    transfer_pct = highly_transferable / max(len(uoc_cluster_counts), 1) * 100

    r1, r2, r3, r4 = st.columns(4)
    eng_card(r1, f"{redundancy_rate:.1f}%", "Redundancy rate")
    eng_card(r2, f"{avg_uniqueness:.2f}", "Avg uniqueness (0-1)")
    eng_card(r3, f"{transfer_pct:.0f}%", "Highly transferable UOCs")
    eng_card(r4, f"{high_sim_pairs:,}", f"Similar pairs (>{sim_threshold})")

    st.divider()
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        # Cluster size distribution
        clust_sizes = df["cluster"].value_counts().reset_index()
        clust_sizes.columns = ["cluster", "count"]
        clust_sizes["label"] = clust_sizes["cluster"].map(res["cluster_labels"])

        fig_cs = px.bar(
            clust_sizes.head(20), x="count", y="label",
            orientation="h",
            color="count", color_continuous_scale="Reds",
            title="Cluster sizes — larger = more redundancy",
            height=400,
        )
        fig_cs.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(13,27,42,1)",
                              font_color="#cfd8dc", showlegend=False,
                              yaxis_title="", xaxis_title="Skill count")
        st.plotly_chart(fig_cs, use_container_width=True)

    with col_r2:
        # Uniqueness distribution
        fig_uniq = px.histogram(
            df, x="uniqueness", nbins=30,
            color="tp_code" if "tp_code" in df else None,
            title="Uniqueness distribution (1 = completely unique)",
            height=400,
        )
        fig_uniq.add_vline(x=avg_uniqueness, line_dash="dash",
                           line_color="#ef9f27",
                           annotation_text=f"avg {avg_uniqueness:.2f}")
        fig_uniq.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(13,27,42,1)",
                                font_color="#cfd8dc")
        st.plotly_chart(fig_uniq, use_container_width=True)

    # TP-level redundancy
    st.markdown('<div class="section-hdr">Redundancy by training package</div>',
                unsafe_allow_html=True)
    tp_red = (
        df.groupby("tp_code")
        .agg(
            total_skills=("skill_statement","count"),
            unique_clusters=("cluster","nunique"),
            avg_uniqueness=("uniqueness","mean"),
            uoc_count=("unit_code","nunique"),
        )
        .reset_index()
    )
    tp_red["redundancy_ratio"] = (
        1 - tp_red["unique_clusters"] / tp_red["total_skills"].clip(lower=1)
    ).round(3)
    tp_red["avg_uniqueness"] = tp_red["avg_uniqueness"].round(3)
    tp_red = tp_red.sort_values("redundancy_ratio", ascending=False)

    st.dataframe(
        tp_red,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tp_code":          st.column_config.TextColumn("TP", width="small"),
            "total_skills":     st.column_config.NumberColumn("Skills"),
            "unique_clusters":  st.column_config.NumberColumn("Unique clusters"),
            "uoc_count":        st.column_config.NumberColumn("UOCs"),
            "avg_uniqueness":   st.column_config.NumberColumn("Avg uniqueness", format="%.3f"),
            "redundancy_ratio": st.column_config.ProgressColumn(
                "Redundancy ratio", min_value=0, max_value=1, format="%.3f"),
        }
    )

    # Export
    st.divider()
    st.download_button(
        "⬇ Export full analysis CSV",
        df[["unit_code","unit_title","tp_code","element_title",
            "skill_statement","cluster","cluster_label",
            "primary_verb","uniqueness"]].to_csv(index=False).encode(),
        "skill_similarity_analysis.csv", "text/csv",
        use_container_width=True,
    )

    # Requirements note
    st.divider()
    st.markdown("""
**Optional: Install UMAP for better embedding quality**

Add to `requirements.txt`:
```
umap-learn>=0.5,<1.0
```

Without it the engine falls back to PCA which is still useful but less spatially accurate.
UMAP produces significantly better cluster separation for skill statements.
    """)
