"""
pages/9_🔗_Role_Skills_Intelligence.py

Five tools for understanding skill connections between occupations.
Similarity is computed from SEMANTIC ANALYSIS of skill statements
(TF-IDF embeddings + cosine similarity) — not UOC overlap.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()
st.set_page_config(page_title="Role Skills Intelligence", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
.intel-card{background:#0a1628;border:1px solid #1a2a4a;border-radius:10px;padding:18px 22px;margin-bottom:10px}
.intel-val{font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;color:#7eb8f7}
.intel-lbl{font-size:0.68rem;color:#4a6fa5;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px}
.section-hdr{font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;
  text-transform:uppercase;color:#4a6fa5;border-bottom:1px solid #1a2a4a;
  padding-bottom:6px;margin:20px 0 12px 0}
.gap-shared{background:#1b3a1b;border-left:3px solid #4caf50;padding:6px 10px;
  border-radius:4px;margin-bottom:4px;font-size:0.82rem;color:#cfd8dc}
.gap-missing{background:#3a1b1b;border-left:3px solid #f44336;padding:6px 10px;
  border-radius:4px;margin-bottom:4px;font-size:0.82rem;color:#cfd8dc}
.gap-unique{background:#1b2a3a;border-left:3px solid #2196f3;padding:6px 10px;
  border-radius:4px;margin-bottom:4px;font-size:0.82rem;color:#cfd8dc}
</style>
""", unsafe_allow_html=True)

st.title("🔗 Role Skills Intelligence")
st.caption("Occupation connections built from **semantic analysis of skill statements** — "
           "not UOC overlap. Two roles are similar if their skills mean the same thing.")

# ── DB ────────────────────────────────────────────────────────────────────────
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

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading occupation-skill data…")
def load_occ_skills() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                o.anzsco_code, o.anzsco_title, o.anzsco_major_group,
                o.confidence, s.unit_code, s.unit_title, s.tp_code,
                s.element_title, s.skill_statement
            FROM uoc_occupation_links o
            JOIN rsd_skill_records s ON s.unit_code = o.uoc_code
            WHERE o.is_primary = TRUE AND o.valid_to IS NULL
              AND o.anzsco_code IS NOT NULL AND o.anzsco_code != ''
              AND s.skill_statement IS NOT NULL AND s.skill_statement != ''
            ORDER BY o.anzsco_code, s.unit_code
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])

df = load_occ_skills()

if df.empty:
    st.warning("No occupation-skill data found. Run the linkage pipeline first.")
    st.stop()

n_occ    = df["anzsco_code"].nunique()
n_uocs   = df["unit_code"].nunique()
n_skills = len(df)

c1, c2, c3 = st.columns(3)
for col, val, lbl in [
    (c1, f"{n_occ:,}", "Occupations"),
    (c2, f"{n_uocs:,}", "UOCs"),
    (c3, f"{n_skills:,}", "Skill statements"),
]:
    col.markdown(
        f'<div class="intel-card"><div class="intel-val">{val}</div>'
        f'<div class="intel-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

# ── Core semantic computation ─────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Computing semantic skill embeddings…")
def compute_semantic_similarity():
    """
    For each occupation, concatenate all skill statements into one document.
    Embed with TF-IDF, then compute cosine similarity between occupation documents.
    Returns: occ_vectors (DataFrame), sim_matrix (ndarray), occ_meta (DataFrame)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    # Build one document per occupation = all skill statements concatenated
    occ_docs = (
        df.groupby("anzsco_code")["skill_statement"]
        .apply(lambda stmts: " ".join(stmts))
        .reset_index()
    )
    occ_docs.columns = ["anzsco_code", "document"]

    occ_meta = (
        df.groupby("anzsco_code")
        .agg(
            title=("anzsco_title", "first"),
            major=("anzsco_major_group", "first"),
            uoc_count=("unit_code", "nunique"),
            skill_count=("skill_statement", "count"),
            tp_count=("tp_code", "nunique"),
        )
        .reset_index()
    )
    occ_meta = occ_meta.merge(occ_docs, on="anzsco_code")

    # TF-IDF embedding
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=15000,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf = vec.fit_transform(occ_meta["document"])
    tfidf_norm = normalize(tfidf)

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_norm)
    np.fill_diagonal(sim_matrix, 0)

    return occ_meta, sim_matrix, tfidf_norm, vec

occ_meta, sim_matrix, tfidf_norm, tfidf_vec = compute_semantic_similarity()
occ_codes = occ_meta["anzsco_code"].tolist()
occ_label = {r["anzsco_code"]: r["title"] for _, r in occ_meta.iterrows()}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌐 Transferability Map",
    "🚀 Career Pathways",
    "📊 Skill Gap Analysis",
    "🗺️ Occupation Clusters",
    "🌉 Cross-TP Bridges",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TRANSFERABILITY MAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-hdr">Skill Transferability Map</div>',
                unsafe_allow_html=True)
    st.caption("Occupations linked by **semantic skill similarity**. "
               "Edges connect roles whose skill statements mean similar things.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        min_sim = st.slider("Min semantic similarity", 0.05, 0.9, 0.25, 0.05,
                            key="map_sim")
    with col_s2:
        major_filter = st.selectbox(
            "Filter major group",
            ["All"] + sorted(df["anzsco_major_group"].dropna().unique()),
            key="map_major",
        )
    with col_s3:
        max_nodes = st.slider("Max occupations", 20, min(200, n_occ), 80, 10,
                              key="map_nodes")

    # Filter occupations
    if major_filter != "All":
        sub_meta = occ_meta[occ_meta["major"] == major_filter].head(max_nodes)
    else:
        sub_meta = occ_meta.nlargest(max_nodes, "skill_count")

    sub_idx   = [occ_codes.index(c) for c in sub_meta["anzsco_code"] if c in occ_codes]
    sub_codes = [occ_codes[i] for i in sub_idx]
    sub_sim   = sim_matrix[np.ix_(sub_idx, sub_idx)]

    if len(sub_codes) < 2:
        st.warning("Not enough occupations. Try changing filters.")
    else:
        import networkx as nx
        G = nx.Graph()
        for c in sub_codes:
            m = sub_meta[sub_meta["anzsco_code"] == c].iloc[0]
            G.add_node(c, title=occ_label.get(c, c),
                       size=float(m["skill_count"]),
                       major=str(m["major"]))

        for i, ci in enumerate(sub_codes):
            for j, cj in enumerate(sub_codes):
                if i < j and sub_sim[i, j] >= min_sim:
                    G.add_edge(ci, cj, weight=float(sub_sim[i, j]))

        pos = nx.spring_layout(G, k=2.5/np.sqrt(max(len(sub_codes), 1)),
                               seed=42, iterations=80,
                               weight="weight")

        major_colors = {
            "Managers":                              "#ef9f27",
            "Professionals":                         "#7eb8f7",
            "Technicians and Trades Workers":        "#a5d6a7",
            "Community and Personal Service Workers":"#ce93d8",
            "Clerical and Administrative Workers":   "#80deea",
            "Sales Workers":                         "#f48fb1",
            "Machinery Operators and Drivers":       "#ffcc80",
            "Labourers":                             "#bcaaa4",
        }

        fig_net = go.Figure()

        # Edges
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d["weight"]
            fig_net.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(0.5, w * 6),
                          color=f"rgba(126,184,247,{min(w * 1.5, 0.55):.2f})"),
                hoverinfo="none", showlegend=False,
            ))

        # Nodes by major group
        for major, color in major_colors.items():
            nodes = [c for c in sub_codes
                     if G.nodes[c].get("major") == major and c in pos]
            if not nodes:
                continue
            node_meta = sub_meta[sub_meta["anzsco_code"].isin(nodes)]
            sizes = []
            hover = []
            for c in nodes:
                m = node_meta[node_meta["anzsco_code"] == c]
                sc = int(m["skill_count"].values[0]) if len(m) else 10
                sizes.append(min(6 + sc / 6, 35))
                hover.append(
                    f"<b>{occ_label.get(c, c)}</b><br>"
                    f"ANZSCO: {c}<br>"
                    f"Skills: {sc}<br>"
                    f"Connections: {G.degree(c)}"
                )
            fig_net.add_trace(go.Scatter(
                x=[pos[c][0] for c in nodes],
                y=[pos[c][1] for c in nodes],
                mode="markers+text",
                name=major,
                marker=dict(size=sizes, color=color, opacity=0.85,
                            line=dict(width=0.8, color="#0a1628")),
                text=[occ_label.get(c, c)[:22] for c in nodes],
                textposition="top center",
                textfont=dict(size=8, color="#cfd8dc"),
                hovertext=hover, hoverinfo="text",
            ))

        fig_net.update_layout(
            height=620, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#080e1a",
            font_color="#7eb8f7",
            legend=dict(x=1.01, y=1, bgcolor="rgba(10,22,40,0.85)",
                        bordercolor="#1a2a4a", font_size=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=220, t=50, b=20),
            title=f"Semantic Skill Network — {len(sub_codes)} occupations, "
                  f"{G.number_of_edges()} connections (sim ≥ {min_sim})",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Top pairs
        st.markdown('<div class="section-hdr">Most semantically similar pairs</div>',
                    unsafe_allow_html=True)
        pairs = []
        for i, ci in enumerate(sub_codes):
            for j, cj in enumerate(sub_codes):
                if i < j and sub_sim[i, j] >= min_sim:
                    pairs.append({
                        "Occupation A": occ_label.get(ci, ci),
                        "Occupation B": occ_label.get(cj, cj),
                        "Semantic similarity": round(float(sub_sim[i, j]), 3),
                    })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values(
                "Semantic similarity", ascending=False).head(20)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True,
                column_config={
                    "Semantic similarity": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.3f"),
                })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CAREER PATHWAYS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-hdr">Career Pathway Engine</div>',
                unsafe_allow_html=True)
    st.caption("Find the closest roles to any occupation based on **what skills mean**, "
               "not just which units appear.")

    sel_occ = st.selectbox(
        "Starting occupation",
        options=occ_codes,
        format_func=lambda c: f"{occ_label.get(c, c)} ({c})",
        key="path_occ",
    )
    top_n = st.slider("Top N pathways", 5, 30, 12, key="path_n")

    if sel_occ in occ_codes:
        idx     = occ_codes.index(sel_occ)
        sims    = sim_matrix[idx]
        top_idx = np.argsort(sims)[::-1][:top_n]

        src_skills = df[df["anzsco_code"] == sel_occ]["skill_statement"].tolist()
        src_uocs   = set(df[df["anzsco_code"] == sel_occ]["unit_code"])

        st.markdown(
            f"**{occ_label.get(sel_occ, sel_occ)}** — "
            f"{len(src_skills)} skills · {len(src_uocs)} UOCs"
        )
        st.divider()

        pathway_rows = []
        for i in top_idx:
            tgt_code   = occ_codes[i]
            sim        = float(sims[i])
            tgt_skills = df[df["anzsco_code"] == tgt_code]["skill_statement"].tolist()
            tgt_uocs   = set(df[df["anzsco_code"] == tgt_code]["unit_code"])
            shared_uocs  = src_uocs & tgt_uocs
            missing_uocs = tgt_uocs - src_uocs
            m = occ_meta[occ_meta["anzsco_code"] == tgt_code].iloc[0]
            pathway_rows.append({
                "code":          tgt_code,
                "title":         occ_label.get(tgt_code, tgt_code),
                "major":         m["major"],
                "semantic_sim":  round(sim, 3),
                "shared_uocs":   len(shared_uocs),
                "missing_uocs":  len(missing_uocs),
                "missing_list":  sorted(missing_uocs),
                "tgt_skill_count": len(tgt_skills),
            })

        # Bar chart
        path_df = pd.DataFrame(pathway_rows)
        fig_path = px.bar(
            path_df, x="semantic_sim", y="title",
            orientation="h",
            color="semantic_sim",
            color_continuous_scale=["#0d47a1", "#1b5e20"],
            range_color=[0, 1],
            title=f"Semantic career pathways from "
                  f"{occ_label.get(sel_occ, sel_occ)[:40]}",
            height=max(300, len(pathway_rows) * 35),
            labels={"semantic_sim": "Semantic similarity", "title": ""},
        )
        fig_path.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#080e1a",
            font_color="#7eb8f7", showlegend=False,
        )
        st.plotly_chart(fig_path, use_container_width=True)

        # Detail expanders
        for p in pathway_rows:
            sim_pct = p["semantic_sim"] * 100
            color = ("#1b5e20" if sim_pct >= 60 else
                     "#0d47a1" if sim_pct >= 35 else "#bf360c")
            with st.expander(
                f"**{p['title']}** — {sim_pct:.0f}% semantic match · "
                f"{p['shared_uocs']} shared UOCs · {p['missing_uocs']} new UOCs",
                expanded=False,
            ):
                st.markdown(
                    f'<div style="background:#0a1628;border-radius:6px;padding:10px">'
                    f'<div style="background:#1a2a4a;border-radius:3px;height:10px">'
                    f'<div style="width:{sim_pct:.0f}%;background:{color};'
                    f'height:10px;border-radius:3px"></div></div>'
                    f'<div style="font-family:monospace;color:{color};margin-top:6px">'
                    f'Semantic similarity: {p["semantic_sim"]:.3f}</div></div>',
                    unsafe_allow_html=True,
                )
                if p["missing_list"]:
                    st.markdown("**New UOCs to gain:**")
                    missing_df = df[df["unit_code"].isin(p["missing_list"])][
                        ["unit_code", "unit_title", "tp_code"]
                    ].drop_duplicates("unit_code")
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # Export
        export_rows = []
        for p in pathway_rows:
            for uoc in p["missing_list"]:
                u = df[df["unit_code"] == uoc][
                    ["unit_code", "unit_title", "tp_code"]
                ].drop_duplicates()
                if not u.empty:
                    export_rows.append({
                        "From Occupation":    occ_label.get(sel_occ, sel_occ),
                        "To Occupation":      p["title"],
                        "Semantic Similarity": p["semantic_sim"],
                        "Gap UOC":            uoc,
                        "Unit Title":         u.iloc[0]["unit_title"],
                        "TP":                 u.iloc[0]["tp_code"],
                    })
        if export_rows:
            st.download_button(
                "⬇ Export pathway gaps CSV",
                pd.DataFrame(export_rows).to_csv(index=False).encode(),
                f"pathway_{sel_occ}.csv", "text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SKILL GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">Skill Gap Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Compare two occupations — semantic similarity, shared UOCs, "
               "and the skill statements that exist in one but not the other.")

    g1, g2 = st.columns(2)
    with g1:
        occ_a = st.selectbox("Occupation A (current)",
            options=occ_codes,
            format_func=lambda c: f"{occ_label.get(c,c)} ({c})",
            key="gap_a")
    with g2:
        occ_b = st.selectbox("Occupation B (target)",
            options=occ_codes,
            format_func=lambda c: f"{occ_label.get(c,c)} ({c})",
            index=min(1, len(occ_codes)-1),
            key="gap_b")

    if occ_a == occ_b:
        st.info("Select two different occupations.")
    else:
        ai = occ_codes.index(occ_a)
        bi = occ_codes.index(occ_b)
        sem_sim = float(sim_matrix[ai, bi])

        uocs_a = set(df[df["anzsco_code"] == occ_a]["unit_code"])
        uocs_b = set(df[df["anzsco_code"] == occ_b]["unit_code"])
        shared  = uocs_a & uocs_b
        only_a  = uocs_a - uocs_b
        only_b  = uocs_b - uocs_a

        skills_a = df[df["anzsco_code"] == occ_a]["skill_statement"].tolist()
        skills_b = df[df["anzsco_code"] == occ_b]["skill_statement"].tolist()

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Semantic similarity", f"{sem_sim:.3f}")
        m2.metric("Shared UOCs", len(shared))
        m3.metric("Skills in A", len(skills_a))
        m4.metric("Skills in B", len(skills_b))

        # Similarity bar
        color = ("#1b5e20" if sem_sim >= 0.6 else
                 "#0d47a1" if sem_sim >= 0.3 else "#bf360c")
        st.markdown(
            f'<div style="background:#1a2a4a;border-radius:6px;height:16px;margin-bottom:16px">'
            f'<div style="width:{sem_sim*100:.0f}%;background:{color};'
            f'height:16px;border-radius:6px"></div></div>'
            f'<div style="font-size:0.75rem;color:{color};margin-bottom:16px">'
            f'Semantic similarity: {sem_sim:.3f} — '
            f'{"High transferability" if sem_sim>=0.6 else "Moderate" if sem_sim>=0.3 else "Low transferability"}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Skill statement semantic matching
        from sklearn.metrics.pairwise import cosine_similarity as cs_fn

        if skills_a and skills_b:
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_stmts = skills_a + skills_b
            v = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                                max_features=5000)
            tm = v.fit_transform(all_stmts)
            sim_ab = cs_fn(tm[:len(skills_a)], tm[len(skills_a):])

            # Find best semantic matches
            st.markdown('<div class="section-hdr">Closest skill statement pairs</div>',
                        unsafe_allow_html=True)
            matches = []
            for i in range(min(len(skills_a), 50)):
                best_j = int(np.argmax(sim_ab[i]))
                best_s = float(sim_ab[i, best_j])
                if best_s >= 0.3:
                    matches.append({
                        "Skill A": skills_a[i][:120],
                        "Skill B": skills_b[best_j][:120],
                        "Similarity": round(best_s, 3),
                    })
            if matches:
                match_df = pd.DataFrame(matches).sort_values(
                    "Similarity", ascending=False).drop_duplicates("Skill A").head(15)
                st.dataframe(match_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Similarity": st.column_config.ProgressColumn(
                            min_value=0, max_value=1, format="%.3f"),
                        "Skill A": st.column_config.TextColumn(width="large"),
                        "Skill B": st.column_config.TextColumn(width="large"),
                    })

        # UOC gap columns
        st.markdown('<div class="section-hdr">UOC breakdown</div>',
                    unsafe_allow_html=True)
        col3a, col3b, col3c = st.columns(3)

        def show_uoc_col(col, uoc_set, label, css):
            with col:
                st.markdown(f"**{label}** ({len(uoc_set)})")
                sub = df[df["unit_code"].isin(uoc_set)][
                    ["unit_code","unit_title","tp_code"]
                ].drop_duplicates("unit_code").head(15)
                for _, row in sub.iterrows():
                    st.markdown(
                        f'<div class="{css}">'
                        f'<code style="color:#7eb8f7;font-size:0.72rem">'
                        f'{row["unit_code"]}</code> {row["unit_title"][:45]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        show_uoc_col(col3a, only_a,
                     f"Only in {occ_label.get(occ_a,'A')[:20]}", "gap-unique")
        show_uoc_col(col3b, shared, "Shared", "gap-shared")
        show_uoc_col(col3c, only_b,
                     f"Gap — {occ_label.get(occ_b,'B')[:20]}", "gap-missing")

        # Export
        gap_rows = []
        for uoc in only_b:
            u = df[df["unit_code"] == uoc][
                ["unit_code","unit_title","tp_code"]].drop_duplicates()
            stmts = df[(df["unit_code"]==uoc) &
                       (df["anzsco_code"]==occ_b)]["skill_statement"].tolist()
            for stmt in stmts[:3]:
                gap_rows.append({
                    "From": occ_label.get(occ_a, occ_a),
                    "To": occ_label.get(occ_b, occ_b),
                    "Semantic similarity": round(sem_sim, 3),
                    "Gap UOC": uoc,
                    "Unit Title": u.iloc[0]["unit_title"] if len(u) else "",
                    "TP": u.iloc[0]["tp_code"] if len(u) else "",
                    "Skill needed": stmt,
                })
        if gap_rows:
            st.download_button(
                "⬇ Export skill gap CSV",
                pd.DataFrame(gap_rows).to_csv(index=False).encode(),
                f"gap_{occ_a}_{occ_b}.csv", "text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — OCCUPATION CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-hdr">Occupation Cluster Map</div>',
                unsafe_allow_html=True)
    st.caption("Occupations grouped by semantic skill similarity — "
               "reveals natural skill families across industry boundaries.")

    n_clust  = st.slider("Clusters", 3, 20, 8, key="clust_n")
    dim_meth = st.radio("Layout", ["UMAP", "PCA"], horizontal=True, key="clust_dim")

    from sklearn.cluster import KMeans

    mat_vals = tfidf_norm.toarray() if hasattr(tfidf_norm, "toarray") else tfidf_norm
    k        = min(n_clust, len(occ_codes)-1)
    km       = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(mat_vals)

    if dim_meth == "UMAP":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=min(15, len(occ_codes)-1))
            coords = reducer.fit_transform(mat_vals)
        except ImportError:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2, random_state=42).fit_transform(mat_vals)
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(mat_vals)

    clust_df = occ_meta.copy()
    clust_df["cluster"] = clusters
    clust_df["x"] = coords[:, 0]
    clust_df["y"] = coords[:, 1]

    # Cluster names from top TF-IDF terms
    fn = tfidf_vec.get_feature_names_out()
    cluster_names = {}
    for c in range(k):
        cidx = np.where(clusters == c)[0]
        if not len(cidx):
            cluster_names[c] = f"Cluster {c}"
            continue
        centroid = mat_vals[cidx].mean(axis=0)
        top = centroid.argsort()[-4:][::-1]
        terms = [fn[i] for i in top if i < len(fn)]
        cluster_names[c] = " · ".join(terms[:3])

    clust_df["cluster_name"] = clust_df["cluster"].map(cluster_names)

    fig_clust = px.scatter(
        clust_df, x="x", y="y",
        color="cluster_name",
        size="skill_count",
        hover_data=["title", "anzsco_code", "major", "uoc_count"],
        text="title",
        height=620,
        title=f"Occupation Skill Families — {k} semantic clusters",
        labels={"x": "Dimension 1", "y": "Dimension 2"},
    )
    fig_clust.update_traces(
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(opacity=0.82, line=dict(width=0.5, color="#0a1628")),
    )
    fig_clust.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080e1a",
        font_color="#7eb8f7",
        xaxis=dict(gridcolor="#1a2a4a"),
        yaxis=dict(gridcolor="#1a2a4a"),
    )
    st.plotly_chart(fig_clust, use_container_width=True)

    st.markdown('<div class="section-hdr">Cluster composition</div>',
                unsafe_allow_html=True)
    for c in range(k):
        sub = clust_df[clust_df["cluster"] == c].sort_values(
            "skill_count", ascending=False)
        if sub.empty: continue
        with st.expander(
            f"**{cluster_names[c]}** — {len(sub)} occupations, "
            f"{int(sub['skill_count'].sum()):,} skills"
        ):
            st.dataframe(
                sub[["title","major","uoc_count","skill_count"]].rename(columns={
                    "title":"Occupation","major":"Major Group",
                    "uoc_count":"UOCs","skill_count":"Skills",
                }),
                use_container_width=True, hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CROSS-TP BRIDGES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">Cross-TP Skill Bridges</div>',
                unsafe_allow_html=True)
    st.caption("Occupations from different training packages that share "
               "semantically similar skills — hidden transferability across industries.")

    tp_list_all = sorted(df["tp_code"].dropna().unique())
    b1, b2 = st.columns(2)
    with b1:
        src_tp = st.selectbox("Source TP", tp_list_all, key="bridge_src")
    with b2:
        tgt_tp = st.selectbox("Target TP",
            [t for t in tp_list_all if t != src_tp],
            key="bridge_tgt")

    min_bridge = st.slider("Min semantic similarity", 0.05, 0.9, 0.20, 0.05,
                           key="bridge_sim")

    src_occs  = df[df["tp_code"] == src_tp]["anzsco_code"].unique()
    tgt_occs  = df[df["tp_code"] == tgt_tp]["anzsco_code"].unique()
    valid_src = [c for c in src_occs if c in occ_codes]
    valid_tgt = [c for c in tgt_occs if c in occ_codes]

    if not valid_src or not valid_tgt:
        st.warning("Not enough occupations in selected TPs.")
    else:
        src_idx = [occ_codes.index(c) for c in valid_src]
        tgt_idx = [occ_codes.index(c) for c in valid_tgt]
        cross   = sim_matrix[np.ix_(src_idx, tgt_idx)]

        bridges = []
        for i, si in enumerate(valid_src):
            for j, tj in enumerate(valid_tgt):
                s = float(cross[i, j])
                if s >= min_bridge:
                    bridges.append({
                        "src_code":  si,
                        "src_title": occ_label.get(si, si),
                        "tgt_code":  tj,
                        "tgt_title": occ_label.get(tj, tj),
                        "similarity": round(s, 3),
                    })

        if not bridges:
            st.info(f"No bridges found at similarity ≥ {min_bridge}. "
                    "Try lowering the threshold.")
        else:
            bridges_df = pd.DataFrame(bridges).sort_values(
                "similarity", ascending=False)

            st.success(
                f"**{len(bridges)}** semantic bridges between "
                f"**{src_tp}** and **{tgt_tp}**"
            )

            # Sankey
            src_labels = bridges_df["src_title"].unique().tolist()
            tgt_labels = bridges_df["tgt_title"].unique().tolist()
            all_labels = src_labels + tgt_labels
            label_idx  = {l: i for i, l in enumerate(all_labels)}

            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    label=all_labels,
                    color=(["#ef9f27"] * len(src_labels) +
                           ["#7eb8f7"] * len(tgt_labels)),
                    pad=15, thickness=20,
                ),
                link=dict(
                    source=[label_idx[r["src_title"]] for _, r in bridges_df.iterrows()],
                    target=[label_idx[r["tgt_title"]] for _, r in bridges_df.iterrows()],
                    value=[int(r["similarity"] * 100) for _, r in bridges_df.iterrows()],
                    color="rgba(126,184,247,0.3)",
                ),
            ))
            fig_sankey.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#7eb8f7",
                title=f"Semantic bridges: {src_tp} → {tgt_tp}",
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            # Detail table
            st.dataframe(
                bridges_df[["src_title","tgt_title","similarity"]].rename(columns={
                    "src_title": src_tp, "tgt_title": tgt_tp,
                    "similarity": "Semantic similarity",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Semantic similarity": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.3f"),
                },
            )
            st.download_button(
                "⬇ Export bridges CSV",
                bridges_df.to_csv(index=False).encode(),
                f"bridges_{src_tp}_{tgt_tp}.csv", "text/csv",
            )
