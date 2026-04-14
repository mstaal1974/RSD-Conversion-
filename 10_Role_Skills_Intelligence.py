"""
pages/9_🔗_Role_Skills_Intelligence.py

Five tools for understanding skill connections between occupations:

1. Skill Transferability Map    — occupation network by shared skills
2. Career Pathway Engine        — closest roles by skill similarity
3. Skill Gap Analysis           — compare two occupations
4. Occupation Cluster Map       — skill families across industries
5. Cross-TP Skill Bridges       — hidden transferability across TPs
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
.gap-shared{background:#1b3a1b;border-left:3px solid #4caf50;padding:6px 10px;border-radius:4px;margin-bottom:4px}
.gap-missing{background:#3a1b1b;border-left:3px solid #f44336;padding:6px 10px;border-radius:4px;margin-bottom:4px}
.gap-unique{background:#1b2a3a;border-left:3px solid #2196f3;padding:6px 10px;border-radius:4px;margin-bottom:4px}
</style>
""", unsafe_allow_html=True)

st.title("🔗 Role Skills Intelligence")
st.caption("Five tools for understanding skill connections and transferability between occupations")

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

# ── Load core dataset ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading occupation-skill data…")
def load_occ_skills() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                o.anzsco_code, o.anzsco_title, o.anzsco_major_group,
                o.confidence, o.mapping_source,
                s.unit_code, s.unit_title, s.tp_code,
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

# Summary
n_occ   = df["anzsco_code"].nunique()
n_uocs  = df["unit_code"].nunique()
n_skills = len(df)

c1,c2,c3 = st.columns(3)
for col, val, lbl in [(c1,f"{n_occ:,}","Occupations"),(c2,f"{n_uocs:,}","UOCs"),(c3,f"{n_skills:,}","Skill statements")]:
    col.markdown(f'<div class="intel-card"><div class="intel-val">{val}</div>'
                 f'<div class="intel-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── Precompute occupation-UOC matrix ─────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Computing skill matrices…")
def compute_matrices(df_hash: int):
    # Occupation × UOC presence matrix
    occ_uoc = df.groupby(["anzsco_code","unit_code"]).size().unstack(fill_value=0)
    occ_uoc_bin = (occ_uoc > 0).astype(int)

    # Occupation metadata
    occ_meta = (
        df.groupby("anzsco_code")
        .agg(
            title=("anzsco_title","first"),
            major=("anzsco_major_group","first"),
            uoc_count=("unit_code","nunique"),
            skill_count=("skill_statement","count"),
            tp_count=("tp_code","nunique"),
        )
        .reset_index()
    )

    # Jaccard similarity between occupations
    mat = occ_uoc_bin.values.astype(float)
    intersection = mat @ mat.T
    row_sums = mat.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union = np.where(union == 0, 1, union)
    jaccard = intersection / union
    np.fill_diagonal(jaccard, 0)

    return occ_uoc_bin, occ_meta, jaccard

occ_uoc_bin, occ_meta, jaccard = compute_matrices(len(df))
occ_list = occ_meta["anzsco_code"].tolist()
occ_label = {r["anzsco_code"]: f"{r['title']}" for _, r in occ_meta.iterrows()}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌐 Transferability Map",
    "🚀 Career Pathways",
    "📊 Skill Gap Analysis",
    "🗺️ Occupation Clusters",
    "🌉 Cross-TP Bridges",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SKILL TRANSFERABILITY MAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-hdr">Skill Transferability Map</div>',
                unsafe_allow_html=True)
    st.caption("Occupations linked by shared skills. Thicker edges = more shared UOCs. "
               "Larger nodes = more skills. Clusters = skill families.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        min_sim = st.slider("Min similarity threshold", 0.05, 0.8, 0.15, 0.05,
                            key="map_sim")
    with col_s2:
        major_filter = st.selectbox("Filter major group",
            ["All"] + sorted(df["anzsco_major_group"].dropna().unique()),
            key="map_major")
    with col_s3:
        max_nodes = st.slider("Max occupations", 20, min(200, n_occ), 60, 10,
                              key="map_nodes")

    # Filter
    if major_filter != "All":
        filtered_codes = occ_meta[occ_meta["major"] == major_filter]["anzsco_code"].tolist()
    else:
        # Take top N by skill count
        filtered_codes = occ_meta.nlargest(max_nodes, "skill_count")["anzsco_code"].tolist()

    filtered_codes = filtered_codes[:max_nodes]
    idx_map = {c: i for i, c in enumerate(filtered_codes)
               if c in occ_uoc_bin.index}
    valid_codes = list(idx_map.keys())

    if len(valid_codes) < 2:
        st.warning("Not enough occupations for selected filters.")
    else:
        sub_idx = [occ_uoc_bin.index.get_loc(c) for c in valid_codes]
        sub_jac = jaccard[np.ix_(sub_idx, sub_idx)]

        # Network layout
        import networkx as nx
        G = nx.Graph()
        for c in valid_codes:
            m = occ_meta[occ_meta["anzsco_code"] == c].iloc[0]
            G.add_node(c, title=occ_label.get(c, c),
                       size=float(m["skill_count"]),
                       major=m["major"])

        for i, ci in enumerate(valid_codes):
            for j, cj in enumerate(valid_codes):
                if i < j and sub_jac[i, j] >= min_sim:
                    G.add_edge(ci, cj, weight=float(sub_jac[i, j]))

        pos = nx.spring_layout(G, k=2/np.sqrt(max(len(valid_codes),1)),
                               seed=42, iterations=50)

        # Build traces
        major_colors = {
            "Managers": "#ef9f27",
            "Professionals": "#7eb8f7",
            "Technicians and Trades Workers": "#a5d6a7",
            "Community and Personal Service Workers": "#ce93d8",
            "Clerical and Administrative Workers": "#80deea",
            "Sales Workers": "#f48fb1",
            "Machinery Operators and Drivers": "#ffcc80",
            "Labourers": "#bcaaa4",
        }

        fig_net = go.Figure()

        # Edges
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            w = d["weight"]
            fig_net.add_trace(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None],
                mode="lines",
                line=dict(width=w*8, color=f"rgba(126,184,247,{min(w*2,0.6)})"),
                hoverinfo="none", showlegend=False,
            ))

        # Nodes per major group
        for major, color in major_colors.items():
            nodes = [c for c in valid_codes
                     if G.nodes[c].get("major") == major and c in pos]
            if not nodes: continue
            m_data = occ_meta[occ_meta["anzsco_code"].isin(nodes)]
            fig_net.add_trace(go.Scatter(
                x=[pos[c][0] for c in nodes],
                y=[pos[c][1] for c in nodes],
                mode="markers+text",
                name=major,
                marker=dict(
                    size=[min(5 + m_data[m_data["anzsco_code"]==c]["skill_count"].values[0]/8, 30)
                          for c in nodes],
                    color=color, opacity=0.85,
                    line=dict(width=1, color="#0a1628"),
                ),
                text=[occ_label.get(c,c)[:20] for c in nodes],
                textposition="top center",
                textfont=dict(size=8, color="#cfd8dc"),
                hovertext=[
                    f"<b>{occ_label.get(c,c)}</b><br>"
                    f"ANZSCO: {c}<br>"
                    f"Skills: {m_data[m_data['anzsco_code']==c]['skill_count'].values[0]}<br>"
                    f"UOCs: {m_data[m_data['anzsco_code']==c]['uoc_count'].values[0]}"
                    for c in nodes
                ],
                hoverinfo="text",
            ))

        fig_net.update_layout(
            height=600, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#080e1a",
            font_color="#7eb8f7",
            legend=dict(x=1.01, y=1, bgcolor="rgba(10,22,40,0.8)",
                        bordercolor="#1a2a4a", font_size=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20,r=200,t=40,b=20),
            title=f"Skill Transferability Network — {len(valid_codes)} occupations, "
                  f"{G.number_of_edges()} connections",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Top transferable pairs
        st.markdown('<div class="section-hdr">Most transferable occupation pairs</div>',
                    unsafe_allow_html=True)
        pairs = []
        for i, ci in enumerate(valid_codes):
            for j, cj in enumerate(valid_codes):
                if i < j and sub_jac[i,j] > 0:
                    shared = int((occ_uoc_bin.loc[ci] & occ_uoc_bin.loc[cj]).sum())
                    pairs.append({
                        "Occupation A": occ_label.get(ci, ci),
                        "Occupation B": occ_label.get(cj, cj),
                        "Shared UOCs": shared,
                        "Similarity": round(float(sub_jac[i,j]), 3),
                    })
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("Similarity", ascending=False).head(20)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True,
                column_config={
                    "Similarity": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.3f"),
                })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CAREER PATHWAY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-hdr">Career Pathway Engine</div>',
                unsafe_allow_html=True)
    st.caption("Starting from any occupation, find the closest roles by skill overlap. "
               "Shows % of skills already held and what's needed to transition.")

    sel_occ = st.selectbox(
        "Starting occupation",
        options=occ_list,
        format_func=lambda c: f"{occ_label.get(c,c)} ({c})",
        key="path_occ",
    )

    top_n_paths = st.slider("Show top N pathways", 5, 30, 10, key="path_n")

    if sel_occ in occ_uoc_bin.index:
        occ_idx = list(occ_uoc_bin.index).index(sel_occ)
        sims    = jaccard[occ_idx]
        top_idx = np.argsort(sims)[::-1][:top_n_paths+1]
        top_idx = [i for i in top_idx if i != occ_idx][:top_n_paths]

        src_uocs  = set(occ_uoc_bin.columns[occ_uoc_bin.loc[sel_occ] > 0])
        src_skills = df[df["anzsco_code"] == sel_occ]["skill_statement"].nunique()

        st.markdown(f"**{occ_label.get(sel_occ, sel_occ)}** — "
                    f"{src_skills} skills · {len(src_uocs)} UOCs")
        st.divider()

        pathway_data = []
        for idx in top_idx:
            tgt_code = occ_uoc_bin.index[idx]
            tgt_uocs = set(occ_uoc_bin.columns[occ_uoc_bin.loc[tgt_code] > 0])
            shared   = src_uocs & tgt_uocs
            missing  = tgt_uocs - src_uocs
            extra    = src_uocs - tgt_uocs
            tgt_meta = occ_meta[occ_meta["anzsco_code"] == tgt_code].iloc[0]
            transferability = len(shared) / max(len(tgt_uocs), 1) * 100

            pathway_data.append({
                "code":            tgt_code,
                "title":           occ_label.get(tgt_code, tgt_code),
                "major":           tgt_meta["major"],
                "similarity":      round(float(sims[idx]), 3),
                "transferability": round(transferability, 1),
                "shared_uocs":     len(shared),
                "missing_uocs":    len(missing),
                "shared_list":     sorted(shared),
                "missing_list":    sorted(missing),
            })

        for p in pathway_data:
            pct   = p["transferability"]
            color = "#1b5e20" if pct >= 70 else "#0d47a1" if pct >= 40 else "#bf360c"
            with st.expander(
                f"**{p['title']}** — {pct:.0f}% transferable · "
                f"{p['shared_uocs']} shared UOCs · {p['missing_uocs']} to gain",
                expanded=False
            ):
                pa, pb = st.columns(2)
                with pa:
                    st.markdown(
                        f'<div style="background:#0a1628;border-radius:6px;padding:10px">'
                        f'<div style="font-size:0.7rem;color:#4a6fa5;margin-bottom:4px">'
                        f'TRANSFERABILITY</div>'
                        f'<div style="background:#1a2a4a;border-radius:3px;height:12px">'
                        f'<div style="width:{pct:.0f}%;background:{color};height:12px;'
                        f'border-radius:3px"></div></div>'
                        f'<div style="font-family:monospace;color:{color};margin-top:4px">'
                        f'{pct:.1f}% of skills already held</div></div>',
                        unsafe_allow_html=True
                    )
                with pb:
                    st.metric("Shared UOCs", p["shared_uocs"])

                if p["missing_list"]:
                    st.markdown("**UOCs needed to transition:**")
                    missing_df = df[df["unit_code"].isin(p["missing_list"])][
                        ["unit_code","unit_title","tp_code"]
                    ].drop_duplicates("unit_code")
                    st.dataframe(missing_df, use_container_width=True,
                                 hide_index=True)

        # Summary chart
        path_df = pd.DataFrame([{
            "Occupation": p["title"][:30],
            "Transferability %": p["transferability"],
            "Shared UOCs": p["shared_uocs"],
            "Missing UOCs": p["missing_uocs"],
        } for p in pathway_data])

        fig_path = px.bar(
            path_df, x="Transferability %", y="Occupation",
            orientation="h", color="Transferability %",
            color_continuous_scale=["#bf360c","#0d47a1","#1b5e20"],
            range_color=[0,100],
            title=f"Career pathways from {occ_label.get(sel_occ,sel_occ)[:40]}",
            height=max(300, len(pathway_data)*35),
        )
        fig_path.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="#080e1a",
                               font_color="#7eb8f7", showlegend=False)
        st.plotly_chart(fig_path, use_container_width=True)

        # Export
        export_rows = []
        for p in pathway_data:
            for uoc in p["missing_list"]:
                u = df[df["unit_code"] == uoc][["unit_code","unit_title","tp_code"]].drop_duplicates()
                if not u.empty:
                    export_rows.append({
                        "From Occupation": occ_label.get(sel_occ, sel_occ),
                        "To Occupation": p["title"],
                        "Transferability %": p["transferability"],
                        "Gap UOC Code": uoc,
                        "Gap UOC Title": u.iloc[0]["unit_title"],
                        "Training Package": u.iloc[0]["tp_code"],
                    })
        if export_rows:
            st.download_button(
                "⬇ Export pathway gap analysis CSV",
                pd.DataFrame(export_rows).to_csv(index=False).encode(),
                f"pathway_gaps_{sel_occ}.csv", "text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SKILL GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">Skill Gap Analysis</div>',
                unsafe_allow_html=True)
    st.caption("Compare two occupations — shared skills, unique skills, and the gap to bridge.")

    g1, g2 = st.columns(2)
    with g1:
        occ_a = st.selectbox("Occupation A (current role)",
            options=occ_list,
            format_func=lambda c: f"{occ_label.get(c,c)} ({c})",
            key="gap_a")
    with g2:
        occ_b = st.selectbox("Occupation B (target role)",
            options=occ_list,
            format_func=lambda c: f"{occ_label.get(c,c)} ({c})",
            index=min(1, len(occ_list)-1),
            key="gap_b")

    if occ_a == occ_b:
        st.info("Select two different occupations to compare.")
    elif occ_a in occ_uoc_bin.index and occ_b in occ_uoc_bin.index:
        uocs_a = set(occ_uoc_bin.columns[occ_uoc_bin.loc[occ_a] > 0])
        uocs_b = set(occ_uoc_bin.columns[occ_uoc_bin.loc[occ_b] > 0])
        shared  = uocs_a & uocs_b
        only_a  = uocs_a - uocs_b
        only_b  = uocs_b - uocs_a

        transferability = len(shared) / max(len(uocs_b), 1) * 100
        ai = list(occ_uoc_bin.index).index(occ_a)
        bi = list(occ_uoc_bin.index).index(occ_b)
        sim = float(jaccard[ai, bi])

        # Header metrics
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Shared UOCs", len(shared))
        m2.metric(f"Only in {occ_a[:6]}…", len(only_a))
        m3.metric(f"Gap to {occ_b[:6]}…", len(only_b))
        m4.metric("Transferability", f"{transferability:.0f}%")

        # Venn-style bar
        total = len(uocs_a | uocs_b)
        st.markdown(
            f'<div style="background:#1a2a4a;border-radius:6px;height:20px;'
            f'display:flex;margin-bottom:16px">'
            f'<div style="width:{len(only_a)/max(total,1)*100:.0f}%;background:#ef9f27;'
            f'border-radius:6px 0 0 6px;height:20px" title="Only in A"></div>'
            f'<div style="width:{len(shared)/max(total,1)*100:.0f}%;background:#4caf50;'
            f'height:20px" title="Shared"></div>'
            f'<div style="width:{len(only_b)/max(total,1)*100:.0f}%;background:#7eb8f7;'
            f'border-radius:0 6px 6px 0;height:20px" title="Only in B"></div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;font-size:0.72rem;color:#7eb8f7;margin-bottom:16px">'
            f'<span>🟠 Only in A: {len(only_a)}</span>'
            f'<span>🟢 Shared: {len(shared)}</span>'
            f'<span>🔵 Gap (only in B): {len(only_b)}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        col3a, col3b, col3c = st.columns(3)

        def show_uoc_list(col, uoc_set, label, css_class):
            with col:
                st.markdown(f"**{label}** ({len(uoc_set)})")
                uoc_df = df[df["unit_code"].isin(uoc_set)][
                    ["unit_code","unit_title","tp_code"]
                ].drop_duplicates("unit_code").head(20)
                for _, row in uoc_df.iterrows():
                    st.markdown(
                        f'<div class="{css_class}" style="font-size:0.78rem;margin-bottom:3px">'
                        f'<code style="color:#7eb8f7">{row["unit_code"]}</code> '
                        f'{row["unit_title"][:50]}</div>',
                        unsafe_allow_html=True
                    )
                if len(uoc_set) > 20:
                    st.caption(f"… and {len(uoc_set)-20} more")

        show_uoc_list(col3a, only_a,
                      f"Only in {occ_label.get(occ_a,occ_a)[:25]}", "gap-unique")
        show_uoc_list(col3b, shared, "Shared by both", "gap-shared")
        show_uoc_list(col3c, only_b,
                      f"Gap — needed for {occ_label.get(occ_b,occ_b)[:20]}", "gap-missing")

        # Skill statement comparison
        st.divider()
        st.markdown("**Skill statement comparison**")
        skills_a = set(df[df["anzsco_code"] == occ_a]["skill_statement"].tolist())
        skills_b = set(df[df["anzsco_code"] == occ_b]["skill_statement"].tolist())

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs

        all_skills = list(skills_a | skills_b)
        if len(all_skills) > 1:
            vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                                  max_features=3000)
            tfidf = vec.fit_transform(all_skills)
            sim_mat = cs(tfidf[:len(skills_a)], tfidf[len(skills_a):])
            avg_semantic_sim = float(sim_mat.mean())
            st.metric("Semantic skill similarity", f"{avg_semantic_sim:.3f}",
                     help="Average cosine similarity between skill statements")

        # Export gap analysis
        gap_export = []
        for uoc in only_b:
            u = df[df["unit_code"] == uoc][["unit_code","unit_title","tp_code"]].drop_duplicates()
            if not u.empty:
                stmts = df[(df["unit_code"] == uoc) & (df["anzsco_code"] == occ_b)]["skill_statement"].tolist()
                for stmt in stmts[:3]:
                    gap_export.append({
                        "From": occ_label.get(occ_a, occ_a),
                        "To": occ_label.get(occ_b, occ_b),
                        "Gap UOC": uoc,
                        "Unit Title": u.iloc[0]["unit_title"],
                        "TP": u.iloc[0]["tp_code"],
                        "Skill needed": stmt,
                    })
        if gap_export:
            st.download_button(
                "⬇ Export skill gap CSV",
                pd.DataFrame(gap_export).to_csv(index=False).encode(),
                f"skill_gap_{occ_a}_to_{occ_b}.csv", "text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — OCCUPATION CLUSTER MAP
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-hdr">Occupation Cluster Map</div>',
                unsafe_allow_html=True)
    st.caption("Occupations grouped by skill similarity — reveals natural 'skill families' "
               "that cross traditional industry boundaries.")

    n_clust = st.slider("Number of skill clusters", 3, 20, 8, key="clust_n")

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    mat_vals = occ_uoc_bin.values.astype(float)
    if mat_vals.shape[0] >= n_clust:
        km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(mat_vals)

        # 2D PCA for visualisation
        n_comp = min(2, mat_vals.shape[1]-1, mat_vals.shape[0]-1)
        pca = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(mat_vals)

        clust_df = occ_meta.copy()
        clust_df["cluster"] = cluster_labels
        if coords.shape[1] >= 1: clust_df["pc1"] = coords[:,0]
        if coords.shape[1] >= 2: clust_df["pc2"] = coords[:,1]

        # Cluster labels from top shared UOCs
        cluster_names = {}
        for c in range(n_clust):
            cidx = np.where(cluster_labels == c)[0]
            if len(cidx) == 0:
                cluster_names[c] = f"Cluster {c}"
                continue
            sub = mat_vals[cidx].sum(axis=0)
            top_uocs = occ_uoc_bin.columns[sub.argsort()[-3:][::-1]].tolist()
            top_tps = [u[:3] for u in top_uocs]
            cluster_names[c] = " · ".join(dict.fromkeys(top_tps))

        clust_df["cluster_name"] = clust_df["cluster"].map(cluster_names)

        fig_clust = px.scatter(
            clust_df, x="pc1", y="pc2",
            color="cluster_name",
            size="skill_count",
            hover_data=["title","anzsco_code","uoc_count","major"],
            text="title",
            height=600,
            title=f"Occupation Skill Families — {n_clust} clusters",
        )
        fig_clust.update_traces(
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(opacity=0.8, line=dict(width=0.5, color="#0a1628")),
        )
        fig_clust.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#080e1a",
            font_color="#7eb8f7",
            xaxis=dict(title="Skill dimension 1", gridcolor="#1a2a4a"),
            yaxis=dict(title="Skill dimension 2", gridcolor="#1a2a4a"),
        )
        st.plotly_chart(fig_clust, use_container_width=True)

        # Cluster details
        st.markdown('<div class="section-hdr">Cluster composition</div>',
                    unsafe_allow_html=True)
        for c in range(n_clust):
            sub = clust_df[clust_df["cluster"] == c].sort_values(
                "skill_count", ascending=False)
            if sub.empty: continue
            with st.expander(
                f"**{cluster_names[c]}** — {len(sub)} occupations, "
                f"{int(sub['skill_count'].sum())} total skills"
            ):
                st.dataframe(
                    sub[["title","major","uoc_count","skill_count","tp_count"]].rename(columns={
                        "title":"Occupation","major":"Major Group",
                        "uoc_count":"UOCs","skill_count":"Skills","tp_count":"TPs"
                    }),
                    use_container_width=True, hide_index=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CROSS-TP SKILL BRIDGES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">Cross-TP Skill Bridges</div>',
                unsafe_allow_html=True)
    st.caption("UOCs from one training package that deliver skills relevant to occupations "
               "in a completely different TP — hidden transferability across industries.")

    tp_list_all = sorted(df["tp_code"].dropna().unique())
    b1, b2 = st.columns(2)
    with b1:
        src_tp = st.selectbox("Source training package", tp_list_all, key="bridge_src")
    with b2:
        tgt_tp = st.selectbox("Target training package",
            [t for t in tp_list_all if t != src_tp],
            key="bridge_tgt")

    min_bridge_sim = st.slider("Min similarity for bridge", 0.1, 0.8, 0.2, 0.05,
                               key="bridge_sim")

    # Get occupations for each TP
    src_occs = df[df["tp_code"] == src_tp]["anzsco_code"].unique()
    tgt_occs = df[df["tp_code"] == tgt_tp]["anzsco_code"].unique()

    valid_src = [c for c in src_occs if c in occ_uoc_bin.index]
    valid_tgt = [c for c in tgt_occs if c in occ_uoc_bin.index]

    if not valid_src or not valid_tgt:
        st.warning("Not enough occupations in selected TPs.")
    else:
        src_idx = [list(occ_uoc_bin.index).index(c) for c in valid_src]
        tgt_idx = [list(occ_uoc_bin.index).index(c) for c in valid_tgt]
        cross_sim = jaccard[np.ix_(src_idx, tgt_idx)]

        bridges = []
        for i, si in enumerate(valid_src):
            for j, tj in enumerate(valid_tgt):
                sim = float(cross_sim[i,j])
                if sim >= min_bridge_sim:
                    shared = set(occ_uoc_bin.columns[occ_uoc_bin.loc[si] > 0]) & \
                             set(occ_uoc_bin.columns[occ_uoc_bin.loc[tj] > 0])
                    bridges.append({
                        "src_code": si,
                        "src_title": occ_label.get(si, si),
                        "tgt_code": tj,
                        "tgt_title": occ_label.get(tj, tj),
                        "similarity": round(sim, 3),
                        "shared_uocs": len(shared),
                        "bridge_uocs": sorted(shared),
                    })

        if not bridges:
            st.info(f"No bridges found between {src_tp} and {tgt_tp} "
                    f"at similarity ≥ {min_bridge_sim}. Try lowering the threshold.")
        else:
            bridges_df = pd.DataFrame(bridges).sort_values("similarity", ascending=False)

            st.success(f"Found **{len(bridges)}** skill bridges between "
                       f"**{src_tp}** and **{tgt_tp}**")

            # Sankey diagram
            src_labels = bridges_df["src_title"].unique().tolist()
            tgt_labels = bridges_df["tgt_title"].unique().tolist()
            all_labels = src_labels + tgt_labels
            label_idx  = {l: i for i, l in enumerate(all_labels)}

            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    label=all_labels,
                    color=["#ef9f27"]*len(src_labels) + ["#7eb8f7"]*len(tgt_labels),
                    pad=15, thickness=20,
                ),
                link=dict(
                    source=[label_idx[r["src_title"]] for _, r in bridges_df.iterrows()],
                    target=[label_idx[r["tgt_title"]] for _, r in bridges_df.iterrows()],
                    value=[r["shared_uocs"] for _, r in bridges_df.iterrows()],
                    color="rgba(126,184,247,0.3)",
                ),
            ))
            fig_sankey.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#7eb8f7",
                font_family="IBM Plex Mono",
                title=f"Skill bridges: {src_tp} → {tgt_tp}",
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            # Bridge details
            st.markdown('<div class="section-hdr">Bridge details</div>',
                        unsafe_allow_html=True)
            for _, row in bridges_df.head(15).iterrows():
                with st.expander(
                    f"**{row['src_title'][:30]}** → **{row['tgt_title'][:30]}** "
                    f"· {row['shared_uocs']} shared UOCs · sim {row['similarity']:.3f}"
                ):
                    bridge_uoc_df = df[df["unit_code"].isin(row["bridge_uocs"])][
                        ["unit_code","unit_title","tp_code"]
                    ].drop_duplicates("unit_code")
                    st.dataframe(bridge_uoc_df, use_container_width=True,
                                 hide_index=True)

            st.download_button(
                "⬇ Export cross-TP bridges CSV",
                bridges_df[["src_title","tgt_title","similarity","shared_uocs"]].to_csv(
                    index=False).encode(),
                f"bridges_{src_tp}_{tgt_tp}.csv", "text/csv",
            )
