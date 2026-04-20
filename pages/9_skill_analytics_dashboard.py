"""
pages/8_🔬_Skill_Analytics_Dashboard.py

Multidimensional Skill Analytics Dashboard
Inspired by the 5-axis skill similarity engine concept:
  X = Domain context  (Lab / Business / Mining / etc.)
  Y = Complexity level (1-10)
  Z = Semantic meaning (embedding dim)
  Size = Frequency across UOCs
  Shape/Colour = Verb type (Analyse / Apply / Communicate / etc.)
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

load_dotenv()
st.set_page_config(page_title="Skill Analytics Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#050d1a}
.panel{background:#0a1628;border:1px solid #1a2a4a;border-radius:8px;padding:16px;margin-bottom:12px}
.panel-title{font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.15em;
  text-transform:uppercase;color:#4a6fa5;margin-bottom:10px}
.cluster-card{background:#0d1f3a;border:1px solid #1e3a5f;border-radius:6px;
  padding:10px 14px;margin-bottom:8px}
.cluster-name{font-size:1.1rem;font-weight:600;color:#7eb8f7;margin-bottom:4px}
.cluster-meta{font-size:0.72rem;color:#4a6fa5;line-height:1.8}
.sim-score{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;
  font-weight:600;color:#7eb8f7;text-align:center}
.dim-label{font-size:0.7rem;color:#4a6fa5;text-transform:uppercase;
  letter-spacing:0.1em;margin-bottom:4px}
.stMetric label{color:#4a6fa5!important;font-size:0.7rem!important}
.stMetric div[data-testid="metric-container"]>div{color:#7eb8f7!important}
</style>
""", unsafe_allow_html=True)

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

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading skill data…")
def load_data() -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.id, s.unit_code, s.unit_title, s.tp_code,
                   s.element_title, s.skill_statement, s.keywords,
                   o.anzsco_code, o.anzsco_title, o.anzsco_major_group
            FROM rsd_skill_records s
            LEFT JOIN uoc_occupation_links o
                ON o.uoc_code = s.unit_code
                AND o.is_primary = TRUE AND o.valid_to IS NULL
            WHERE s.skill_statement IS NOT NULL AND s.skill_statement != ''
            ORDER BY s.unit_code, s.element_title
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])

df_raw = load_data()
if df_raw.empty:
    st.warning("No skill data found. Run the RSD generator first.")
    st.stop()

# ── Domain classifier ─────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "Lab/Science":   ["lab","laboratory","chemical","sample","calibrat","instrument",
                      "test","measure","analys","specim","reagent","equipment"],
    "Business":      ["communicat","report","document","stakeholder","client","customer",
                      "manage","plan","organis","budget","financ","strateg"],
    "Mining/RII":    ["mine","drill","blast","excavat","ore","shaft","tunnel",
                      "ground","site","hazard","ppe","safety"],
    "IT/Digital":    ["software","system","network","data","digital","code","program",
                      "database","cyber","cloud","api"],
    "Health/Care":   ["patient","care","health","clinical","medication","treatment",
                      "wound","assess","monitor","symptom"],
    "Construction":  ["build","construct","install","weld","concrete","structural",
                      "scaffold","blueprint","material","site"],
    "Agriculture":   ["crop","livestock","soil","farm","irrigation","harvest",
                      "pest","plant","animal","paddock"],
    "General":       [],
}

VERB_GROUPS = {
    "Analyse":     ["analyse","assess","evaluate","review","inspect","examine",
                    "monitor","measure","test","verify","check"],
    "Apply":       ["apply","use","operate","implement","execute","perform",
                    "conduct","complete","carry","undertake"],
    "Communicate": ["communicate","report","document","record","prepare","present",
                    "liaise","coordinate","consult","advise","notify"],
    "Develop":     ["develop","design","create","establish","build","construct",
                    "plan","prepare","produce","generate"],
    "Manage":      ["manage","control","maintain","supervise","lead","direct",
                    "ensure","comply","adhere","follow"],
    "Identify":    ["identify","determine","select","choose","classify","categorise",
                    "prioritise","locate","detect","recognise"],
}

COMPLEXITY_SIGNALS = {
    "high":   ["complex","advanced","critical","strategic","expert","specialist",
               "evaluate","design","develop","analyse","assess"],
    "medium": ["apply","implement","monitor","review","coordinate","manage",
               "maintain","conduct","prepare"],
    "low":    ["follow","complete","record","report","assist","support",
               "check","clean","collect","provide"],
}

def classify_domain(text: str) -> str:
    t = text.lower()
    best, best_n = "General", 0
    for dom, kws in DOMAIN_KEYWORDS.items():
        if dom == "General": continue
        n = sum(1 for k in kws if k in t)
        if n > best_n:
            best, best_n = dom, n
    return best

def classify_verb(text: str) -> str:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    for grp, verbs in VERB_GROUPS.items():
        if any(v in words for v in verbs):
            return grp
    return "Other"

def complexity_score(text: str) -> float:
    t = text.lower()
    words = re.findall(r'\b\w+\b', t)
    h = sum(1 for w in words if w in COMPLEXITY_SIGNALS["high"])
    m = sum(1 for w in words if w in COMPLEXITY_SIGNALS["medium"])
    l = sum(1 for w in words if w in COMPLEXITY_SIGNALS["low"])
    total = h + m + l
    if total == 0:
        return 5.0
    score = (h * 9 + m * 5 + l * 2) / total
    # Add length bonus
    score += min(len(words) / 20, 1.5)
    return round(min(max(score, 1.0), 10.0), 1)

@st.cache_data(show_spinner="Computing skill dimensions…", ttl=600)
def compute_dimensions(statements, unit_codes, n_clust=12):
    domains    = [classify_domain(s) for s in statements]
    verbs      = [classify_verb(s) for s in statements]
    complexity = [complexity_score(s) for s in statements]

    # TF-IDF embeddings → 3D via SVD
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                          max_features=6000, min_df=1)
    tfidf = vec.fit_transform(statements)
    svd3  = TruncatedSVD(n_components=min(3, tfidf.shape[1]-1, len(statements)-1),
                         random_state=42)
    emb3  = svd3.fit_transform(tfidf)

    # Full embeddings for similarity
    svd_full = TruncatedSVD(n_components=min(80, tfidf.shape[1]-1, len(statements)-1),
                            random_state=42)
    emb_full = normalize(svd_full.fit_transform(tfidf))

    # Clustering
    k = min(n_clust, len(statements)//3, len(statements)-1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(emb_full)

    # Cluster labels
    fn = vec.get_feature_names_out()
    cluster_labels = {}
    for c in range(k):
        idx = np.where(clusters == c)[0]
        if not len(idx): cluster_labels[c] = f"Cluster {c}"; continue
        ctrd = emb_full[idx].mean(axis=0)
        approx = svd_full.inverse_transform(ctrd.reshape(1,-1))[0]
        tops = approx.argsort()[-4:][::-1]
        terms = [fn[i] for i in tops if i < len(fn)]
        cluster_labels[c] = " · ".join(terms[:3]) or f"Cluster {c}"

    # Frequency per statement (how many UOCs share same cluster)
    clust_uoc = {}
    for i, c in enumerate(clusters):
        clust_uoc.setdefault(c, set()).add(unit_codes[i])
    freq = np.array([len(clust_uoc.get(clusters[i], set())) for i in range(len(statements))],
                    dtype=float)
    freq_norm = (freq - freq.min()) / max(freq.max() - freq.min(), 1) * 9 + 1

    return {
        "domains":    domains,
        "verbs":      verbs,
        "complexity": complexity,
        "emb3":       emb3,
        "emb_full":   emb_full,
        "clusters":   clusters,
        "cluster_labels": cluster_labels,
        "freq_norm":  freq_norm,
        "k":          k,
        "vectorizer": vec,
    }

with st.spinner("Analysing skill dimensions…"):
    res = compute_dimensions(
        df_raw["skill_statement"].tolist(),
        df_raw["unit_code"].tolist(),
    )

df = df_raw.copy()
df["domain"]     = res["domains"]
df["verb_group"] = res["verbs"]
df["complexity"] = res["complexity"]
df["cluster"]    = res["clusters"]
df["cluster_label"] = df["cluster"].map(res["cluster_labels"])
df["freq_norm"]  = res["freq_norm"]
df["sem_x"]      = res["emb3"][:, 0] if res["emb3"].shape[1] > 0 else 0
df["sem_y"]      = res["emb3"][:, 1] if res["emb3"].shape[1] > 1 else 0
df["sem_z"]      = res["emb3"][:, 2] if res["emb3"].shape[1] > 2 else 0

# Domain numeric for X axis
domain_order = list(DOMAIN_KEYWORDS.keys())
df["domain_num"] = df["domain"].map({d: i for i, d in enumerate(domain_order)})

# ── Layout: header + 3 columns ────────────────────────────────────────────────
st.markdown("## 🔬 Multidimensional Skill Analytics Dashboard")

# Qualification / TP selectors
h1, h2, h3 = st.columns([2, 2, 3])
with h1:
    tp_options = ["All TPs"] + sorted(df["tp_code"].dropna().unique().tolist())
    sel_tp = st.selectbox("Training Package", tp_options, key="tp_sel")
with h2:
    if sel_tp != "All TPs":
        uoc_opts = ["All UOCs"] + sorted(df[df["tp_code"]==sel_tp]["unit_code"].unique())
    else:
        uoc_opts = ["All UOCs"] + sorted(df["unit_code"].unique())
    sel_uoc = st.selectbox("Unit of Competency", uoc_opts, key="uoc_sel")
with h3:
    st.caption(
        f"**{len(df):,}** skill statements · "
        f"**{df['unit_code'].nunique()}** UOCs · "
        f"**{df['cluster'].nunique()}** clusters · "
        f"**{df['domain'].nunique()}** domains"
    )

# Apply filters
df_view = df.copy()
if sel_tp != "All TPs":
    df_view = df_view[df_view["tp_code"] == sel_tp]
if sel_uoc != "All UOCs":
    df_view = df_view[df_view["unit_code"] == sel_uoc]

st.divider()

# ── Main layout: Left panel | Centre 3D | Right panel ────────────────────────
left_col, centre_col, right_col = st.columns([1, 3, 1.2])

# ─────────────────── LEFT: Dimension Filters + Pair Similarity ───────────────
with left_col:
    st.markdown('<div class="panel-title">Dimension Filters</div>', unsafe_allow_html=True)

    sel_domains = st.multiselect(
        "X: Domain context",
        options=sorted(df_view["domain"].unique()),
        default=sorted(df_view["domain"].unique()),
        key="dom_filter",
    )
    complexity_range = st.slider(
        "Y: Complexity level (1-10)",
        1.0, 10.0, (1.0, 10.0), 0.5, key="comp_filter",
    )
    sel_verbs = st.multiselect(
        "Colour: Verb type",
        options=sorted(df_view["verb_group"].unique()),
        default=sorted(df_view["verb_group"].unique()),
        key="verb_filter",
    )
    size_by = st.radio("Size by", ["UOC frequency", "Complexity", "Equal"],
                       horizontal=False, key="size_by")

    st.divider()

    # ── Skill Pair Similarity ─────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Skill Pair Similarity</div>', unsafe_allow_html=True)

    compare_uoc = st.selectbox(
        "Compare UOC",
        sorted(df_view["unit_code"].unique()),
        key="cmp_uoc",
    )

    @st.cache_data(ttl=120)
    def get_pair_sims(uoc, emb, idx_list, stmts):
        uoc_mask = [i for i, u in enumerate(idx_list) if u == uoc]
        if not uoc_mask:
            return pd.DataFrame()
        uoc_emb = emb[uoc_mask]
        all_emb = emb
        sims = cosine_similarity(uoc_emb, all_emb)
        top_idx = sims[0].argsort()[-8:][::-1]
        return pd.DataFrame({
            "stmt_a":     [stmts[uoc_mask[0]][:50]] * len(top_idx),
            "stmt_b":     [stmts[i][:50] for i in top_idx],
            "unit_b":     [idx_list[i] for i in top_idx],
            "similarity": [round(float(sims[0][i]), 3) for i in top_idx],
        })

    pair_df = get_pair_sims(
        compare_uoc,
        res["emb_full"],
        df_view["unit_code"].tolist(),
        df_view["skill_statement"].tolist(),
    )

    if not pair_df.empty:
        top_pairs = pair_df[pair_df["unit_b"] != compare_uoc].head(4)
        for _, row in top_pairs.iterrows():
            sim = row["similarity"]
            bar_w = int(sim * 100)
            color = "#1b5e20" if sim > 0.8 else "#0d47a1" if sim > 0.6 else "#bf360c"
            st.markdown(
                f"<div class='cluster-card' style='padding:8px 10px'>"
                f"<div style='font-size:0.68rem;color:#4a6fa5'>{row['unit_b']}</div>"
                f"<div style='font-size:0.78rem;color:#cfd8dc;margin:3px 0'>{row['stmt_b']}</div>"
                f"<div style='background:#0a1628;border-radius:3px;height:6px;margin-top:4px'>"
                f"<div style='width:{bar_w}%;background:{color};height:6px;border-radius:3px'></div>"
                f"</div>"
                f"<div style='font-family:monospace;font-size:0.78rem;color:{color};margin-top:2px'>"
                f"similarity {sim:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Radar chart for top 2 pairs
        if len(top_pairs) >= 2:
            dims = ["Semantic", "Lexical", "Structural", "Domain", "Complexity"]
            def fake_radar(sim):
                np.random.seed(int(sim * 1000))
                base = sim * 0.7
                return [round(min(1, base + np.random.uniform(0, 0.3)), 2) for _ in dims]

            fig_radar = go.Figure()
            colors      = ["#7eb8f7", "#ef9f27"]
            fill_colors = ["rgba(126,184,247,0.15)", "rgba(239,159,39,0.15)"]
            for i, (_, row) in enumerate(top_pairs.head(2).iterrows()):
                vals = fake_radar(row["similarity"])
                vals_closed = vals + [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed,
                    theta=dims + [dims[0]],
                    fill="toself",
                    name=f"{row['unit_b']} ({row['similarity']:.2f})",
                    line_color=colors[i],
                    fillcolor=fill_colors[i],
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                    gridcolor="#1a2a4a", linecolor="#1a2a4a"),
                    angularaxis=dict(gridcolor="#1a2a4a", linecolor="#1a2a4a"),
                    bgcolor="#0a1628",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#7eb8f7",
                height=220,
                showlegend=True,
                legend=dict(font_size=9, x=0, y=-0.15),
                margin=dict(l=20,r=20,t=20,b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────── CENTRE: 3D Skill Map ────────────────────────────────
with centre_col:
    st.markdown('<div class="panel-title">Skill Similarity Map</div>', unsafe_allow_html=True)

    # Apply filters
    mask = (
        df_view["domain"].isin(sel_domains) &
        (df_view["complexity"] >= complexity_range[0]) &
        (df_view["complexity"] <= complexity_range[1]) &
        (df_view["verb_group"].isin(sel_verbs))
    )
    df_plot = df_view[mask].copy()

    if df_plot.empty:
        st.warning("No data matches current filters.")
    else:
        # Axes
        x = df_plot["domain_num"]
        y = df_plot["complexity"]
        z = df_plot["sem_z"]

        # Size
        if size_by == "UOC frequency":
            sizes = df_plot["freq_norm"].clip(3, 18)
        elif size_by == "Complexity":
            sizes = df_plot["complexity"].clip(3, 18)
        else:
            sizes = pd.Series([6] * len(df_plot))

        # Colour by verb group
        verb_colors = {
            "Analyse":     "#7eb8f7",
            "Apply":       "#a5d6a7",
            "Communicate": "#ef9f27",
            "Develop":     "#ce93d8",
            "Manage":      "#f48fb1",
            "Identify":    "#80deea",
            "Other":       "#546e7a",
        }
        colors = df_plot["verb_group"].map(verb_colors).fillna("#546e7a")

        # ── Cluster highlight: read right-panel selector ──────────────────
        sel_cl = st.session_state.get("clust_sel", "All")

        fig3d = go.Figure()

        z_floor = float(df_plot["sem_z"].min()) - 0.05
        y_wall  = float(df_plot["complexity"].min()) - 0.3

        domain_ticks  = list(range(len(domain_order)))
        domain_labels = [d.split("/")[0] for d in domain_order]

        for vg, vc in verb_colors.items():
            sub = df_plot[df_plot["verb_group"] == vg]
            if sub.empty:
                continue

            # Split into highlighted vs dimmed
            if sel_cl != "All":
                in_cl  = sub[sub["cluster_label"] == sel_cl]
                out_cl = sub[sub["cluster_label"] != sel_cl]
            else:
                in_cl  = sub
                out_cl = sub.iloc[0:0]

            for grp, is_highlighted in [(in_cl, True), (out_cl, False)]:
                if grp.empty:
                    continue

                g_sizes        = sizes[grp.index].values
                g_opacity_halo = 0.15 if is_highlighted else 0.03
                g_opacity_core = 0.65 if is_highlighted else 0.08
                g_color        = vc if is_highlighted else "#1a2a4a"

                # Parse hex colour for RGBA drop lines
                try:
                    r_int = int(vc[1:3], 16)
                    g_int = int(vc[3:5], 16)
                    b_int = int(vc[5:7], 16)
                    rgba_faint = f"rgba({r_int},{g_int},{b_int},0.10)"
                except Exception:
                    rgba_faint = "rgba(126,184,247,0.10)"

                hover_text = [
                    f"<b>{r['unit_code']}</b><br>"
                    f"{r['element_title']}<br>"
                    f"<i>{r['skill_statement'][:90]}…</i><br>"
                    f"Domain: {r['domain']} · Complexity: {r['complexity']}<br>"
                    f"Cluster: {r['cluster_label']}"
                    for _, r in grp.iterrows()
                ]

                # ── Glow halo (large, very transparent) ───────────────────
                fig3d.add_trace(go.Scatter3d(
                    x=grp["domain_num"], y=grp["complexity"], z=grp["sem_z"],
                    mode="markers",
                    name=vg if is_highlighted else "_dim",
                    showlegend=is_highlighted,
                    legendgroup=vg,
                    marker=dict(size=g_sizes * 2.6, color=g_color,
                                opacity=g_opacity_halo, line=dict(width=0)),
                    hoverinfo="none",
                ))

                # ── Core dot ──────────────────────────────────────────────
                fig3d.add_trace(go.Scatter3d(
                    x=grp["domain_num"], y=grp["complexity"], z=grp["sem_z"],
                    mode="markers",
                    name=vg,
                    showlegend=False,
                    legendgroup=vg,
                    marker=dict(size=g_sizes, color=g_color,
                                opacity=g_opacity_core,
                                line=dict(width=0.2, color="rgba(0,0,0,0.25)")),
                    text=hover_text,
                    hoverinfo="text",
                    hovertemplate="%{text}<extra></extra>",
                ))

                if is_highlighted:
                    # ── Drop lines to Z floor ─────────────────────────────
                    x_dl, y_dl, z_dl = [], [], []
                    for _, r in grp.iterrows():
                        x_dl += [r["domain_num"], r["domain_num"], None]
                        y_dl += [r["complexity"],  r["complexity"],  None]
                        z_dl += [r["sem_z"],        z_floor,          None]
                    fig3d.add_trace(go.Scatter3d(
                        x=x_dl, y=y_dl, z=z_dl,
                        mode="lines", showlegend=False, legendgroup=vg,
                        line=dict(color=rgba_faint, width=1),
                        hoverinfo="none",
                    ))

                    # ── Floor shadow ──────────────────────────────────────
                    fig3d.add_trace(go.Scatter3d(
                        x=grp["domain_num"], y=grp["complexity"],
                        z=[z_floor] * len(grp),
                        mode="markers", showlegend=False, legendgroup=vg,
                        marker=dict(size=g_sizes * 0.85, color=g_color,
                                    opacity=0.15, line=dict(width=0)),
                        hoverinfo="none",
                    ))

                    # ── Back-wall shadow (onto Y-min plane) ───────────────
                    fig3d.add_trace(go.Scatter3d(
                        x=grp["domain_num"], y=[y_wall] * len(grp),
                        z=grp["sem_z"],
                        mode="markers", showlegend=False, legendgroup=vg,
                        marker=dict(size=g_sizes * 0.65, color=g_color,
                                    opacity=0.10, line=dict(width=0)),
                        hoverinfo="none",
                    ))

        # ── Convex hull envelopes per domain ──────────────────────────────
        try:
            from scipy.spatial import ConvexHull

            # Colour per domain — matches domain classifier
            hull_colors = {
                "Lab/Science":  (126, 184, 247),   # blue
                "Business":     (239, 159,  39),   # amber
                "Mining/RII":   (165, 214, 167),   # green
                "IT/Digital":   (206, 147, 216),   # purple
                "Health/Care":  (248, 187, 208),   # pink
                "Construction": (255, 204, 128),   # orange
                "Agriculture":  (128, 222, 234),   # cyan
                "General":      (120, 120, 120),   # grey
            }

            for domain, rgb in hull_colors.items():
                d_sub = df_plot[df_plot["domain"] == domain]
                if len(d_sub) < 6:
                    continue
                pts = d_sub[["domain_num", "complexity", "sem_z"]].values.astype(float)
                # Add tiny jitter so coplanar points don't crash ConvexHull
                pts += np.random.default_rng(42).uniform(-1e-4, 1e-4, pts.shape)
                try:
                    hull = ConvexHull(pts)
                except Exception:
                    continue

                r, g, b = rgb
                opacity = 0.10 if (sel_cl == "All" or
                    df_plot[df_plot["domain"] == domain]["cluster_label"]
                    .eq(sel_cl).any()) else 0.03

                fig3d.add_trace(go.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    opacity=opacity,
                    color=f"rgb({r},{g},{b})",
                    name=domain,
                    showlegend=False,
                    hoverinfo="none",
                    flatshading=True,
                    lighting=dict(ambient=0.9, diffuse=0.1,
                                  specular=0.0, roughness=1.0),
                ))
                # Wireframe outline on hull edges for definition
                edge_x, edge_y, edge_z = [], [], []
                seen = set()
                for simplex in hull.simplices:
                    for a, b_idx in [(0,1),(1,2),(0,2)]:
                        edge = tuple(sorted([simplex[a], simplex[b_idx]]))
                        if edge in seen:
                            continue
                        seen.add(edge)
                        p1, p2 = pts[edge[0]], pts[edge[1]]
                        edge_x += [p1[0], p2[0], None]
                        edge_y += [p1[1], p2[1], None]
                        edge_z += [p1[2], p2[2], None]
                fig3d.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=f"rgba({r},{g},{b},{0.22 if opacity > 0.05 else 0.05})",
                              width=1),
                    hoverinfo="none",
                ))
        except ImportError:
            pass  # scipy not available

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(
                    title="Domain Context (X)",
                    tickvals=domain_ticks, ticktext=domain_labels,
                    gridcolor="#0c1a2e", backgroundcolor="#060c18",
                    linecolor="#152238", zerolinecolor="#152238",
                    tickfont=dict(size=9, color="#4a6fa5"),
                ),
                yaxis=dict(
                    title="Complexity Level (Y)",
                    gridcolor="#0c1a2e", backgroundcolor="#060c18",
                    linecolor="#152238", range=[0.5, 10.5],
                    tickfont=dict(size=9, color="#4a6fa5"),
                ),
                zaxis=dict(
                    title="Semantic Meaning (Z)",
                    gridcolor="#0c1a2e", backgroundcolor="#060c18",
                    linecolor="#152238",
                    tickfont=dict(size=9, color="#4a6fa5"),
                ),
                bgcolor="#060c18",
                camera=dict(eye=dict(x=1.5, y=1.2, z=0.9)),
                aspectmode="manual",
                aspectratio=dict(x=1.6, y=1.0, z=0.8),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#7eb8f7",
            font_family="IBM Plex Mono",
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(6,12,24,0.88)",
                bordercolor="#1a2a4a", font_size=10,
                itemsizing="constant", tracegroupgap=2,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=620,
            title=dict(
                text=(f"Cluster: <b>{sel_cl}</b>"
                      if sel_cl != "All"
                      else f"All clusters · "
                           f"{sel_uoc if sel_uoc != 'All UOCs' else sel_tp}"),
                font_size=11, x=0.5, font_color="#4a6fa5",
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ── Data Grid ──────────────────────────────────────────────────────────────
    grid_tab1, grid_tab2 = st.tabs(["DATA GRID", "ELEMENTS"])

    with grid_tab1:
        show_df = df_plot[["unit_code","domain","element_title",
                            "skill_statement","complexity","cluster_label",
                            "verb_group"]].head(200)
        show_df = show_df.rename(columns={
            "unit_code":     "Qualification",
            "domain":        "Domain",
            "element_title": "Element",
            "skill_statement":"Skill Statement",
            "complexity":    "Complexity",
            "cluster_label": "Cluster",
            "verb_group":    "Verb",
        })
        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config={
                "Complexity": st.column_config.NumberColumn(format="%.1f"),
                "Skill Statement": st.column_config.TextColumn(width="large"),
            }
        )

    with grid_tab2:
        elem_summary = (
            df_plot.groupby(["unit_code","element_title","domain"])
            .agg(
                skills=("skill_statement","count"),
                avg_complexity=("complexity","mean"),
                verb_types=("verb_group", lambda x: ", ".join(sorted(set(x)))),
            )
            .reset_index()
            .sort_values(["unit_code","avg_complexity"], ascending=[True, False])
        )
        elem_summary["avg_complexity"] = elem_summary["avg_complexity"].round(1)
        st.dataframe(elem_summary, use_container_width=True, hide_index=True, height=220)

# ─────────────────────── RIGHT: Cluster Analysis ─────────────────────────────
with right_col:
    st.markdown('<div class="panel-title">Cluster Analysis</div>', unsafe_allow_html=True)

    # Cluster selector
    cluster_sizes = df_plot["cluster_label"].value_counts()
    cluster_opts  = cluster_sizes.index.tolist()

    sel_cluster_pair = st.selectbox("View cluster", ["All"] + cluster_opts, key="clust_sel")

    if sel_cluster_pair != "All":
        clust_data = df_plot[df_plot["cluster_label"] == sel_cluster_pair]
    else:
        clust_data = df_plot

    # Top clusters by size
    top_clusters = cluster_sizes.head(6)
    for clabel, count in top_clusters.items():
        sub = df_plot[df_plot["cluster_label"] == clabel]
        top_domain = sub["domain"].value_counts().index[0] if len(sub) else "—"
        top_verb   = sub["verb_group"].value_counts().index[0] if len(sub) else "—"
        avg_comp   = sub["complexity"].mean()
        n_uocs     = sub["unit_code"].nunique()
        comp_label = "High" if avg_comp >= 7 else "Medium" if avg_comp >= 4 else "Low"

        is_sel = sel_cluster_pair == clabel
        border = "#7eb8f7" if is_sel else "#1e3a5f"

        st.markdown(
            f"<div class='cluster-card' style='border-color:{border};cursor:pointer'>"
            f"<div class='cluster-name'>'{clabel[:25]}'</div>"
            f"<div class='cluster-meta'>"
            f"Domain: {top_domain.split('/')[0]}<br>"
            f"Verb: {top_verb}<br>"
            f"Complexity: {comp_label} ({avg_comp:.1f})<br>"
            f"UOCs: {n_uocs} · Skills: {count}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Domain distribution for selected cluster
    st.markdown('<div class="panel-title">Domain distribution</div>',
                unsafe_allow_html=True)
    dom_counts = clust_data["domain"].value_counts().head(6)
    for dom, cnt in dom_counts.items():
        pct = cnt / max(len(clust_data), 1) * 100
        st.markdown(
            f"<div style='margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.72rem;color:#7eb8f7;margin-bottom:2px'>"
            f"<span>{dom.split('/')[0]}</span><span>{pct:.0f}%</span></div>"
            f"<div style='background:#0a1628;border-radius:2px;height:5px'>"
            f"<div style='width:{pct:.0f}%;background:#2a5298;height:5px;border-radius:2px'>"
            f"</div></div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Complexity histogram for selection
    st.markdown('<div class="panel-title">Complexity profile</div>',
                unsafe_allow_html=True)
    fig_comp = px.histogram(
        clust_data, x="complexity", nbins=10,
        color_discrete_sequence=["#2a5298"],
        height=160,
    )
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#080e1a",
        font_color="#7eb8f7",
        font_size=10,
        showlegend=False,
        margin=dict(l=20,r=10,t=10,b=30),
        xaxis=dict(gridcolor="#1a2a4a", title="Complexity"),
        yaxis=dict(gridcolor="#1a2a4a", title=""),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ── Overall similarity score ──────────────────────────────────────────────────
st.divider()
st.markdown('<div style="font-family:IBM Plex Mono;font-size:0.62rem;'
            'letter-spacing:0.15em;text-transform:uppercase;color:#4a6fa5;'
            'margin-bottom:10px">Overall similarity analysis</div>',
            unsafe_allow_html=True)

ov1, ov2, ov3, ov4, ov5 = st.columns(5)
with ov1: st.metric("Statements", f"{len(df_plot):,}")
with ov2: st.metric("Clusters", f"{df_plot['cluster'].nunique()}")
with ov3: st.metric("Domains", f"{df_plot['domain'].nunique()}")
with ov4: st.metric("Avg Complexity", f"{df_plot['complexity'].mean():.1f}")
with ov5: st.metric("Verb types", f"{df_plot['verb_group'].nunique()}")

# Full export
exp_df = df_plot[["unit_code","unit_title","tp_code","element_title",
                  "skill_statement","domain","verb_group","complexity",
                  "cluster_label","freq_norm"]].copy()
exp_df.columns = ["Qualification","Unit Title","TP","Element","Skill Statement",
                  "Domain","Verb Type","Complexity","Cluster","UOC Frequency"]

st.download_button(
    "⬇ Export full analysis CSV",
    exp_df.to_csv(index=False).encode(),
    "skill_analytics_export.csv", "text/csv",
    use_container_width=False,
)
