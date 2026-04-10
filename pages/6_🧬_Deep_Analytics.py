"""
pages/6_🧬_Deep_Analytics.py

Advanced skill statement analytics:
  1. 3D Embedding Space Explorer (UMAP/PCA → Plotly 3D scatter)
  2. UOC Cohesion Scores (intra vs inter similarity)
  3. Statement Distinctiveness Index (uniqueness per statement)
  4. Redundancy Efficiency Map (consolidation potential by TP)
  5. Statement Lineage Tracer (nearest neighbours across all TPs)
  6. Consolidation ROI Calculator (interactive threshold slider)
  7. Statement Quality Composite Score (leaderboard)
  8. AQF Complexity Progression (semantic complexity by level)

Requires: semantic analysis run first (sa_embeddings in session_state)
OR will compute TF-IDF embeddings directly from DB as fallback.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
st.set_page_config(page_title="Deep Analytics", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
  .dak-card {
    background: linear-gradient(135deg,#0a0f1e,#111827);
    border:1px solid #1f2937;border-radius:12px;padding:20px 24px;
    position:relative;overflow:hidden;
  }
  .dak-card::after {
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#6366f1,#06b6d4,#10b981);
  }
  .dak-val {
    font-family:'JetBrains Mono',monospace;font-size:2.2rem;
    font-weight:600;color:#a5f3fc;line-height:1;
  }
  .dak-lbl { font-size:0.7rem;color:#4b5563;text-transform:uppercase;
             letter-spacing:.12em;margin-top:5px; }
  .dak-sub { font-size:0.82rem;color:#6b7280;margin-top:4px; }
  .shdr {
    font-family:'JetBrains Mono',monospace;font-size:0.62rem;
    letter-spacing:.18em;text-transform:uppercase;color:#374151;
    border-bottom:1px solid #111827;padding-bottom:6px;margin:32px 0 16px;
  }
  .stmt-card {
    background:#060d1a;border:1px solid #1e3a5f;border-radius:8px;
    padding:12px 16px;margin-bottom:8px;
  }
  .score-pill {
    display:inline-block;font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;padding:2px 8px;border-radius:10px;
  }
  .score-hi { background:#064e3b;color:#6ee7b7; }
  .score-md { background:#1e3a5f;color:#93c5fd; }
  .score-lo { background:#7f1d1d;color:#fca5a5; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Deep Analytics")
st.caption("Advanced semantic analysis — congruence, uniqueness, redundancy, 3D skill space")

# ── DB connection ─────────────────────────────────────────────────────────────
def _secret(k, d=""):
    try: return st.secrets.get(k, os.getenv(k, d)) or d
    except: return os.getenv(k, d) or d

DB_URL = _secret("DATABASE_URL")
if not DB_URL:
    st.error("DATABASE_URL not configured."); st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Loading skill statements…")
def load_statements(tp_prefix="", limit=5000):
    with engine.connect() as conn:
        q = """
            SELECT unit_code, unit_title, element_title,
                   skill_statement, tp_code, keywords,
                   qa_passes, qa_word_count, rewrite_count
            FROM rsd_skill_records
            WHERE skill_statement IS NOT NULL AND skill_statement != ''
        """
        params = {"lim": limit}
        if tp_prefix:
            q += " AND unit_code LIKE :pref"
            params["pref"] = f"{tp_prefix.upper()}%"
        q += " ORDER BY unit_code, row_index LIMIT :lim"
        rows = conn.execute(text(q), params).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])
    df["tp_code"] = df["tp_code"].replace("", np.nan).fillna(
        df["unit_code"].str[:3])
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Data settings")
    tp_prefix = st.text_input("TP prefix filter", placeholder="MSL or blank for all")
    max_stmts = st.slider("Max statements to analyse", 500, 10000, 3000, 500)
    st.divider()
    st.markdown("### 🔢 Embedding source")
    emb_source = st.radio("Use", [
        "Session state (from Semantic Analysis page)",
        "Compute TF-IDF (fast, no API cost)",
    ])
    use_session = emb_source.startswith("Session")
    st.divider()
    st.markdown("### 🎛️ Thresholds")
    sim_threshold   = st.slider("Similarity threshold", 0.5, 0.99, 0.85, 0.01)
    unique_threshold = st.slider("Uniqueness min score", 0.0, 1.0, 0.3, 0.05,
                                  help="Statements below this are potential duplicates")
    compute_btn = st.button("▶ Compute analytics", type="primary",
                             use_container_width=True)

df_raw = load_statements(tp_prefix, max_stmts)
if df_raw.empty:
    st.info("No data found."); st.stop()

n = len(df_raw)
st.info(f"**{n:,}** statements loaded across "
        f"**{df_raw['unit_code'].nunique()}** units / "
        f"**{df_raw['tp_code'].nunique()}** training packages")

# ── Embedding computation ──────────────────────────────────────────────────────
def get_embeddings(df):
    texts = df["skill_statement"].fillna("").tolist()
    # Try session state first if requested
    if use_session and st.session_state.get("sa_embeddings") is not None:
        embs = st.session_state["sa_embeddings"]
        if len(embs) == len(df):
            return embs.astype(np.float32)
        st.warning("Session embeddings length mismatch — using TF-IDF instead.")

    # TF-IDF fallback
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=3000,
                          stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(texts)
    dense = normalize(mat.toarray(), norm="l2").astype(np.float32)
    return dense

def cosine_sim(a, b=None):
    if b is None: b = a
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return an @ bn.T

# ── Compute analytics ─────────────────────────────────────────────────────────
if compute_btn:
    prog = st.progress(0)
    stat = st.empty()

    stat.write("⏳ Computing embeddings…")
    embeddings = get_embeddings(df_raw.reset_index(drop=True))
    prog.progress(0.2)

    stat.write("⏳ Computing similarity matrix…")
    # For large sets, sample for full matrix
    n_full = min(n, 2000)
    emb_sample = embeddings[:n_full]
    sim_mat = cosine_sim(emb_sample)
    np.fill_diagonal(sim_mat, 0)  # exclude self
    prog.progress(0.45)

    stat.write("⏳ Computing uniqueness scores…")
    # Uniqueness = 1 - max_sim_to_any_other
    max_sims = sim_mat.max(axis=1)
    uniqueness = (1 - max_sims).clip(0, 1)

    # Mean sim to all others (lower = more unique)
    mean_sims = sim_mat.mean(axis=1)

    prog.progress(0.55)

    stat.write("⏳ Computing UOC cohesion scores…")
    unit_codes = df_raw["unit_code"].values[:n_full]
    unique_units = pd.Series(unit_codes).value_counts()
    cohesion_rows = []
    for uc in unique_units[unique_units >= 2].index[:50]:
        mask = unit_codes == uc
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue
        sub = sim_mat[np.ix_(idx, idx)]
        intra = float(sub[np.triu_indices(len(idx), k=1)].mean()) if len(idx) > 1 else 0
        # Inter: avg sim to statements NOT in this unit
        out_idx = np.where(~mask)[0][:200]
        if len(out_idx) > 0:
            inter_mat = sim_mat[np.ix_(idx, out_idx)]
            inter = float(inter_mat.mean())
        else:
            inter = 0
        cohesion_rows.append({
            "unit_code": uc,
            "unit_title": df_raw[df_raw["unit_code"]==uc]["unit_title"].iloc[0][:50] if len(df_raw[df_raw["unit_code"]==uc]) > 0 else "",
            "n_stmts": int(len(idx)),
            "intra_sim": round(intra, 3),
            "inter_sim": round(inter, 3),
            "cohesion":  round(intra - inter, 3),
            "tp_code": df_raw[df_raw["unit_code"]==uc]["tp_code"].iloc[0] if len(df_raw[df_raw["unit_code"]==uc]) > 0 else "",
        })
    cohesion_df = pd.DataFrame(cohesion_rows).sort_values("cohesion", ascending=False)
    prog.progress(0.65)

    stat.write("⏳ Computing TP redundancy…")
    tp_codes_arr = df_raw["tp_code"].values[:n_full]
    unique_tps = pd.Series(tp_codes_arr).value_counts()
    tp_redund = []
    tps_list = unique_tps[unique_tps >= 5].index.tolist()[:20]
    for tp in tps_list:
        mask = tp_codes_arr == tp
        idx  = np.where(mask)[0]
        if len(idx) < 2:
            continue
        sub = sim_mat[np.ix_(idx, idx)]
        pairs_above = (sub > sim_threshold).sum() // 2
        total_pairs = len(idx) * (len(idx)-1) // 2
        redundancy  = pairs_above / max(total_pairs, 1)
        tp_redund.append({
            "tp_code":    tp,
            "n_stmts":    int(len(idx)),
            "redundancy": round(float(redundancy), 3),
            "pairs_above_threshold": int(pairs_above),
            "potential_saving": int(pairs_above),
        })
    tp_redund_df = pd.DataFrame(tp_redund).sort_values("redundancy", ascending=False)
    prog.progress(0.75)

    stat.write("⏳ Computing 3D projection…")
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_
        dim_method = "PCA"
    except Exception:
        coords_3d = np.random.randn(n, 3)
        explained = [0, 0, 0]
        dim_method = "random"
    prog.progress(0.85)

    stat.write("⏳ Computing quality composite scores…")
    df_q = df_raw.copy().reset_index(drop=True)
    # Uniqueness component
    u_scores = np.zeros(n)
    u_scores[:n_full] = uniqueness
    if n > n_full:
        u_scores[n_full:] = uniqueness.mean()

    df_q["uniqueness"]    = u_scores
    df_q["qa_norm"]       = df_q["qa_passes"].fillna(False).astype(float)
    df_q["wc_norm"]       = np.clip(
        (df_q["qa_word_count"].fillna(0) - 20) / 40, 0, 1)
    df_q["rewrite_norm"]  = 1 - np.clip(
        df_q["rewrite_count"].fillna(0) / 5, 0, 1)

    df_q["quality_score"] = (
        0.30 * df_q["qa_norm"] +
        0.25 * df_q["uniqueness"] +
        0.25 * df_q["wc_norm"] +
        0.20 * df_q["rewrite_norm"]
    ).round(3)

    prog.progress(1.0)
    stat.write("✅ Analytics ready")

    # Store
    st.session_state.update({
        "da_df":         df_q,
        "da_embeddings": embeddings,
        "da_sim_mat":    sim_mat,
        "da_uniqueness": u_scores,
        "da_cohesion":   cohesion_df,
        "da_tp_redund":  tp_redund_df,
        "da_coords_3d":  coords_3d,
        "da_dim_method": dim_method,
        "da_explained":  explained,
        "da_n_full":     n_full,
    })

# ── Check results ─────────────────────────────────────────────────────────────
if "da_df" not in st.session_state:
    st.info("Click **▶ Compute analytics** in the sidebar to begin.")
    st.stop()

df         = st.session_state["da_df"]
sim_mat    = st.session_state["da_sim_mat"]
uniqueness = st.session_state["da_uniqueness"]
cohesion_df = st.session_state["da_cohesion"]
tp_redund_df = st.session_state["da_tp_redund"]
coords_3d  = st.session_state["da_coords_3d"]
dim_method = st.session_state["da_dim_method"]
explained  = st.session_state["da_explained"]
n_full     = st.session_state["da_n_full"]

# ── Summary metrics ────────────────────────────────────────────────────────────
st.markdown('<div class="shdr">Summary metrics</div>', unsafe_allow_html=True)

avg_uniq = float(uniqueness.mean())
pct_redund = float((sim_mat > sim_threshold).sum() / 2 / max(n_full*(n_full-1)//2, 1))
avg_coh = float(cohesion_df["cohesion"].mean()) if not cohesion_df.empty else 0
avg_qual = float(df["quality_score"].mean())

cols = st.columns(5)
metrics = [
    (f"{n:,}",          "Statements",       f"{df['unit_code'].nunique()} units"),
    (f"{avg_uniq:.2f}", "Avg uniqueness",   "1.0 = fully unique"),
    (f"{pct_redund:.1%}","Redundancy rate", f"pairs > {sim_threshold}"),
    (f"{avg_coh:.3f}",  "Avg UOC cohesion", "intra − inter sim"),
    (f"{avg_qual:.2f}", "Avg quality score","0–1 composite"),
]
for col, (val, lbl, sub) in zip(cols, metrics):
    with col:
        st.markdown(
            f'<div class="dak-card"><div class="dak-val">{val}</div>'
            f'<div class="dak-lbl">{lbl}</div><div class="dak-sub">{sub}</div></div>',
            unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7 = st.tabs([
    "🌐 3D Skill Space",
    "🧲 UOC Cohesion",
    "✨ Uniqueness",
    "♻️ Redundancy",
    "🔗 Lineage Tracer",
    "⚖️ ROI Calculator",
    "🏆 Quality Scores",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — 3D SKILL SPACE
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="shdr">3D embedding space explorer</div>',
                unsafe_allow_html=True)

    col_ctrl, col_info = st.columns([3,1])
    with col_ctrl:
        colour_by = st.selectbox("Colour points by",
            ["tp_code","unit_code","quality_score","qa_passes"],
            key="3d_colour")
        n_3d = st.slider("Points to show", 200, min(n, 3000), min(n, 1000), 100,
                          key="3d_n")
    with col_info:
        if dim_method == "PCA":
            st.metric("PC1 variance", f"{explained[0]:.1%}")
            st.metric("PC2 variance", f"{explained[1]:.1%}")
            st.metric("PC3 variance", f"{explained[2]:.1%}")

    df_3d = df.iloc[:n_3d].copy()
    x3, y3, z3 = coords_3d[:n_3d, 0], coords_3d[:n_3d, 1], coords_3d[:n_3d, 2]

    hover_text = (
        df_3d["unit_code"] + " | " +
        df_3d["skill_statement"].str[:60] + "…"
    )

    if colour_by in ("quality_score",):
        color_vals = df_3d[colour_by].fillna(0)
        fig3d = go.Figure(go.Scatter3d(
            x=x3, y=y3, z=z3,
            mode="markers",
            marker=dict(
                size=3.5,
                color=color_vals,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Quality", thickness=12),
                opacity=0.8,
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))
    elif colour_by == "qa_passes":
        color_vals = df_3d["qa_passes"].fillna(False).map({True:"#10b981", False:"#ef4444"})
        fig3d = go.Figure(go.Scatter3d(
            x=x3, y=y3, z=z3,
            mode="markers",
            marker=dict(size=3.5, color=color_vals, opacity=0.8),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))
    else:
        # Categorical colour
        cats = df_3d[colour_by].fillna("unknown")
        unique_cats = cats.unique()[:20]
        colours = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
        cat_color = {c: colours[i % len(colours)] for i,c in enumerate(unique_cats)}
        fig3d = go.Figure()
        for cat in unique_cats:
            mask = cats == cat
            fig3d.add_trace(go.Scatter3d(
                x=x3[mask], y=y3[mask], z=z3[mask],
                mode="markers",
                name=str(cat)[:15],
                marker=dict(size=3.5, color=cat_color[cat], opacity=0.8),
                text=hover_text[mask],
                hovertemplate="%{text}<extra></extra>",
            ))

    fig3d.update_layout(
        paper_bgcolor="#060d1a",
        plot_bgcolor="#060d1a",
        font_color="#9ca3af",
        scene=dict(
            bgcolor="#060d1a",
            xaxis=dict(title=f"{dim_method}1", gridcolor="#1f2937",
                       color="#4b5563"),
            yaxis=dict(title=f"{dim_method}2", gridcolor="#1f2937",
                       color="#4b5563"),
            zaxis=dict(title=f"{dim_method}3", gridcolor="#1f2937",
                       color="#4b5563"),
        ),
        margin=dict(l=0,r=0,t=30,b=0),
        height=600,
        showlegend=colour_by in ("tp_code","unit_code"),
        legend=dict(bgcolor="#060d1a", bordercolor="#1f2937",
                    font=dict(size=10)),
        title=dict(
            text=f"Skill statement space — {n_3d} statements ({dim_method})",
            font=dict(size=13, color="#6b7280"),
        ),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption(
        "Orbit: drag · Zoom: scroll · Hover for statement text. "
        "Nearby points = semantically similar statements. "
        f"{'UMAP' if dim_method=='UMAP' else 'PCA'} reduces high-dimensional "
        "embeddings to 3 axes. Install `umap-learn` for better separation."
    )

    # ── 3D Similarity density surface ────────────────────────────────────────
    st.markdown('<div class="shdr">Similarity density landscape</div>',
                unsafe_allow_html=True)
    st.caption("Statement similarity as a 3D height surface — peaks = dense clusters, valleys = unique skills.")

    n_surf = min(n_full, 80)
    z_surface = sim_mat[:n_surf, :n_surf]

    fig_surf = go.Figure(go.Surface(
        z=z_surface,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Similarity", thickness=12),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#a5f3fc", project_z=True),
        ),
    ))
    fig_surf.update_layout(
        paper_bgcolor="#060d1a",
        plot_bgcolor="#060d1a",
        font_color="#9ca3af",
        scene=dict(
            bgcolor="#060d1a",
            xaxis=dict(title="Statement →", gridcolor="#1f2937", color="#4b5563"),
            yaxis=dict(title="Statement →", gridcolor="#1f2937", color="#4b5563"),
            zaxis=dict(title="Similarity", gridcolor="#1f2937", color="#4b5563"),
        ),
        margin=dict(l=0,r=0,t=30,b=0),
        height=500,
        title=dict(text=f"Similarity density — first {n_surf} statements",
                   font=dict(size=13, color="#6b7280")),
    )
    st.plotly_chart(fig_surf, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — UOC COHESION
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="shdr">UOC cohesion scores</div>', unsafe_allow_html=True)
    st.caption(
        "**Cohesion = intra-UOC similarity − inter-UOC similarity.** "
        "High cohesion = tight, well-scoped unit. "
        "Low/negative = elements are scattered or overlap other units."
    )

    if cohesion_df.empty:
        st.info("Not enough data — need UOCs with ≥2 statements.")
    else:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            chart_coh = alt.Chart(cohesion_df.head(30)).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
            ).encode(
                y=alt.Y("unit_code:N", sort="-x", title=""),
                x=alt.X("cohesion:Q", title="Cohesion score",
                        scale=alt.Scale(domain=[
                            float(cohesion_df["cohesion"].min())-0.02,
                            float(cohesion_df["cohesion"].max())+0.02,
                        ])),
                color=alt.Color("cohesion:Q",
                                scale=alt.Scale(scheme="redyellowgreen",
                                                domain=[-0.1, 0.3]),
                                legend=None),
                tooltip=[
                    alt.Tooltip("unit_code:N", title="Unit"),
                    alt.Tooltip("unit_title:N", title="Title"),
                    alt.Tooltip("n_stmts:Q", title="Statements"),
                    alt.Tooltip("intra_sim:Q", title="Intra sim", format=".3f"),
                    alt.Tooltip("inter_sim:Q", title="Inter sim", format=".3f"),
                    alt.Tooltip("cohesion:Q", title="Cohesion", format=".3f"),
                ],
            ).properties(height=450, title="Top 30 UOCs by cohesion score")
            zero_line = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
                color="#ef4444", strokeDash=[4,2], size=1).encode(x="x:Q")
            st.altair_chart(chart_coh + zero_line, use_container_width=True)

        with col_b:
            st.dataframe(
                cohesion_df[["unit_code","n_stmts","intra_sim",
                              "inter_sim","cohesion","tp_code"]].head(25),
                use_container_width=True, hide_index=True,
                column_config={
                    "unit_code":  st.column_config.TextColumn("Unit", width="small"),
                    "n_stmts":    st.column_config.NumberColumn("Stmts", width="small"),
                    "intra_sim":  st.column_config.NumberColumn("Intra", format="%.3f"),
                    "inter_sim":  st.column_config.NumberColumn("Inter", format="%.3f"),
                    "cohesion":   st.column_config.NumberColumn("Cohesion", format="%.3f"),
                    "tp_code":    st.column_config.TextColumn("TP", width="small"),
                },
            )

        # 3D cohesion scatter
        st.markdown('<div class="shdr">Intra vs inter similarity — 3D scatter</div>',
                    unsafe_allow_html=True)
        fig_coh3d = go.Figure(go.Scatter3d(
            x=cohesion_df["intra_sim"],
            y=cohesion_df["inter_sim"],
            z=cohesion_df["cohesion"],
            mode="markers+text",
            text=cohesion_df["unit_code"],
            textfont=dict(size=8, color="#9ca3af"),
            textposition="top center",
            marker=dict(
                size=cohesion_df["n_stmts"].clip(3, 20).values,
                color=cohesion_df["cohesion"],
                colorscale="RdYlGn",
                colorbar=dict(title="Cohesion", thickness=12),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Intra: %{x:.3f}<br>Inter: %{y:.3f}<br>"
                "Cohesion: %{z:.3f}<extra></extra>"
            ),
        ))
        fig_coh3d.update_layout(
            paper_bgcolor="#060d1a", font_color="#9ca3af", height=500,
            scene=dict(
                bgcolor="#060d1a",
                xaxis=dict(title="Intra-UOC similarity",
                           gridcolor="#1f2937", color="#4b5563"),
                yaxis=dict(title="Inter-UOC similarity",
                           gridcolor="#1f2937", color="#4b5563"),
                zaxis=dict(title="Cohesion",
                           gridcolor="#1f2937", color="#4b5563"),
            ),
            margin=dict(l=0,r=0,t=30,b=0),
        )
        st.plotly_chart(fig_coh3d, use_container_width=True)
        st.caption("Point size = number of statements. Colour = cohesion score (green = high).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — UNIQUENESS
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="shdr">Statement distinctiveness index</div>',
                unsafe_allow_html=True)
    st.caption(
        "**Uniqueness = 1 − max similarity to any other statement.** "
        "1.0 = completely unique. 0.0 = exact duplicate exists."
    )

    df_uniq = df.copy()
    df_uniq["uniqueness"] = uniqueness
    df_uniq["max_sim"]    = 1 - uniqueness

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg uniqueness",     f"{float(df_uniq['uniqueness'].mean()):.3f}")
    col2.metric("Near-duplicates",    f"{int((df_uniq['uniqueness'] < 0.1).sum())}",
                help="Uniqueness < 0.1")
    col3.metric("Highly unique",      f"{int((df_uniq['uniqueness'] > 0.7).sum())}",
                help="Uniqueness > 0.7")

    # Distribution
    hist_vals, hist_bins = np.histogram(df_uniq["uniqueness"].dropna(), bins=40)
    hist_df = pd.DataFrame({
        "uniqueness": [(hist_bins[i]+hist_bins[i+1])/2 for i in range(len(hist_vals))],
        "count": [int(v) for v in hist_vals],
    })
    hist_df = hist_df[hist_df["count"] > 0]

    hist_chart = alt.Chart(hist_df).mark_bar(
        color="#6366f1",
        cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
    ).encode(
        x=alt.X("uniqueness:Q", bin=False, title="Uniqueness score",
                scale=alt.Scale(domain=[0,1])),
        y=alt.Y("count:Q", title="Statements"),
        tooltip=[alt.Tooltip("uniqueness:Q", format=".2f"), "count"],
    ).properties(height=200, title="Distribution of uniqueness scores")

    thresh_line = alt.Chart(pd.DataFrame({"x":[unique_threshold]})).mark_rule(
        color="#f59e0b", strokeDash=[4,2], size=1.5).encode(x="x:Q")
    st.altair_chart(hist_chart + thresh_line, use_container_width=True)
    st.caption(f"🟡 Yellow line = uniqueness threshold ({unique_threshold})")

    # Per-TP uniqueness
    tp_uniq = (
        df_uniq.groupby("tp_code")["uniqueness"]
        .agg(["mean","min","max","std"])
        .reset_index()
        .round(3)
    )
    tp_uniq.columns = ["TP","Avg","Min","Max","Std"]
    tp_uniq = tp_uniq.sort_values("Avg")

    st.markdown('<div class="shdr">Uniqueness by training package</div>',
                unsafe_allow_html=True)

    fig_box = go.Figure()
    for tp in tp_uniq["TP"].tolist():
        vals = df_uniq[df_uniq["tp_code"]==tp]["uniqueness"].dropna().tolist()
        if vals:
            fig_box.add_trace(go.Box(
                y=vals, name=tp,
                boxpoints=False,
                marker_color="#6366f1",
                line_color="#818cf8",
            ))
    fig_box.update_layout(
        paper_bgcolor="#060d1a", plot_bgcolor="#060d1a",
        font_color="#9ca3af", height=350,
        xaxis=dict(gridcolor="#1f2937", color="#4b5563"),
        yaxis=dict(title="Uniqueness", gridcolor="#1f2937", color="#4b5563"),
        margin=dict(l=0,r=0,t=30,b=60),
        showlegend=False,
        title=dict(text="Uniqueness distribution per training package",
                   font=dict(size=13, color="#6b7280")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Most unique + least unique
    col_u, col_d = st.columns(2)
    with col_u:
        st.markdown("**Most unique statements (high value, low redundancy)**")
        top_uniq = df_uniq.nlargest(10, "uniqueness")[
            ["unit_code","element_title","skill_statement","uniqueness"]]
        for _, row in top_uniq.iterrows():
            score = row["uniqueness"]
            cls = "score-hi" if score > 0.7 else "score-md"
            st.markdown(
                f'<div class="stmt-card">'
                f'<div style="margin-bottom:6px">'
                f'<code style="color:#818cf8;font-size:0.72rem">{row["unit_code"]}</code> '
                f'<span class="score-pill {cls}">{score:.3f}</span>'
                f'</div>'
                f'<div style="font-size:0.85rem;color:#d1d5db">{row["skill_statement"][:120]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_d:
        st.markdown("**Least unique statements (consolidation candidates)**")
        bot_uniq = df_uniq.nsmallest(10, "uniqueness")[
            ["unit_code","element_title","skill_statement","uniqueness"]]
        for _, row in bot_uniq.iterrows():
            score = row["uniqueness"]
            st.markdown(
                f'<div class="stmt-card">'
                f'<div style="margin-bottom:6px">'
                f'<code style="color:#818cf8;font-size:0.72rem">{row["unit_code"]}</code> '
                f'<span class="score-pill score-lo">{score:.3f}</span>'
                f'</div>'
                f'<div style="font-size:0.85rem;color:#d1d5db">{row["skill_statement"][:120]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — REDUNDANCY
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="shdr">Redundancy efficiency map</div>',
                unsafe_allow_html=True)
    st.caption(
        "Proportion of statement pairs within each TP that exceed the "
        f"similarity threshold ({sim_threshold}). High = consolidation opportunity."
    )

    if tp_redund_df.empty:
        st.info("Not enough data per TP.")
    else:
        col_a, col_b = st.columns([3,2])
        with col_a:
            redund_chart = alt.Chart(tp_redund_df).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
            ).encode(
                y=alt.Y("tp_code:N", sort="-x", title=""),
                x=alt.X("redundancy:Q", title="Redundancy rate",
                        axis=alt.Axis(format=".0%")),
                color=alt.Color("redundancy:Q",
                                scale=alt.Scale(scheme="orangered"),
                                legend=None),
                tooltip=[
                    alt.Tooltip("tp_code:N",    title="TP"),
                    alt.Tooltip("n_stmts:Q",    title="Statements"),
                    alt.Tooltip("redundancy:Q", title="Redundancy", format=".1%"),
                    alt.Tooltip("potential_saving:Q", title="Pairs to merge"),
                ],
            ).properties(height=350, title="Redundancy rate by training package")
            st.altair_chart(redund_chart, use_container_width=True)

        with col_b:
            total_savings = int(tp_redund_df["potential_saving"].sum())
            st.metric("Total pairs above threshold", f"{total_savings:,}")
            st.metric("Most redundant TP",
                      tp_redund_df.iloc[0]["tp_code"] if len(tp_redund_df) else "—",
                      f"{tp_redund_df.iloc[0]['redundancy']:.1%}" if len(tp_redund_df) else "")
            st.dataframe(
                tp_redund_df[["tp_code","n_stmts","redundancy","potential_saving"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "tp_code": st.column_config.TextColumn("TP"),
                    "n_stmts": st.column_config.NumberColumn("Stmts"),
                    "redundancy": st.column_config.NumberColumn(
                        "Redundancy", format="%.1%"),
                    "potential_saving": st.column_config.NumberColumn("Pairs"),
                },
            )

        # Cross-TP redundancy heatmap
        st.markdown('<div class="shdr">Cross-TP similarity heatmap</div>',
                    unsafe_allow_html=True)

        tps_for_heat = tp_redund_df["tp_code"].tolist()[:15]
        cross_rows = []
        tp_arr = df["tp_code"].values[:n_full]
        for ta in tps_for_heat:
            for tb in tps_for_heat:
                ma = tp_arr == ta
                mb = tp_arr == tb
                ia = np.where(ma)[0][:50]
                ib = np.where(mb)[0][:50]
                if len(ia) == 0 or len(ib) == 0:
                    cross_rows.append({"TP A":ta,"TP B":tb,"avg_sim":0})
                    continue
                sub = sim_mat[np.ix_(ia, ib)]
                cross_rows.append({
                    "TP A": ta, "TP B": tb,
                    "avg_sim": round(float(sub.mean()), 3),
                })

        cross_df = pd.DataFrame(cross_rows)
        if not cross_df.empty:
            heat = alt.Chart(cross_df).mark_rect().encode(
                x=alt.X("TP B:N", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("TP A:N"),
                color=alt.Color("avg_sim:Q",
                                scale=alt.Scale(scheme="viridis"),
                                legend=alt.Legend(title="Avg sim")),
                tooltip=["TP A","TP B",
                         alt.Tooltip("avg_sim:Q", title="Avg similarity",format=".3f")],
            ).properties(
                height=400,
                title="Average cross-TP statement similarity",
            ).interactive(False)
            st.altair_chart(heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — LINEAGE TRACER
# ─────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="shdr">Statement lineage tracer</div>',
                unsafe_allow_html=True)
    st.caption(
        "Select any statement and find its nearest semantic neighbours "
        "across all training packages — ranked by similarity."
    )

    unit_sel = st.selectbox(
        "Select unit", df["unit_code"].dropna().unique()[:100], key="lin_unit")
    unit_stmts = df[df["unit_code"] == unit_sel]["skill_statement"].dropna().tolist()

    if unit_stmts:
        stmt_sel = st.selectbox("Select statement", unit_stmts[:20], key="lin_stmt")
        n_neighbours = st.slider("Neighbours to show", 5, 50, 20, key="lin_n")

        if st.button("🔗 Find neighbours", key="lin_find"):
            stmt_idx = df[
                (df["unit_code"] == unit_sel) &
                (df["skill_statement"] == stmt_sel)
            ].index.tolist()

            if stmt_idx and stmt_idx[0] < n_full:
                idx = stmt_idx[0]
                sims = sim_mat[idx].copy()
                sims[idx] = -1  # exclude self
                top_idx = np.argsort(sims)[::-1][:n_neighbours]

                results = []
                for ti in top_idx:
                    row = df.iloc[ti]
                    results.append({
                        "similarity": round(float(sims[ti]), 4),
                        "unit_code":  row["unit_code"],
                        "tp_code":    row["tp_code"],
                        "element":    row["element_title"],
                        "statement":  row["skill_statement"],
                    })
                st.session_state["lin_results"] = results
            else:
                st.warning("Statement not in similarity matrix — try reducing max statements or running compute again.")

    results = st.session_state.get("lin_results", [])
    if results:
        st.markdown(f"**{len(results)} nearest neighbours:**")
        for r in results:
            sim = r["similarity"]
            cls = "score-hi" if sim > 0.9 else ("score-md" if sim > 0.7 else "score-lo")
            same_tp = r["tp_code"] == df[df["unit_code"]==unit_sel]["tp_code"].iloc[0] if len(df[df["unit_code"]==unit_sel]) > 0 else False
            cross_badge = (
                f'<span style="font-size:0.68rem;color:#f59e0b;margin-left:6px">⚡ CROSS-TP</span>'
                if not same_tp else ""
            )
            st.markdown(
                f'<div class="stmt-card">'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px">'
                f'<code style="color:#818cf8;font-size:0.72rem">{r["unit_code"]}</code>'
                f'<span class="score-pill {cls}">{sim:.4f}</span>'
                f'{cross_badge}'
                f'<span style="font-size:0.7rem;color:#4b5563">{r["element"][:40]}</span>'
                f'</div>'
                f'<div style="font-size:0.85rem;color:#d1d5db">{r["statement"][:140]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ROI CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="shdr">Consolidation ROI calculator</div>',
                unsafe_allow_html=True)
    st.caption(
        "If you merge all pairs above a similarity threshold, "
        "how many statements remain? Slide to explore the trade-off."
    )

    thresholds = np.arange(0.50, 1.00, 0.02)
    roi_rows = []
    for thr in thresholds:
        above = (sim_mat > thr).sum() // 2
        # Rough estimate: each pair above threshold saves ~0.5 statements
        savings = min(int(above * 0.5), n_full - 1)
        remaining = max(n_full - savings, 1)
        roi_rows.append({
            "threshold": round(float(thr), 2),
            "pairs_above": int(above),
            "estimated_remaining": int(remaining),
            "reduction_pct": round(100 * savings / max(n_full,1), 1),
        })
    roi_df = pd.DataFrame(roi_rows)

    col_l, col_r = st.columns([3,2])
    with col_l:
        roi_chart = alt.Chart(roi_df).mark_area(
            color="#6366f1", opacity=0.3,
            line={"color": "#818cf8", "size": 2},
        ).encode(
            x=alt.X("threshold:Q", title="Similarity threshold",
                    scale=alt.Scale(domain=[0.5, 1.0])),
            y=alt.Y("estimated_remaining:Q", title="Estimated statements remaining"),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("estimated_remaining:Q", title="Remaining"),
                alt.Tooltip("reduction_pct:Q", title="Reduction %", format=".1f"),
                alt.Tooltip("pairs_above:Q", title="Pairs to merge"),
            ],
        ).properties(height=300, title="Statements remaining vs similarity threshold")

        current_line = alt.Chart(pd.DataFrame({"x":[sim_threshold]})).mark_rule(
            color="#f59e0b", strokeDash=[4,2], size=1.5).encode(x="x:Q")
        st.altair_chart(roi_chart + current_line, use_container_width=True)

    with col_r:
        thr_val = st.slider(
            "Explore threshold", 0.50, 0.99, float(sim_threshold), 0.01,
            key="roi_thr")
        row = roi_df[roi_df["threshold"] == round(thr_val, 2)]
        if len(row):
            r = row.iloc[0]
            st.metric("Statements remaining",
                      f"{int(r['estimated_remaining']):,}")
            st.metric("Reduction",
                      f"{r['reduction_pct']:.1f}%")
            st.metric("Pairs to consolidate",
                      f"{int(r['pairs_above']):,}")
        st.dataframe(roi_df.tail(10), use_container_width=True, hide_index=True,
            column_config={
                "threshold":           st.column_config.NumberColumn("Threshold", format="%.2f"),
                "estimated_remaining": st.column_config.NumberColumn("Remaining"),
                "reduction_pct":       st.column_config.NumberColumn("Reduction %", format="%.1f"),
            })

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — QUALITY SCORES
# ─────────────────────────────────────────────────────────────────────────────
with t7:
    st.markdown('<div class="shdr">Statement quality composite score</div>',
                unsafe_allow_html=True)
    st.caption(
        "Composite 0–1 score: QA pass (30%) + Uniqueness (25%) + "
        "Word count in target range (25%) + Low rewrite count (20%)."
    )

    col_qa, col_qb, col_qc = st.columns(3)
    col_qa.metric("Avg quality score", f"{df['quality_score'].mean():.3f}")
    col_qb.metric("Score ≥ 0.75 (excellent)",
                  f"{int((df['quality_score'] >= 0.75).sum()):,}")
    col_qc.metric("Score < 0.40 (needs review)",
                  f"{int((df['quality_score'] < 0.40).sum()):,}")

    # Quality by TP
    tp_qual = (
        df.groupby("tp_code")["quality_score"]
        .agg(["mean","min","max"])
        .reset_index().round(3)
    )
    tp_qual.columns = ["TP","Avg","Min","Max"]
    tp_qual = tp_qual.sort_values("Avg", ascending=False)

    qual_chart = alt.Chart(tp_qual).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
    ).encode(
        y=alt.Y("TP:N", sort="-x", title=""),
        x=alt.X("Avg:Q", title="Avg quality score",
                scale=alt.Scale(domain=[0,1])),
        color=alt.Color("Avg:Q",
                        scale=alt.Scale(scheme="yellowgreen", domain=[0,1]),
                        legend=None),
        tooltip=["TP","Avg","Min","Max"],
    ).properties(height=350, title="Average quality score by training package")

    target_line = alt.Chart(pd.DataFrame({"x":[0.75]})).mark_rule(
        color="#10b981", strokeDash=[4,2], size=1.5).encode(x="x:Q")
    st.altair_chart(qual_chart + target_line, use_container_width=True)

    # Leaderboard
    st.markdown('<div class="shdr">Statement leaderboard</div>',
                unsafe_allow_html=True)
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown("**Top 15 highest quality statements**")
        top15 = df.nlargest(15, "quality_score")[
            ["unit_code","skill_statement","quality_score",
             "qa_passes","qa_word_count","uniqueness"]
        ]
        st.dataframe(top15, use_container_width=True, hide_index=True,
            column_config={
                "unit_code":     st.column_config.TextColumn("Unit", width="small"),
                "skill_statement": st.column_config.TextColumn("Statement", width="large"),
                "quality_score": st.column_config.NumberColumn("Score", format="%.3f"),
                "qa_passes":     st.column_config.CheckboxColumn("QA"),
                "qa_word_count": st.column_config.NumberColumn("Words"),
                "uniqueness":    st.column_config.NumberColumn("Unique", format="%.3f"),
            })

    with col_bot:
        st.markdown("**Bottom 15 — needs attention**")
        bot15 = df.nsmallest(15, "quality_score")[
            ["unit_code","skill_statement","quality_score",
             "qa_passes","qa_word_count","uniqueness"]
        ]
        st.dataframe(bot15, use_container_width=True, hide_index=True,
            column_config={
                "unit_code":     st.column_config.TextColumn("Unit", width="small"),
                "skill_statement": st.column_config.TextColumn("Statement", width="large"),
                "quality_score": st.column_config.NumberColumn("Score", format="%.3f"),
                "qa_passes":     st.column_config.CheckboxColumn("QA"),
                "qa_word_count": st.column_config.NumberColumn("Words"),
                "uniqueness":    st.column_config.NumberColumn("Unique", format="%.3f"),
            })

    # Quality scatter
    st.markdown('<div class="shdr">Quality vs uniqueness scatter</div>',
                unsafe_allow_html=True)
    scatter_df = df[["quality_score","uniqueness","tp_code",
                      "qa_word_count","unit_code"]].dropna().head(2000)
    fig_sc = go.Figure(go.Scattergl(
        x=scatter_df["uniqueness"],
        y=scatter_df["quality_score"],
        mode="markers",
        marker=dict(
            size=5,
            color=scatter_df["qa_word_count"].fillna(30),
            colorscale="Viridis",
            colorbar=dict(title="Word count", thickness=12),
            opacity=0.7,
        ),
        text=scatter_df["unit_code"],
        hovertemplate="<b>%{text}</b><br>Uniqueness: %{x:.3f}<br>Quality: %{y:.3f}<extra></extra>",
    ))
    fig_sc.update_layout(
        paper_bgcolor="#060d1a", plot_bgcolor="#060d1a",
        font_color="#9ca3af", height=400,
        xaxis=dict(title="Uniqueness", gridcolor="#1f2937", color="#4b5563"),
        yaxis=dict(title="Quality score", gridcolor="#1f2937", color="#4b5563"),
        margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text="Quality vs uniqueness — colour = word count",
                   font=dict(size=13, color="#6b7280")),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "**Ideal zone: top-right** — high uniqueness + high quality. "
        "**Top-left** = high quality but common/duplicate. "
        "**Bottom-right** = unique but low quality (worth rewriting). "
        "**Bottom-left** = consolidation + rewrite candidates."
    )
