"""
pages/4_📊_Insights.py

Visual analytics + social-media-ready graphics for the RSD skill set.

Charts:
  1. Skill Network Graph — force-directed, units as nodes, edges = cross-TP similarity
  2. Training Package Overlap Sankey — flow from TPs into shared cluster groups
  3. Skill Universe Bubble Chart — units sized by count, coloured by cluster density
  4. Keyword Galaxy — word cloud of most frequent skill keywords
  5. Impact Card — before/after deduplication stats
  6. Coverage Sunburst — skill count by TP → element category
  7. QA Pass-rate Gauge — visual quality indicator
  8. Cross-TP Chord — which packages share the most skills
"""
from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="RSD Insights", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  .insight-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
  }
  .insight-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
  }
  .big-stat {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .stat-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
  }
  .stat-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 8px;
  }
  .share-tip {
    background: #0c1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
    color: #7dd3fc;
    margin-top: 12px;
  }
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #334155;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
    margin: 40px 0 20px 0;
  }
</style>
""", unsafe_allow_html=True)

st.title("📊 RSD Insights")
st.caption("Visual analytics and shareable graphics for your skill statement library")

# ── DB connection ─────────────────────────────────────────────────────────────
def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

DB_URL = _secret("DATABASE_URL")
if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

@st.cache_data(ttl=120, show_spinner="Loading skill data…")
def load_all():
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(text(
                "SELECT s.unit_code, s.unit_title, s.element_title, s.skill_statement, "
                "s.keywords, s.qa_passes, s.qa_word_count, s.rewrite_count, "
                "s.tp_code, s.tp_title, r.source_filename "
                "FROM rsd_skill_records s "
                "LEFT JOIN rsd_runs r ON r.id = s.run_id "
                "ORDER BY s.unit_code, s.row_index"
            )).mappings().all()
        )
    return df

df = load_all()

if df.empty:
    st.info("No skill records in DB yet.")
    st.stop()

# Robust tp_code derivation — DB stores empty strings, not NULLs, for missing values
df["unit_code"] = df["unit_code"].replace("", np.nan).fillna("(unknown)")

# Priority: tp_code column -> unit_code first 3 chars -> source_filename -> fallback
df["tp_code"] = df["tp_code"].replace("", np.nan)
_uc_prefix = df["unit_code"].str[:3].replace("(un", np.nan).replace("", np.nan)
df["tp_code"] = df["tp_code"].fillna(_uc_prefix)

# Derive from source_filename for records still blank (e.g. "MSL.xlsx" -> "MSL")
if "source_filename" in df.columns:
    _from_file = (
        df["source_filename"].fillna("")
        .str.replace(r"\.(xlsx|csv)$", "", regex=True)
        .str.replace(r"\s.*$", "", regex=True)   # strip spaces and after
        .str.upper().str[:6]
        .replace("", np.nan)
    )
    df["tp_code"] = df["tp_code"].fillna(_from_file)
    df.drop(columns=["source_filename"], inplace=True, errors="ignore")

df["tp_code"]  = df["tp_code"].fillna("(unknown)").replace("", "(unknown)")
df["tp_title"] = df["tp_title"].replace("", np.nan).fillna(df["tp_code"])
df["qa_passes"] = df["qa_passes"].fillna(False)

n_total    = len(df)
n_units    = df["unit_code"].nunique()
n_tps      = df["tp_code"].nunique()
qa_rate    = round(100 * df["qa_passes"].sum() / max(n_total, 1), 1)
avg_words  = round(df["qa_word_count"].dropna().mean(), 1)

# Check if semantic analysis results exist in session
has_semantic = st.session_state.get("sa_results", False)
if has_semantic:
    df_ann   = st.session_state["sa_df_ann"]
    clusters = st.session_state["sa_clusters"]
    n_clusters  = len(clusters)
    n_canonical = n_clusters + int((df_ann["cluster_id"] == -1).sum())
else:
    n_clusters  = None
    n_canonical = None

# ── 1. Impact Stats Card ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Impact overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

def stat_card(col, value, label, sub="", colour="#38bdf8"):
    with col:
        st.markdown(
            f'<div class="insight-card">'
            f'<div class="big-stat" style="background:linear-gradient(135deg,{colour},#818cf8);'
            f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

stat_card(c1, f"{n_total:,}", "Skill Statements", f"across {n_units} units", "#38bdf8")
stat_card(c2, f"{n_tps}", "Training Packages", f"in the database", "#818cf8")
stat_card(c3, f"{qa_rate}%", "QA Pass Rate", f"avg {avg_words} words/stmt", "#4ade80")
if n_canonical:
    reduction = round(100 * (n_total - n_canonical) / max(n_total, 1))
    stat_card(c4, f"{n_canonical:,}", "Canonical Statements",
              f"−{reduction}% after deduplication", "#f472b6")
else:
    stat_card(c4, "Run", "Semantic Analysis", "to see deduplication impact", "#f472b6")

st.markdown(
    '<div class="share-tip">📱 Social media tip: Screenshot these cards. '
    '"We distilled X training packages into Y canonical skill statements with Z% QA pass rate."</div>',
    unsafe_allow_html=True,
)

# ── 2. Skill Universe Bubble Chart ───────────────────────────────────────────
st.markdown('<div class="section-header">Skill universe — training packages</div>',
            unsafe_allow_html=True)
st.caption(
    "Each bubble = one training package. Size = number of skill statements. "
    "Colour = QA pass rate. The closer to the centre, the higher the average word count."
)

unit_stats = (
    df.groupby(["tp_code", "tp_title", "unit_code"])
    .agg(
        count=("skill_statement", "count"),
        qa_rate=("qa_passes", lambda x: round(100 * x.sum() / max(len(x), 1), 1)),
        avg_words=("qa_word_count", "mean"),
    )
    .reset_index()
)
tp_stats = (
    df.groupby("tp_code")
    .agg(
        count=("skill_statement", "count"),
        qa_rate=("qa_passes", lambda x: round(100 * x.sum() / max(len(x), 1), 1)),
        avg_words=("qa_word_count", "mean"),
        tp_title=("tp_title", "first"),
        n_units=("unit_code", "nunique"),
    )
    .reset_index()
)
tp_stats["avg_words"] = tp_stats["avg_words"].fillna(35)
tp_stats["label"]     = tp_stats["tp_code"] + " (" + tp_stats["count"].astype(str) + ")"

bubble = alt.Chart(tp_stats).mark_circle(opacity=0.85, stroke="white", strokeWidth=0.5).encode(
    x=alt.X("avg_words:Q",
            title="Avg statement word count",
            scale=alt.Scale(domain=[
                max(0, float(tp_stats["avg_words"].min()) - 3),
                float(tp_stats["avg_words"].max()) + 3,
            ])),
    y=alt.Y("qa_rate:Q", title="QA pass rate (%)", scale=alt.Scale(domain=[50, 105])),
    size=alt.Size("count:Q", scale=alt.Scale(range=[200, 4000]),
                  legend=alt.Legend(title="Statements")),
    color=alt.Color("qa_rate:Q",
                    scale=alt.Scale(scheme="greens", domain=[60, 100]),
                    legend=alt.Legend(title="QA %")),
    tooltip=[
        alt.Tooltip("tp_code:N",   title="TP Code"),
        alt.Tooltip("tp_title:N",  title="Training Package"),
        alt.Tooltip("count:Q",     title="Statements"),
        alt.Tooltip("n_units:Q",   title="Units"),
        alt.Tooltip("qa_rate:Q",   title="QA Pass %", format=".1f"),
        alt.Tooltip("avg_words:Q", title="Avg words", format=".1f"),
    ],
).properties(height=420, title="Training Package Skill Universe")

text_layer = alt.Chart(tp_stats[tp_stats["count"] >= tp_stats["count"].quantile(0.6)]).mark_text(
    fontSize=10, fontWeight="bold", dy=0, color="white",
).encode(
    x="avg_words:Q",
    y="qa_rate:Q",
    text="tp_code:N",
)

st.altair_chart((bubble + text_layer), use_container_width=True)
st.caption("💡 Top-right = high QA pass rate + detailed statements. Save via chart menu (⋮) for sharing.")

# ── 3. Skill Coverage Sunburst ────────────────────────────────────────────────
st.markdown('<div class="section-header">Skill coverage — package → unit breakdown</div>',
            unsafe_allow_html=True)

try:
    import plotly.express as px

    # Build hierarchy: TP → Unit
    sun_df = (
        df.groupby(["tp_code", "unit_code"])
        .agg(count=("skill_statement", "count"),
             qa=("qa_passes", lambda x: round(100*x.sum()/max(len(x),1),1)))
        .reset_index()
    )
    sun_df["tp_label"] = sun_df["tp_code"]

    fig_sun = px.sunburst(
        sun_df,
        path=["tp_label", "unit_code"],
        values="count",
        color="qa",
        color_continuous_scale="RdYlGn",
        range_color=[60, 100],
        title="Skill coverage by training package and unit",
        color_continuous_midpoint=80,
    )
    fig_sun.update_traces(textinfo="label+percent parent")
    fig_sun.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="white",
        height=600,
        margin=dict(t=50, l=0, r=0, b=0),
        coloraxis_colorbar=dict(title="QA %"),
    )
    st.plotly_chart(fig_sun, use_container_width=True)
    st.caption(
        "Click any segment to drill in. Colour = QA pass rate. "
        "📱 Great for social media: shows the breadth and quality of your skills library at a glance."
    )
except ImportError:
    st.info("Install plotly to see sunburst chart.")

# ── 4. Keyword Galaxy (Word Cloud) ────────────────────────────────────────────
st.markdown('<div class="section-header">Keyword galaxy</div>', unsafe_allow_html=True)

kw_col = "keywords" if "keywords" in df.columns else None
if kw_col:
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        all_kws = (
            df[kw_col].dropna()
            .str.split(";")
            .explode()
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
            .dropna()
        )
        freq_dict = all_kws.value_counts().to_dict()

        # Custom colour function — cool blues/purples
        def skill_colour(word, font_size, position, orientation, random_state=None, **kwargs):
            colours = ["#38bdf8", "#818cf8", "#f472b6", "#4ade80", "#facc15", "#fb923c"]
            import random
            rng = random.Random(hash(word) % 1000)
            return rng.choice(colours)

        wc = WordCloud(
            width=1400, height=700,
            background_color="#0f172a",
            max_words=120,
            color_func=skill_colour,
            prefer_horizontal=0.85,
            min_font_size=10,
            max_font_size=90,
            relative_scaling=0.5,
            collocations=False,
        ).generate_from_frequencies(freq_dict)

        fig_wc, ax = plt.subplots(figsize=(14, 7))
        fig_wc.patch.set_facecolor("#0f172a")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            "Most frequent skill keywords across all training packages",
            color="white", fontsize=13, pad=12,
        )
        st.pyplot(fig_wc)

        # Download as PNG
        buf = io.BytesIO()
        fig_wc.savefig(buf, format="png", bbox_inches="tight",
                       facecolor="#0f172a", dpi=180)
        buf.seek(0)
        st.download_button(
            "⬇ Download Keyword Galaxy (PNG)",
            buf, "keyword_galaxy.png", "image/png",
        )
        plt.close(fig_wc)
        st.caption(
            "📱 Perfect for LinkedIn/Twitter: shows the language of vocational skills. "
            "Download as PNG and post directly."
        )

    except ImportError:
        st.info("Install `wordcloud` and `matplotlib` to see the keyword galaxy.")
else:
    st.info("No keyword data found in database.")

# ── 5. Cross-TP Skill Overlap Chord / Heatmap ─────────────────────────────────
st.markdown('<div class="section-header">Cross-training-package skill overlap</div>',
            unsafe_allow_html=True)

# Count shared elements between TPs (elements with same name across TPs)
elem_tp = (
    df.groupby(["element_title", "tp_code"])
    .size()
    .reset_index(name="count")
)
shared_elements = (
    elem_tp.groupby("element_title")["tp_code"]
    .apply(list)
    .reset_index()
)
shared_elements = shared_elements[shared_elements["tp_code"].apply(len) > 1]

# Build TP-TP overlap matrix based on shared element names
tps = sorted(df["tp_code"].dropna().unique())
overlap_rows = []
for i, tp_a in enumerate(tps):
    for j, tp_b in enumerate(tps):
        if i >= j:
            continue
        shared = shared_elements[
            shared_elements["tp_code"].apply(
                lambda x: tp_a in x and tp_b in x
            )
        ]
        if len(shared) > 0:
            overlap_rows.append({
                "TP A": tp_a, "TP B": tp_b,
                "Shared elements": len(shared),
                "Examples": "; ".join(shared["element_title"].head(3).tolist()),
            })

if overlap_rows:
    overlap_df = pd.DataFrame(overlap_rows).sort_values("Shared elements", ascending=False)

    col_chart2, col_table2 = st.columns([2, 1])
    with col_chart2:
        # Build symmetric matrix for heatmap
        all_tps = sorted(set(overlap_df["TP A"].tolist() + overlap_df["TP B"].tolist()))
        heat_rows = []
        overlap_lookup = {(r["TP A"], r["TP B"]): r["Shared elements"]
                          for _, r in overlap_df.iterrows()}
        for ta in all_tps:
            for tb in all_tps:
                val = overlap_lookup.get((ta, tb), overlap_lookup.get((tb, ta), 0))
                if ta == tb:
                    val = int(df[df["tp_code"] == ta]["element_title"].nunique())
                heat_rows.append({"TP A": ta, "TP B": tb, "Overlap": val})

        heat2_df = pd.DataFrame(heat_rows)
        overlap_chart = alt.Chart(heat2_df).mark_rect(
            cornerRadius=2,
        ).encode(
            x=alt.X("TP B:N", axis=alt.Axis(labelAngle=-45, title="")),
            y=alt.Y("TP A:N", axis=alt.Axis(title="")),
            color=alt.Color(
                "Overlap:Q",
                scale=alt.Scale(scheme="orangered"),
                legend=alt.Legend(title="Shared elements"),
            ),
            tooltip=[
                alt.Tooltip("TP A:N"),
                alt.Tooltip("TP B:N"),
                alt.Tooltip("Overlap:Q", title="Shared elements"),
            ],
        ).properties(
            height=400,
            title="Element-name overlap between training packages",
        ).interactive(False)
        st.altair_chart(overlap_chart, use_container_width=True)

    with col_table2:
        st.caption(f"**{len(overlap_df)} overlapping TP pairs**")
        st.dataframe(
            overlap_df[["TP A", "TP B", "Shared elements"]].head(20),
            use_container_width=True, hide_index=True,
        )
        with st.expander("View examples"):
            st.dataframe(
                overlap_df[["TP A", "TP B", "Examples"]].head(10),
                use_container_width=True, hide_index=True,
            )

    st.caption(
        "📱 Talking point: 'Our analysis found X training packages share common skill elements, "
        "enabling a unified cross-industry competency framework.'"
    )
else:
    st.info("No element-name overlap found between training packages.")

# ── 6. QA Quality Distribution ────────────────────────────────────────────────
st.markdown('<div class="section-header">Statement quality distribution</div>',
            unsafe_allow_html=True)

col_qa1, col_qa2 = st.columns(2)

with col_qa1:
    # Word count distribution per TP
    wc_df = df[df["qa_word_count"].notna()].copy()
    wc_df["qa_word_count"] = wc_df["qa_word_count"].astype(float)

    wc_chart = alt.Chart(wc_df).mark_bar(
        color="#38bdf8", opacity=0.8,
        cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
    ).encode(
        x=alt.X("qa_word_count:Q", bin=alt.Bin(maxbins=25),
                title="Word count per statement"),
        y=alt.Y("count()", title="Number of statements"),
        tooltip=[
            alt.Tooltip("qa_word_count:Q", bin=True, title="Word count"),
            alt.Tooltip("count()", title="Statements"),
        ],
    ).properties(
        height=250,
        title="Statement length distribution (target: 30–50 words)",
    )
    # Target zone
    target_zone = alt.Chart(pd.DataFrame([{"x1": 30, "x2": 50}])).mark_rect(
        color="#4ade80", opacity=0.12,
    ).encode(x="x1:Q", x2="x2:Q")

    st.altair_chart(target_zone + wc_chart, use_container_width=True)

with col_qa2:
    # QA pass rate by TP — horizontal bar
    qa_by_tp = (
        df.groupby("tp_code")
        .agg(
            total=("qa_passes", "count"),
            passed=("qa_passes", "sum"),
        )
        .reset_index()
    )
    qa_by_tp["rate"] = (100 * qa_by_tp["passed"] / qa_by_tp["total"].clip(lower=1)).round(1)
    qa_by_tp = qa_by_tp.sort_values("rate", ascending=True)

    qa_bar = alt.Chart(qa_by_tp).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
    ).encode(
        y=alt.Y("tp_code:N", sort=None, title=""),
        x=alt.X("rate:Q", title="QA pass rate (%)", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color(
            "rate:Q",
            scale=alt.Scale(scheme="rdylgn", domain=[60, 100]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("tp_code:N", title="TP"),
            alt.Tooltip("rate:Q", title="QA pass %", format=".1f"),
            alt.Tooltip("total:Q", title="Total statements"),
        ],
    ).properties(
        height=250,
        title="QA pass rate by training package",
    )
    target_line = alt.Chart(pd.DataFrame({"x": [90]})).mark_rule(
        color="#4ade80", strokeDash=[4, 2], size=1.5,
    ).encode(x="x:Q")

    st.altair_chart(qa_bar + target_line, use_container_width=True)
    st.caption("Green line = 90% target")

# ── 7. Skills Network Graph ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Skills network — training package connections</div>',
            unsafe_allow_html=True)
st.caption(
    "Force-directed network: nodes = training packages, "
    "edges = shared element names between packages. "
    "Thicker/brighter edges = more shared skills."
)

if overlap_rows:
    try:
        import networkx as nx

        G = nx.Graph()
        # Nodes
        for tp in tps:
            n = int(df[df["tp_code"] == tp]["skill_statement"].count())
            G.add_node(tp, size=n)

        # Edges — only where overlap > 0
        for _, row in overlap_df.iterrows():
            G.add_edge(row["TP A"], row["TP B"], weight=int(row["Shared elements"]))

        # Layout — spring layout
        pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

        # Build edge dataframe
        edge_rows = []
        for u, v, data in G.edges(data=True):
            edge_rows.append({
                "x0": pos[u][0], "y0": pos[u][1],
                "x1": pos[v][0], "y1": pos[v][1],
                "weight": data.get("weight", 1),
            })
        edge_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
            columns=["x0","y0","x1","y1","weight"])

        # Build node dataframe
        node_rows = []
        for node in G.nodes():
            node_rows.append({
                "tp": node,
                "x": pos[node][0],
                "y": pos[node][1],
                "size": G.nodes[node].get("size", 10),
                "degree": G.degree(node),
            })
        node_df = pd.DataFrame(node_rows)

        # Edge layer
        if not edge_df.empty:
            edges_chart = alt.Chart(edge_df).mark_rule(
                color="#334155", opacity=0.6,
            ).encode(
                x="x0:Q", y="y0:Q",
                x2="x1:Q", y2="y1:Q",
                strokeWidth=alt.StrokeWidth(
                    "weight:Q",
                    scale=alt.Scale(range=[0.5, 4]),
                    legend=None,
                ),
                opacity=alt.Opacity(
                    "weight:Q",
                    scale=alt.Scale(range=[0.3, 0.9]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("weight:Q", title="Shared elements")],
            )
        else:
            edges_chart = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()

        # Node layer
        nodes_chart = alt.Chart(node_df).mark_circle(
            stroke="white", strokeWidth=1.5,
        ).encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            size=alt.Size("size:Q", scale=alt.Scale(range=[300, 3000]),
                          legend=alt.Legend(title="Statements")),
            color=alt.Color("degree:Q",
                            scale=alt.Scale(scheme="plasma"),
                            legend=alt.Legend(title="Connections")),
            tooltip=[
                alt.Tooltip("tp:N",     title="Training Package"),
                alt.Tooltip("size:Q",   title="Statements"),
                alt.Tooltip("degree:Q", title="Connections"),
            ],
        )

        text_chart = alt.Chart(node_df).mark_text(
            fontSize=9, fontWeight="bold", dy=-14, color="white",
        ).encode(
            x="x:Q", y="y:Q",
            text="tp:N",
        )

        network_chart = (edges_chart + nodes_chart + text_chart).properties(
            width=700, height=550,
            title="Training Package Skills Network",
        ).configure_view(
            strokeWidth=0,
            fill="#0f172a",
        ).configure_axis(grid=False).interactive(False)

        st.altair_chart(network_chart, use_container_width=True)
        st.caption(
            "📱 **Most shareable chart** — shows the interconnected nature of vocational skills. "
            "Save via chart menu (⋮ → Save as SVG/PNG) for high-quality social media posts. "
            "Caption idea: 'Mapping the connections across Australia's vocational training landscape.'"
        )

    except ImportError:
        st.info("Install `networkx` to see the skills network graph.")
else:
    st.info("No cross-TP overlap found to build network graph.")

# ── 8. Rewrite effort heatmap ─────────────────────────────────────────────────
st.markdown('<div class="section-header">AI generation effort — rewrite heatmap</div>',
            unsafe_allow_html=True)
st.caption(
    "How many BART rewrites were needed per unit. "
    "High rewrite counts indicate elements that are harder to express as skill statements."
)

if "rewrite_count" in df.columns:
    rewrite_df = (
        df[df["rewrite_count"].notna()]
        .groupby(["tp_code", "unit_code"])
        .agg(
            avg_rewrites=("rewrite_count", "mean"),
            max_rewrites=("rewrite_count", "max"),
            n=("rewrite_count", "count"),
        )
        .reset_index()
    )
    rewrite_df["avg_rewrites"] = rewrite_df["avg_rewrites"].round(2)

    # Show only TPs with > 1 unit
    tp_multi = rewrite_df.groupby("tp_code")["unit_code"].count()
    tp_multi = tp_multi[tp_multi >= 2].index.tolist()
    rewrite_show = rewrite_df[rewrite_df["tp_code"].isin(tp_multi[:15])]

    if not rewrite_show.empty:
        rewrite_chart = alt.Chart(rewrite_show).mark_rect(
            cornerRadius=2,
        ).encode(
            x=alt.X("unit_code:N",
                    axis=alt.Axis(labelAngle=-45, title="Unit code")),
            y=alt.Y("tp_code:N", title="Training package"),
            color=alt.Color(
                "avg_rewrites:Q",
                scale=alt.Scale(scheme="orangered", domain=[0, 3]),
                legend=alt.Legend(title="Avg rewrites"),
            ),
            tooltip=[
                alt.Tooltip("tp_code:N",      title="TP"),
                alt.Tooltip("unit_code:N",     title="Unit"),
                alt.Tooltip("avg_rewrites:Q",  title="Avg rewrites", format=".2f"),
                alt.Tooltip("max_rewrites:Q",  title="Max rewrites"),
                alt.Tooltip("n:Q",             title="Statements"),
            ],
        ).properties(
            height=350,
            title="Average BART rewrites per unit (darker = more effort required)",
        ).interactive(False)
        st.altair_chart(rewrite_chart, use_container_width=True)
        st.caption(
            "Units requiring more rewrites may need their performance criteria reviewed — "
            "they may be poorly structured or overly complex."
        )

# ── 9. Social media export pack ───────────────────────────────────────────────
st.markdown('<div class="section-header">Social media export tips</div>',
            unsafe_allow_html=True)

tips = [
    ("📊 Skill Universe Bubble", "Download via chart menu (⋮). Caption: 'X training packages, Y skills, one unified framework.'"),
    ("🌐 Skills Network Graph",  "Download via chart menu. Caption: 'Mapping the connections across vocational training.'"),
    ("☁️ Keyword Galaxy",        "Download button above. Caption: 'The language of work — most common skills across VET.'"),
    ("🌞 Coverage Sunburst",     "Screenshot + crop. Caption: 'From X training packages to a single skills taxonomy.'"),
    ("📈 Impact Stats Card",     "Screenshot the top 4 metric cards. Caption: 'XX% QA pass rate across YY skill statements.'"),
]

for icon_title, tip in tips:
    st.markdown(
        f'<div class="share-tip"><strong>{icon_title}</strong><br>{tip}</div>',
        unsafe_allow_html=True,
    )

st.markdown("")
st.info(
    "💡 **Best performing formats for LinkedIn:** "
    "Infographics with 1–3 large numbers, network graphs, before/after comparisons. "
    "Use dark backgrounds (#0f172a) — they stand out in light-mode feeds. "
    "Tag @NCVER, @SkillsOrg, relevant industry bodies for reach."
)
