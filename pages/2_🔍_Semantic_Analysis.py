"""
pages/2_🔍_Semantic_Analysis.py

Semantic analysis of skill statements:
  - Embedding-based similarity using OpenAI text-embedding-3-small
  - DBSCAN clustering to group similar statements
  - Near-duplicate detection
  - Canonical statement selection per cluster
  - Export of deduplicated / aligned RSD set
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="RSD Semantic Analysis", layout="wide")
st.title("🔍 Semantic Analysis — Align & Deduplicate Skill Statements")

st.markdown("""
Generates embeddings for every skill statement, clusters semantically similar ones,
identifies near-duplicates, and lets you pick a canonical statement per group —
producing a clean, aligned RSD set ready for export.
""")

# ── Connections ───────────────────────────────────────────────────────────────
def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

DB_URL      = _secret("DATABASE_URL")
OPENAI_KEY  = _secret("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.error("OPENAI_API_KEY not configured — required for embeddings.")
    st.stop()

if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

@st.cache_resource
def get_provider(key):
    from core.providers.openai_provider import OpenAIProvider
    return OpenAIProvider(key)

engine   = get_engine(DB_URL)
provider = get_provider(OPENAI_KEY)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Analysis settings")

    cluster_threshold = st.slider(
        "Cluster similarity threshold",
        min_value=0.70, max_value=0.98, value=0.85, step=0.01,
        help="Statements with similarity ≥ this value are grouped together. "
             "Higher = tighter groups, fewer clusters."
    )
    duplicate_threshold = st.slider(
        "Near-duplicate threshold",
        min_value=0.85, max_value=0.99, value=0.92, step=0.01,
        help="Pairs with similarity ≥ this are flagged as near-duplicates."
    )
    min_cluster_size = st.slider(
        "Min cluster size", min_value=2, max_value=5, value=2,
        help="Minimum number of statements to form a cluster."
    )
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )

    st.divider()
    st.header("Filter source data")

    # Load run list
    with engine.connect() as conn:
        runs = conn.execute(text(
            "SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC"
        )).mappings().all()
    runs = [dict(r) for r in runs]

    run_options = ["All runs"] + [f"{r['id'][:8]}… {r['source_filename']}" for r in runs]
    run_ids     = [None] + [r["id"] for r in runs]
    selected_run_label = st.selectbox("Source run", run_options)
    selected_run_id    = run_ids[run_options.index(selected_run_label)]

    unit_filter = st.text_input("Filter by unit code (optional)", placeholder="e.g. BSB")

    st.divider()
    run_analysis = st.button("▶ Run analysis", type="primary")

# ── Load skill statements ─────────────────────────────────────────────────────
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

df_raw = load_statements(selected_run_id, unit_filter)

if df_raw.empty:
    st.info("No skill statements found. Run some batches first.")
    st.stop()

st.info(f"**{len(df_raw)}** skill statements loaded — ready for analysis.")

# Show sample
with st.expander("Preview loaded statements"):
    st.dataframe(
        df_raw[["unit_code", "unit_title", "element_title", "skill_statement"]].head(20),
        use_container_width=True, hide_index=True,
    )

# ── Run analysis ──────────────────────────────────────────────────────────────
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if run_analysis:
    if len(df_raw) > 2000:
        st.warning(
            f"You have {len(df_raw)} statements. Analysis on large sets takes longer "
            "and uses more OpenAI API credits. Consider filtering by unit code first."
        )

    from core.semantic import analyse_statements

    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(pct)
        status_text.write(msg)

    try:
        results = analyse_statements(
            df_raw,
            provider=provider,
            embedding_model=embedding_model,
            cluster_threshold=cluster_threshold,
            duplicate_threshold=duplicate_threshold,
            min_cluster_size=min_cluster_size,
            progress_callback=update_progress,
        )
        st.session_state["analysis_results"] = results
        status_text.write("Analysis complete ✅")
        progress_bar.progress(1.0)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# ── Display results ───────────────────────────────────────────────────────────
results = st.session_state.get("analysis_results")

if results is None:
    st.info("Configure settings in the sidebar and click **▶ Run analysis** to begin.")
    st.stop()

df_ann       = results["df_annotated"]
clusters     = results["clusters"]
near_dupes   = results["near_duplicates"]
labels       = results["labels"]

n_clustered  = int((labels >= 0).sum())
n_singletons = int((labels == -1).sum())
n_clusters   = len(clusters)
n_dupes      = len(near_dupes)

# Summary metrics
st.divider()
st.subheader("Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total statements",   len(df_ann))
c2.metric("Clusters found",     n_clusters)
c3.metric("Statements clustered", n_clustered)
c4.metric("Singletons",         n_singletons)
c5.metric("Near-duplicate pairs", n_dupes)

potential_reduction = n_clustered - n_clusters
st.success(
    f"**Potential reduction: {potential_reduction} statements** — "
    f"keeping one canonical per cluster reduces {len(df_ann)} → "
    f"{n_singletons + n_clusters} statements "
    f"({round(100 * potential_reduction / max(len(df_ann), 1))}% reduction)."
)

st.divider()

# ── Cluster browser ───────────────────────────────────────────────────────────
st.subheader("Cluster browser")
st.caption("Each cluster groups semantically similar statements. The ⭐ canonical is the one closest to the cluster centre.")

if not clusters:
    st.info("No clusters found — try lowering the similarity threshold.")
else:
    # Sort clusters by size descending
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]["size"], reverse=True)

    for cluster_id, info in sorted_clusters:
        members = df_ann[df_ann.index.isin(info["member_indices"])]
        canonical_idx = info["canonical_idx"]

        with st.expander(
            f"Cluster {cluster_id} — {info['size']} statements "
            f"| ⭐ {info['canonical_text'][:80]}…"
        ):
            # Show all members with canonical highlighted
            for _, mem_row in members.iterrows():
                is_canon = mem_row.name == canonical_idx
                prefix = "⭐ **CANONICAL**" if is_canon else "○"
                unit = mem_row.get("unit_code", "")
                elem = mem_row.get("element_title", "")
                stmt = mem_row.get("skill_statement", "")
                st.markdown(
                    f"{prefix} `{unit}` — *{elem}*\n\n> {stmt}"
                )
                st.divider()

            # Allow user to override canonical choice
            member_options = {
                f"{df_ann.loc[i, 'unit_code']} — {df_ann.loc[i, 'skill_statement'][:60]}…": i
                for i in info["member_indices"]
            }
            chosen_label = st.selectbox(
                "Override canonical statement",
                list(member_options.keys()),
                index=list(member_options.values()).index(canonical_idx),
                key=f"canon_{cluster_id}",
            )
            chosen_idx = member_options[chosen_label]
            if chosen_idx != canonical_idx:
                # Update session state override
                overrides = st.session_state.get("canonical_overrides", {})
                overrides[cluster_id] = chosen_idx
                st.session_state["canonical_overrides"] = overrides
                st.success("Override saved — will use this statement in export.")

st.divider()

# ── Near-duplicate browser ────────────────────────────────────────────────────
st.subheader(f"Near-duplicate pairs ({n_dupes})")
st.caption(f"Statement pairs with similarity ≥ {duplicate_threshold}")

if not near_dupes:
    st.success("No near-duplicates found at this threshold.")
else:
    dupes_df = pd.DataFrame(near_dupes)

    # Add unit code context
    dupes_df["unit_a"] = dupes_df["idx_a"].apply(
        lambda i: df_ann.loc[i, "unit_code"] if i < len(df_ann) else ""
    )
    dupes_df["unit_b"] = dupes_df["idx_b"].apply(
        lambda i: df_ann.loc[i, "unit_code"] if i < len(df_ann) else ""
    )

    st.dataframe(
        dupes_df[["similarity", "unit_a", "text_a", "unit_b", "text_b"]].head(100),
        use_container_width=True,
        column_config={
            "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
            "unit_a":     st.column_config.TextColumn("Unit A", width="small"),
            "text_a":     st.column_config.TextColumn("Statement A", width="large"),
            "unit_b":     st.column_config.TextColumn("Unit B", width="small"),
            "text_b":     st.column_config.TextColumn("Statement B", width="large"),
        },
        hide_index=True,
    )

st.divider()

# ── Similarity heatmap (top N) ────────────────────────────────────────────────
st.subheader("Similarity heatmap (top 50 statements)")
st.caption("Darker = more similar. Diagonal is always 1.0.")

sim_matrix = results["sim_matrix"]
n_show = min(50, len(df_ann))
sim_slice = sim_matrix[:n_show, :n_show]

import altair as alt

# Build long-form dataframe for Altair
rows_heat = []
labels_heat = [
    f"{df_ann.iloc[i]['unit_code']}:{i}"
    for i in range(n_show)
]
for i in range(n_show):
    for j in range(n_show):
        rows_heat.append({"x": labels_heat[j], "y": labels_heat[i], "sim": float(sim_slice[i, j])})

heat_df = pd.DataFrame(rows_heat)
chart = alt.Chart(heat_df).mark_rect().encode(
    x=alt.X("x:O", sort=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y("y:O", sort=None, axis=alt.Axis(labels=False, ticks=False)),
    color=alt.Color("sim:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Similarity")),
    tooltip=["x", "y", alt.Tooltip("sim:Q", format=".3f")],
).properties(width=600, height=600)
st.altair_chart(chart, use_container_width=True)

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export deduplicated statements")

overrides = st.session_state.get("canonical_overrides", {})

# Build export dataframe — one row per cluster (canonical) + all singletons
canonical_indices = set()
for cluster_id, info in clusters.items():
    override_idx = overrides.get(cluster_id)
    canonical_indices.add(override_idx if override_idx is not None else info["canonical_idx"])

# Singletons (not in any cluster)
singleton_mask = df_ann["cluster_id"] == -1

# Combined
export_mask = df_ann.index.isin(canonical_indices) | singleton_mask
df_export = df_ann[export_mask].copy()

st.info(
    f"Export will contain **{len(df_export)} statements** "
    f"({len(df_ann) - len(df_export)} removed as non-canonical duplicates)."
)

# Show export preview
st.dataframe(
    df_export[["unit_code", "unit_title", "element_title", "skill_statement", "cluster_id", "is_canonical"]].head(30),
    use_container_width=True, hide_index=True,
)

# Download buttons
from core.exporters import to_rsd_rows, to_traceability, to_osmt_rows

c1, c2, c3 = st.columns(3)
c1.download_button(
    "⬇ Deduplicated OSMT CSV",
    to_osmt_rows(df_export).to_csv(index=False).encode(),
    "deduplicated_osmt.csv", "text/csv",
)
c2.download_button(
    "⬇ Deduplicated RSD CSV",
    to_rsd_rows(df_export).to_csv(index=False).encode(),
    "deduplicated_rsd.csv", "text/csv",
)
c3.download_button(
    "⬇ Cluster analysis CSV",
    df_ann[["unit_code", "unit_title", "element_title", "skill_statement",
             "cluster_id", "is_canonical", "is_singleton"]].to_csv(index=False).encode(),
    "cluster_analysis.csv", "text/csv",
)

st.caption(
    "The **Cluster analysis CSV** includes every statement with its cluster_id, "
    "is_canonical flag, and is_singleton flag — useful for reviewing groupings in Excel."
)
