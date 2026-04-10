"""
pages/3_🌐_ESCO_Alignment.py

Match skill statements to ESCO skills and associate with ESCO occupations.

Uses the free ESCO REST API (ec.europa.eu/esco/api) — no API key required.

Workflow:
  1. Load skill statements from DB
  2. For each statement, search ESCO /search?type=skill
  3. For top match, fetch essential + optional occupations
  4. Display results and allow export with ESCO enrichment
  5. Optionally save ESCO data back to DB
"""
from __future__ import annotations
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="ESCO Alignment", layout="wide")
st.title("🌐 ESCO Skills & Occupation Alignment")

st.markdown("""
Match each RSD skill statement against the **ESCO taxonomy** (European Skills,
Competences, Qualifications and Occupations) to identify the closest standardised
skill and the occupations that require it.

Uses the free [ESCO REST API](https://ec.europa.eu/esco/api) — no key required.
""")

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ESCO matching settings")

    top_n_skills = st.slider(
        "Top N ESCO skills to search",
        min_value=1, max_value=10, value=3,
        help="Number of ESCO skills returned per search. Only the top match is used for occupation lookup."
    )
    top_n_occupations = st.slider(
        "Max occupations per skill",
        min_value=3, max_value=20, value=8,
        help="Maximum number of ESCO occupations fetched per matched skill."
    )
    min_score = st.slider(
        "Minimum match score",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Discard ESCO matches below this relevance score. 0 = accept all matches."
    )
    language = st.selectbox(
        "ESCO language",
        ["en", "de", "fr", "es", "it", "nl", "pl", "pt"],
        index=0,
    )

    st.divider()
    st.header("Filter source data")

    with engine.connect() as conn:
        runs = conn.execute(text(
            "SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC"
        )).mappings().all()
    runs = [dict(r) for r in runs]

    run_options = ["All runs"] + [f"{str(r['id'])[:8]}… {r['source_filename']}" for r in runs]
    run_ids     = [None] + [str(r["id"]) for r in runs]
    selected_run_label = st.selectbox("Source run", run_options)
    selected_run_id    = run_ids[run_options.index(selected_run_label)]

    unit_filter = st.text_input("Filter by unit code", placeholder="e.g. BSB")

    batch_size = st.number_input(
        "Batch size (statements per run)",
        min_value=10, max_value=500, value=50, step=10,
        help="Process this many statements at a time to manage API rate limits."
    )

    st.divider()
    run_matching = st.button("▶ Run ESCO matching", type="primary")
    save_to_db   = st.button("💾 Save ESCO results to DB")

# ── Load statements ───────────────────────────────────────────────────────────
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

st.info(f"**{len(df_raw)}** skill statements loaded.")

# Estimate time
est_seconds = len(df_raw) * 0.7  # ~0.7s per statement (2 API calls + delays)
est_minutes = round(est_seconds / 60, 1)
st.caption(
    f"Estimated time: ~{est_minutes} minutes for all {len(df_raw)} statements "
    f"(ESCO API rate limiting applies — ~2 calls per statement)."
)

# Preview
with st.expander("Preview statements to match"):
    st.dataframe(
        df_raw[["unit_code", "unit_title", "element_title", "skill_statement"]].head(10),
        use_container_width=True, hide_index=True,
    )

# ── Run matching ──────────────────────────────────────────────────────────────
if "esco_results" not in st.session_state:
    st.session_state["esco_results"] = None
if "esco_offset" not in st.session_state:
    st.session_state["esco_offset"] = 0

if run_matching:
    from core.esco import batch_match

    offset = st.session_state.get("esco_offset", 0)
    end    = min(offset + int(batch_size), len(df_raw))
    batch  = df_raw.iloc[offset:end].copy()

    st.info(f"Processing statements {offset + 1}–{end} of {len(df_raw)}…")
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(min(pct, 1.0))
        status_text.write(msg)

    try:
        batch_results = batch_match(
            batch,
            top_n_skills=top_n_skills,
            top_n_occupations=top_n_occupations,
            min_score=min_score,
            progress_callback=update_progress,
        )

        # Merge with existing results
        existing = st.session_state.get("esco_results")
        if existing is not None:
            combined = pd.concat([existing, batch_results], ignore_index=True)
        else:
            combined = batch_results

        st.session_state["esco_results"] = combined
        st.session_state["esco_offset"]  = end

        status_text.write(f"✅ Batch complete — {len(combined)} statements matched so far.")
        progress_bar.progress(1.0)

        remaining = len(df_raw) - end
        if remaining > 0:
            st.info(
                f"**{remaining} statements remaining.** "
                f"Click **▶ Run ESCO matching** again to continue."
            )
        else:
            st.success("✅ All statements matched!")
            st.session_state["esco_offset"] = 0  # reset for next run

    except Exception as e:
        st.error(f"ESCO matching failed: {e}")
        st.caption("This may be a temporary ESCO API issue. Try again in a few seconds.")

# ── Display results ───────────────────────────────────────────────────────────
results_df = st.session_state.get("esco_results")

if results_df is None or results_df.empty:
    st.info("Configure settings in the sidebar and click **▶ Run ESCO matching** to begin.")
    st.stop()

st.divider()
st.subheader(f"ESCO matching results — {len(results_df)} statements")

# Summary metrics
matched = results_df[results_df["esco_skill_uri"] != ""]
unmatched = results_df[results_df["esco_skill_uri"] == ""]
with_occs = results_df[results_df["esco_occupation_titles"] != ""]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Statements processed", len(results_df))
c2.metric("Matched to ESCO skill", len(matched),
          delta=f"{round(100*len(matched)/max(len(results_df),1))}%")
c3.metric("With occupations",     len(with_occs))
c4.metric("Unmatched",            len(unmatched))

if len(matched) > 0:
    avg_score = round(matched["esco_skill_score"].mean(), 3)
    st.caption(f"Average ESCO match score: **{avg_score}** (higher = closer match)")

st.divider()

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Results browser")

score_filter = st.slider(
    "Show results with ESCO score ≥",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
)
show_df = results_df[results_df["esco_skill_score"] >= score_filter].copy()

view_cols = [c for c in [
    "unit_code", "element_title", "skill_statement",
    "esco_skill_title", "esco_skill_score",
    "esco_occupation_titles",
] if c in show_df.columns]

st.dataframe(
    show_df[view_cols],
    use_container_width=True,
    column_config={
        "unit_code":             st.column_config.TextColumn("Unit", width="small"),
        "element_title":         st.column_config.TextColumn("Element"),
        "skill_statement":       st.column_config.TextColumn("Skill statement", width="large"),
        "esco_skill_title":      st.column_config.TextColumn("ESCO skill match", width="medium"),
        "esco_skill_score":      st.column_config.NumberColumn("Score", format="%.3f"),
        "esco_occupation_titles":st.column_config.TextColumn("Associated occupations", width="large"),
    },
    hide_index=True,
)

st.divider()

# ── Occupation frequency analysis ────────────────────────────────────────────
st.subheader("Occupation frequency")
st.caption("Which ESCO occupations appear most frequently across all matched skills?")

all_occ_titles = (
    results_df["esco_occupation_titles"]
    .dropna()
    .str.split("; ")
    .explode()
    .str.strip()
    .replace("", pd.NA)
    .dropna()
)

if len(all_occ_titles) > 0:
    occ_counts = all_occ_titles.value_counts().reset_index()
    occ_counts.columns = ["Occupation", "Count"]

    col1, col2 = st.columns([2, 3])
    with col1:
        st.caption(f"{len(occ_counts)} unique occupations identified")
        st.dataframe(occ_counts.head(30), use_container_width=True, hide_index=True)
    with col2:
        st.caption("Top 20 occupations by frequency")
        st.bar_chart(occ_counts.head(20).set_index("Occupation"))
else:
    st.info("No occupation data yet — run matching first.")

st.divider()

# ── By unit code analysis ─────────────────────────────────────────────────────
st.subheader("ESCO coverage by unit")

if "unit_code" in results_df.columns and len(matched) > 0:
    unit_summary = (
        results_df.groupby("unit_code")
        .agg(
            total=("skill_statement", "count"),
            matched=("esco_skill_uri", lambda x: (x != "").sum()),
            avg_score=("esco_skill_score", "mean"),
            top_esco=("esco_skill_title", lambda x: x.mode()[0] if len(x.mode()) > 0 else ""),
        )
        .reset_index()
    )
    unit_summary["match_rate"] = (
        100 * unit_summary["matched"] / unit_summary["total"]
    ).round(1).astype(str) + "%"
    unit_summary["avg_score"] = unit_summary["avg_score"].round(3)

    st.dataframe(
        unit_summary,
        use_container_width=True,
        column_config={
            "unit_code":   st.column_config.TextColumn("Unit"),
            "total":       st.column_config.NumberColumn("Statements"),
            "matched":     st.column_config.NumberColumn("Matched"),
            "match_rate":  st.column_config.TextColumn("Match rate"),
            "avg_score":   st.column_config.NumberColumn("Avg score"),
            "top_esco":    st.column_config.TextColumn("Most common ESCO skill", width="large"),
        },
        hide_index=True,
    )

st.divider()

# ── Save to DB ────────────────────────────────────────────────────────────────
if save_to_db:
    if results_df is None or results_df.empty:
        st.warning("No results to save yet.")
    else:
        # Ensure columns exist in DB
        with engine.begin() as conn:
            for col, defn in [
                ("esco_skill_uri",         "TEXT"),
                ("esco_skill_title",       "TEXT"),
                ("esco_skill_score",       "REAL"),
                ("esco_occupation_titles", "TEXT"),
                ("esco_occupation_uris",   "TEXT"),
            ]:
                conn.execute(text(f"""
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='rsd_skill_records' AND column_name='{col}'
                      ) THEN
                        ALTER TABLE rsd_skill_records ADD COLUMN {col} {defn};
                      END IF;
                    END $$;
                """))

        # Update records
        updated = 0
        with engine.begin() as conn:
            for _, row in results_df.iterrows():
                if not row.get("id"):
                    continue
                conn.execute(
                    text("""
                        UPDATE rsd_skill_records SET
                            esco_skill_uri         = :eski_uri,
                            esco_skill_title       = :esco_title,
                            esco_skill_score       = :esco_score,
                            esco_occupation_titles = :occ_titles,
                            esco_occupation_uris   = :occ_uris
                        WHERE id = :rec_id
                    """),
                    {
                        "eski_uri":   str(row.get("esco_skill_uri", "") or ""),
                        "esco_title": str(row.get("esco_skill_title", "") or ""),
                        "esco_score": float(row.get("esco_skill_score", 0) or 0),
                        "occ_titles": str(row.get("esco_occupation_titles", "") or ""),
                        "occ_uris":   str(row.get("esco_occupation_uris", "") or ""),
                        "rec_id":     str(row["id"]),
                    }
                )
                updated += 1

        st.success(f"✅ Saved ESCO data for {updated} records to DB.")
        st.cache_data.clear()

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export with ESCO enrichment")

if results_df is not None and not results_df.empty:
    # Build enriched export
    export_cols = [c for c in [
        "unit_code", "unit_title", "element_title",
        "skill_statement", "keywords",
        "esco_skill_uri", "esco_skill_title", "esco_skill_score",
        "esco_occupation_titles", "esco_occupation_uris",
        "qa_passes", "qa_word_count",
    ] if c in results_df.columns]

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ ESCO-enriched RSD CSV",
        results_df[export_cols].to_csv(index=False).encode(),
        "esco_enriched_rsd.csv",
        "text/csv",
    )
    c2.download_button(
        "⬇ Occupation mapping CSV",
        results_df[[c for c in [
            "unit_code", "element_title", "skill_statement",
            "esco_skill_title", "esco_skill_score",
            "esco_occupation_titles",
        ] if c in results_df.columns]].to_csv(index=False).encode(),
        "occupation_mapping.csv",
        "text/csv",
    )

    st.caption(
        "The **ESCO-enriched RSD CSV** includes the matched ESCO skill URI and title, "
        "relevance score, and all associated ESCO occupations for each skill statement. "
        "The **Occupation mapping CSV** is a clean view for stakeholder review."
    )
