"""
pages/1_📊_Dashboard.py

DB dashboard — works with rsd_runs / rsd_skill_records schema.
"""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="RSD Dashboard", layout="wide")
st.title("📊 RSD Database Dashboard")

# ── DB connection ─────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

DB_URL = _secret("DATABASE_URL")

if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url: str):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

try:
    engine = get_engine(DB_URL)
except Exception as e:
    st.error(f"DB connection failed: {e}")
    st.stop()

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def load_runs() -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT
                r.id                                                AS run_id,
                r.created_at,
                r.updated_at,
                r.status,
                r.source_filename,
                r.extractor_name,
                r.model,
                r.provider,
                COUNT(s.id)                                         AS total_records,
                SUM(CASE WHEN s.qa_passes THEN 1 ELSE 0 END)       AS qa_passed,
                SUM(CASE WHEN s.error_message != '' AND s.error_message IS NOT NULL
                         THEN 1 ELSE 0 END)                         AS errors
            FROM rsd_runs r
            LEFT JOIN rsd_skill_records s ON s.run_id = r.id
            GROUP BY r.id
            ORDER BY r.created_at DESC
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def load_skill_records(run_id: str | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        if run_id:
            res = conn.execute(
                text("SELECT * FROM rsd_skill_records WHERE run_id=:rid ORDER BY row_index"),
                {"rid": run_id},
            ).mappings().all()
        else:
            res = conn.execute(
                text("SELECT * FROM rsd_skill_records ORDER BY unit_code, row_index")
            ).mappings().all()
    return pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def load_summary_stats() -> dict:
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(DISTINCT r.id)                                    AS total_runs,
                COUNT(s.id)                                             AS total_skills,
                COUNT(DISTINCT s.unit_code)                             AS total_units,
                SUM(CASE WHEN s.qa_passes THEN 1 ELSE 0 END)           AS qa_passed,
                SUM(CASE WHEN NOT s.qa_passes THEN 1 ELSE 0 END)       AS qa_failed,
                SUM(CASE WHEN s.error_message != '' AND s.error_message IS NOT NULL
                         THEN 1 ELSE 0 END)                             AS total_errors,
                ROUND(AVG(s.qa_word_count)::numeric, 1)                 AS avg_word_count,
                ROUND(AVG(s.rewrite_count)::numeric, 2)                 AS avg_rewrites,
                ROUND(AVG(s.bart_temperature)::numeric, 2)              AS avg_temperature
            FROM rsd_runs r
            LEFT JOIN rsd_skill_records s ON s.run_id = r.id
        """)).mappings().first()
    return dict(stats) if stats else {}


@st.cache_data(ttl=30, show_spinner=False)
def load_unit_summary() -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT
                unit_code,
                unit_title,
                COUNT(*)                                              AS elements,
                SUM(CASE WHEN qa_passes THEN 1 ELSE 0 END)           AS qa_passed,
                SUM(CASE WHEN NOT qa_passes THEN 1 ELSE 0 END)       AS qa_failed,
                SUM(CASE WHEN error_message != '' AND error_message IS NOT NULL
                         THEN 1 ELSE 0 END)                           AS errors,
                ROUND(AVG(qa_word_count)::numeric, 1)                 AS avg_words,
                ROUND(AVG(rewrite_count)::numeric, 2)                 AS avg_rewrites,
                MAX(created_at)                                       AS last_updated
            FROM rsd_skill_records
            GROUP BY unit_code, unit_title
            ORDER BY unit_code
        """)).mappings().all()
    return pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()


def refresh():
    st.cache_data.clear()
    st.rerun()


# ── Refresh button ────────────────────────────────────────────────────────────
col_title, col_btn = st.columns([6, 1])
with col_btn:
    if st.button("🔄 Refresh", use_container_width=True):
        refresh()

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Overview")
stats = load_summary_stats()

if not stats or stats.get("total_skills") in (None, 0):
    st.info("No skill records in the database yet.")
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total runs",       stats.get("total_runs", 0))
    c2.metric("Skill statements", stats.get("total_skills", 0))
    c3.metric("Units",            stats.get("total_units", 0))
    c4.metric("QA passed",
              stats.get("qa_passed", 0),
              delta=f"{round(100 * int(stats.get('qa_passed',0)) / max(int(stats.get('total_skills',1)),1))}%")
    c5.metric("QA failed",   stats.get("qa_failed", 0))
    c6.metric("Errors",      stats.get("total_errors", 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg word count",  stats.get("avg_word_count", "-"))
    c2.metric("Avg rewrites",    stats.get("avg_rewrites", "-"))
    c3.metric("Avg temperature", stats.get("avg_temperature", "-"))

st.divider()

# ── Run history ───────────────────────────────────────────────────────────────
st.subheader("Run history")
runs_df = load_runs()

if runs_df.empty:
    st.info("No runs found.")
else:
    runs_df["run_id"] = runs_df["run_id"].astype(str)

    def _status_icon(s: str) -> str:
        return {"completed": "✅", "running": "🔄", "created": "⏳"}.get(str(s), "❓")

    display_runs = runs_df.copy()
    display_runs["status"] = display_runs["status"].apply(lambda s: f"{_status_icon(s)} {s}")
    display_runs["qa_rate"] = display_runs.apply(
        lambda r: f"{round(100 * int(r.get('qa_passed',0)) / max(int(r.get('total_records',1)),1))}%"
        if r.get("total_records") else "—", axis=1,
    )

    show_cols = [c for c in [
        "run_id", "created_at", "status", "source_filename",
        "model", "provider", "total_records", "qa_passed", "qa_rate", "errors",
    ] if c in display_runs.columns]

    st.dataframe(
        display_runs[show_cols],
        use_container_width=True,
        column_config={
            "run_id":        st.column_config.TextColumn("Run ID", width="small"),
            "created_at":    st.column_config.DatetimeColumn("Created", format="DD/MM/YY HH:mm"),
            "total_records": st.column_config.NumberColumn("Records"),
            "qa_passed":     st.column_config.NumberColumn("QA ✓"),
            "qa_rate":       st.column_config.TextColumn("QA rate"),
            "errors":        st.column_config.NumberColumn("Errors"),
        },
        hide_index=True,
    )

    with st.expander("🗑 Delete a run"):
        del_run_id = st.selectbox(
            "Select run to delete",
            runs_df["run_id"].tolist(),
            format_func=lambda r: f"{r[:8]}… — {runs_df[runs_df['run_id']==r]['source_filename'].iloc[0]}"
                if not runs_df.empty else r,
        )
        if st.button("Delete run (and all its records)", type="primary"):
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM rsd_runs WHERE id=:id"), {"id": del_run_id})
            st.success(f"Deleted run {del_run_id[:8]}…")
            refresh()

st.divider()

# ── Unit summary ──────────────────────────────────────────────────────────────
st.subheader("By unit / training package")
unit_df = load_unit_summary()

if not unit_df.empty:
    st.dataframe(
        unit_df,
        use_container_width=True,
        column_config={
            "unit_code":    st.column_config.TextColumn("Unit code"),
            "unit_title":   st.column_config.TextColumn("Unit title", width="large"),
            "elements":     st.column_config.NumberColumn("Elements"),
            "qa_passed":    st.column_config.NumberColumn("QA ✓"),
            "qa_failed":    st.column_config.NumberColumn("QA ✗"),
            "errors":       st.column_config.NumberColumn("Errors"),
            "avg_words":    st.column_config.NumberColumn("Avg words"),
            "avg_rewrites": st.column_config.NumberColumn("Avg rewrites"),
            "last_updated": st.column_config.DatetimeColumn("Last updated", format="DD/MM/YY HH:mm"),
        },
        hide_index=True,
    )
    st.caption("QA pass rate by unit")
    chart_df = unit_df[["unit_code", "qa_passed", "qa_failed"]].set_index("unit_code")
    st.bar_chart(chart_df)

st.divider()

# ── Skill statement browser ───────────────────────────────────────────────────
st.subheader("Skill statement browser")

f1, f2, f3 = st.columns(3)
with f1:
    run_ids = runs_df["run_id"].astype(str).tolist() if not runs_df.empty else []
    run_filter = st.selectbox(
        "Filter by run",
        ["All runs"] + run_ids,
        format_func=lambda r: "All runs" if r == "All runs"
            else f"{r[:8]}… — {runs_df[runs_df['run_id']==r]['source_filename'].iloc[0]}"
                 if not runs_df.empty else r,
    )
with f2:
    qa_filter = st.selectbox("QA filter", ["All", "Passed only", "Failed only", "Errors only"])
with f3:
    search = st.text_input("Search skill statements", placeholder="keyword…")

skills_df = load_skill_records(None if run_filter == "All runs" else run_filter)

if not skills_df.empty:
    if qa_filter == "Passed only":
        skills_df = skills_df[skills_df["qa_passes"] == True]
    elif qa_filter == "Failed only":
        skills_df = skills_df[skills_df["qa_passes"] == False]
    elif qa_filter == "Errors only":
        skills_df = skills_df[
            skills_df["error_message"].notna() & (skills_df["error_message"] != "")
        ]
    if search:
        skills_df = skills_df[
            skills_df["skill_statement"].str.contains(search, case=False, na=False)
        ]

    st.caption(f"Showing {len(skills_df)} records")

    edit_cols = [c for c in [
        "unit_code", "unit_title", "element_title",
        "skill_statement", "keywords",
        "qa_passes", "qa_word_count", "rewrite_count", "error_message",
    ] if c in skills_df.columns]

    edited = st.data_editor(
        skills_df[edit_cols].reset_index(drop=True),
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "unit_code":     st.column_config.TextColumn("Unit code", width="small"),
            "unit_title":    st.column_config.TextColumn("Unit title"),
            "element_title": st.column_config.TextColumn("Element"),
            "skill_statement": st.column_config.TextColumn("Skill statement", width="large"),
            "keywords":      st.column_config.TextColumn("Keywords"),
            "qa_passes":     st.column_config.CheckboxColumn("QA ✓"),
            "qa_word_count": st.column_config.NumberColumn("Words"),
            "rewrite_count": st.column_config.NumberColumn("Rewrites"),
            "error_message": st.column_config.TextColumn("Error"),
        },
        hide_index=True,
        key="skill_editor",
    )

    if st.button("💾 Save edits to DB", type="primary"):
        updated = 0
        with engine.begin() as conn:
            for i, row in edited.iterrows():
                original_row = skills_df.iloc[i]
                conn.execute(
                    text("""
                        UPDATE rsd_skill_records
                        SET skill_statement = :skill,
                            keywords        = :kw,
                            updated_at      = NOW()
                        WHERE run_id = :rid
                          AND id     = :recid
                    """),
                    {
                        "skill": row.get("skill_statement", ""),
                        "kw":    row.get("keywords", ""),
                        "rid":   str(original_row["run_id"]),
                        "recid": str(original_row["id"]),
                    },
                )
                updated += 1
        st.success(f"Saved {updated} records ✅")
        refresh()
else:
    st.info("No records match the current filters.")

st.divider()

# ── Data quality checks ───────────────────────────────────────────────────────
st.subheader("Data quality checks")
all_skills = load_skill_records()

if not all_skills.empty:
    issues = []

    empty_skills = all_skills[
        all_skills["skill_statement"].isna() |
        (all_skills["skill_statement"].str.strip() == "") |
        all_skills["skill_statement"].str.startswith("ERROR:")
    ]
    if len(empty_skills):
        issues.append(f"⚠️ **{len(empty_skills)}** skill statements are empty or errored")

    qa_fails = all_skills[all_skills["qa_passes"] == False]
    if len(qa_fails):
        issues.append(f"⚠️ **{len(qa_fails)}** statements failed QA checks")

    if "qa_word_count" in all_skills.columns:
        wc = all_skills["qa_word_count"].dropna()
        short = all_skills[(wc < 30).values]
        long_ = all_skills[(wc > 60).values]
        if len(short):
            issues.append(f"⚠️ **{len(short)}** statements are under 30 words")
        if len(long_):
            issues.append(f"⚠️ **{len(long_)}** statements are over 60 words")

    kw_col = "keywords" if "keywords" in all_skills.columns else None
    if kw_col:
        no_kw = all_skills[all_skills[kw_col].isna() | (all_skills[kw_col].str.strip() == "")]
        if len(no_kw):
            issues.append(f"ℹ️ **{len(no_kw)}** statements have no keywords")

    dupes = all_skills[all_skills.duplicated(subset=["skill_statement"], keep=False)]
    if len(dupes):
        issues.append(f"⚠️ **{len(dupes)}** duplicate skill statements detected")

    if issues:
        for issue in issues:
            st.markdown(issue)
    else:
        st.success("✅ No data quality issues found")

    if len(qa_fails) > 0:
        with st.expander(f"View {len(qa_fails)} QA failures"):
            show = [c for c in ["unit_code", "element_title", "skill_statement",
                                 "qa_word_count", "qa_has_method", "qa_has_outcome"]
                    if c in qa_fails.columns]
            st.dataframe(qa_fails[show], use_container_width=True, hide_index=True)

    if len(dupes) > 0:
        with st.expander(f"View {len(dupes)} duplicates"):
            show = [c for c in ["unit_code", "element_title", "skill_statement"]
                    if c in dupes.columns]
            st.dataframe(dupes[show], use_container_width=True, hide_index=True)
else:
    st.info("No records to check yet.")

st.divider()

# ── Keyword analysis ──────────────────────────────────────────────────────────
st.subheader("Keyword analysis")
all_skills = load_skill_records()
kw_col = "keywords" if "keywords" in all_skills.columns else None

if not all_skills.empty and kw_col:
    all_kws = (
        all_skills[kw_col]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
        .dropna()
    )
    if len(all_kws):
        kw_counts = all_kws.value_counts().reset_index()
        kw_counts.columns = ["Keyword", "Count"]
        col1, col2 = st.columns([2, 3])
        with col1:
            st.caption(f"{len(kw_counts)} unique keywords across {len(all_skills)} statements")
            st.dataframe(kw_counts.head(30), use_container_width=True, hide_index=True)
        with col2:
            st.caption("Top 20 keywords")
            st.bar_chart(kw_counts.head(20).set_index("Keyword"))
    else:
        st.info("No keywords found.")
else:
    st.info("No keyword data available yet.")

st.divider()

# ── Export all DB records ─────────────────────────────────────────────────────
st.subheader("Export all DB records")
all_skills = load_skill_records()

if not all_skills.empty:
    from core.exporters import to_rsd_rows, to_traceability, to_osmt_rows
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇ All records — OSMT CSV",
        to_osmt_rows(all_skills).to_csv(index=False).encode(),
        "all_osmt.csv", "text/csv",
    )
    c2.download_button(
        "⬇ All records — RSD CSV",
        to_rsd_rows(all_skills).to_csv(index=False).encode(),
        "all_rsd.csv", "text/csv",
    )
    c3.download_button(
        "⬇ All records — Traceability CSV",
        to_traceability(all_skills).to_csv(index=False).encode(),
        "all_traceability.csv", "text/csv",
    )
else:
    st.info("No records to export yet.")
