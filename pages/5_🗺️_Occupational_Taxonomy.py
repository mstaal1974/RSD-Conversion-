"""
pages/5_🗺️_Occupational_Taxonomy.py

UOC → ANZSCO/ASCED Occupational Taxonomy Pipeline

Tabs:
  1. Dashboard — coverage metrics and ANZSCO distribution
  2. Pipeline  — seed, TGA ingest, linkage engine, ASC upload
  3. Browse    — enriched skill statements with occupational context
  4. VC Record — W3C Verifiable Credential aligned JSON
  5. Export    — CSV/OSMT with ANZSCO fields
"""
from __future__ import annotations
import json
import os
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
st.set_page_config(page_title="Occupational Taxonomy", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .tax-card {
    background: linear-gradient(135deg, #0c1445 0%, #1a237e 100%);
    border: 1px solid #283593;
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
  }
  .tax-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #82b1ff;
    line-height: 1;
  }
  .tax-card .lbl {
    font-size: 0.7rem;
    color: #7986cb;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
  }
  .tax-card .sub {
    font-size: 0.8rem;
    color: #9fa8da;
    margin-top: 4px;
  }
  .conf-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
  }
  .conf-high  { background:#1b5e20; color:#a5d6a7; }
  .conf-med   { background:#0d47a1; color:#90caf9; }
  .conf-low   { background:#bf360c; color:#ffcc80; }
  .conf-vlow  { background:#212121; color:#bdbdbd; }
  .section-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #37474f;
    border-bottom: 1px solid #1a237e;
    padding-bottom: 6px; margin: 28px 0 16px 0;
  }
  .pipeline-step {
    background: #0d1b2a;
    border-left: 3px solid #3f51b5;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.87rem;
    color: #b0bec5;
  }
  .pipeline-step .step-num {
    font-family: 'IBM Plex Mono', monospace;
    color: #82b1ff;
    font-weight: 600;
    margin-right: 8px;
  }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ Occupational Taxonomy")
st.caption("Link UOCs to ANZSCO/ASCED occupations via TGA packaging rules and the Australian Skills Classification")

# ── Connections ───────────────────────────────────────────────────────────────
def _secret(k, d=""):
    try:
        return st.secrets.get(k, os.getenv(k, d)) or d
    except Exception:
        return os.getenv(k, d) or d

DB_URL   = _secret("DATABASE_URL")
TGA_USER = _secret("TGA_USERNAME")
TGA_PASS = _secret("TGA_PASSWORD")

if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

# ── Taxonomy module check ─────────────────────────────────────────────────────
try:
    from core.db import init_taxonomy_db, init_refinements
    init_taxonomy_db(engine)
    init_refinements(engine)
except Exception as _e:
    st.error(
        f"Taxonomy module not ready: {_e}. "
        "Ensure core/linkage_engine.py, core/rsd_record.py and "
        "core/tga_ingestor.py are committed to GitHub."
    )
    st.stop()

# ── Tables existence check — must happen before any tab tries to query ─────────
try:
    from core.linkage_engine import LinkageEngine
except ImportError as _e:
    st.error(f"core/linkage_engine.py missing from repo: {_e}")
    st.stop()

try:
    with engine.connect() as _conn:
        _row = _conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'uoc_occupation_links'
        """)).fetchone()
    _tables_ready = bool(_row and _row[0] > 0)
except Exception:
    _tables_ready = False

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "⚙️ Pipeline",
    "🔎 Browse",
    "📜 VC Record",
    "⬇ Export",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not _tables_ready:
        st.warning(
            "⚠️ Taxonomy tables do not exist yet. "
            "Run `migrations/001_create_taxonomy_tables.sql` against your database "
            "in Cloud SQL Studio, then refresh this page."
        )
        st.stop()

    @st.cache_data(ttl=60, show_spinner=False)
    def get_coverage():
        return LinkageEngine.coverage_stats(engine)

    stats = get_coverage()

    st.markdown('<div class="section-hdr">Coverage overview</div>',
                unsafe_allow_html=True)

    if not stats or not stats.get("total_uocs"):
        st.info("No taxonomy data yet. Run the pipeline on the ⚙️ Pipeline tab.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)

        def card(col, val, lbl, sub=""):
            with col:
                st.markdown(
                    f'<div class="tax-card"><div class="val">{val}</div>'
                    f'<div class="lbl">{lbl}</div>'
                    f'<div class="sub">{sub}</div></div>',
                    unsafe_allow_html=True)

        total   = int(stats.get("total_uocs") or 0)
        linked  = int(stats.get("linked_uocs") or 0)
        hconf   = int(stats.get("high_conf_uocs") or 0)
        uanzsco = int(stats.get("unique_anzsco") or 0)
        majors  = int(stats.get("major_groups") or 0)
        avgc    = float(stats.get("avg_confidence") or 0)

        card(c1, f"{total:,}",  "Total UOCs",      "in DB")
        card(c2, f"{linked:,}", "Linked to ANZSCO",
             f"{round(100*linked/max(total,1))}% coverage")
        card(c3, f"{hconf:,}",  "High confidence",  "conf ≥ 0.70")
        card(c4, f"{uanzsco}",  "ANZSCO codes",     f"{majors} major groups")
        card(c5, f"{avgc:.2f}", "Avg confidence",   "0=low, 1=direct")

        st.divider()

        # ── ANZSCO distribution ───────────────────────────────────────────────
        st.markdown('<div class="section-hdr">ANZSCO major group distribution</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=60)
        def load_anzsco_dist():
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT o.anzsco_major_group,
                               COUNT(DISTINCT o.uoc_code) AS uoc_count,
                               ROUND(AVG(o.confidence)::numeric, 3) AS avg_conf
                        FROM uoc_occupation_links o
                        WHERE o.is_primary = TRUE AND o.valid_to IS NULL
                          AND o.anzsco_major_group != ''
                        GROUP BY o.anzsco_major_group
                        ORDER BY uoc_count DESC
                    """)).fetchall()
                return pd.DataFrame(rows,
                    columns=["Major Group", "UOC Count", "Avg Confidence"])
            except Exception:
                return pd.DataFrame()

        anzsco_df = load_anzsco_dist()
        if not anzsco_df.empty:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                chart = alt.Chart(anzsco_df).mark_bar(
                    cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                ).encode(
                    y=alt.Y("Major Group:N", sort="-x", title=""),
                    x=alt.X("UOC Count:Q", title="UOCs"),
                    color=alt.Color("Avg Confidence:Q",
                        scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=alt.Legend(title="Avg conf")),
                    tooltip=["Major Group", "UOC Count",
                             alt.Tooltip("Avg Confidence:Q", format=".3f")],
                ).properties(height=300, title="UOCs per ANZSCO major group")
                st.altair_chart(chart, use_container_width=True)
            with col_b:
                st.dataframe(anzsco_df, use_container_width=True, hide_index=True)

        # ── Source breakdown ──────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Mapping source breakdown</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=60)
        def load_source_dist():
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT mapping_source,
                               COUNT(DISTINCT uoc_code) AS uocs,
                               ROUND(AVG(confidence)::numeric, 3) AS avg_conf
                        FROM uoc_occupation_links
                        WHERE is_primary = TRUE AND valid_to IS NULL
                        GROUP BY mapping_source ORDER BY avg_conf DESC
                    """)).fetchall()
                return pd.DataFrame(rows, columns=["Source", "UOCs", "Avg Confidence"])
            except Exception:
                return pd.DataFrame()

        src_df = load_source_dist()
        if not src_df.empty:
            src_chart = alt.Chart(src_df).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                color="#3f51b5",
            ).encode(
                y=alt.Y("Source:N", sort="-x", title=""),
                x=alt.X("UOCs:Q"),
                tooltip=["Source", "UOCs",
                         alt.Tooltip("Avg Confidence:Q", format=".3f")],
            ).properties(height=220, title="UOCs mapped by source (priority)")
            st.altair_chart(src_chart, use_container_width=True)

            confidence_key = {
                "direct_uoc_classification": ("1.00", "Direct on UOC"),
                "core_native":               ("0.85", "Core — same TP"),
                "core_imported":             ("0.70", "Core — imported"),
                "elective_native":           ("0.50", "Elective — same TP"),
                "elective_imported":         ("0.40", "Elective — imported"),
                "asc_specialist_task":       ("0.30", "ASC keyword match"),
            }
            st.caption(" · ".join(
                f"**{src}** = {conf} ({desc})"
                for src, (conf, desc) in confidence_key.items()
            ))

        # ── TP coverage table ─────────────────────────────────────────────────
        st.markdown('<div class="section-hdr">Coverage by training package</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=60)
        def load_tp_coverage():
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT s.tp_code,
                               COUNT(DISTINCT s.unit_code)  AS total_uocs,
                               COUNT(DISTINCT o.uoc_code)   AS linked_uocs,
                               ROUND(AVG(o.confidence)::numeric, 3) AS avg_conf,
                               COUNT(DISTINCT o.anzsco_code) AS anzsco_count
                        FROM rsd_skill_records s
                        LEFT JOIN uoc_occupation_links o
                            ON o.uoc_code = s.unit_code
                            AND o.is_primary = TRUE AND o.valid_to IS NULL
                        WHERE s.tp_code IS NOT NULL AND s.tp_code != ''
                        GROUP BY s.tp_code ORDER BY total_uocs DESC
                    """)).fetchall()
                df = pd.DataFrame(rows, columns=[
                    "TP", "Total UOCs", "Linked", "Avg Conf", "ANZSCO Count"])
                df["Coverage %"] = (
                    100 * df["Linked"] / df["Total UOCs"].clip(lower=1)
                ).round(1).astype(str) + "%"
                return df
            except Exception:
                return pd.DataFrame()

        tp_df = load_tp_coverage()
        if not tp_df.empty:
            st.dataframe(
                tp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "TP":           st.column_config.TextColumn("TP", width="small"),
                    "Total UOCs":   st.column_config.NumberColumn("UOCs"),
                    "Linked":       st.column_config.NumberColumn("Linked"),
                    "Avg Conf":     st.column_config.NumberColumn("Avg Conf", format="%.3f"),
                    "ANZSCO Count": st.column_config.NumberColumn("ANZSCO codes"),
                    "Coverage %":   st.column_config.TextColumn("Coverage"),
                },
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if not _tables_ready:
        st.warning(
            "⚠️ Taxonomy tables do not exist yet. "
            "Run `migrations/001_create_taxonomy_tables.sql` in Cloud SQL Studio first."
        )
        st.stop()

    st.markdown('<div class="section-hdr">Pipeline steps</div>',
                unsafe_allow_html=True)

    for step, label, desc in [
        ("1", "Seed from DB",
         "Bootstrap uoc_registry from your existing rsd_skill_records — instant, no API needed."),
        ("2", "TGA Ingestion (optional)",
         "Fetch ContentBundle from TGA SOAP API to get packaging rules and ANZSCO classifications."),
        ("3", "Run Linkage Engine",
         "Apply 4-priority algorithm: Direct → Core (native/imported) → Elective → ASC keyword match."),
        ("4", "ASC Upload (optional)",
         "Upload the JSA Australian Skills Classification CSV to enable Priority 4 matching."),
    ]:
        st.markdown(
            f'<div class="pipeline-step">'
            f'<span class="step-num">Step {step}:</span>'
            f'<strong>{label}</strong> — {desc}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Step 1: Seed ──────────────────────────────────────────────────────────
    st.subheader("Step 1 — Seed UOC registry from DB")
    st.caption("Reads all unique unit_codes from rsd_skill_records and registers them.")

    if st.button("▶ Seed from rsd_skill_records", type="primary"):
        from core.db import start_pipeline_run, finish_pipeline_run
        from tga_ingestor import TGAIngestor

        rid = start_pipeline_run(engine, run_type="seed")
        ingestor = TGAIngestor(engine, TGA_USER, TGA_PASS, pipeline_run_id=rid)
        with st.spinner("Seeding…"):
            n = ingestor.seed_from_rsd_records()
        finish_pipeline_run(engine, rid, "success", uocs=n)
        st.success(f"✅ Seeded **{n}** UOCs into uoc_registry")
        st.cache_data.clear()

    st.divider()

    # ── Step 2: TGA ingestion ─────────────────────────────────────────────────
    st.subheader("Step 2 — TGA ingestion (optional)")

    tga_col1, tga_col2 = st.columns(2)
    with tga_col1:
        tga_user_input = st.text_input("TGA username", value=TGA_USER or "")
        tga_pass_input = st.text_input("TGA password", value=TGA_PASS or "",
                                       type="password")
    with tga_col2:
        tp_filter_input = st.text_input(
            "Training packages (comma-separated, blank = all)",
            placeholder="MSL,BSB,RII",
        )
        use_soap = st.checkbox("Use SOAP API", value=True,
            help="Uncheck to use REST-only mode (no packaging rules)")

    st.caption(
        "TGA credentials are stored in Cloud Run secrets as TGA_USERNAME and TGA_PASSWORD. "
        "Contact NTRReform@dewr.gov.au to request production API access."
    )

    if st.button("▶ Run TGA ingestion"):
        from core.db import start_pipeline_run, finish_pipeline_run
        from core.tga_ingestor import TGAIngestor

        tp_list = ([t.strip().upper() for t in tp_filter_input.split(",") if t.strip()]
                   if tp_filter_input.strip() else None)

        rid = start_pipeline_run(engine, run_type="tga_ingest",
                                 tp_scope=",".join(tp_list) if tp_list else "all")
        ingestor = TGAIngestor(engine, tga_user_input, tga_pass_input,
                               pipeline_run_id=rid)
        progress = st.progress(0)
        status_t = st.empty()

        def cb(pct, msg):
            progress.progress(min(pct, 1.0))
            status_t.write(msg)

        try:
            counts = ingestor.run(tp_codes=tp_list, use_soap=use_soap,
                                  progress_callback=cb)
            finish_pipeline_run(engine, rid, "success",
                                quals=counts.get("quals", 0),
                                uocs=counts.get("uocs", 0))
            st.success(
                f"✅ Ingested **{counts.get('quals',0)}** qualifications, "
                f"**{counts.get('uocs',0)}** UOCs"
            )
        except Exception as e:
            finish_pipeline_run(engine, rid, "failed", error=str(e))
            st.error(f"Ingestion failed: {e}")
        finally:
            st.cache_data.clear()

    st.divider()

    # ── Step 3: Linkage engine ────────────────────────────────────────────────
    st.subheader("Step 3 — Run linkage engine")

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        link_tp = st.text_input(
            "Limit to training packages (blank = all)",
            placeholder="MSL,BSB", key="link_tp",
        )
        run_asc_match = st.checkbox(
            "Enable Priority 4 (ASC keyword matching)", value=True,
            help="Requires ASC data loaded in Step 4. Skipped if no data found."
        )
    with link_col2:
        st.markdown("**Confidence thresholds:**")
        st.markdown("""
| Priority | Source | Confidence |
|---|---|---|
| P1 | Direct UOC classification | **1.00** |
| P2 | Core — native TP | **0.85** |
| P2 | Core — imported | **0.70** |
| P3 | Elective — native | **0.50** |
| P3 | Elective — imported | **0.40** |
| P4 | ASC keyword match | **0.30** |
        """)

    if st.button("▶ Run linkage engine", type="primary"):
        from core.db import start_pipeline_run, finish_pipeline_run

        uoc_filter = None
        if link_tp.strip():
            tp_list = [t.strip().upper() for t in link_tp.split(",") if t.strip()]
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT uoc_code FROM uoc_registry
                    WHERE tp_code = ANY(:tps)
                    AND usage_recommendation = 'Current'
                """), {"tps": tp_list}).fetchall()
            uoc_filter = [r[0] for r in rows]
            if not uoc_filter:
                st.warning(f"No UOCs found for TPs: {tp_list}. Run Step 1 first.")
                st.stop()

        rid = start_pipeline_run(engine, run_type="linkage",
                                 tp_scope=link_tp or "all")
        engine_obj = LinkageEngine(engine, rid)
        progress = st.progress(0)
        status_t = st.empty()

        def cb2(pct, msg):
            progress.progress(min(pct, 1.0))
            status_t.write(msg)

        try:
            counts = engine_obj.run(uoc_codes=uoc_filter,
                                    run_asc=run_asc_match,
                                    progress_callback=cb2)
            finish_pipeline_run(engine, rid, "success",
                                links_created=counts.get("links_created", 0),
                                links_updated=counts.get("links_updated", 0))
            st.success(f"✅ Created **{counts.get('links_created',0)}** occupation links")
        except Exception as e:
            finish_pipeline_run(engine, rid, "failed", error=str(e))
            st.error(f"Linkage failed: {e}")
        finally:
            st.cache_data.clear()

    st.divider()

    # ── Step 4: ASC upload ────────────────────────────────────────────────────
    st.subheader("Step 4 — Load ASC Specialist Tasks (optional)")
    st.caption(
        "Download from Jobs and Skills Australia: jobsandskills.gov.au → "
        "Data and research → Australian Skills Classification. "
        "Expected columns: task_id, task_description, anzsco_code, anzsco_title"
    )

    asc_file = st.file_uploader("Upload asc_specialist_tasks.csv", type=["csv"])
    if asc_file:
        try:
            asc_df = pd.read_csv(asc_file)
            st.dataframe(asc_df.head(5), use_container_width=True, hide_index=True)

            required = {"task_id", "task_description", "anzsco_code", "anzsco_title"}
            if not required.issubset(set(asc_df.columns)):
                st.error(f"Missing columns: {required - set(asc_df.columns)}")
            else:
                if st.button("💾 Load ASC data into DB"):
                    with engine.begin() as conn:
                        for _, row in asc_df.iterrows():
                            conn.execute(text("""
                                INSERT INTO asc_specialist_tasks
                                    (task_id, task_description, anzsco_code, anzsco_title)
                                VALUES (:tid, :tdesc, :ac, :at)
                                ON CONFLICT (task_id) DO UPDATE SET
                                    task_description = EXCLUDED.task_description
                            """), {
                                "tid":   str(row.get("task_id", "")),
                                "tdesc": str(row.get("task_description", "")),
                                "ac":    str(row.get("anzsco_code", "")),
                                "at":    str(row.get("anzsco_title", "")),
                            })
                    st.success(f"✅ Loaded {len(asc_df)} ASC tasks")
        except Exception as e:
            st.error(f"Error reading ASC file: {e}")

    st.divider()

    # ── Pipeline run history ──────────────────────────────────────────────────
    st.subheader("Pipeline run history")

    @st.cache_data(ttl=30)
    def load_pipeline_runs():
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT id, run_type, tp_scope, status,
                           started_at, completed_at,
                           quals_processed, uocs_processed,
                           links_created, error_message
                    FROM pipeline_runs
                    ORDER BY started_at DESC LIMIT 20
                """)).mappings().all()
            return pd.DataFrame([dict(r) for r in rows])
        except Exception:
            return pd.DataFrame()

    runs_df = load_pipeline_runs()
    if not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True, hide_index=True,
            column_config={
                "id":            st.column_config.NumberColumn("Run ID", width="small"),
                "run_type":      st.column_config.TextColumn("Type"),
                "tp_scope":      st.column_config.TextColumn("Scope"),
                "status":        st.column_config.TextColumn("Status"),
                "started_at":    st.column_config.DatetimeColumn("Started"),
                "links_created": st.column_config.NumberColumn("Links created"),
                "error_message": st.column_config.TextColumn("Error", width="large"),
            })
    else:
        st.info("No pipeline runs yet.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BROWSE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not _tables_ready:
        st.warning("⚠️ Taxonomy tables not found. Run the migration first.")
        st.stop()

    st.markdown('<div class="section-hdr">Enriched skill statements</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        unit_filter = st.text_input("Filter by unit code", placeholder="MSL975058")
    with f2:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    with f3:
        limit_n = st.selectbox("Max rows", [100, 500, 1000, 5000], index=1)

    @st.cache_data(ttl=30)
    def load_linked(uc, mc, lim):
        return LinkageEngine.get_linked_records(
            engine, unit_code=uc or None,
            min_confidence=mc, limit=lim,
        )

    df_linked = load_linked(unit_filter, min_conf, limit_n)

    if df_linked.empty:
        st.info("No linked records found. Run the pipeline first.")
    else:
        linked_count = df_linked["anzsco_code"].notna().sum()
        st.info(
            f"**{len(df_linked)}** records — "
            f"**{linked_count}** ({round(100*linked_count/max(len(df_linked),1))}%) "
            f"have ANZSCO occupational links"
        )

        def conf_badge(v) -> str:
            if pd.isna(v):
                return "—"
            v = float(v)
            if v >= 0.85:
                return f'<span class="conf-badge conf-high">{v:.2f}</span>'
            elif v >= 0.70:
                return f'<span class="conf-badge conf-med">{v:.2f}</span>'
            elif v >= 0.40:
                return f'<span class="conf-badge conf-low">{v:.2f}</span>'
            return f'<span class="conf-badge conf-vlow">{v:.2f}</span>'

        for _, row in df_linked.head(50).iterrows():
            anzsco   = row.get("anzsco_code") or ""
            anzsco_t = row.get("anzsco_title") or ""
            conf     = row.get("confidence")
            src      = row.get("mapping_source") or "—"
            unit     = row.get("unit_code") or ""
            stmt     = row.get("skill_statement") or ""
            major    = row.get("anzsco_major_group") or ""
            ind      = row.get("industry_sector") or ""
            badge    = conf_badge(conf) if anzsco else "—"
            footer   = (f"<div style='font-size:0.72rem;color:#37474f;margin-top:4px'>"
                        f"{major}{' · ' + ind if ind else ''}</div>") if major else ""

            st.markdown(
                f"<div style='background:#0d1b2a;border:1px solid #1a237e;"
                f"border-radius:8px;padding:12px 16px;margin-bottom:8px'>"
                f"<div style='display:flex;gap:12px;align-items:center;"
                f"flex-wrap:wrap;margin-bottom:8px'>"
                f"<code style='color:#82b1ff;font-size:0.75rem'>{unit}</code>"
                f"<span style='color:#5c6bc0;font-size:0.75rem'>{anzsco} {anzsco_t}</span>"
                f"{badge}"
                f"<span style='color:#37474f;font-size:0.72rem'>{src}</span>"
                f"</div>"
                f"<div style='font-size:0.87rem;color:#cfd8dc'>{stmt}</div>"
                f"{footer}</div>",
                unsafe_allow_html=True,
            )

        if len(df_linked) > 50:
            st.caption(f"Showing 50 of {len(df_linked)} — use Export tab for full set")

        with st.expander("Full data table"):
            st.dataframe(
                df_linked,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "unit_code":       st.column_config.TextColumn("Unit", width="small"),
                    "unit_title":      st.column_config.TextColumn("Unit title"),
                    "element_title":   st.column_config.TextColumn("Element"),
                    "skill_statement": st.column_config.TextColumn("Skill statement", width="large"),
                    "anzsco_code":     st.column_config.TextColumn("ANZSCO", width="small"),
                    "anzsco_title":    st.column_config.TextColumn("ANZSCO title"),
                    "confidence":      st.column_config.NumberColumn("Conf", format="%.3f"),
                    "mapping_source":  st.column_config.TextColumn("Source"),
                },
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VC RECORD
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if not _tables_ready:
        st.warning("⚠️ Taxonomy tables not found. Run the migration first.")
        st.stop()

    from core.rsd_record import (
        build_uoc_record, records_to_jsonl,
        anzsco_uri, aqf_to_skill_label,
    )

    st.markdown(
        '<div class="section-hdr">W3C Verifiable Credential — Fully linked UOC record</div>',
        unsafe_allow_html=True)
    st.caption(
        "Builds the canonical JSON record for any UOC, aligned to "
        "W3C Verifiable Credentials v1.1, Open Skills Network RSD, "
        "and Blocksure.com.au evidence standards."
    )

    col_vc1, col_vc2 = st.columns([2, 3])

    with col_vc1:
        @st.cache_data(ttl=60)
        def load_uoc_list():
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT DISTINCT s.unit_code, s.unit_title
                    FROM rsd_skill_records s
                    WHERE s.unit_code IS NOT NULL AND s.unit_code != ''
                    ORDER BY s.unit_code LIMIT 500
                """)).fetchall()
            return [(r[0], r[1] or r[0]) for r in rows]

        uoc_list = load_uoc_list()
        if not uoc_list:
            st.info("No UOC data found.")
        else:
            uoc_options = {f"{code} — {title[:50]}": code for code, title in uoc_list}
            sel_uoc_label = st.selectbox("Select UOC", list(uoc_options.keys()),
                                         key="vc_uoc_sel")
            sel_uoc_code  = uoc_options[sel_uoc_label]
            include_all   = st.checkbox("Include all occupation links", value=False,
                help="Show every occupation link, not just primary")

            if st.button("▶ Build VC Record", type="primary", key="vc_build"):
                with st.spinner("Building record…"):
                    record = build_uoc_record(engine, sel_uoc_code,
                                              include_all_links=include_all)
                st.session_state["vc_record"] = record
                st.session_state["vc_uoc"]    = sel_uoc_code

        st.divider()

        # Evidence management
        st.subheader("Evidence linking (Ref 3)")
        ev_unit  = st.text_input("Unit code", placeholder="MSL975058", key="ev_unit")
        ev_elem  = st.text_input("Element title (partial match)",
                                 placeholder="Process samples", key="ev_elem")
        ev_hash  = st.text_input("Evidence hash",
                                 placeholder="Qm... or blocksure:abc123", key="ev_hash")
        ev_uri   = st.text_input("Evidence URI",
                                 placeholder="https://blocksure.com.au/cert/abc123",
                                 key="ev_uri")
        ev_type  = st.selectbox("Evidence type", [
            "Simulation Performance Data",
            "Workplace Assessment",
            "Portfolio Evidence",
            "Third-Party Verification",
            "Recognition of Prior Learning",
        ], key="ev_type")

        if st.button("💾 Save evidence link", key="ev_save"):
            if ev_unit and ev_hash:
                with engine.begin() as conn:
                    filter_sql = "AND element_title ILIKE :elem" if ev_elem else ""
                    conn.execute(text(f"""
                        UPDATE rsd_skill_records
                        SET evidence_hash = :hash,
                            evidence_uri  = :uri,
                            evidence_type = :etype
                        WHERE unit_code = :uc {filter_sql}
                    """), {
                        "hash":  ev_hash,
                        "uri":   ev_uri or "",
                        "etype": ev_type,
                        "uc":    ev_unit,
                        "elem":  f"%{ev_elem}%" if ev_elem else None,
                    })
                st.success(f"✅ Evidence linked to {ev_unit}")
            else:
                st.warning("Unit code and hash are required.")

        st.divider()
        st.subheader("Assign element IDs")
        if st.button("▶ Generate element IDs for all records", key="gen_eids"):
            from core.rsd_record import build_element_id
            from collections import defaultdict

            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT id, unit_code, element_title, row_index
                    FROM rsd_skill_records
                    WHERE element_id IS NULL OR element_id = ''
                    ORDER BY unit_code, row_index
                """)).fetchall()

            unit_idx: dict = defaultdict(int)
            with engine.begin() as conn:
                for row in rows:
                    rec_id, uc, elem, _ = row
                    eid = build_element_id(uc or "", elem or "", unit_idx[uc])
                    unit_idx[uc] += 1
                    conn.execute(text(
                        "UPDATE rsd_skill_records SET element_id=:eid WHERE id=:rid"
                    ), {"eid": eid, "rid": str(rec_id)})
            st.success(f"✅ Generated {len(rows)} element IDs")

    with col_vc2:
        record = st.session_state.get("vc_record")
        if not record:
            st.info("Select a UOC and click **▶ Build VC Record** to generate the JSON.")
        else:
            uoc_c = st.session_state.get("vc_uoc", "")
            uh    = record.get("unit_header", {})
            imp   = record.get("import_status", {})
            pocc  = record.get("primary_occupation", {})
            ta    = pocc.get("taxonomic_alignment", {})

            s1, s2, s3, s4 = st.columns(4)
            def mini_card(col, val, lbl):
                with col:
                    st.markdown(
                        f'<div class="tax-card" style="padding:14px">'
                        f'<div class="val" style="font-size:1.4rem">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

            mini_card(s1, uh.get("aqf_level", "—")[:15], "AQF Level")
            mini_card(s2, uh.get("skill_level_label", "—")[:20], "Skill Level")
            mini_card(s3, "✓" if not imp.get("is_imported") else "↗ IMPORTED", "Origin")
            mini_card(s4, f"{ta.get('confidence', 0):.2f}", "Conf")

            st.markdown("")

            if ta:
                st.markdown("**Taxonomic alignment (W3C VC)**")
                ta_df = pd.DataFrame([{
                    "Field": k,
                    "Value": str(v) if not isinstance(v, dict) else json.dumps(v),
                } for k, v in ta.items()])
                st.dataframe(ta_df, use_container_width=True, hide_index=True)
                if ta.get("targetUrl"):
                    st.caption(f"🔗 ABS canonical URI: {ta['targetUrl']}")

            st.divider()

            stmts = record.get("rsd_skill_statements", [])
            st.markdown(f"**{len(stmts)} skill statements**")
            for stmt in stmts[:5]:
                ev    = stmt.get("evidence_requirements", {})
                qa    = stmt.get("qa_status", "")
                eid   = stmt.get("element_id", "")
                badge_col = "#1b5e20" if ev.get("verified") else "#bf360c"
                badge_lbl = "✓ Verified" if ev.get("verified") else "⚠ Placeholder"

                st.markdown(
                    f'<div style="background:#0d1b2a;border:1px solid #1a237e;'
                    f'border-radius:8px;padding:12px;margin-bottom:8px">'
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px">'
                    f'<code style="color:#82b1ff;font-size:0.72rem">{eid}</code>'
                    f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
                    f'background:{badge_col};color:white;padding:2px 6px;'
                    f'border-radius:3px">{badge_lbl}</span>'
                    f'<span style="font-size:0.7rem;color:#546e7a">{qa}</span>'
                    f'</div>'
                    f'<div style="font-size:0.87rem;color:#cfd8dc;margin-bottom:6px">'
                    f'{stmt.get("skill_statement","")}</div>'
                    f'<div style="font-size:0.7rem;color:#37474f">'
                    f'Evidence: {ev.get("type","")} · Hash: {str(ev.get("hash",""))[:40]}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            if len(stmts) > 5:
                st.caption(f"… and {len(stmts)-5} more. Download JSON for full record.")

            st.divider()
            with st.expander("📄 Raw JSON record"):
                st.json(record)

            json_str = json.dumps(record, indent=2, ensure_ascii=False, default=str)
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "⬇ Download VC JSON", json_str.encode(),
                f"{uoc_c}_vc_record.json", "application/json",
                use_container_width=True,
            )

            tp_for_batch = record.get("tp_code", "")
            if tp_for_batch and st.button(
                    f"⬇ Build JSONL batch for all {tp_for_batch} UOCs",
                    key="vc_batch"):
                with engine.connect() as conn:
                    batch_codes = [r[0] for r in conn.execute(text("""
                        SELECT DISTINCT unit_code FROM rsd_skill_records
                        WHERE unit_code LIKE :prefix ORDER BY unit_code
                    """), {"prefix": f"{tp_for_batch}%"}).fetchall()]
                with st.spinner(f"Building {len(batch_codes)} records…"):
                    jsonl = records_to_jsonl(engine, batch_codes)
                dl2.download_button(
                    f"⬇ {tp_for_batch} JSONL ({len(batch_codes)} UOCs)",
                    jsonl.encode(), f"{tp_for_batch}_vc_batch.jsonl",
                    "application/x-ndjson", use_container_width=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    if not _tables_ready:
        st.warning("⚠️ Taxonomy tables not found. Run the migration first.")
        st.stop()

    st.markdown('<div class="section-hdr">Export enriched skill statements</div>',
                unsafe_allow_html=True)

    @st.cache_data(ttl=60)
    def load_full_export():
        return LinkageEngine.get_linked_records(engine, limit=100000)

    export_df = load_full_export()

    if export_df.empty:
        st.info("No linked records to export. Run the pipeline first.")
    else:
        st.info(f"**{len(export_df):,}** enriched skill statements ready for export")

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "⬇ Full enriched CSV",
            export_df.to_csv(index=False).encode(),
            "rsd_enriched_occupational.csv", "text/csv",
            use_container_width=True,
        )

        osmt_cols = {
            "unit_code":          "Standards",
            "unit_title":         "Category",
            "element_title":      "Sub-Category",
            "skill_statement":    "Skill Statement",
            "anzsco_code":        "ANZSCO Code",
            "anzsco_title":       "Occupation",
            "anzsco_major_group": "ANZSCO Major Group",
            "industry_sector":    "Industry Sector",
            "confidence":         "Taxonomy Confidence",
            "mapping_source":     "Mapping Source",
        }
        available = {k: v for k, v in osmt_cols.items() if k in export_df.columns}
        osmt_df   = export_df[list(available.keys())].rename(columns=available)

        c2.download_button(
            "⬇ OSMT + ANZSCO CSV",
            osmt_df.to_csv(index=False).encode(),
            "osmt_with_anzsco.csv", "text/csv",
            use_container_width=True,
        )

        summary = (
            export_df[
                export_df["anzsco_code"].notna() &
                (export_df["anzsco_code"] != "")
            ]
            .groupby(["anzsco_code", "anzsco_title", "anzsco_major_group"])
            .agg(
                uoc_count=("unit_code", "nunique"),
                stmt_count=("skill_statement", "count"),
                avg_confidence=("confidence", "mean"),
                training_packages=(
                    "unit_code",
                    lambda x: "; ".join(sorted(set(c[:3] for c in x if c)))),
            )
            .reset_index()
        )
        summary["avg_confidence"] = pd.to_numeric(summary["avg_confidence"], errors="coerce").round(3)
        summary = summary.sort_values("stmt_count", ascending=False)

        c3.download_button(
            "⬇ Occupation summary CSV",
            summary.to_csv(index=False).encode(),
            "occupation_summary.csv", "text/csv",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Occupation summary")
        if not summary.empty:
            st.dataframe(
                summary.head(30),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "anzsco_code":       st.column_config.TextColumn("ANZSCO"),
                    "anzsco_title":      st.column_config.TextColumn("Occupation"),
                    "anzsco_major_group":st.column_config.TextColumn("Major group"),
                    "uoc_count":         st.column_config.NumberColumn("UOCs"),
                    "stmt_count":        st.column_config.NumberColumn("Statements"),
                    "avg_confidence":    st.column_config.NumberColumn("Avg conf", format="%.3f"),
                    "training_packages": st.column_config.TextColumn("TPs"),
                },
            )

            top_occs  = summary.head(20)
            occ_chart = alt.Chart(top_occs).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
            ).encode(
                y=alt.Y("anzsco_title:N", sort="-x", title=""),
                x=alt.X("stmt_count:Q", title="Skill statements"),
                color=alt.Color("avg_confidence:Q",
                    scale=alt.Scale(scheme="blues", domain=[0, 1]),
                    legend=alt.Legend(title="Avg conf")),
                tooltip=[
                    alt.Tooltip("anzsco_code:N",   title="ANZSCO"),
                    alt.Tooltip("anzsco_title:N",   title="Occupation"),
                    alt.Tooltip("stmt_count:Q",     title="Statements"),
                    alt.Tooltip("uoc_count:Q",      title="UOCs"),
                    alt.Tooltip("avg_confidence:Q", title="Avg conf", format=".3f"),
                ],
            ).properties(height=420, title="Top 20 occupations by statement count")
            st.altair_chart(occ_chart, use_container_width=True)
