"""
pages/6_📋_Occupation_Skills_Profiles.py

Output: For each ANZSCO occupation — the skills required and the UOCs they come from.

Structure:
  Occupation (ANZSCO code + title)
    └── Skill statement
          └── UOC code + title + element
"""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
import altair as alt

load_dotenv()
st.set_page_config(page_title="Occupation Skills Profiles", layout="wide")

st.title("📋 Occupation Skills Profiles")
st.caption(
    "For each ANZSCO occupation — the required skills and the Units of Competency "
    "they are derived from."
)

# ── Connection ────────────────────────────────────────────────────────────────
def _secret(k, d=""):
    try:
        return st.secrets.get(k, os.getenv(k, d)) or d
    except Exception:
        return os.getenv(k, d) or d

DB_URL = _secret("DATABASE_URL")
if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()

@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

engine = get_engine(DB_URL)

# ── Check tables exist ────────────────────────────────────────────────────────
try:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'uoc_occupation_links'
        """)).fetchone()
    if not row or row[0] == 0:
        st.warning("⚠️ Taxonomy tables not found. Run the migration and pipeline first.")
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# ── Load full occupation-skills-UOC dataset ───────────────────────────────────
@st.cache_data(ttl=120, show_spinner="Loading occupation profiles…")
def load_profiles(min_conf: float) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                o.anzsco_code,
                o.anzsco_title,
                o.anzsco_major_group,
                o.confidence,
                o.mapping_source,
                s.unit_code,
                s.unit_title,
                s.tp_code,
                s.element_title,
                s.skill_statement,
                s.keywords
            FROM uoc_occupation_links o
            JOIN rsd_skill_records s
                ON s.unit_code = o.uoc_code
            WHERE o.is_primary = TRUE
              
              AND o.confidence >= :minc
              AND o.anzsco_code IS NOT NULL
              AND o.anzsco_code != ''
              AND s.skill_statement IS NOT NULL
              AND s.skill_statement != ''
            ORDER BY
                o.anzsco_major_group,
                o.anzsco_code,
                o.confidence DESC,
                s.unit_code,
                s.element_title
        """), {"minc": min_conf}).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.30, 0.05)

    df_all = load_profiles(min_conf)

    if df_all.empty:
        st.warning("No data. Run the linkage pipeline first.")
        st.stop()

    major_groups = ["All"] + sorted(df_all["anzsco_major_group"].dropna().unique().tolist())
    sel_major = st.selectbox("ANZSCO major group", major_groups)

    if sel_major != "All":
        df_filtered = df_all[df_all["anzsco_major_group"] == sel_major]
    else:
        df_filtered = df_all

    occupations = sorted(df_filtered["anzsco_code"].unique().tolist(),
                         key=lambda c: df_filtered[df_filtered["anzsco_code"]==c]["anzsco_title"].iloc[0])
    occ_options = {f"{row['anzsco_code']} — {row['anzsco_title']}": row['anzsco_code']
                   for _, row in df_filtered.drop_duplicates("anzsco_code").iterrows()}

    sel_occ_label = st.selectbox("Occupation", list(occ_options.keys()))
    sel_occ_code  = occ_options[sel_occ_label]

    st.divider()
    tp_codes = sorted(df_all["tp_code"].dropna().unique().tolist())
    sel_tps = st.multiselect("Filter by training package", tp_codes, default=[])

# ── Filter to selected occupation ─────────────────────────────────────────────
df_occ = df_filtered[df_filtered["anzsco_code"] == sel_occ_code].copy()
if sel_tps:
    df_occ = df_occ[df_occ["tp_code"].isin(sel_tps)]

if df_occ.empty:
    st.info("No skills found for this occupation with current filters.")
    st.stop()

occ_title   = df_occ["anzsco_title"].iloc[0]
major_group = df_occ["anzsco_major_group"].iloc[0]
avg_conf    = df_occ["confidence"].mean()
n_skills    = len(df_occ)
n_uocs      = df_occ["unit_code"].nunique()
n_tps       = df_occ["tp_code"].nunique()

# ── Header cards ──────────────────────────────────────────────────────────────
st.markdown(f"## {sel_occ_code} — {occ_title}")
st.caption(f"ANZSCO major group: **{major_group}**")

c1, c2, c3, c4 = st.columns(4)
def metric(col, val, lbl):
    with col:
        st.metric(lbl, val)

metric(c1, f"{n_skills:,}",   "Skill statements")
metric(c2, f"{n_uocs}",       "Units of competency")
metric(c3, f"{n_tps}",        "Training packages")
metric(c4, f"{avg_conf:.2f}", "Avg confidence")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📄 Skills by UOC", "🗂️ Skills by Element", "⬇ Export"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Skills grouped by UOC
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"**{n_skills}** skill statements from **{n_uocs}** units of competency "
        f"linked to occupation **{sel_occ_code}**"
    )

    for unit_code, grp in df_occ.groupby("unit_code"):
        unit_title = grp["unit_title"].iloc[0]
        tp         = grp["tp_code"].iloc[0] or ""
        n_stmts    = len(grp)
        conf       = grp["confidence"].iloc[0]

        conf_color = (
            "#1b5e20" if conf >= 0.85 else
            "#0d47a1" if conf >= 0.70 else
            "#bf360c" if conf >= 0.40 else
            "#212121"
        )

        with st.expander(
            f"**{unit_code}** — {unit_title}  ·  {n_stmts} skills  ·  TP: {tp}",
            expanded=False
        ):
            st.markdown(
                f'<span style="font-family:monospace;font-size:0.75rem;'
                f'background:{conf_color};color:white;padding:2px 8px;'
                f'border-radius:4px">conf {conf:.2f}</span>',
                unsafe_allow_html=True
            )
            st.markdown("")

            for _, row in grp.iterrows():
                elem  = row.get("element_title") or ""
                stmt  = row.get("skill_statement") or ""
                st.markdown(
                    f"<div style='background:#0d1b2a;border-left:3px solid #1a237e;"
                    f"border-radius:4px;padding:8px 12px;margin-bottom:6px'>"
                    f"<div style='font-size:0.7rem;color:#5c6bc0;margin-bottom:4px'>"
                    f"{elem}</div>"
                    f"<div style='font-size:0.87rem;color:#cfd8dc'>{stmt}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Skills grouped by Element
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    for elem_title, grp in df_occ.groupby("element_title"):
        n_e = len(grp)
        with st.expander(f"**{elem_title}**  ·  {n_e} skills", expanded=False):
            for _, row in grp.iterrows():
                unit  = row.get("unit_code") or ""
                stmt  = row.get("skill_statement") or ""
                st.markdown(
                    f"<div style='background:#0d1b2a;border-left:3px solid #283593;"
                    f"border-radius:4px;padding:8px 12px;margin-bottom:6px'>"
                    f"<div style='font-size:0.7rem;color:#5c6bc0;margin-bottom:4px'>"
                    f"<code>{unit}</code></div>"
                    f"<div style='font-size:0.87rem;color:#cfd8dc'>{stmt}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Export
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Export options")

    # Single occupation CSV
    export_cols = {
        "anzsco_code":    "ANZSCO Code",
        "anzsco_title":   "Occupation",
        "anzsco_major_group": "Major Group",
        "unit_code":      "Unit Code",
        "unit_title":     "Unit Title",
        "tp_code":        "Training Package",
        "element_title":  "Element",
        "skill_statement":"Skill Statement",
        "confidence":     "Taxonomy Confidence",
        "mapping_source": "Mapping Source",
    }
    export_df = df_occ[[c for c in export_cols if c in df_occ.columns]].rename(columns=export_cols)

    st.download_button(
        f"⬇ Download {sel_occ_code} skills profile CSV",
        export_df.to_csv(index=False).encode(),
        f"{sel_occ_code}_skills_profile.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()

    # Full all-occupations export (Occupation → Skills → UOC)
    st.subheader("Full occupational skills matrix")
    st.caption("All occupations × all skill statements × all UOCs in one file.")

    @st.cache_data(ttl=120)
    def load_full_matrix(mc):
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    o.anzsco_code,
                    o.anzsco_title,
                    o.anzsco_major_group,
                    o.confidence,
                    o.mapping_source,
                    s.unit_code,
                    s.unit_title,
                    s.tp_code,
                    s.element_title,
                    s.skill_statement
                FROM uoc_occupation_links o
                JOIN rsd_skill_records s ON s.unit_code = o.uoc_code
                WHERE o.is_primary = TRUE
                  
                  AND o.confidence >= :mc
                  AND o.anzsco_code IS NOT NULL
                  AND o.anzsco_code != ''
                  AND s.skill_statement IS NOT NULL
                ORDER BY o.anzsco_major_group, o.anzsco_code, s.unit_code
            """), {"mc": mc}).mappings().all()
        return pd.DataFrame([dict(r) for r in rows])

    full_df = load_full_matrix(min_conf)

    col_map = {
        "anzsco_code":      "ANZSCO Code",
        "anzsco_title":     "Occupation",
        "anzsco_major_group": "Major Group",
        "unit_code":        "Unit Code",
        "unit_title":       "Unit Title",
        "tp_code":          "Training Package",
        "element_title":    "Element",
        "skill_statement":  "Skill Statement",
        "confidence":       "Taxonomy Confidence",
        "mapping_source":   "Mapping Source",
    }
    full_export = full_df[[c for c in col_map if c in full_df.columns]].rename(columns=col_map)

    st.info(
        f"**{len(full_export):,}** rows — "
        f"**{full_df['anzsco_code'].nunique()}** occupations × "
        f"**{full_df['unit_code'].nunique()}** UOCs"
    )

    st.download_button(
        "⬇ Download full occupational skills matrix CSV",
        full_export.to_csv(index=False).encode(),
        "occupational_skills_matrix.csv",
        "text/csv",
        use_container_width=True,
    )

    # UOC-first summary pivot
    st.divider()
    st.subheader("Occupation × UOC summary")
    pivot = (
        full_df.groupby(["anzsco_code", "anzsco_title", "unit_code", "unit_title", "tp_code"])
        .agg(
            skill_count=("skill_statement", "count"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    pivot["avg_confidence"] = pd.to_numeric(pivot["avg_confidence"], errors="coerce").round(3)
    pivot = pivot.sort_values(["anzsco_code", "skill_count"], ascending=[True, False])

    st.dataframe(
        pivot,
        use_container_width=True,
        hide_index=True,
        column_config={
            "anzsco_code":    st.column_config.TextColumn("ANZSCO"),
            "anzsco_title":   st.column_config.TextColumn("Occupation"),
            "unit_code":      st.column_config.TextColumn("Unit", width="small"),
            "unit_title":     st.column_config.TextColumn("Unit title"),
            "tp_code":        st.column_config.TextColumn("TP", width="small"),
            "skill_count":    st.column_config.NumberColumn("Skills"),
            "avg_confidence": st.column_config.NumberColumn("Avg conf", format="%.3f"),
        }
    )

    st.download_button(
        "⬇ Download occupation × UOC summary CSV",
        pivot.to_csv(index=False).encode(),
        "occupation_uoc_summary.csv",
        "text/csv",
        use_container_width=True,
    )
