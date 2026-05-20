"""
pages/13_🇦🇺_Occupational_Profiles.py

For each UOC (or whole qualification), build an occupational profile by
threading the ESCO matches in rsd_skill_records through the ISCO ↔ ANZSCO
crosswalk: ESCO skill → ESCO occupation → ISCO-08 → ANZSCO.

Prereqs:
  • core.esco_local is loaded (data/esco/ populated, model bundled)
  • data/esco/occupations_en.csv present (for the ISCO group on each ESCO occupation)
  • ANZSCO crosswalk at data/anzsco/isco_to_anzsco.csv  (see data/anzsco/README.md)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from core import anzsco_crosswalk, esco_local, profiles

load_dotenv()
st.set_page_config(page_title="Occupational Profiles", layout="wide")

st.title("🇦🇺 ESCO → ANZSCO Occupational Profiles")
st.caption(
    "Threads every UOC's ESCO matches through the ISCO ↔ ANZSCO crosswalk "
    "to build an occupational profile grounded in EU skills data but expressed "
    "in Australian occupation codes."
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


# ── Preflight ─────────────────────────────────────────────────────────────────
if not esco_local.is_available():
    st.error("ESCO CSVs not found. See data/esco/README.md.")
    st.stop()

if not anzsco_crosswalk.is_available():
    st.warning(
        "ANZSCO crosswalk not found at `data/anzsco/isco_to_anzsco.csv`. "
        "See `data/anzsco/README.md` for how to prepare it from the ABS release."
    )
    st.stop()


# ── Scope picker ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scope")
    scope = st.radio("Profile granularity", ["Single UOC", "Whole qualification"])

    if scope == "Single UOC":
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT unit_code FROM rsd_skill_records "
                    "WHERE esco_skill_uri IS NOT NULL AND esco_skill_uri <> '' "
                    "ORDER BY unit_code"
                )
            ).all()
        uoc_codes = [r[0] for r in rows]
        if not uoc_codes:
            st.info("No ESCO-matched statements yet. Run the ESCO Alignment page first.")
            st.stop()
        target_uoc = st.selectbox("UOC", uoc_codes)
        target_qual = None
    else:
        with engine.connect() as conn:
            try:
                rows = conn.execute(
                    text(
                        "SELECT qual_code, qual_title FROM qual_registry "
                        "WHERE status = 'Current' ORDER BY qual_code"
                    )
                ).all()
            except Exception:
                rows = []
        if not rows:
            st.info("No qualifications in qual_registry. Ingest a qualification first.")
            st.stop()
        labels = [f"{c} — {t}" for c, t in rows]
        sel = st.selectbox("Qualification", labels)
        target_qual = rows[labels.index(sel)][0]
        target_uoc = None

    st.divider()
    top_n = st.slider("Top N ANZSCO occupations", 5, 50, 15)
    min_score = st.slider("Min aggregate score", 0.0, 5.0, 0.0, 0.1)


# ── Build profile ─────────────────────────────────────────────────────────────
with st.spinner("Building profile…"):
    if target_uoc:
        prof = profiles.uoc_profile(engine, target_uoc)
        scope_label = f"UOC **{target_uoc}**"
    else:
        prof = profiles.qualification_profile(engine, target_qual)
        scope_label = f"Qualification **{target_qual}**"

cov = prof["coverage"]
anzsco_rows = prof["anzsco_rows"]
statements_df = prof["statements"]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Skill statements (matched)", cov["n_statements"])
col_b.metric("Statements with ANZSCO mapping", cov["n_with_anzsco"])
col_c.metric("Distinct ANZSCO occupations", len(anzsco_rows))
col_d.metric(
    "Crosswalk coverage",
    f"{(cov['n_with_anzsco'] / cov['n_statements'] * 100):.0f}%" if cov["n_statements"] else "—",
)

if not anzsco_rows:
    st.info(
        "No ANZSCO occupations resolved for this scope. Likely missing "
        "ISCO codes on the matched ESCO occupations, or no overlapping rows "
        "in the crosswalk."
    )
    st.stop()

filtered = [r for r in anzsco_rows if r["score"] >= min_score][:top_n]
df_rank = pd.DataFrame(filtered)

# ── Ranked bar chart ──────────────────────────────────────────────────────────
st.subheader(f"Top ANZSCO occupations for {scope_label}")
chart_df = df_rank.assign(
    label=lambda d: d["anzsco_code"].astype(str) + " — " + d["anzsco_title"]
)[["label", "score", "n_statements"]]
st.bar_chart(chart_df.set_index("label")["score"])

# ── Detail table ──────────────────────────────────────────────────────────────
st.subheader("Ranked detail")
display = df_rank[[
    "anzsco_code", "anzsco_title", "score", "n_statements",
    "supporting_esco_occupations",
]].copy()
display["supporting_esco_occupations"] = display["supporting_esco_occupations"].apply(
    lambda xs: ", ".join(xs[:6]) + (f" (+{len(xs) - 6})" if len(xs) > 6 else "")
)
display = display.rename(columns={
    "anzsco_code": "ANZSCO",
    "anzsco_title": "Title",
    "score": "Score",
    "n_statements": "Skills",
    "supporting_esco_occupations": "Top ESCO occupations behind this",
})
st.dataframe(display, use_container_width=True, hide_index=True, height=400)

# ── Drill-down: pick an ANZSCO, see the statements driving it ─────────────────
st.subheader("Evidence")
chosen = st.selectbox(
    "Inspect which ANZSCO occupation?",
    [f"{r['anzsco_code']} — {r['anzsco_title']}" for r in filtered],
)
chosen_code = chosen.split(" — ")[0]
chosen_row = next(r for r in filtered if r["anzsco_code"] == chosen_code)

driving_ids = set(chosen_row["supporting_skill_statements"])
evidence_df = statements_df[statements_df["id"].isin(driving_ids)][[
    "unit_code", "element_title", "skill_statement",
    "esco_skill_title", "esco_skill_score",
]].sort_values("esco_skill_score", ascending=False)

st.caption(
    f"{len(evidence_df)} skill statements drive **{chosen_code} — {chosen_row['anzsco_title']}** "
    f"(aggregate score {chosen_row['score']:.2f})."
)
st.dataframe(evidence_df, use_container_width=True, hide_index=True, height=400)


# ── Downloads ─────────────────────────────────────────────────────────────────
st.subheader("Download")
out = df_rank.copy()
out["supporting_esco_occupations"] = out["supporting_esco_occupations"].apply(lambda xs: " | ".join(xs))
out["supporting_skill_statements"] = out["supporting_skill_statements"].apply(lambda xs: " | ".join(str(i) for i in xs))

st.download_button(
    "⬇ Profile CSV",
    out.to_csv(index=False).encode(),
    file_name=f"profile_{target_uoc or target_qual}.csv",
    mime="text/csv",
)
