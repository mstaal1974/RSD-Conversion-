"""
pages/15_🔎_Skills_by_Occupation.py

Reverse of page 13: search by Australian occupation name → show every
RSD skill statement in the corpus that contributes to it via the
ANZSCO → ISCO → ESCO chain.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import anzsco_crosswalk, esco_local, profiles
from core.db import get_engine

st.set_page_config(page_title="Skills by Occupation", layout="wide")
st.title("🔎 Skills by Occupation")
st.caption(
    "Type an Australian occupation (ANZSCO) — the page reverses the "
    "Occupational Profiles chain and lists every matched skill statement "
    "that contributes to it, ranked by contribution score."
)

# ── Preconditions ──────────────────────────────────────────────────────────────
if not esco_local.is_available():
    st.error("ESCO CSVs not found. See data/esco/README.md.")
    st.stop()
if not anzsco_crosswalk.is_available():
    st.warning(
        "ANZSCO crosswalk not found at `data/anzsco/isco_to_anzsco.csv`. "
        "See `data/anzsco/README.md`."
    )
    st.stop()

engine = get_engine()

# ── Occupation picker ─────────────────────────────────────────────────────────
query = st.text_input(
    "Search occupation name",
    placeholder="e.g. electrician, registered nurse, plumber, software engineer…",
)

if not query:
    st.info("Start typing to search ANZSCO titles.")
    st.stop()

hits = anzsco_crosswalk.search_titles(query, limit=50)
if not hits:
    st.warning(f"No ANZSCO occupations match “{query}”.")
    st.stop()

labels = [f"{h['anzsco_code']} — {h['anzsco_title']}" for h in hits]
sel = st.selectbox(
    f"Matches ({len(hits)})  — pick one",
    labels,
    index=0,
)
chosen = hits[labels.index(sel)]
anzsco_code = chosen["anzsco_code"]

# ── Build reverse profile ─────────────────────────────────────────────────────
with st.spinner("Tracing skill statements…"):
    result = profiles.statements_for_anzsco(engine, anzsco_code)

st.subheader(f"{anzsco_code} — {result['anzsco_title']}")

stmts_df = result["statements"]
esco_occs = result["esco_occupations"]
isco_codes = result["isco_codes"]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Skill statements", len(stmts_df))
col_b.metric("Distinct UOCs", stmts_df["unit_code"].nunique() if not stmts_df.empty else 0)
col_c.metric("ESCO occupations in chain", len(esco_occs))
col_d.metric("ISCO unit groups", len(isco_codes))

if stmts_df.empty:
    st.info(
        "No matched skill statements in the corpus map to this occupation yet. "
        "Either the chain has no ESCO occupations under it, or the ESCO matcher "
        "hasn't yet linked any statements to those occupations."
    )
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    min_score = st.slider(
        "Min contribution score", 0.0, 1.0,
        float(round(stmts_df["contribution_score"].min(), 2)), 0.05,
    )
    only_exact = st.checkbox(
        "Only via exact ISCO ↔ ANZSCO links (weight = 1.0)", value=False,
    )
    group_by_uoc = st.checkbox("Group by UOC", value=False)

filtered = stmts_df[stmts_df["contribution_score"] >= min_score]
if only_exact:
    filtered = filtered[filtered["crosswalk_weight"] >= 1.0]

st.caption(
    f"Showing **{len(filtered)} of {len(stmts_df)}** statements "
    f"(min score {min_score:.2f}{', exact only' if only_exact else ''})."
)

# ── Render ────────────────────────────────────────────────────────────────────
if group_by_uoc:
    grouped = filtered.groupby("unit_code").agg(
        n_statements=("id", "count"),
        max_contribution=("contribution_score", "max"),
        mean_contribution=("contribution_score", "mean"),
    ).reset_index().sort_values("max_contribution", ascending=False)
    st.dataframe(grouped, use_container_width=True, hide_index=True)
    chosen_uoc = st.selectbox("Drill into UOC", grouped["unit_code"].tolist())
    if chosen_uoc:
        sub = filtered[filtered["unit_code"] == chosen_uoc][[
            "element_title", "skill_statement",
            "esco_skill_title", "esco_skill_score",
            "via_esco_title", "via_isco", "crosswalk_weight", "contribution_score",
        ]]
        st.dataframe(sub, use_container_width=True, hide_index=True, height=500)
else:
    display = filtered[[
        "unit_code", "element_title", "skill_statement",
        "esco_skill_title", "esco_skill_score",
        "via_esco_title", "via_isco", "crosswalk_weight", "contribution_score",
    ]].rename(columns={
        "via_esco_title": "via ESCO occupation",
        "via_isco": "via ISCO",
        "crosswalk_weight": "weight",
    })
    st.dataframe(display, use_container_width=True, hide_index=True, height=600)

# ── Provenance ────────────────────────────────────────────────────────────────
with st.expander("Chain detail — ANZSCO → ISCO → ESCO occupations", expanded=False):
    st.write(f"**ISCO unit groups behind {anzsco_code}**: {', '.join(isco_codes) or '—'}")
    if esco_occs:
        chain_df = pd.DataFrame(esco_occs).sort_values(["isco_code", "title"])
        st.dataframe(chain_df, use_container_width=True, hide_index=True)

# ── Export ────────────────────────────────────────────────────────────────────
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered statements (CSV)",
    csv,
    file_name=f"skills_for_{anzsco_code}.csv",
    mime="text/csv",
)
