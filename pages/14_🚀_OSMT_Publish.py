"""
pages/14_🚀_OSMT_Publish.py

Push generated RSDs to an OSMT (Open Skills Management Tool) instance
via its REST API. Closes the export loop so curators don't have to
download CSVs and re-upload through OSMT's bulk import UI.

Requires:
  OSMT_BASE_URL   — e.g. https://osmt.example.org
  OSMT_API_TOKEN  — bearer token (preferred), OR
  OSMT_USERNAME + OSMT_PASSWORD
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

from core.exporters import to_osmt_rows
from core.osmt_client import OSMTClient, OSMTError

load_dotenv()
st.set_page_config(page_title="OSMT Publish", layout="wide")

st.title("🚀 Publish to OSMT")
st.caption(
    "Push RSDs from this platform directly to an Open Skills Management Tool "
    "instance over its REST API."
)


# ── Secrets / config ──────────────────────────────────────────────────────────
def _secret(k, d=""):
    try:
        return st.secrets.get(k, os.getenv(k, d)) or d
    except Exception:
        return os.getenv(k, d) or d


OSMT_URL   = _secret("OSMT_BASE_URL")
HAS_TOKEN  = bool(_secret("OSMT_API_TOKEN"))
HAS_LOGIN  = bool(_secret("OSMT_USERNAME") and _secret("OSMT_PASSWORD"))
DB_URL     = _secret("DATABASE_URL")

if not OSMT_URL:
    st.error("`OSMT_BASE_URL` is not configured.")
    st.stop()

if not (HAS_TOKEN or HAS_LOGIN):
    st.error(
        "OSMT credentials missing — set `OSMT_API_TOKEN` (preferred) or both "
        "`OSMT_USERNAME` and `OSMT_PASSWORD`."
    )
    st.stop()


@st.cache_resource
def get_client() -> OSMTClient:
    return OSMTClient.from_env()


@st.cache_resource
def get_engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)


# ── Connection test ───────────────────────────────────────────────────────────
st.subheader("1. Connection")

col_a, col_b = st.columns([1, 3])
if col_a.button("🔌 Test connection", type="primary"):
    try:
        resp = get_client().health()
        col_b.success(f"✅ Reached `{OSMT_URL}` — auth OK.")
        with col_b.expander("Raw health response"):
            st.json(resp)
    except OSMTError as e:
        col_b.error(f"❌ {e}")
        st.stop()


# ── Source selection ──────────────────────────────────────────────────────────
st.subheader("2. Select RSDs to push")

source = st.radio(
    "Source",
    ["Current database run", "Uploaded JSONL"],
    horizontal=True,
)

records: list[dict] = []

if source == "Current database run":
    if not DB_URL:
        st.error("`DATABASE_URL` not set.")
        st.stop()
    engine = get_engine(DB_URL)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC LIMIT 50"
            )
        ).mappings().all()
    if not rows:
        st.info("No runs in the database.")
        st.stop()
    run_options = [f"{str(r['id'])[:8]}… {r['source_filename']}" for r in rows]
    sel = st.selectbox("Run", run_options)
    sel_run_id = rows[run_options.index(sel)]["id"]

    qa_only = st.checkbox("Only QA-passing rows", value=True)
    fid_only = st.checkbox("Only PC-fidelity-passing rows", value=True)
    cap = st.number_input("Max rows", 1, 100_000, 500, 50)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                SELECT unit_code, unit_title, element_title, skill_statement,
                       keywords, tp_code, tp_title, esco_skill_uri, esco_skill_title,
                       qa_passes, qa_pc_cosine_pass
                  FROM rsd_skill_records
                 WHERE run_id = :rid
                   AND skill_statement IS NOT NULL AND skill_statement <> ''
                   {'AND qa_passes = TRUE' if qa_only else ''}
                   {'AND COALESCE(qa_pc_cosine_pass, TRUE) = TRUE' if fid_only else ''}
                 ORDER BY unit_code, row_index
                 LIMIT :cap
                """
            ),
            conn,
            params={"rid": sel_run_id, "cap": int(cap)},
        )
    st.caption(f"{len(df)} rows selected.")
    if not df.empty:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        records = to_osmt_rows(df).to_dict(orient="records")

else:
    f = st.file_uploader("RSD JSONL file", type=["jsonl", "json"])
    if f is not None:
        for line in f.read().decode("utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        st.caption(f"{len(records)} records loaded.")


# ── Push ──────────────────────────────────────────────────────────────────────
st.subheader("3. Push to OSMT")

batch_size = st.slider("Batch size", 10, 200, 50, 10)
dry_run = st.checkbox(
    "Dry run (preview the first record only, no API call)",
    value=True,
)

if dry_run and records:
    st.markdown("**Preview of first record:**")
    st.json(records[0])

if not dry_run and records:
    if st.button("🚀 Push now", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        total = len(records)

        def cb(done, msg):
            bar.progress(min(done / total, 1.0), text=msg)

        try:
            summary = get_client().bulk_upsert_skills(
                records, batch_size=batch_size, progress_callback=cb
            )
            bar.progress(1.0, text="Done")
            if summary["errors"]:
                st.warning(
                    f"Pushed {summary['pushed']} / {total} records. "
                    f"{summary['errors']} errored."
                )
                with st.expander("Error detail"):
                    for msg in summary["error_messages"]:
                        st.code(msg)
            else:
                st.success(f"✅ Pushed all {summary['pushed']} records to OSMT.")
        except OSMTError as e:
            st.error(f"Push failed: {e}")
