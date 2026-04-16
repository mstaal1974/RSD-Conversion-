"""
pages/3_🌐_ESCO_Alignment.py

Match RSD skill statements against the ESCO taxonomy and associate with occupations.
Uses the free ESCO REST API (ec.europa.eu/esco/api) — no key required.

PERSISTENCE DESIGN
──────────────────
Progress is stored *only* in the database — never in session_state.
On every page load (including after a refresh) the app queries:
  • load_progress() — how many rows are total / already matched
  • load_todo()     — rows WHERE esco_skill_uri IS NULL (uncached, always fresh)

This means:
  • Refresh mid-run → resume from exact row that failed, zero loss
  • Close tab, come back tomorrow → same thing
  • No need to click "resume" — just click Run and it skips done rows
"""
from __future__ import annotations

import os
import time
import json
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
st.set_page_config(page_title="ESCO Alignment", layout="wide")

st.title("🌐 ESCO Skills & Occupation Alignment")
st.caption(
    "Match each RSD skill statement against the ESCO taxonomy. "
    "Progress is saved to the database after every batch — "
    "safe to refresh or close at any time."
)

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def load_progress(run_id=None, unit_prefix=""):
    """Return (total, done) counts — always live from DB, never cached."""
    base_filter = ""
    params: dict = {}
    if run_id:
        base_filter += " AND run_id = :run_id"
        params["run_id"] = run_id
    if unit_prefix:
        base_filter += " AND unit_code ILIKE :up"
        params["up"] = unit_prefix.upper() + "%"

    with engine.connect() as conn:
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM rsd_skill_records WHERE 1=1 {base_filter}"),
            params,
        ).scalar() or 0
        done = conn.execute(
            text(
                f"SELECT COUNT(*) FROM rsd_skill_records "
                f"WHERE esco_skill_uri IS NOT NULL AND esco_skill_uri <> '' {base_filter}"
            ),
            params,
        ).scalar() or 0
    return int(total), int(done)


def load_todo(run_id=None, unit_prefix="", limit=None):
    """
    Load ONLY unmatched rows (esco_skill_uri IS NULL or empty).
    NOT cached — must always reflect the current DB state.
    """
    base_filter = " AND (esco_skill_uri IS NULL OR esco_skill_uri = '')"
    params: dict = {}
    if run_id:
        base_filter += " AND run_id = :run_id"
        params["run_id"] = run_id
    if unit_prefix:
        base_filter += " AND unit_code ILIKE :up"
        params["up"] = unit_prefix.upper() + "%"

    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT * FROM rsd_skill_records "
                f"WHERE 1=1 {base_filter} "
                f"ORDER BY unit_code, row_index {limit_clause}"
            ),
            params,
        ).mappings().all()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def save_batch_to_db(records: list[dict]):
    """
    Upsert ESCO results for a list of records.
    Each record must have keys: id, uri, title, score, occ_t, occ_u
    """
    if not records:
        return
    with engine.begin() as conn:
        for rec in records:
            conn.execute(
                text(
                    """
                    UPDATE rsd_skill_records SET
                        esco_skill_uri        = :uri,
                        esco_skill_title      = :title,
                        esco_skill_score      = :score,
                        esco_occupation_titles = :occ_t,
                        esco_occupation_uris   = :occ_u
                    WHERE id = :id
                    """
                ),
                rec,
            )


def ensure_esco_columns():
    """Add ESCO columns to rsd_skill_records if they don't exist yet."""
    cols = {
        "esco_skill_uri": "TEXT",
        "esco_skill_title": "TEXT",
        "esco_skill_score": "FLOAT",
        "esco_occupation_titles": "TEXT",
        "esco_occupation_uris": "TEXT",
    }
    with engine.begin() as conn:
        for col, dtype in cols.items():
            try:
                conn.execute(
                    text(
                        f"ALTER TABLE rsd_skill_records ADD COLUMN IF NOT EXISTS {col} {dtype}"
                    )
                )
            except Exception:
                pass  # column already exists


ensure_esco_columns()


# ─────────────────────────────────────────────────────────────────────────────
# ESCO API
# ─────────────────────────────────────────────────────────────────────────────

ESCO_BASE = "https://ec.europa.eu/esco/api"
REQUEST_DELAY = 0.35   # seconds between calls


def _esco_get(url: str, retries: int = 3) -> dict:
    headers = {"Accept": "application/json", "Accept-Language": "en"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** (attempt + 1))
            elif e.code >= 500:
                time.sleep(2)
            else:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return {}


def match_statement(
    statement: str,
    top_n_skills: int = 3,
    top_n_occupations: int = 8,
    min_score: float = 0.0,
    language: str = "en",
) -> dict:
    """
    Match one skill statement to ESCO. Returns a flat dict with:
      top_skill_uri, top_skill_title, top_skill_score,
      all_occupation_titles (pipe-separated), all_occupation_uris (pipe-separated)
    """
    empty = dict(
        top_skill_uri="",
        top_skill_title="",
        top_skill_score=0.0,
        all_occupation_titles="",
        all_occupation_uris="",
    )

    if not statement or not statement.strip():
        return empty

    # ── Step 1: search skills ──────────────────────────────────────────────
    params = urllib.parse.urlencode(
        {"text": statement[:500], "type": "skill", "language": language, "limit": top_n_skills}
    )
    time.sleep(REQUEST_DELAY)
    try:
        data = _esco_get(f"{ESCO_BASE}/search?{params}")
    except Exception:
        return empty

    results = data.get("_embedded", {}).get("results", [])
    if not results:
        return empty

    top = results[0]
    top_uri = top.get("uri", "")
    top_title = top.get("title", "")
    top_score = round(float(top.get("score", 0)), 4)

    if top_score < min_score or not top_uri:
        return empty

    # ── Step 2: fetch occupations ──────────────────────────────────────────
    occ_titles: list[str] = []
    occ_uris: list[str] = []

    for relation in ("isEssentialForOccupation", "isOptionalForOccupation"):
        occ_params = urllib.parse.urlencode(
            {"uri": top_uri, "relation": relation, "language": language, "limit": top_n_occupations}
        )
        time.sleep(REQUEST_DELAY)
        try:
            occ_data = _esco_get(f"{ESCO_BASE}/resource/related?{occ_params}")
            for occ in occ_data.get("_embedded", {}).get("occupationList", []):
                uri = occ.get("uri", "")
                title = occ.get("title", "")
                if uri and uri not in occ_uris:
                    occ_uris.append(uri)
                    occ_titles.append(title)
        except Exception:
            pass

    return dict(
        top_skill_uri=top_uri,
        top_skill_title=top_title,
        top_skill_score=top_score,
        all_occupation_titles=" | ".join(occ_titles[:top_n_occupations]),
        all_occupation_uris=" | ".join(occ_uris[:top_n_occupations]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Matching settings")
    top_n_skills = st.slider("Top N ESCO skills", 1, 10, 3)
    top_n_occ = st.slider("Max occupations per skill", 3, 20, 8)
    min_score = st.slider("Min match score", 0.0, 1.0, 0.0, 0.05)
    language = st.selectbox("ESCO language", ["en", "de", "fr", "es", "it", "nl", "pl", "pt"])

    st.divider()
    st.header("Filter source data")

    try:
        with engine.connect() as conn:
            runs = conn.execute(
                text("SELECT id, source_filename FROM rsd_runs ORDER BY created_at DESC")
            ).mappings().all()
        runs = [dict(r) for r in runs]
        run_options = ["All runs"] + [f"{str(r['id'])[:8]}… {r['source_filename']}" for r in runs]
        run_ids = [None] + [r["id"] for r in runs]
        sel_run_label = st.selectbox("Source run", run_options)
        sel_run_id = run_ids[run_options.index(sel_run_label)]
    except Exception:
        sel_run_id = None

    unit_filter = st.text_input("Filter by unit code prefix", placeholder="e.g. BSB")
    batch_size = st.number_input(
        "Batch size", 10, 500, 250, 10,
        help="Statements per batch. Saves to DB after each batch."
    )

    st.divider()
    st.header("Danger zone")
    if st.button("🗑 Reset all ESCO data", type="secondary"):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE rsd_skill_records SET "
                    "esco_skill_uri=NULL, esco_skill_title=NULL, esco_skill_score=NULL, "
                    "esco_occupation_titles=NULL, esco_occupation_uris=NULL"
                )
            )
        st.success("All ESCO data cleared.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Progress header — always live from DB
# ─────────────────────────────────────────────────────────────────────────────

total_rows, done_rows = load_progress(sel_run_id, unit_filter)
todo_rows = total_rows - done_rows

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total statements", total_rows)
col_b.metric("Already matched ✅", done_rows)
col_c.metric("Remaining ⏳", todo_rows)

if total_rows > 0:
    st.progress(done_rows / total_rows, text=f"{done_rows}/{total_rows} matched")

if todo_rows == 0 and total_rows > 0:
    st.success("🎉 All statements have been matched! See results below.")

# ─────────────────────────────────────────────────────────────────────────────
# Run controls
# ─────────────────────────────────────────────────────────────────────────────

st.divider()

run_col1, run_col2 = st.columns([2, 1])

with run_col1:
    run_all = st.button(
        "▶ Run all remaining batches",
        type="primary",
        disabled=(todo_rows == 0),
        help="Processes all unmatched statements, saving after each batch. Safe to stop and restart.",
    )

with run_col2:
    run_one = st.button(
        "▶ Run one batch",
        disabled=(todo_rows == 0),
        help=f"Process the next {batch_size} unmatched statements only.",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run logic
# ─────────────────────────────────────────────────────────────────────────────

if run_all or run_one:
    # Load only unmatched rows — never cached, always fresh from DB
    df_todo = load_todo(sel_run_id, unit_filter, limit=batch_size if run_one else None)

    if df_todo.empty:
        st.info("Nothing left to process.")
    else:
        n_todo = len(df_todo)
        n_batches = max(1, (n_todo + int(batch_size) - 1) // int(batch_size))

        st.info(
            f"Processing **{n_todo}** remaining statements in **{n_batches}** batch(es) "
            f"of {batch_size}. Progress is saved after each batch."
        )

        overall_bar = st.progress(0, text="Starting…")
        batch_label = st.empty()
        batch_bar = st.progress(0)
        status_line = st.empty()
        errors: list[str] = []

        for batch_num in range(n_batches):
            start = batch_num * int(batch_size)
            end = min(start + int(batch_size), n_todo)
            df_batch = df_todo.iloc[start:end]

            batch_label.markdown(
                f"**Batch {batch_num + 1}/{n_batches}** — rows {start + 1}–{end}"
            )
            batch_bar.progress(0)

            batch_records: list[dict] = []
            batch_errors: list[str] = []

            for i, (_, row) in enumerate(df_batch.iterrows()):
                stmt = str(row.get("skill_statement", "") or "").strip()

                try:
                    result = match_statement(
                        stmt,
                        top_n_skills=top_n_skills,
                        top_n_occupations=top_n_occ,
                        min_score=min_score,
                        language=language,
                    )
                except Exception as e:
                    batch_errors.append(f"Row {row.get('id')}: {e}")
                    result = dict(
                        top_skill_uri="",
                        top_skill_title="",
                        top_skill_score=0.0,
                        all_occupation_titles="",
                        all_occupation_uris="",
                    )

                batch_records.append(
                    {
                        "id": row["id"],
                        "uri": result["top_skill_uri"],
                        "title": result["top_skill_title"],
                        "score": result["top_skill_score"],
                        "occ_t": result["all_occupation_titles"],
                        "occ_u": result["all_occupation_uris"],
                    }
                )

                pct = (i + 1) / len(df_batch)
                batch_bar.progress(pct, text=f"{i + 1}/{len(df_batch)} in this batch")

            # ── Save batch to DB immediately ───────────────────────────────
            try:
                save_batch_to_db(batch_records)
                saved_note = f"✅ Batch {batch_num + 1} saved ({len(batch_records)} rows)"
            except Exception as db_err:
                saved_note = f"⚠️ Batch {batch_num + 1} DB save failed: {db_err}"
                errors.append(saved_note)

            errors.extend(batch_errors)

            # ── Update overall progress ────────────────────────────────────
            _, current_done = load_progress(sel_run_id, unit_filter)
            overall_pct = current_done / total_rows if total_rows else 1.0
            overall_bar.progress(
                overall_pct,
                text=f"Overall: {current_done}/{total_rows} matched ({overall_pct * 100:.1f}%)",
            )
            status_line.markdown(saved_note)

        # ── All done ───────────────────────────────────────────────────────
        _, final_done = load_progress(sel_run_id, unit_filter)
        overall_bar.progress(1.0 if final_done >= total_rows else final_done / total_rows,
                             text=f"Done — {final_done}/{total_rows} matched")

        if errors:
            with st.expander(f"⚠️ {len(errors)} error(s) — click to view"):
                for e in errors:
                    st.caption(e)
        else:
            st.success(
                f"✅ Batch run complete — **{final_done}/{total_rows}** statements matched. "
                "Refresh the page or click Run again to continue remaining rows."
            )

        st.rerun()   # Refresh metrics at top of page


# ─────────────────────────────────────────────────────────────────────────────
# Results display (always from DB)
# ─────────────────────────────────────────────────────────────────────────────

if done_rows > 0:
    st.divider()
    st.subheader("📊 Results")

    @st.cache_data(ttl=30, show_spinner=False)
    def load_results(run_id, unit_prefix):
        base_filter = " AND esco_skill_uri IS NOT NULL AND esco_skill_uri <> ''"
        params: dict = {}
        if run_id:
            base_filter += " AND run_id = :run_id"
            params["run_id"] = run_id
        if unit_prefix:
            base_filter += " AND unit_code ILIKE :up"
            params["up"] = unit_prefix.upper() + "%"
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT unit_code, element_title, skill_statement, "
                    f"esco_skill_uri, esco_skill_title, esco_skill_score, "
                    f"esco_occupation_titles, esco_occupation_uris "
                    f"FROM rsd_skill_records WHERE 1=1 {base_filter} "
                    f"ORDER BY unit_code, esco_skill_score DESC"
                ),
                params,
            ).mappings().all()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    df_results = load_results(sel_run_id, unit_filter)

    if not df_results.empty:
        # ── Occupation frequency chart ─────────────────────────────────────
        all_occ = []
        for v in df_results["esco_occupation_titles"].dropna():
            all_occ.extend([o.strip() for o in str(v).split("|") if o.strip()])
        if all_occ:
            from collections import Counter
            top_occ = pd.DataFrame(
                Counter(all_occ).most_common(20), columns=["Occupation", "Count"]
            )
            st.markdown("**Top 20 ESCO occupations across all matched skills**")
            st.bar_chart(top_occ.set_index("Occupation"))

        # ── Data table ────────────────────────────────────────────────────
        st.dataframe(
            df_results[
                ["unit_code", "skill_statement", "esco_skill_title",
                 "esco_skill_score", "esco_occupation_titles"]
            ],
            use_container_width=True,
            height=400,
        )

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown("**Download results**")
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇ ESCO-enriched CSV (all columns)",
            df_results.to_csv(index=False).encode(),
            "esco_enriched_rsd.csv",
            "text/csv",
            use_container_width=True,
        )
        occ_cols = ["unit_code", "element_title", "skill_statement",
                    "esco_skill_title", "esco_skill_score", "esco_occupation_titles"]
        dl2.download_button(
            "⬇ Occupation mapping CSV",
            df_results[[c for c in occ_cols if c in df_results.columns]]
            .to_csv(index=False)
            .encode(),
            "occupation_mapping.csv",
            "text/csv",
            use_container_width=True,
        )
