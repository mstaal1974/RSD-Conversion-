"""
rsd-convert — Training Package → Element-level Skill Statements (BART)

Key improvements over v1:
  • Multi-provider: OpenAI or Anthropic (or both with fallback)
  • Temperature correctly passed to the generator
  • Concurrent batch processing (ThreadPoolExecutor)
  • Per-element error handling + retry — a single API failure never kills a batch
  • XLSX + CSV upload support
  • Content-hash fingerprinting (not just shape)
  • Downloads always available via session-state fallback when DB is absent
  • Inline st.data_editor editing before download
  • Session-scoped run IDs — users can only resume their own runs
"""
from __future__ import annotations
import io
import os
import uuid
import hashlib
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.extractor import normalize_training_package_csv, build_registry, content_fingerprint
from core.bart_generator import generate_skill_statement
from core.keyword_generator import generate_keywords
from core.exporters import to_rsd_rows, to_traceability
from core.providers import OpenAIProvider, AnthropicProvider, with_fallback
from core.providers.openai_provider import OpenAIProvider as _OAI
from core.providers.anthropic_provider import AnthropicProvider as _ANT

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RSD Convert — BART Skill Statements", layout="wide")
st.title("Training Package → Element-level Skill Statements (BART)")

# ── Secrets / env ────────────────────────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

OPENAI_KEY  = _secret("OPENAI_API_KEY")
ANTHROPIC_KEY = _secret("ANTHROPIC_API_KEY")
DB_URL      = _secret("DATABASE_URL")


# ── Session initialisation ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "session_token": str(uuid.uuid4()),   # unique per browser session
        "norm_df": None,
        "extractor_used": None,
        "scorecard": None,
        "last_fingerprint": None,
        "run_id": None,
        "next_index_ui": 0,
        "batch_results_df": None,   # in-memory fallback when DB absent
        "all_results_df": None,     # accumulated across batches in-memory
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── DB setup ─────────────────────────────────────────────────────────────────
engine = None
db_ready = False
db_error = None

if DB_URL:
    try:
        from core.db import (
            get_engine, init_db, create_run, validate_run_owner,
            upsert_skill_records, get_next_index, update_run_status,
            fetch_run_records,
        )
        engine = get_engine(DB_URL)
        init_db(engine)
        db_ready = True
    except Exception as exc:
        db_error = str(exc)
        db_ready = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Provider")

    provider_choice = st.radio(
        "LLM provider",
        ["OpenAI", "Anthropic", "OpenAI → Anthropic fallback"],
        index=0,
    )

    if provider_choice in ("OpenAI", "OpenAI → Anthropic fallback"):
        from core.providers.openai_provider import _DEFAULT_MODELS as _oai_models
        default_model = _oai_models[1]  # gpt-4.1-mini
    else:
        from core.providers.anthropic_provider import _DEFAULT_MODELS as _ant_models
        default_model = _ant_models[0]  # claude-sonnet-4-6

    model = st.text_input("Model", value=_secret("OPENAI_MODEL", default_model))

    st.divider()
    st.header("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_fixes   = st.slider("Max auto-fixes per element", 0, 3, 1)
    max_workers = st.slider("Concurrent workers", 1, 16, 5)
    st.caption("Higher workers = faster batches; watch your API rate limits.")

    st.divider()
    st.header("Batching / Resume")
    batch_size = st.number_input("Batch size (elements)", 10, 1000, 100, 10)

    st.divider()
    st.header("Keywords")
    generate_kw = st.toggle("Generate AI keywords per skill", value=False)

    st.divider()
    st.header("Extractor")
    reg = build_registry()
    extractor_mode = st.radio("Mode", ["Auto-detect", "Choose"], index=0)
    forced = None
    if extractor_mode == "Choose":
        forced = st.selectbox("Extractor", reg.list_names())

    st.divider()
    st.header("Database")
    if not DB_URL:
        st.warning("DATABASE_URL not set — results stored in session only.")
    elif not db_ready:
        st.error("DB connection failed.")
        st.code(db_error)
    else:
        st.success("DB connected ✅")
        resume_run_id = st.text_input("Resume run ID (optional)")
        use_resume = st.toggle("Resume from DB", value=False)

    st.divider()
    colA, colB = st.columns(2)
    run_batch  = colA.button("▶ Run next batch", type="primary")
    reset_run  = colB.button("↺ Reset run")

if reset_run:
    st.session_state.update({
        "run_id": None,
        "next_index_ui": 0,
        "batch_results_df": None,
        "all_results_df": None,
    })
    st.success("Run state reset.")


# ── Provider factory ──────────────────────────────────────────────────────────
@st.cache_resource
def _make_provider(choice: str, oai_key: str, ant_key: str):
    if choice == "OpenAI":
        if not oai_key:
            return None, "OPENAI_API_KEY not set"
        return _OAI(oai_key), None
    if choice == "Anthropic":
        if not ant_key:
            return None, "ANTHROPIC_API_KEY not set"
        return _ANT(ant_key), None
    # Fallback
    if not oai_key or not ant_key:
        return None, "Both OPENAI_API_KEY and ANTHROPIC_API_KEY required for fallback mode"
    return with_fallback(_OAI(oai_key), _ANT(ant_key)), None

provider, provider_err = _make_provider(provider_choice, OPENAI_KEY, ANTHROPIC_KEY)


# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload training package (CSV or XLSX)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a training package file to begin.")
    st.stop()

@st.cache_data(show_spinner="Reading file…")
def _read_upload(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    if name.endswith(".xlsx"):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

raw_df = _read_upload(uploaded.name, uploaded.getvalue())
fp = content_fingerprint(raw_df)  # MD5 of actual cell data

# Reset on new/changed file
if st.session_state["last_fingerprint"] != fp:
    st.session_state.update({
        "last_fingerprint": fp,
        "norm_df": None,
        "extractor_used": None,
        "scorecard": None,
        "run_id": None,
        "next_index_ui": 0,
        "batch_results_df": None,
        "all_results_df": None,
    })

with st.expander("Preview raw data (first 20 rows)"):
    st.dataframe(raw_df.head(20), use_container_width=True)


# ── Normalise ─────────────────────────────────────────────────────────────────
if st.session_state["norm_df"] is None:
    try:
        with st.spinner("Detecting format and normalising…"):
            norm_df, ext_used, scorecard = normalize_training_package_csv(raw_df, forced)
        st.session_state.update({
            "norm_df": norm_df,
            "extractor_used": ext_used,
            "scorecard": scorecard,
        })
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        st.stop()

norm_df      = st.session_state["norm_df"]
extractor_used = st.session_state["extractor_used"]
scorecard    = st.session_state["scorecard"]

st.success(f"Extractor: **{extractor_used}** — {len(norm_df)} elements detected")

if scorecard is not None:
    with st.expander("Extractor scorecard"):
        st.dataframe(scorecard, use_container_width=True)

with st.expander("Normalised data preview"):
    st.dataframe(norm_df.head(30), use_container_width=True)
    # Diagnostics
    import re
    bad = norm_df[norm_df["element_title"].apply(lambda v: bool(re.search(r"\b\d+\.\d+\b", str(v))))]
    if len(bad):
        st.warning(f"⚠️ {len(bad)} element titles contain PC tokens — check extractor output.")
    else:
        st.caption("✅ Element titles are clean (no PC tokens).")

total = len(norm_df)
if total == 0:
    st.error("No elements found after normalisation.")
    st.stop()


# ── Run / resume tracking ─────────────────────────────────────────────────────
if db_ready:
    # Resume logic
    if "use_resume" in dir() and use_resume and resume_run_id.strip():
        candidate = resume_run_id.strip()
        if validate_run_owner(engine, candidate, st.session_state["session_token"]):
            st.session_state["run_id"] = candidate
        else:
            st.error("Run ID not found or not owned by this session.")

    # Create run record on first batch
    if st.session_state["run_id"] is None and run_batch:
        settings = dict(
            batch_size=int(batch_size),
            max_fixes=int(max_fixes),
            temperature=float(temperature),
            max_workers=int(max_workers),
            generate_keywords=bool(generate_kw),
        )
        st.session_state["run_id"] = create_run(
            engine,
            session_token=st.session_state["session_token"],
            source_filename=uploaded.name,
            source_fingerprint=fp,
            extractor_name=extractor_used,
            extractor_version="1.1.0",
            sil_version="1.0.0",
            model=model,
            provider=provider_choice,
            settings=settings,
        )

run_id = st.session_state.get("run_id")

# Compute start index
if db_ready and run_id:
    start_index = int(get_next_index(engine, run_id))
else:
    start_index = int(st.session_state["next_index_ui"])

end_index = min(total, start_index + int(batch_size))

st.info(
    f"Batch: rows **{start_index} → {end_index - 1}** of {total} "
    f"| Session: `{st.session_state['session_token'][:8]}…`"
    + (f" | Run: `{run_id[:8]}…`" if run_id else "")
)

# Partial downloads from DB
if db_ready and run_id:
    with st.expander("Stored results so far (DB)"):
        db_df = fetch_run_records(engine, run_id)
        if len(db_df):
            st.dataframe(db_df[["row_index","unit_code","element_title","qa_passes"]].head(50), use_container_width=True)
            _rsd = to_rsd_rows(db_df)
            _tr  = to_traceability(db_df)
            c1, c2 = st.columns(2)
            c1.download_button("⬇ Partial RSD CSV", _rsd.to_csv(index=False).encode(), "rsd_partial.csv", "text/csv")
            c2.download_button("⬇ Partial traceability CSV", _tr.to_csv(index=False).encode(), "traceability_partial.csv", "text/csv")
        else:
            st.caption("No stored rows yet.")


# ── Run batch ─────────────────────────────────────────────────────────────────
if not run_batch:
    # Show any accumulated in-memory results even without running
    _all = st.session_state.get("all_results_df")
    if _all is not None and len(_all):
        st.subheader("Accumulated results (in-session)")
        _show_cols = ["unit_code", "element_title", "skill_statement", "qa_passes"]
        edited = st.data_editor(_all[[c for c in _show_cols if c in _all.columns]], use_container_width=True, num_rows="fixed", key="editor_idle")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ RSD CSV", to_rsd_rows(_all).to_csv(index=False).encode(), "rsd_output.csv", "text/csv")
        c2.download_button("⬇ Traceability CSV", to_traceability(_all).to_csv(index=False).encode(), "traceability.csv", "text/csv")
    st.stop()

if provider_err:
    st.error(f"Provider error: {provider_err}")
    st.stop()

if start_index >= total:
    st.success("All elements already processed ✅")
    if db_ready and run_id:
        update_run_status(engine, run_id, "completed")
    st.stop()

batch_df = norm_df.iloc[start_index:end_index].reset_index(drop=True)

st.subheader(f"Running batch ({len(batch_df)} elements, {max_workers} workers)")
progress_bar = st.progress(0)
status_text  = st.empty()
error_box    = st.empty()

completed_count = 0
completed_lock  = threading.Lock()
errors: list[str] = []


def _process_row(idx: int, row: pd.Series) -> dict:
    """Worker function — runs in thread pool."""
    result = dict(
        _idx=idx,
        unit_code=str(row.get("unit_code", "")),
        unit_title=str(row.get("unit_title", "")),
        element_title=str(row.get("element_title", "")),
        pcs_text=str(row.get("pcs_text", "")),
        skill_statement="",
        bart_prompt="",
        qa_one_sentence=False,
        qa_word_count=0,
        qa_has_method=False,
        qa_has_outcome=False,
        qa_passes=False,
        rewrite_count=0,
        bart_model=model,
        bart_temperature=float(temperature),
        keywords="",
        error_message="",
    )
    try:
        skill, qa, bart_prompt = generate_skill_statement(
            provider=provider,
            model=model,
            unit_code=result["unit_code"],
            unit_title=result["unit_title"],
            element_title=result["element_title"],
            pcs_text=result["pcs_text"],
            max_fixes=int(max_fixes),
            temperature=float(temperature),
        )
        result.update(
            skill_statement=skill,
            bart_prompt=bart_prompt,
            qa_one_sentence=bool(qa.get("one_sentence")),
            qa_word_count=int(qa.get("word_count", 0)),
            qa_has_method=bool(qa.get("has_method_phrase")),
            qa_has_outcome=bool(qa.get("has_outcome_phrase")),
            qa_passes=bool(qa.get("passes")),
            rewrite_count=int(qa.get("rewrite_count", 0)),
        )
        if generate_kw:
            result["keywords"] = generate_keywords(
                provider=provider,
                model=model,
                skill_statement=skill,
                pcs_text=result["pcs_text"],
                temperature=float(temperature),
            )
    except Exception as exc:
        result["skill_statement"] = f"ERROR: {exc}"
        result["error_message"] = str(exc)
        logging.exception("Error processing row %s", idx)
    return result


# Submit all futures
futures = {}
with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
    for i, (_, row) in enumerate(batch_df.iterrows()):
        futures[executor.submit(_process_row, i, row)] = i

    results_by_idx: dict[int, dict] = {}
    for future in as_completed(futures):
        res = future.result()
        results_by_idx[res["_idx"]] = res
        with completed_lock:
            completed_count += 1
            cnt = completed_count
        if res["error_message"]:
            errors.append(f"Row {start_index + res['_idx']}: {res['error_message']}")
        progress_bar.progress(cnt / len(batch_df))
        status_text.write(f"Completed {cnt}/{len(batch_df)}…")

status_text.write(f"Batch complete ✅  ({len(errors)} errors)")
if errors:
    error_box.warning("Some rows had errors:\n" + "\n".join(errors[:10]))

# Rebuild ordered DataFrame
ordered = [results_by_idx[i] for i in range(len(batch_df))]
result_df = pd.DataFrame(ordered).drop(columns=["_idx"])


# ── Persist to DB (if available) ───────────────────────────────────────────────
if db_ready and run_id:
    try:
        upsert_skill_records(engine, run_id, result_df, start_index)
        st.success("Batch saved to DB ✅")
        new_next = int(get_next_index(engine, run_id))
        st.session_state["next_index_ui"] = new_next
        if new_next >= total:
            update_run_status(engine, run_id, "completed")
        else:
            update_run_status(engine, run_id, "running")
    except Exception as exc:
        st.error(f"DB write failed: {exc} — results are still available in-session below.")
else:
    st.session_state["next_index_ui"] = end_index
    if not db_ready:
        st.warning("DB not configured — results available for download this session only.")

# Accumulate in-memory across batches
prev = st.session_state.get("all_results_df")
if prev is not None and len(prev):
    st.session_state["all_results_df"] = pd.concat([prev, result_df], ignore_index=True)
else:
    st.session_state["all_results_df"] = result_df

all_results = st.session_state["all_results_df"]


# ── Results view + inline editor ───────────────────────────────────────────────
st.subheader("Batch results — edit before downloading")

_show = ["unit_code", "unit_title", "element_title", "skill_statement", "qa_passes", "rewrite_count"]
if generate_kw:
    _show.append("keywords")
_show = [c for c in _show if c in result_df.columns]

edited_df = st.data_editor(
    result_df[_show],
    use_container_width=True,
    num_rows="fixed",
    column_config={"qa_passes": st.column_config.CheckboxColumn("QA ✓")},
    key="editor_batch",
)

# Merge edits back
for col in ["skill_statement", "keywords"]:
    if col in edited_df.columns:
        result_df[col] = edited_df[col]


# ── Downloads ─────────────────────────────────────────────────────────────────
st.subheader("Downloads")

# Prefer full DB set if available, else accumulated in-memory
if db_ready and run_id:
    _full_df = fetch_run_records(engine, run_id)
    done = len(_full_df) >= total
else:
    _full_df = all_results
    done = int(st.session_state["next_index_ui"]) >= total

rsd_df   = to_rsd_rows(_full_df)
trace_df = to_traceability(_full_df)
suffix   = "" if done else "_in_progress"

c1, c2 = st.columns(2)
c1.download_button(
    f"⬇ RSD output CSV{'  ✅' if done else '  (partial)'}",
    rsd_df.to_csv(index=False).encode(),
    f"rsd_output{suffix}.csv",
    "text/csv",
)
c2.download_button(
    f"⬇ Traceability CSV{'  ✅' if done else '  (partial)'}",
    trace_df.to_csv(index=False).encode(),
    f"traceability{suffix}.csv",
    "text/csv",
)

if done:
    st.success(f"All {total} elements processed ✅")
else:
    remaining = total - int(st.session_state.get("next_index_ui", end_index))
    st.info(f"{remaining} elements remaining — click **▶ Run next batch** to continue.")
