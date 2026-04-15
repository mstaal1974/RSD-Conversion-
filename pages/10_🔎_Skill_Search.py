"""
pages/7_🔎_Skill_Search.py

Semantic Skill Search
---------------------
Type any phrase or statement and find the most semantically similar
skill statements across all training packages in the database.

Features:
  - Free-text semantic search using OpenAI embeddings
  - Filter by training package, similarity threshold, min score
  - Results ranked by cosine similarity with score badges
  - Export matched results to CSV
  - Embeddings cached in session_state to avoid re-computing on every search
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

st.set_page_config(page_title="Skill Search", layout="wide", page_icon="🔎")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #07090f;
    color: #c9d1d9;
  }

  .search-hero {
    background: linear-gradient(160deg, #0d1117 0%, #0f1923 60%, #0a1628 100%);
    border: 1px solid #1c2a3a;
    border-radius: 14px;
    padding: 32px 36px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .search-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #06b6d4);
  }
  .search-hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #e2e8f0;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
  }
  .search-hero p {
    font-size: 0.9rem;
    color: #64748b;
    margin: 0;
  }

  .result-card {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
    position: relative;
  }
  .result-card:hover { border-color: #0ea5e9; }

  .result-card .stmt {
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.6;
    margin-bottom: 10px;
  }
  .result-card .meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
  .result-card .meta span { color: #94a3b8; }

  .score-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    float: right;
    margin-top: -2px;
  }
  .score-95  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .score-85  { background: #0c2340; color: #60a5fa; border: 1px solid #1e40af; }
  .score-75  { background: #2d1f0e; color: #fbbf24; border: 1px solid #92400e; }
  .score-low { background: #1c1917; color: #78716c; border: 1px solid #44403c; }

  .stat-box {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .stat-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #38bdf8;
    line-height: 1;
  }
  .stat-box .lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-top: 5px;
  }

  .section-rule {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #334155;
    border-bottom: 1px solid #1c2a3a;
    padding-bottom: 6px;
    margin: 28px 0 16px;
  }

  div[data-testid="stTextInput"] input {
    background: #0d1117 !important;
    border: 1px solid #1c2a3a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 28px !important;
    letter-spacing: 0.04em !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
    transform: translateY(-1px);
  }

  .cache-info {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #22c55e;
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 6px;
    padding: 6px 12px;
    display: inline-block;
    margin-bottom: 12px;
  }
  .no-results {
    text-align: center;
    padding: 60px 20px;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)


# ── DB + OpenAI setup ─────────────────────────────────────────────────────────
def _secret(key, default=""):
    try:
        return st.secrets.get(key, os.getenv(key, default)) or default
    except Exception:
        return os.getenv(key, default) or default

DB_URL     = _secret("DATABASE_URL")
OPENAI_KEY = _secret("OPENAI_API_KEY")

if not DB_URL:
    st.error("DATABASE_URL not configured.")
    st.stop()
if not OPENAI_KEY:
    st.error("OPENAI_API_KEY not configured.")
    st.stop()

@st.cache_resource
def get_engine(url: str):
    from sqlalchemy import create_engine
    return create_engine(url, pool_pre_ping=True)

@st.cache_resource
def get_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

engine = get_engine(DB_URL)
oai    = get_openai_client(OPENAI_KEY)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Loading skill statements from database…")
def load_statements() -> pd.DataFrame:
    """Load all skill statements with metadata from the DB.
    Introspects actual columns first so it never breaks on schema differences."""
    with engine.connect() as conn:
        # Find out what columns actually exist
        actual_cols = pd.read_sql(
            text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'rsd_skill_records'
            """),
            conn,
        )["column_name"].tolist()

    # Always-required columns
    wanted = ["id", "skill_statement", "unit_code", "unit_title", "tp_code", "tp_title"]
    # Optional — include only if they exist
    optional = ["qa_status", "element_name", "qa_score", "run_id"]
    wanted += [c for c in optional if c in actual_cols]
    # Only select columns that actually exist
    select_cols = [c for c in wanted if c in actual_cols]

    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"""
                SELECT {', '.join(select_cols)}
                FROM rsd_skill_records
                WHERE skill_statement IS NOT NULL
                  AND skill_statement <> ''
                ORDER BY tp_code, unit_code
            """),
            conn,
        )
    return df


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed texts using OpenAI text-embedding-3-small with rate-limit retries."""
    import time
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t.strip() or "empty" for t in texts[i : i + batch_size]]
        for attempt in range(5):
            try:
                resp = oai.embeddings.create(input=batch, model="text-embedding-3-small")
                all_embeddings.extend([item.embedding for item in resp.data])
                time.sleep(0.3)
                break
            except Exception as e:
                wait = 2 ** attempt
                st.warning(f"Batch {i//batch_size+1} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed after 5 attempts at batch {i}")
    return np.array(all_embeddings, dtype=np.float32)


def cosine_sim(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity of one query vector against a corpus matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = corpus_vecs / norms
    return (normed @ q).astype(float)


def score_class(score: float) -> str:
    if score >= 0.95: return "score-95"
    if score >= 0.85: return "score-85"
    if score >= 0.75: return "score-75"
    return "score-low"

def score_label(score: float) -> str:
    if score >= 0.95: return "Excellent match"
    if score >= 0.85: return "Strong match"
    if score >= 0.75: return "Good match"
    return "Partial match"


# ── Build / retrieve corpus embeddings ───────────────────────────────────────
def get_corpus_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Return embeddings for the full corpus, cached in session_state.
    Uses small batches with rate-limit protection and exponential backoff.
    """
    import time
    cache_key = "corpus_embeddings"
    count_key = "corpus_embedding_count"
    cached_n  = st.session_state.get(count_key, 0)

    if cache_key in st.session_state and cached_n == len(df):
        return st.session_state[cache_key]

    texts      = df["skill_statement"].tolist()
    n          = len(texts)
    batch_size = 100
    n_batches  = (n + batch_size - 1) // batch_size

    progress = st.progress(0, text=f"Building embeddings for {n:,} statements ({n_batches} batches)…")
    embeddings_list = []

    for i in range(0, n, batch_size):
        batch = [t.strip() or "empty" for t in texts[i : i + batch_size]]
        for attempt in range(5):
            try:
                resp = oai.embeddings.create(input=batch, model="text-embedding-3-small")
                embeddings_list.extend([item.embedding for item in resp.data])
                time.sleep(0.3)
                break
            except Exception as e:
                wait = 2 ** attempt
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed after 5 attempts at batch {i}")

        pct = min((i + batch_size) / n, 1.0)
        progress.progress(pct, text=f"Embedded {min(i+batch_size, n):,} / {n:,}…")

    progress.empty()
    emb = np.array(embeddings_list, dtype=np.float32)
    st.session_state[cache_key] = emb
    st.session_state[count_key] = n
    return emb



# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="search-hero">
  <h1>🔎 Semantic Skill Search</h1>
  <p>Search across all skill statements using natural language.
     Finds semantically similar statements even when the exact words differ.</p>
</div>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
df_all = load_statements()

if df_all.empty:
    st.warning("No skill statements found in the database. Run a batch first.")
    st.stop()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Search Settings")

    top_n = st.slider("Max results", 5, 100, 20, 5)

    min_score = st.slider(
        "Min similarity score",
        0.50, 0.99, 0.70, 0.01,
        help="Only show results with cosine similarity above this threshold."
    )

    st.divider()
    st.markdown("### 🗂️ Filter by Training Package")

    tp_options = sorted(df_all["tp_code"].dropna().unique().tolist())
    selected_tps = st.multiselect(
        "Training packages",
        options=tp_options,
        default=[],
        placeholder="All packages (leave blank)"
    )

    st.divider()
    st.markdown("### 🔄 Embedding Cache")

    cached_n = st.session_state.get("corpus_embedding_count", 0)
    if cached_n:
        st.markdown(
            f'<div class="cache-info">✓ {cached_n:,} statements cached</div>',
            unsafe_allow_html=True
        )
        if st.button("🗑 Clear cache", use_container_width=True):
            st.session_state.pop("corpus_embeddings", None)
            st.session_state.pop("corpus_embedding_count", None)
            st.rerun()
    else:
        st.caption("Embeddings will be built on first search and cached for the session.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Uses OpenAI `text-embedding-3-small` to encode your query and every "
        "skill statement into a 1536-dimension vector, then ranks by cosine similarity. "
        "The first search builds the corpus cache (~1-2 min for 48k statements)."
    )


# ── Stats row ─────────────────────────────────────────────────────────────────
df_filtered = df_all if not selected_tps else df_all[df_all["tp_code"].isin(selected_tps)]

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df_filtered):,}",                  "Statements in scope"),
    (c2, f"{df_filtered['unit_code'].nunique():,}", "Unique units"),
    (c3, f"{df_filtered['tp_code'].nunique():,}",  "Training packages"),
    (c4, f"{min_score:.0%}",                        "Min similarity"),
]:
    col.markdown(
        f'<div class="stat-box"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── Search input ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-rule">Search Query</div>', unsafe_allow_html=True)

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    query = st.text_input(
        "query",
        label_visibility="collapsed",
        placeholder='e.g.  "Identify communication requirements"  or just  "communication"',
        key="search_query",
    )
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

# Example chips
st.markdown(
    "<div style='font-size:0.78rem;color:#475569;margin-top:-8px;margin-bottom:20px;'>"
    "Try: &nbsp;"
    "<code>identify communication requirements</code> &nbsp;·&nbsp; "
    "<code>workplace health and safety</code> &nbsp;·&nbsp; "
    "<code>assess client needs</code> &nbsp;·&nbsp; "
    "<code>documentation</code>"
    "</div>",
    unsafe_allow_html=True,
)


# ── Search execution ──────────────────────────────────────────────────────────
if search_clicked and query.strip():

    with st.spinner("Building corpus embeddings (cached after first search)…"):
        corpus_emb = get_corpus_embeddings(df_all)

    # Filter corpus to selected TPs if needed
    if selected_tps:
        mask       = df_all["tp_code"].isin(selected_tps)
        df_search  = df_all[mask].reset_index(drop=True)
        search_emb = corpus_emb[mask.values]
    else:
        df_search  = df_all.reset_index(drop=True)
        search_emb = corpus_emb

    # Embed the query
    with st.spinner("Embedding query…"):
        q_vec = embed_texts([query.strip()])[0]

    # Compute similarities
    scores = cosine_sim(q_vec, search_emb)

    # Rank and threshold
    ranked_idx = np.argsort(scores)[::-1]
    ranked_idx = [i for i in ranked_idx if scores[i] >= min_score][:top_n]

    st.markdown(
        f'<div class="section-rule">Results — {len(ranked_idx)} matches '
        f'(similarity ≥ {min_score:.0%})</div>',
        unsafe_allow_html=True
    )

    if not ranked_idx:
        st.markdown(
            '<div class="no-results">'
            '⊘ No matches above the similarity threshold.<br>'
            '<small>Try lowering the min score or broadening your query.</small>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        results_rows = []

        for rank, idx in enumerate(ranked_idx, 1):
            row   = df_search.iloc[idx]
            score = float(scores[idx])
            sc    = score_class(score)
            sl    = score_label(score)

            tp_display  = f"{row.get('tp_code','') or ''} — {row.get('tp_title','') or ''}"
            unit_display = f"{row.get('unit_code','') or ''} {row.get('unit_title','') or ''}"
            elem_display = ""

            st.markdown(f"""
            <div class="result-card">
              <span class="score-badge {sc}">{score:.1%} · {sl}</span>
              <div class="stmt">{row['skill_statement']}</div>
              <div class="meta">
                <span>#{rank}</span>
                <span>📦 {tp_display}</span>
                <span>📄 {unit_display}</span>
                {'<span>⚡ ' + elem_display + '</span>' if elem_display else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)

            results_rows.append({
                "Rank":             rank,
                "Similarity":       round(score, 4),
                "Skill Statement":  row["skill_statement"],
                "Unit Code":        row.get("unit_code",""),
                "Unit Title":       row.get("unit_title",""),
                "TP Code":          row.get("tp_code",""),
                "TP Title":         row.get("tp_title",""),
                "QA Status":        row.get("qa_status",""),
            })

        # Export button
        if results_rows:
            st.markdown("<br>", unsafe_allow_html=True)
            results_df = pd.DataFrame(results_rows)
            st.download_button(
                label=f"⬇ Export {len(results_rows)} results to CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name=f"skill_search_{query[:30].replace(' ','_')}.csv",
                mime="text/csv",
            )

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")
