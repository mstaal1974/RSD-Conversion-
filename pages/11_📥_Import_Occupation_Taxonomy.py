"""
pages/7_📥_Import_Occupation_Taxonomy.py

Import qualification → occupation taxonomy from Excel into the DB.

Parses comma-separated Industry Sector and Occupation fields into
individual rows in qual_taxonomy_links so the linkage engine can use them.

Expected Excel columns (header on row 2):
  Qualification Code | Qualification Name | ANZSCO Identifier |
  Taxonomy - Industry Sector | Taxonomy - Occupation
"""
from __future__ import annotations

import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
st.set_page_config(page_title="Import Occupation Taxonomy", layout="wide")

st.title("📥 Import Occupation Taxonomy")
st.caption(
    "Load qualification occupation taxonomy from Excel into the database. "
    "Comma-separated Industry Sectors and Occupations are split into individual rows."
)

# ─────────────────────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────────────────────

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


def ensure_tables():
    """
    Create or migrate qual_registry and qual_taxonomy_links.
    Each statement runs in its OWN transaction so a failure on one
    (e.g. table already exists with a different schema) cannot poison
    the others.
    """
    ddl_statements = [
        # Create qual_registry if missing
        """
        CREATE TABLE IF NOT EXISTS qual_registry (
            qual_code   TEXT PRIMARY KEY,
            qual_title  TEXT,
            status      TEXT DEFAULT 'Current',
            updated_at  TIMESTAMP DEFAULT NOW()
        )
        """,
        # Add updated_at if table existed without it
        "ALTER TABLE qual_registry ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
        # Create qual_taxonomy_links — no FK constraint so it works even if
        # qual_registry was just created in this same boot
        """
        CREATE TABLE IF NOT EXISTS qual_taxonomy_links (
            id        SERIAL PRIMARY KEY,
            qual_code TEXT NOT NULL,
            scheme    TEXT NOT NULL,
            value     TEXT NOT NULL,
            UNIQUE (qual_code, scheme, value)
        )
        """,
        # Add any columns that may be missing on an older version of the table
        "ALTER TABLE qual_taxonomy_links ADD COLUMN IF NOT EXISTS scheme TEXT",
        "ALTER TABLE qual_taxonomy_links ADD COLUMN IF NOT EXISTS value  TEXT",
        # Drop the wrong single-column unique constraint if it exists
        # (old schema had UNIQUE(qual_code) which blocks multiple rows per qual)
        """
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'qual_taxonomy_links_qual_code_key'
                  AND conrelid = 'qual_taxonomy_links'::regclass
            ) THEN
                ALTER TABLE qual_taxonomy_links
                DROP CONSTRAINT qual_taxonomy_links_qual_code_key;
            END IF;
        END $$
        """,
        # Add the correct compound unique constraint if not already present
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'qual_taxonomy_links_qual_code_scheme_value_key'
                  AND conrelid = 'qual_taxonomy_links'::regclass
            ) THEN
                ALTER TABLE qual_taxonomy_links
                ADD CONSTRAINT qual_taxonomy_links_qual_code_scheme_value_key
                UNIQUE (qual_code, scheme, value);
            END IF;
        END $$
        """,
        # Indexes — safe to re-run
        "CREATE INDEX IF NOT EXISTS idx_qtl_qual_code ON qual_taxonomy_links (qual_code)",
        "CREATE INDEX IF NOT EXISTS idx_qtl_scheme    ON qual_taxonomy_links (scheme)",
    ]
    for sql in ddl_statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql.strip()))
        except Exception:
            pass  # already exists or column already present — safe to ignore


ensure_tables()


# ─────────────────────────────────────────────────────────────────────────────
# Parse helper
# ─────────────────────────────────────────────────────────────────────────────

def parse_excel(uploaded) -> pd.DataFrame:
    """
    Read the Excel, normalise columns, expand comma-separated taxonomy fields.
    Returns a flat DataFrame with one taxonomy value per row.
    """
    raw = pd.read_excel(uploaded, header=1)

    # Normalise column names
    raw.columns = [c.strip() for c in raw.columns]
    col_map = {
        "Qualification Code": "qual_code",
        "Qualification Code ": "qual_code",
        "Qualification Name": "qual_title",
        "Qualification Name ": "qual_title",
        "ANZSCO Identifier": "anzsco",
        "Taxonomy - Industry Sector": "industry_sector",
        "Taxonomy - Occupation": "occupation",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    raw = raw.dropna(subset=["qual_code"])
    raw["qual_code"] = raw["qual_code"].astype(str).str.strip()
    raw["qual_title"] = raw.get("qual_title", pd.Series("", index=raw.index)).fillna("").astype(str).str.strip()

    # Convert ANZSCO to clean string (drop .0)
    def clean_anzsco(v):
        if pd.isna(v):
            return ""
        try:
            return str(int(float(v)))
        except Exception:
            return str(v).strip()

    raw["anzsco"] = raw.get("anzsco", pd.Series("", index=raw.index)).apply(clean_anzsco)

    rows = []

    for _, r in raw.iterrows():
        qual_code  = r["qual_code"]
        qual_title = r.get("qual_title", "")

        # ── ANZSCO ────────────────────────────────────────────────────────
        anzsco = r.get("anzsco", "")
        if anzsco:
            rows.append({
                "qual_code":  qual_code,
                "qual_title": qual_title,
                "scheme":     "ANZSCO Identifier",
                "value":      anzsco,
            })

        # ── Industry Sector (split on comma) ──────────────────────────────
        industry_raw = str(r.get("industry_sector", "") or "")
        for sector in [s.strip() for s in industry_raw.split(",") if s.strip()]:
            rows.append({
                "qual_code":  qual_code,
                "qual_title": qual_title,
                "scheme":     "Taxonomy-Industry Sector",
                "value":      sector,
            })

        # ── Occupation (split on comma) ────────────────────────────────────
        occ_raw = str(r.get("occupation", "") or "")
        for occ in [o.strip() for o in occ_raw.split(",") if o.strip()]:
            if occ.lower() in ("no specific job role", "entry level and support animal carer roles"):
                continue  # skip generic placeholders
            rows.append({
                "qual_code":  qual_code,
                "qual_title": qual_title,
                "scheme":     "Taxonomy-Occupation",
                "value":      occ,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Current DB stats
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def db_stats():
    with engine.connect() as conn:
        quals   = conn.execute(text("SELECT COUNT(*) FROM qual_registry")).scalar() or 0
        links   = conn.execute(text("SELECT COUNT(*) FROM qual_taxonomy_links")).scalar() or 0
        schemes = conn.execute(
            text("SELECT scheme, COUNT(*) as n FROM qual_taxonomy_links GROUP BY scheme ORDER BY n DESC")
        ).mappings().all()
    return int(quals), int(links), [dict(r) for r in schemes]


n_quals, n_links, schemes = db_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Quals in DB", n_quals)
col2.metric("Taxonomy links in DB", n_links)
col3.metric("Scheme types", len(schemes))

if schemes:
    with st.expander("Current scheme breakdown"):
        for s in schemes:
            st.caption(f"{s['scheme']}: {s['n']:,} rows")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload Occupation Taxonomy Excel",
    type=["xlsx", "xls"],
    help="Expects columns: Qualification Code, Qualification Name, "
         "ANZSCO Identifier, Taxonomy - Industry Sector, Taxonomy - Occupation",
)

if uploaded:
    with st.spinner("Parsing Excel…"):
        try:
            df = parse_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to parse Excel: {e}")
            st.stop()

    # ── Preview ───────────────────────────────────────────────────────────
    qual_count = df["qual_code"].nunique()
    st.success(
        f"Parsed **{qual_count:,}** qualifications → "
        f"**{len(df):,}** taxonomy rows "
        f"({df[df.scheme=='ANZSCO Identifier'].shape[0]:,} ANZSCO, "
        f"{df[df.scheme=='Taxonomy-Industry Sector'].shape[0]:,} industry sectors, "
        f"{df[df.scheme=='Taxonomy-Occupation'].shape[0]:,} occupations)"
    )

    # Show per-scheme breakdown
    scheme_counts = df.groupby("scheme").size().reset_index(name="rows")
    st.dataframe(scheme_counts, use_container_width=False, hide_index=True)

    st.markdown("**Sample rows (first 20)**")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.divider()

    # ── Import button ─────────────────────────────────────────────────────
    col_a, col_b = st.columns([2, 1])

    with col_a:
        mode = st.radio(
            "Import mode",
            ["Upsert (add new, update existing)", "Replace all (delete first, then insert)"],
            help="Upsert is safe for incremental updates. Replace clears all existing taxonomy data first.",
        )

    with col_b:
        st.write("")
        st.write("")
        do_import = st.button("⬆ Import to database", type="primary", use_container_width=True)

    if do_import:
        replace = "Replace" in mode
        progress = st.progress(0, text="Starting import…")
        status = st.empty()

        try:
            with engine.begin() as conn:
                if replace:
                    conn.execute(text("DELETE FROM qual_taxonomy_links"))
                    conn.execute(text("DELETE FROM qual_registry"))
                    status.info("Cleared existing taxonomy data.")

                # ── Upsert qual_registry ─────────────────────────────────
                qual_rows = (
                    df[["qual_code", "qual_title"]]
                    .drop_duplicates("qual_code")
                    .to_dict("records")
                )
                progress.progress(0.1, text=f"Upserting {len(qual_rows):,} qualifications…")

                for i, qr in enumerate(qual_rows):
                    conn.execute(
                        text("""
                            INSERT INTO qual_registry (qual_code, qual_title, updated_at)
                            VALUES (:qual_code, :qual_title, NOW())
                            ON CONFLICT (qual_code) DO UPDATE
                            SET qual_title = EXCLUDED.qual_title,
                                updated_at = NOW()
                        """),
                        qr,
                    )
                    if i % 100 == 0:
                        progress.progress(
                            0.1 + 0.3 * (i / len(qual_rows)),
                            text=f"Qualifications: {i}/{len(qual_rows)}…"
                        )

                # ── Upsert qual_taxonomy_links ───────────────────────────
                link_rows = df[["qual_code", "scheme", "value"]].to_dict("records")
                progress.progress(0.4, text=f"Inserting {len(link_rows):,} taxonomy links…")

                inserted = 0
                skipped = 0
                for i, lr in enumerate(link_rows):
                    result = conn.execute(
                        text("""
                            INSERT INTO qual_taxonomy_links (qual_code, scheme, value)
                            SELECT :qual_code, :scheme, :value
                            WHERE NOT EXISTS (
                                SELECT 1 FROM qual_taxonomy_links
                                WHERE qual_code = :qual_code
                                  AND scheme    = :scheme
                                  AND value     = :value
                            )
                        """),
                        lr,
                    )
                    if result.rowcount:
                        inserted += 1
                    else:
                        skipped += 1

                    if i % 200 == 0:
                        progress.progress(
                            0.4 + 0.55 * (i / len(link_rows)),
                            text=f"Links: {i}/{len(link_rows)} ({inserted:,} new, {skipped:,} existing)…"
                        )

            progress.progress(1.0, text="Import complete ✅")
            st.success(
                f"✅ Import done — "
                f"**{len(qual_rows):,}** qualifications and "
                f"**{inserted:,}** taxonomy links saved "
                f"({skipped:,} already existed, skipped)."
            )
            st.cache_data.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Import failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Browse existing data
# ─────────────────────────────────────────────────────────────────────────────

if n_links > 0:
    st.divider()
    st.subheader("🔍 Browse taxonomy")

    search = st.text_input("Search by qualification code or occupation", placeholder="e.g. BSB50420 or Accountant")

    if search:
        @st.cache_data(ttl=30, show_spinner=False)
        def search_taxonomy(q):
            with engine.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT r.qual_code, r.qual_title, t.scheme, t.value
                        FROM qual_taxonomy_links t
                        JOIN qual_registry r ON r.qual_code = t.qual_code
                        WHERE r.qual_code ILIKE :q
                           OR t.value ILIKE :q
                        ORDER BY r.qual_code, t.scheme, t.value
                        LIMIT 200
                    """),
                    {"q": f"%{q}%"},
                ).mappings().all()
            return pd.DataFrame([dict(r) for r in rows])

        results = search_taxonomy(search)
        if results.empty:
            st.info("No results found.")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)
