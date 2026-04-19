"""
load_tga_scrape.py
==================
Loads the output of tga_scraper.py (tga_qualifications_updated.xlsx) into
the PostgreSQL database, populating:

  qual_taxonomy_links    — ANZSCO / Industry Sector / Occupation per qualification
  uoc_qual_memberships   — which units are core/elective in which qualification
  uoc_occupation_links   — occupation links inherited by unit from its qualifications

Run from the project root:
    python load_tga_scrape.py

Or with a custom file path:
    python load_tga_scrape.py --input tga_qualifications_updated.xlsx
"""
from __future__ import annotations
import argparse
import os
import sys
import logging

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── DB connection ─────────────────────────────────────────────────────────────

def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        # Try reading from .streamlit/secrets.toml
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            url = secrets.get("DATABASE_URL")
        except Exception:
            pass
    if not url:
        sys.exit("❌  DATABASE_URL not found. Set it as an env var or in .streamlit/secrets.toml")
    return create_engine(url, pool_pre_ping=True)


# ── Schema helpers ────────────────────────────────────────────────────────────

def ensure_tables(engine):
    """
    Create or migrate all taxonomy tables to match linkage_engine.py expectations.
    Each statement in its own transaction so failures don't poison others.
    """
    statements = [
        # qual_registry — one row per qualification
        """CREATE TABLE IF NOT EXISTS qual_registry (
            qual_code   TEXT PRIMARY KEY,
            qual_title  TEXT,
            status      TEXT DEFAULT 'Current',
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        )""",

        # qual_taxonomy_links — scheme/value rows (many per qual)
        # Drop old single-column unique constraint if present
        """DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'qual_taxonomy_links_qual_code_key'
                  AND conrelid = 'qual_taxonomy_links'::regclass
            ) THEN
                ALTER TABLE qual_taxonomy_links DROP CONSTRAINT qual_taxonomy_links_qual_code_key;
            END IF;
        END $$""",

        """CREATE TABLE IF NOT EXISTS qual_taxonomy_links (
            id        BIGSERIAL PRIMARY KEY,
            qual_code TEXT NOT NULL,
            scheme    TEXT NOT NULL,
            value     TEXT NOT NULL,
            UNIQUE (qual_code, scheme, value)
        )""",

        "ALTER TABLE qual_taxonomy_links ADD COLUMN IF NOT EXISTS scheme TEXT",
        "ALTER TABLE qual_taxonomy_links ADD COLUMN IF NOT EXISTS value  TEXT",

        # Add compound unique constraint if missing
        """DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'qual_taxonomy_links_qual_code_scheme_value_key'
                  AND conrelid = 'qual_taxonomy_links'::regclass
            ) THEN
                ALTER TABLE qual_taxonomy_links
                ADD CONSTRAINT qual_taxonomy_links_qual_code_scheme_value_key
                UNIQUE (qual_code, scheme, value);
            END IF;
        END $$""",

        # uoc_qual_memberships — linkage engine uses uoc_code/is_imported/owner_tp_code
        """CREATE TABLE IF NOT EXISTS uoc_qual_memberships (
            id              BIGSERIAL PRIMARY KEY,
            uoc_code        TEXT NOT NULL,
            qual_code       TEXT NOT NULL,
            membership_type TEXT NOT NULL,
            is_imported     BOOLEAN DEFAULT FALSE,
            owner_tp_code   TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (uoc_code, qual_code, membership_type)
        )""",

        # Migrate old column names if table existed with wrong schema
        "ALTER TABLE uoc_qual_memberships RENAME COLUMN unit_code TO uoc_code",
        "ALTER TABLE uoc_qual_memberships RENAME COLUMN is_native TO is_imported",
        "ALTER TABLE uoc_qual_memberships ADD COLUMN IF NOT EXISTS owner_tp_code TEXT",

        # uoc_occupation_links — must have valid_to, pipeline_run_id, source_qual_code
        """CREATE TABLE IF NOT EXISTS uoc_occupation_links (
            id                  BIGSERIAL PRIMARY KEY,
            uoc_code            TEXT NOT NULL,
            anzsco_code         TEXT NOT NULL DEFAULT '',
            anzsco_title        TEXT NOT NULL DEFAULT '',
            anzsco_major_group  TEXT NOT NULL DEFAULT '',
            asced_code          TEXT,
            asced_title         TEXT,
            industry_sector     TEXT,
            occupation_titles   TEXT,
            confidence          NUMERIC(5,3) NOT NULL,
            mapping_source      TEXT NOT NULL,
            source_qual_code    TEXT,
            source_asc_task_id  TEXT,
            is_primary          BOOLEAN DEFAULT FALSE,
            is_imported         BOOLEAN DEFAULT FALSE,
            owner_tp_code       TEXT,
            home_tp_title       TEXT,
            anzsco_uri          TEXT DEFAULT '',
            vc_context          TEXT DEFAULT 'https://www.w3.org/2018/credentials/v1',
            vc_type             TEXT DEFAULT 'TaxonomicAlignment',
            aqf_level           TEXT,
            skill_level_label   TEXT,
            valid_from          TIMESTAMPTZ DEFAULT NOW(),
            valid_to            TIMESTAMPTZ,
            pipeline_run_id     BIGINT,
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )""",

        # Add any columns missing from older versions
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS pipeline_run_id BIGINT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS source_qual_code TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS source_asc_task_id TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS is_imported BOOLEAN DEFAULT FALSE",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS owner_tp_code TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS home_tp_title TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS anzsco_uri TEXT DEFAULT ''",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS vc_context TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS vc_type TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS aqf_level TEXT",
        "ALTER TABLE uoc_occupation_links ADD COLUMN IF NOT EXISTS skill_level_label TEXT",

        # Rename old qual_code column to source_qual_code if needed
        "ALTER TABLE uoc_occupation_links RENAME COLUMN qual_code TO source_qual_code",

        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_uol_uoc_code ON uoc_occupation_links (uoc_code)",
        "CREATE INDEX IF NOT EXISTS idx_uol_valid ON uoc_occupation_links (uoc_code, is_primary, valid_to)",
        "CREATE INDEX IF NOT EXISTS idx_qtl_qual ON qual_taxonomy_links (qual_code)",
        "CREATE INDEX IF NOT EXISTS idx_uqm_uoc ON uoc_qual_memberships (uoc_code)",
    ]

    for sql in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql.strip()))
        except Exception:
            pass  # column already exists, constraint already present, etc.

        # Index for fast lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uoc_occ_links_uoc_code
            ON uoc_occupation_links (uoc_code)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uoc_qual_unit
            ON uoc_qual_memberships (unit_code)
        """))

    log.info("Tables verified / created.")


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_unit_codes(raw: str) -> list[str]:
    """Parse a pipe-separated list of unit codes, stripping blanks."""
    if not raw or pd.isna(raw):
        return []
    return [c.strip() for c in str(raw).split("|") if c.strip()]


# ── Main loaders ──────────────────────────────────────────────────────────────


<?xml version='1.0' encoding='UTF-8'?><Error><Code>AccessDenied</Code><Message>Access denied.</Message><Details>Anonymous caller does not have storage.objects.list access to the Google Cloud Storage bucket. Permission 'storage.objects.list' denied on resource (or it may not exist).</Details></Error>
        return 0

    with engine.begin() as conn:
        for qr in qual_rows:
            conn.execute(text("""
                INSERT INTO qual_registry (qual_code, qual_title, status, updated_at)
                VALUES (:qual_code, :qual_title, 'Current', NOW())
                ON CONFLICT (qual_code) DO UPDATE
                SET qual_title = EXCLUDED.qual_title, updated_at = NOW()
            """), qr)

    inserted = 0
    with engine.begin() as conn:
        for lr in link_rows:
            result = conn.execute(text("""
                INSERT INTO qual_taxonomy_links (qual_code, scheme, value)
                SELECT :qual_code, :scheme, :value
                WHERE NOT EXISTS (
                    SELECT 1 FROM qual_taxonomy_links
                    WHERE qual_code = :qual_code
                      AND scheme    = :scheme
                      AND value     = :value
                )
            """), lr)
            inserted += result.rowcount

    log.info(f"  qual_registry: {len(qual_rows):,} quals | qual_taxonomy_links: {inserted:,} new rows")
    return inserted


def load_memberships(engine, df: pd.DataFrame) -> int:
    """Populate uoc_qual_memberships from Core_Unit_Codes / Elective_Unit_Codes."""
    rows = []
    for _, row in df.iterrows():
        qual_code = str(row.get("Qualification Code", "") or "").strip()
        if not qual_code:
            continue

        for unit_code in parse_unit_codes(row.get("Core_Unit_Codes", "")):
            rows.append({
                "uoc_code":        unit_code,
                "qual_code":       qual_code,
                "membership_type": "core",
                "is_imported":     False,
            })

        for unit_code in parse_unit_codes(row.get("Elective_Unit_Codes", "")):
            rows.append({
                "uoc_code":        unit_code,
                "qual_code":       qual_code,
                "membership_type": "elective",
                "is_imported":     False,
            })

    if not rows:
        log.warning("No unit membership rows found — Core_Unit_Codes / Elective_Unit_Codes may be empty.")
        log.warning("The scraper may not have extracted units successfully from TGA.")
        return 0

    # Batch upsert
    BATCH = 500
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            conn.execute(text("""
                INSERT INTO uoc_qual_memberships
                    (uoc_code, qual_code, membership_type, is_imported)
                VALUES
                    (:uoc_code, :qual_code, :membership_type, :is_imported)
                ON CONFLICT (uoc_code, qual_code, membership_type) DO NOTHING
            """), batch)
            total += len(batch)

    log.info(f"  ✓  uoc_qual_memberships: {total:,} rows inserted.")
    return total


def load_occupation_links(engine, df: pd.DataFrame) -> int:
    """
    Derive uoc_occupation_links from the membership + taxonomy data.

    Confidence scores mirror the linkage engine:
      core unit     → 0.50  (elective_native baseline from scraper HTML)
      elective unit → 0.40
    """
    CONF = {"core": 0.50, "elective": 0.40}

    rows = []
    for _, row in df.iterrows():
        qual_code        = str(row.get("Qualification Code", "") or "").strip()
        occupation_titles = str(row.get("Taxonomy_Occupation", "") or "").strip()
        industry_sector  = str(row.get("Taxonomy_Industry_Sector", "") or "").strip()
        anzsco           = str(row.get("ANZSCO_Identifier", "") or "").strip()

        if not qual_code or not occupation_titles or occupation_titles == "N/A":
            continue

        for membership_type, code_col in [("core", "Core_Unit_Codes"),
                                           ("elective", "Elective_Unit_Codes")]:
            for unit_code in parse_unit_codes(row.get(code_col, "")):
                rows.append({
                    "uoc_code":         unit_code,
                    "anzsco_code":      anzsco if anzsco != "N/A" else "",
                    "anzsco_title":     "",
                    "anzsco_major_group": "",
                    "industry_sector":  industry_sector,
                    "occupation_titles": occupation_titles,
                    "confidence":       CONF[membership_type],
                    "mapping_source":   f"{membership_type}_native",
                    "qual_code":        qual_code,
                    "is_primary":       False,
                })

    if not rows:
        log.warning("No occupation link rows to insert.")
        return 0

    # Mark primary: highest-confidence link per unit
    from collections import defaultdict
    best: dict[str, float] = defaultdict(float)
    for r in rows:
        best[r["uoc_code"]] = max(best[r["uoc_code"]], r["confidence"])
    for r in rows:
        r["is_primary"] = (r["confidence"] == best[r["uoc_code"]])

    BATCH = 500
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            conn.execute(text("""
                INSERT INTO uoc_occupation_links
                    (uoc_code, anzsco_code, anzsco_title, anzsco_major_group,
                     industry_sector, occupation_titles,
                     confidence, mapping_source, qual_code, is_primary)
                VALUES
                    (:uoc_code, :anzsco_code, :anzsco_title, :anzsco_major_group,
                     :industry_sector, :occupation_titles,
                     :confidence, :mapping_source, :qual_code, :is_primary)
                ON CONFLICT (uoc_code, qual_code, mapping_source) DO UPDATE SET
                    occupation_titles = EXCLUDED.occupation_titles,
                    industry_sector   = EXCLUDED.industry_sector,
                    anzsco_code       = EXCLUDED.anzsco_code,
                    confidence        = EXCLUDED.confidence,
                    is_primary        = EXCLUDED.is_primary
            """), batch)
            total += len(batch)

    log.info(f"  ✓  uoc_occupation_links: {total:,} rows upserted.")
    return total


def update_rsd_records(engine) -> int:
    """
    Back-fill occupation_titles on rsd_skill_records from uoc_occupation_links
    where the is_primary link exists.
    """
    # Check if rsd_skill_records has an occupation_titles column; add if missing
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE rsd_skill_records
            ADD COLUMN IF NOT EXISTS occupation_titles TEXT
        """))
        result = conn.execute(text("""
            UPDATE rsd_skill_records r
            SET occupation_titles = u.occupation_titles
            FROM uoc_occupation_links u
            WHERE r.unit_code = u.uoc_code
              AND u.is_primary = TRUE
              AND (r.occupation_titles IS NULL OR r.occupation_titles = '')
        """))
        updated = result.rowcount

    log.info(f"  ✓  rsd_skill_records.occupation_titles: {updated:,} rows updated.")
    return updated


# ── Entry point ───────────────────────────────────────────────────────────────

def load_excel(input_path: str) -> pd.DataFrame:
    if input_path.startswith("gs://"):
        import tempfile
        from google.cloud import storage
        log.info(f"Downloading from GCS: {input_path}")
        path = input_path[5:]
        bucket_name, blob_name = path.split("/", 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(tmp.name)
        log.info(f"Downloaded to {tmp.name}")
        return pd.read_excel(tmp.name)
    else:
        return pd.read_excel(input_path)


def main(input_path: str) -> None:
    log.info(f"Loading: {input_path}")
    df = load_excel(input_path)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Sanity check
    required = ["Qualification Code", "Taxonomy_Occupation"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"❌  Missing columns in Excel: {missing}\n"
                 f"    Available: {df.columns.tolist()}")

    engine = get_engine()
    log.info("Connected to database.")

    ensure_tables(engine)

    log.info("Step 1/4 — Loading qualification taxonomy…")
    load_qual_taxonomy(engine, df)

    log.info("Step 2/4 — Loading unit memberships (core / elective)…")
    n_memberships = load_memberships(engine, df)

    log.info("Step 3/4 — Deriving occupation links…")
    load_occupation_links(engine, df)

    log.info("Step 4/4 — Back-filling rsd_skill_records…")
    update_rsd_records(engine)

    log.info("✅  Done.")
    if n_memberships == 0:
        log.warning(
            "⚠  No unit memberships were loaded — this means the scraper didn't "
            "extract Core_Unit_Codes / Elective_Unit_Codes from TGA.\n"
            "   The qual_taxonomy_links and occupation_titles columns will still "
            "be populated, but the unit→qualification mapping won't be available.\n"
            "   Check a few rows of the Excel to confirm those columns have data."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load TGA scrape results into the database.")
    parser.add_argument(
        "--input", "-i",
        default="tga_qualifications_updated.xlsx",
        help="Path to the scraper output Excel file"
    )
    args = parser.parse_args()
    main(args.input)
