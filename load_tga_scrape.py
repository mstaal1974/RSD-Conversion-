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
    """Create the required tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS qual_taxonomy_links (
                id                  BIGSERIAL PRIMARY KEY,
                qual_code           TEXT NOT NULL,
                qual_title          TEXT,
                anzsco_identifier   TEXT,
                industry_sector     TEXT,
                occupation_titles   TEXT,
                created_at          TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (qual_code)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS uoc_qual_memberships (
                id              BIGSERIAL PRIMARY KEY,
                unit_code       TEXT NOT NULL,
                qual_code       TEXT NOT NULL,
                membership_type TEXT NOT NULL,   -- 'core' or 'elective'
                is_native       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (unit_code, qual_code, membership_type)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS uoc_occupation_links (
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
                qual_code           TEXT,
                is_primary          BOOLEAN DEFAULT FALSE,
                created_at          TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (uoc_code, qual_code, mapping_source)
            )
        """))

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

def load_qual_taxonomy(engine, df: pd.DataFrame) -> int:
    """Upsert qualification taxonomy rows."""
    rows = []
    for _, row in df.iterrows():
        qual_code = str(row.get("Qualification Code", "") or "").strip()
        if not qual_code:
            continue
        rows.append({
            "qual_code":         qual_code,
            "qual_title":        str(row.get("Qualification Title", "") or ""),
            "anzsco_identifier": str(row.get("ANZSCO_Identifier", "") or ""),
            "industry_sector":   str(row.get("Taxonomy_Industry_Sector", "") or ""),
            "occupation_titles": str(row.get("Taxonomy_Occupation", "") or ""),
        })

    if not rows:
        log.warning("No qualification taxonomy rows to insert.")
        return 0

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO qual_taxonomy_links
                (qual_code, qual_title, anzsco_identifier, industry_sector, occupation_titles)
            VALUES
                (:qual_code, :qual_title, :anzsco_identifier, :industry_sector, :occupation_titles)
            ON CONFLICT (qual_code) DO UPDATE SET
                qual_title        = EXCLUDED.qual_title,
                anzsco_identifier = EXCLUDED.anzsco_identifier,
                industry_sector   = EXCLUDED.industry_sector,
                occupation_titles = EXCLUDED.occupation_titles,
                created_at        = NOW()
        """), rows)

    log.info(f"  ✓  qual_taxonomy_links: {len(rows):,} rows upserted.")
    return len(rows)


def load_memberships(engine, df: pd.DataFrame) -> int:
    """Populate uoc_qual_memberships from Core_Unit_Codes / Elective_Unit_Codes."""
    rows = []
    for _, row in df.iterrows():
        qual_code = str(row.get("Qualification Code", "") or "").strip()
        if not qual_code:
            continue

        for unit_code in parse_unit_codes(row.get("Core_Unit_Codes", "")):
            rows.append({
                "unit_code":       unit_code,
                "qual_code":       qual_code,
                "membership_type": "core",
                "is_native":       True,
            })

        for unit_code in parse_unit_codes(row.get("Elective_Unit_Codes", "")):
            rows.append({
                "unit_code":       unit_code,
                "qual_code":       qual_code,
                "membership_type": "elective",
                "is_native":       True,
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
                    (unit_code, qual_code, membership_type, is_native)
                VALUES
                    (:unit_code, :qual_code, :membership_type, :is_native)
                ON CONFLICT (unit_code, qual_code, membership_type) DO NOTHING
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

def main(input_path: str) -> None:
    log.info(f"Loading: {input_path}")
    df = pd.read_excel(input_path)
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
