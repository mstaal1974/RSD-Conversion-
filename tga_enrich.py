"""
tga_enrich.py
=============
Two jobs in one:

  1. PARSE UNITS — extracts Core_Unit_Codes and Elective_Unit_Codes from the
     Contents column already present in the source Excel. No network calls.

  2. FETCH TAXONOMY — calls the TGA production SOAP API for each qualification
     to get ANZSCO Identifier, Industry Sector, and Occupation taxonomy.
     Must run inside Cloud Run (production API allowlists Cloud Run IPs).

Then loads everything into the database tables:
  qual_taxonomy_links, uoc_qual_memberships, uoc_occupation_links,
  and back-fills rsd_skill_records.occupation_titles

Usage (local, units-only — no SOAP needed):
    python tga_enrich.py --input tga_qualifications_updated.xlsx --skip-soap

Usage (Cloud Run job — full enrichment):
    python tga_enrich.py --input gs://rsd-convert-data/tga_qualifications_updated.xlsx
"""
from __future__ import annotations
import argparse
import logging
import os
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── SOAP config ───────────────────────────────────────────────────────────────
SOAP_URL = (
    "https://ws.training.gov.au/Deewr.Tga.Webservices/"
    "TrainingComponentService.svc/Training12"
)
SOAP_ACTION = (
    "http://training.gov.au/services/trainingcomponent/12/"
    "ITrainingComponentService/GetDetails"
)
SOAP_USER = os.getenv("TGA_USERNAME", "WebService.Read")
SOAP_PASS = os.getenv("TGA_PASSWORD", "Asdf098")

SOAP_BODY = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
               xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
  <soap:Header>
    <wsse:Security>
      <wsse:UsernameToken>
        <wsse:Username>{user}</wsse:Username>
        <wsse:Password>{password}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
  <soap:Body>
    <GetDetails xmlns="http://training.gov.au/services/trainingcomponent/12">
      <request>
        <Code>{code}</Code>
        <InformationRequest>
          <Classifications>true</Classifications>
        </InformationRequest>
      </request>
    </GetDetails>
  </soap:Body>
</soap:Envelope>"""

# ── DB setup ──────────────────────────────────────────────────────────────────

def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            import toml
            url = toml.load(".streamlit/secrets.toml").get("DATABASE_URL")
        except Exception:
            pass
    if not url:
        sys.exit("❌  DATABASE_URL not set.")
    return create_engine(url, pool_pre_ping=True)


def ensure_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS qual_taxonomy_links CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS uoc_qual_memberships CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS uoc_occupation_links CASCADE"))

        conn.execute(text("""
            CREATE TABLE qual_taxonomy_links (
                id                BIGSERIAL PRIMARY KEY,
                qual_code         TEXT NOT NULL,
                qual_title        TEXT,
                anzsco_identifier TEXT,
                industry_sector   TEXT,
                occupation_titles TEXT,
                created_at        TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (qual_code)
            )
        """))
        conn.execute(text("""
            CREATE TABLE uoc_qual_memberships (
                id              BIGSERIAL PRIMARY KEY,
                unit_code       TEXT NOT NULL,
                qual_code       TEXT NOT NULL,
                membership_type TEXT NOT NULL,
                is_native       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (unit_code, qual_code, membership_type)
            )
        """))
        conn.execute(text("""
            CREATE TABLE uoc_occupation_links (
                id                BIGSERIAL PRIMARY KEY,
                uoc_code          TEXT NOT NULL,
                anzsco_code       TEXT NOT NULL DEFAULT '',
                anzsco_title      TEXT NOT NULL DEFAULT '',
                anzsco_major_group TEXT NOT NULL DEFAULT '',
                industry_sector   TEXT,
                occupation_titles TEXT,
                confidence        NUMERIC(5,3) NOT NULL,
                mapping_source    TEXT NOT NULL,
                qual_code         TEXT,
                is_primary        BOOLEAN DEFAULT FALSE,
                created_at        TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (uoc_code, qual_code, mapping_source)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_uoc_occ_links_uoc
            ON uoc_occupation_links (uoc_code)
        """))
    log.info("Tables ready.")


# ── GCS loader ────────────────────────────────────────────────────────────────

def load_excel(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        import tempfile
        from google.cloud import storage
        log.info(f"Downloading from GCS: {path}")
        bucket_name, blob_name = path[5:].split("/", 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(tmp.name)
        return pd.read_excel(tmp.name)
    return pd.read_excel(path)


# ── Unit code parser ──────────────────────────────────────────────────────────

UNIT_RE = re.compile(r'^([A-Z]{2,8}\d{3,6}[A-Z]?)\s*$')


def parse_units(text: str) -> tuple[list[str], list[str]]:
    """Extract core and elective unit codes from a Contents text block."""
    if not text or pd.isna(text):
        return [], []
    core, elective, current = [], [], None
    for line in str(text).splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith('core unit'):
            current = 'core'
        elif low.startswith('elective unit') or low.startswith('group '):
            current = 'elective'
        elif UNIT_RE.match(line):
            if current == 'core':
                core.append(line)
            elif current == 'elective':
                elective.append(line)
    return core, elective


# ── SOAP taxonomy fetcher ─────────────────────────────────────────────────────

def fetch_taxonomy_soap(qual_code: str, session) -> dict:
    """Call TGA SOAP API and return taxonomy dict. Returns N/A values on failure."""
    result = {
        "ANZSCO_Identifier": "N/A",
        "Taxonomy_Industry_Sector": "N/A",
        "Taxonomy_Occupation": "N/A",
    }
    try:
        body = SOAP_BODY.format(user=SOAP_USER, password=SOAP_PASS, code=qual_code)
        r = session.post(
            SOAP_URL,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": SOAP_ACTION,
            },
            timeout=20,
        )
        if r.status_code != 200:
            log.warning(f"  {qual_code}: HTTP {r.status_code}")
            return result

        import xml.etree.ElementTree as ET
        ns = {
            "tc": "http://training.gov.au/services/trainingcomponent/12",
            "tc_types": "http://training.gov.au/services/trainingcomponent/12/types",
        }
        root = ET.fromstring(r.content)

        # Classifications are in Classification elements
        occupations, industry, anzsco = [], [], []
        for cl in root.iter():
            tag = cl.tag.split("}")[-1] if "}" in cl.tag else cl.tag
            if tag == "ClassificationScheme":
                scheme = cl.findtext(".//{*}Scheme") or ""
                value  = cl.findtext(".//{*}Value") or ""
                desc   = cl.findtext(".//{*}Description") or ""
                display = desc if desc else value
                if "ANZSCO" in scheme:
                    anzsco.append(display)
                elif "Industry" in scheme or "Sector" in scheme:
                    industry.append(display)
                elif "Occupation" in scheme or "Taxonomy" in scheme:
                    occupations.append(display)

        if anzsco:
            result["ANZSCO_Identifier"] = ", ".join(anzsco)
        if industry:
            result["Taxonomy_Industry_Sector"] = ", ".join(industry)
        if occupations:
            result["Taxonomy_Occupation"] = ", ".join(occupations)

    except Exception as e:
        log.warning(f"  {qual_code} SOAP error: {e}")
    return result


# ── DB loaders ────────────────────────────────────────────────────────────────

def load_qual_taxonomy(engine, df: pd.DataFrame) -> int:
    rows = []
    for _, row in df.iterrows():
        qc = str(row.get("Qualification Code", "") or "").strip()
        if not qc:
            continue
        rows.append({
            "qual_code":         qc,
            "qual_title":        str(row.get("Qualification Title", "") or ""),
            "anzsco_identifier": str(row.get("ANZSCO_Identifier", "") or ""),
            "industry_sector":   str(row.get("Taxonomy_Industry_Sector", "") or ""),
            "occupation_titles": str(row.get("Taxonomy_Occupation", "") or ""),
        })
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO qual_taxonomy_links
                (qual_code, qual_title, anzsco_identifier, industry_sector, occupation_titles)
            VALUES (:qual_code, :qual_title, :anzsco_identifier, :industry_sector, :occupation_titles)
            ON CONFLICT (qual_code) DO UPDATE SET
                qual_title = EXCLUDED.qual_title,
                anzsco_identifier = EXCLUDED.anzsco_identifier,
                industry_sector = EXCLUDED.industry_sector,
                occupation_titles = EXCLUDED.occupation_titles
        """), rows)
    log.info(f"  ✓  qual_taxonomy_links: {len(rows):,} rows")
    return len(rows)


def load_memberships(engine, df: pd.DataFrame) -> int:
    rows = []
    for _, row in df.iterrows():
        qc = str(row.get("Qualification Code", "") or "").strip()
        for mtype, col in [("core", "Core_Unit_Codes"), ("elective", "Elective_Unit_Codes")]:
            raw = str(row.get(col, "") or "")
            for uc in [c.strip() for c in raw.split("|") if c.strip()]:
                rows.append({"unit_code": uc, "qual_code": qc,
                             "membership_type": mtype, "is_native": True})
    if not rows:
        log.warning("  No unit memberships found.")
        return 0
    BATCH = 500
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            conn.execute(text("""
                INSERT INTO uoc_qual_memberships (unit_code, qual_code, membership_type, is_native)
                VALUES (:unit_code, :qual_code, :membership_type, :is_native)
                ON CONFLICT DO NOTHING
            """), rows[i:i+BATCH])
    log.info(f"  ✓  uoc_qual_memberships: {len(rows):,} rows")
    return len(rows)


def load_occupation_links(engine, df: pd.DataFrame) -> int:
    CONF = {"core": 0.50, "elective": 0.40}
    rows = []
    for _, row in df.iterrows():
        qc  = str(row.get("Qualification Code", "") or "").strip()
        occ = str(row.get("Taxonomy_Occupation", "") or "").strip()
        ind = str(row.get("Taxonomy_Industry_Sector", "") or "").strip()
        anz = str(row.get("ANZSCO_Identifier", "") or "").strip()
        if not qc or not occ or occ in ("N/A", "nan", ""):
            continue
        for mtype, col in [("core", "Core_Unit_Codes"), ("elective", "Elective_Unit_Codes")]:
            raw = str(row.get(col, "") or "")
            for uc in [c.strip() for c in raw.split("|") if c.strip()]:
                rows.append({
                    "uoc_code": uc, "anzsco_code": anz if anz != "N/A" else "",
                    "anzsco_title": "", "anzsco_major_group": "",
                    "industry_sector": ind, "occupation_titles": occ,
                    "confidence": CONF[mtype], "mapping_source": f"{mtype}_native",
                    "qual_code": qc, "is_primary": False,
                })
    if not rows:
        log.warning("  No occupation links to insert.")
        return 0
    from collections import defaultdict
    best: dict[str, float] = defaultdict(float)
    for r in rows:
        best[r["uoc_code"]] = max(best[r["uoc_code"]], r["confidence"])
    for r in rows:
        r["is_primary"] = (r["confidence"] == best[r["uoc_code"]])
    BATCH = 500
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            conn.execute(text("""
                INSERT INTO uoc_occupation_links
                    (uoc_code, anzsco_code, anzsco_title, anzsco_major_group,
                     industry_sector, occupation_titles, confidence,
                     mapping_source, qual_code, is_primary)
                VALUES
                    (:uoc_code, :anzsco_code, :anzsco_title, :anzsco_major_group,
                     :industry_sector, :occupation_titles, :confidence,
                     :mapping_source, :qual_code, :is_primary)
                ON CONFLICT (uoc_code, qual_code, mapping_source) DO UPDATE SET
                    occupation_titles = EXCLUDED.occupation_titles,
                    industry_sector   = EXCLUDED.industry_sector,
                    is_primary        = EXCLUDED.is_primary
            """), rows[i:i+BATCH])
    log.info(f"  ✓  uoc_occupation_links: {len(rows):,} rows")
    return len(rows)


def backfill_rsd(engine) -> int:
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE rsd_skill_records ADD COLUMN IF NOT EXISTS occupation_titles TEXT"
        ))
        result = conn.execute(text("""
            UPDATE rsd_skill_records r
            SET occupation_titles = u.occupation_titles
            FROM uoc_occupation_links u
            WHERE r.unit_code = u.uoc_code AND u.is_primary = TRUE
              AND (r.occupation_titles IS NULL OR r.occupation_titles = '')
        """))
    log.info(f"  ✓  rsd_skill_records back-filled: {result.rowcount:,} rows")
    return result.rowcount


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_path: str, skip_soap: bool) -> None:
    log.info(f"Loading: {input_path}")
    df = load_excel(input_path)
    log.info(f"  {len(df):,} rows, {len(df.columns)} columns")

    # ── Step 1: Parse unit codes from Contents ────────────────────────────────
    log.info("Step 1 — Parsing unit codes from Contents column…")
    core_list, elec_list = [], []
    for _, row in df.iterrows():
        core, elec = parse_units(row.get("Contents", ""))
        core_list.append("|".join(core))
        elec_list.append("|".join(elec))
    df["Core_Unit_Codes"]     = core_list
    df["Elective_Unit_Codes"] = elec_list
    n_core = sum(1 for x in core_list if x)
    log.info(f"  ✓  {n_core:,} qualifications have core unit codes")

    # ── Step 2: Fetch taxonomy via SOAP (Cloud Run only) ──────────────────────
    if not skip_soap:
        import requests
        session = requests.Session()
        log.info(f"Step 2 — Fetching taxonomy from TGA SOAP API for {len(df):,} qualifications…")
        log.info("  (This takes ~40 mins at 2s/qual — run as a Cloud Run job)")

        anzsco_list, sector_list, occ_list = [], [], []
        for i, (_, row) in enumerate(df.iterrows()):
            qc = str(row.get("Qualification Code", "") or "").strip()
            tax = fetch_taxonomy_soap(qc, session)
            anzsco_list.append(tax["ANZSCO_Identifier"])
            sector_list.append(tax["Taxonomy_Industry_Sector"])
            occ_list.append(tax["Taxonomy_Occupation"])
            if (i + 1) % 50 == 0:
                log.info(f"  {i+1}/{len(df)} done…")
            time.sleep(2)

        df["ANZSCO_Identifier"]        = anzsco_list
        df["Taxonomy_Industry_Sector"] = sector_list
        df["Taxonomy_Occupation"]      = occ_list
        n_occ = sum(1 for x in occ_list if x and x != "N/A")
        log.info(f"  ✓  {n_occ:,} qualifications have occupation taxonomy")
    else:
        log.info("Step 2 — Skipping SOAP fetch (--skip-soap flag set)")

    # ── Step 3-6: Load into DB ────────────────────────────────────────────────
    engine = get_engine()
    log.info("Connected to database.")
    ensure_tables(engine)

    log.info("Step 3 — Loading qualification taxonomy…")
    load_qual_taxonomy(engine, df)

    log.info("Step 4 — Loading unit memberships…")
    load_memberships(engine, df)

    log.info("Step 5 — Deriving occupation links…")
    load_occupation_links(engine, df)

    log.info("Step 6 — Back-filling rsd_skill_records…")
    backfill_rsd(engine)

    log.info("✅  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="tga_qualifications_updated.xlsx")
    parser.add_argument("--skip-soap", action="store_true",
                        help="Skip SOAP taxonomy fetch (use existing taxonomy in Excel)")
    args = parser.parse_args()
    main(args.input, args.skip_soap)
