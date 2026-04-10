"""
core/rsd_record.py

Builds the canonical fully-linked UOC record as a W3C Verifiable
Credential-aligned JSON object.

Implements all four technical refinements:
  Ref 1 — taxonomic_alignment object with ANZSCO URI (W3C VC pattern)
  Ref 2 — aqf_level + skill_level_label from qual_registry
  Ref 3 — evidence_requirements with IPFS hash placeholder
  Ref 4 — is_imported flag + home_tp_code in UOC header
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from sqlalchemy import text, Engine

# ── ANZSCO URI builder ────────────────────────────────────────────────────────
ABS_ANZSCO_BASE = "https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations"

def anzsco_uri(code: str) -> str:
    """Return the canonical ABS URI for an ANZSCO code."""
    if not code:
        return ""
    clean = code.replace(" ", "").replace("-", "")
    return f"https://www.abs.gov.au/anzsco/{clean}"


# ── AQF level mapping ─────────────────────────────────────────────────────────
AQF_SKILL_LABELS = {
    "Certificate I":   "Foundation",
    "Certificate II":  "Entry Level",
    "Certificate III": "Trades / Operational",
    "Certificate IV":  "Advanced Trades / Paraprofessional",
    "Diploma":         "Paraprofessional / Technician",
    "Advanced Diploma":"Advanced Technician / Associate Professional",
    "Graduate Certificate": "Graduate Specialist",
    "Graduate Diploma":     "Graduate Specialist",
    "Bachelor Degree":      "Professional",
    "Honours Degree":       "Honours Professional",
    "Graduate Certificate": "Postgraduate Specialist",
    "Masters Degree":       "Expert / Researcher",
    "Doctoral Degree":      "Research Expert",
}


def aqf_to_skill_label(aqf_level: str | None) -> str:
    if not aqf_level:
        return "Unspecified"
    for key, label in AQF_SKILL_LABELS.items():
        if key.lower() in (aqf_level or "").lower():
            return label
    return aqf_level or "Unspecified"


# ── Element ID builder ────────────────────────────────────────────────────────
def build_element_id(uoc_code: str, element_title: str, index: int) -> str:
    """Build a stable element ID: MSL975058-E01, MSL975058-E02 etc."""
    return f"{uoc_code}-E{index+1:02d}"


# ── Evidence hash ─────────────────────────────────────────────────────────────
def build_evidence_stub(
    skill_statement: str,
    evidence_hash: str | None = None,
    evidence_uri: str | None = None,
    evidence_type: str | None = None,
) -> dict:
    """
    Build a W3C VC evidence block.
    If no hash is provided, a deterministic SHA-256 of the statement is used
    as a placeholder until real Blocksure/IPFS data is linked.
    """
    if not evidence_hash:
        # Deterministic placeholder — not a real proof, signals "unverified"
        evidence_hash = "sha256:" + hashlib.sha256(
            skill_statement.encode()
        ).hexdigest()
        verified = False
    else:
        verified = True

    return {
        "type":           evidence_type or "Simulation Performance Data",
        "hash":           evidence_hash,
        "uri":            evidence_uri or "",    # IPFS or Blocksure URL when available
        "verified":       verified,
        "placeholder":    not verified,
        "blocksure_ready": bool(evidence_uri and "blocksure" in (evidence_uri or "")),
    }


# ── Taxonomic alignment (W3C VC pattern) ─────────────────────────────────────
def build_taxonomic_alignment(
    anzsco_code:    str,
    anzsco_title:   str,
    confidence:     float,
    mapping_source: str,
    asced_code:     str | None = None,
    asced_title:    str | None = None,
) -> dict:
    """
    Structured taxonomic alignment following W3C Verifiable Credentials
    and Open Skills Network (OSN) alignment patterns.
    """
    alignment = {
        "type":            "TaxonomicAlignment",
        "@context":        "https://www.w3.org/2018/credentials/v1",
        "targetFramework": {
            "id":    "ANZSCO",
            "name":  "Australian and New Zealand Standard Classification of Occupations",
            "version": "2022",
            "publisher": "Australian Bureau of Statistics",
        },
        "targetCode":  anzsco_code,
        "targetName":  anzsco_title,
        "targetUrl":   anzsco_uri(anzsco_code),
        "confidence":  round(float(confidence), 3),
        "alignmentType": "relatedTo",
        "mappingSource": mapping_source,
    }
    if asced_code or asced_title:
        alignment["asced"] = {
            "targetFramework": "ASCED — Australian Standard Classification of Education",
            "targetCode":  asced_code or "",
            "targetName":  asced_title or "",
        }
    return alignment


# ── Full UOC record builder ───────────────────────────────────────────────────
def build_uoc_record(
    engine: Engine,
    uoc_code: str,
    include_all_links: bool = False,
) -> dict:
    """
    Build the complete W3C VC-aligned JSON record for a single UOC.

    Structure:
    {
      "@context": [...],
      "type": "RichSkillDescriptor",
      "uoc_code": ...,
      "unit_header": {   <- Ref 2, 4
        "aqf_level": ..., "skill_level_label": ...,
        "is_imported": ..., "owner_tp_code": ..., "home_tp_title": ...
      },
      "primary_occupation": {  <- Ref 1
        "taxonomic_alignment": { W3C VC object }
      },
      "rsd_skill_statements": [  <- Ref 1, 3
        {
          "element_id": ...,
          "skill_statement": ...,
          "taxonomic_alignment": { ... },
          "evidence_requirements": { ... },
          "qa_status": ...
        }
      ],
      "qualification_memberships": [...],
      "all_occupation_links": [...],
      "metadata": {...}
    }
    """
    with engine.connect() as conn:

        # ── UOC base info ─────────────────────────────────────────────────────
        uoc_row = conn.execute(text("""
            SELECT u.uoc_code, u.uoc_title, u.tp_code, u.tp_title,
                   u.usage_recommendation
            FROM uoc_registry u
            WHERE u.uoc_code = :uc
        """), {"uc": uoc_code}).fetchone()

        if not uoc_row:
            return {"error": f"UOC {uoc_code} not found in registry"}

        # ── Qualification memberships + AQF levels (Ref 2) ───────────────────
        mem_rows = conn.execute(text("""
            SELECT m.qual_code, q.qual_title, q.aqf_level,
                   m.membership_type, m.is_imported,
                   m.owner_tp_code, m.elective_group
            FROM uoc_qual_memberships m
            JOIN qual_registry q ON q.qual_code = m.qual_code
            WHERE m.uoc_code = :uc
            ORDER BY m.membership_type, q.aqf_level
        """), {"uc": uoc_code}).fetchall()

        # ── Primary occupation link (Ref 1) ───────────────────────────────────
        occ_rows = conn.execute(text("""
            SELECT o.anzsco_code, o.anzsco_title, o.anzsco_major_group,
                   o.asced_code, o.asced_title,
                   o.industry_sector, o.occupation_titles,
                   o.confidence, o.mapping_source,
                   o.source_qual_code, o.is_primary,
                   o.is_imported, o.owner_tp_code, o.home_tp_title,
                   o.aqf_level, o.skill_level_label
            FROM uoc_occupation_links o
            WHERE o.uoc_code = :uc AND o.valid_to IS NULL
            ORDER BY o.confidence DESC
        """), {"uc": uoc_code}).fetchall()

        # ── Skill statements (Ref 3) ──────────────────────────────────────────
        stmt_rows = conn.execute(text("""
            SELECT s.element_title, s.skill_statement,
                   s.qa_passes, s.keywords,
                   s.evidence_hash, s.evidence_type, s.evidence_uri,
                   s.element_id
            FROM rsd_skill_records s
            WHERE s.unit_code = :uc
            AND s.skill_statement IS NOT NULL AND s.skill_statement != ''
            ORDER BY s.row_index
        """), {"uc": uoc_code}).fetchall()

    # ── Derive AQF level from highest-level qual (Ref 2) ─────────────────────
    AQF_ORDER = [
        "Doctoral", "Masters", "Graduate Diploma", "Graduate Certificate",
        "Honours", "Bachelor", "Advanced Diploma", "Diploma",
        "Certificate IV", "Certificate III", "Certificate II", "Certificate I",
    ]
    best_aqf = None
    for level_key in AQF_ORDER:
        for mem in mem_rows:
            if mem[2] and level_key.lower() in mem[2].lower():
                best_aqf = mem[2]
                break
        if best_aqf:
            break

    skill_label = aqf_to_skill_label(best_aqf)

    # ── Is this UOC imported? (Ref 4) ─────────────────────────────────────────
    is_imported  = any(bool(m[4]) for m in mem_rows)
    owner_tp     = uoc_row[2]  # always the code's own TP
    home_tp_title = uoc_row[3] or ""

    # If the unit appears ONLY as imported across all quals, flag it
    native_memberships = [m for m in mem_rows if not m[4]]
    imported_only = is_imported and len(native_memberships) == 0

    # ── Primary occupation ────────────────────────────────────────────────────
    primary_occ = next((o for o in occ_rows if o[10]), None)  # is_primary=True
    if not primary_occ and occ_rows:
        primary_occ = occ_rows[0]

    primary_block = {}
    if primary_occ:
        primary_block = {
            "anzsco_code":      primary_occ[0],
            "anzsco_title":     primary_occ[1],
            "anzsco_major_group": primary_occ[2],
            "industry_sector":  primary_occ[6],
            "occupation_titles": (primary_occ[7] or "").split("; ") if primary_occ[7] else [],
            "taxonomic_alignment": build_taxonomic_alignment(
                anzsco_code=primary_occ[0] or "",
                anzsco_title=primary_occ[1] or "",
                confidence=float(primary_occ[8] or 0),
                mapping_source=primary_occ[9] or "",
                asced_code=primary_occ[3],
                asced_title=primary_occ[4],
            ),
        }

    # ── All occupation links ──────────────────────────────────────────────────
    all_links = []
    for o in occ_rows:
        link = {
            "taxonomic_alignment": build_taxonomic_alignment(
                anzsco_code=o[0] or "",
                anzsco_title=o[1] or "",
                confidence=float(o[8] or 0),
                mapping_source=o[9] or "",
                asced_code=o[3],
                asced_title=o[4],
            ),
            "source_qual": o[10] if include_all_links else None,
            "is_primary":  bool(o[10]),
        }
        if include_all_links:
            all_links.append(link)
        elif o[10]:  # only primary if not full mode
            all_links.append(link)

    # ── Skill statements with evidence (Ref 1, 3) ────────────────────────────
    # Get primary ANZSCO for each statement's taxonomic_alignment
    stmt_anzsco_code  = primary_occ[0] if primary_occ else ""
    stmt_anzsco_title = primary_occ[1] if primary_occ else ""
    stmt_confidence   = float(primary_occ[8] or 0) if primary_occ else 0.0
    stmt_source       = primary_occ[9] if primary_occ else ""
    stmt_asced_c      = primary_occ[3] if primary_occ else None
    stmt_asced_t      = primary_occ[4] if primary_occ else None

    rsd_statements = []
    for idx, s in enumerate(stmt_rows):
        (elem_title, skill_stmt, qa_passes, keywords,
         ev_hash, ev_type, ev_uri, elem_id) = s

        # Use stored element_id or build one
        eid = elem_id or build_element_id(uoc_code, elem_title or "", idx)

        rsd_statements.append({
            "element_id":   eid,
            "element_title": elem_title or "",
            "skill_statement": skill_stmt,
            "keywords":     (keywords or "").split(";") if keywords else [],
            "taxonomic_alignment": build_taxonomic_alignment(
                anzsco_code=stmt_anzsco_code,
                anzsco_title=stmt_anzsco_title,
                confidence=stmt_confidence,
                mapping_source=stmt_source,
                asced_code=stmt_asced_c,
                asced_title=stmt_asced_t,
            ) if stmt_anzsco_code else None,
            "evidence_requirements": build_evidence_stub(
                skill_statement=skill_stmt,
                evidence_hash=ev_hash,
                evidence_uri=ev_uri,
                evidence_type=ev_type,
            ),
            "qa_status": "Verified" if qa_passes else "Pending Review",
        })

    # ── Qualification memberships (Ref 2, 4) ──────────────────────────────────
    qual_memberships = [
        {
            "qual_code":      m[0],
            "qual_title":     m[1],
            "aqf_level":      m[2],
            "skill_level_label": aqf_to_skill_label(m[2]),
            "membership_type": m[3],
            "is_imported":    bool(m[4]),
            "owner_tp_code":  m[5],
            "elective_group": m[6],
        }
        for m in mem_rows
    ]

    # ── Assemble final record ─────────────────────────────────────────────────
    record = {
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.3.json",
            "https://w3id.org/openbadges/v2",
        ],
        "type": ["VerifiableCredential", "RichSkillDescriptor"],
        "id":   f"https://rsd.example.org/uoc/{uoc_code}",   # replace with your domain

        # ── UOC header ────────────────────────────────────────────────────────
        "uoc_code":  uoc_code,
        "uoc_title": uoc_row[1],
        "tp_code":   uoc_row[2],
        "tp_title":  uoc_row[3] or "",
        "usage_recommendation": uoc_row[4] or "Current",

        # Ref 2: AQF / skill level
        "unit_header": {
            "aqf_level":         best_aqf or "Unspecified",
            "skill_level_label": skill_label,
            "aqf_note":          (
                "Derived from highest AQF-level Current qualification "
                "containing this UOC as Core or Elective."
            ),
        },

        # Ref 4: Imported unit flag
        "import_status": {
            "is_imported":    is_imported,
            "imported_only":  imported_only,
            "owner_tp_code":  owner_tp,
            "home_tp_title":  home_tp_title,
            "note": (
                f"This unit is owned by {owner_tp} "
                "but appears in qualifications from other training packages."
                if is_imported else
                "This unit is native to its training package."
            ),
        },

        # ── Primary occupation (Ref 1) ────────────────────────────────────────
        "primary_occupation": primary_block,

        # ── Skill statements (Ref 1, 3) ───────────────────────────────────────
        "rsd_skill_statements": rsd_statements,

        # ── Qualification memberships (Ref 2, 4) ──────────────────────────────
        "qualification_memberships": qual_memberships,

        # ── All occupation links ───────────────────────────────────────────────
        "all_occupation_links": all_links if include_all_links else [
            link for link in all_links if link.get("is_primary")
        ],

        # ── Metadata ──────────────────────────────────────────────────────────
        "metadata": {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "tga_source":       "training.gov.au National Training Register",
            "anzsco_version":   "ANZSCO 2022 (ABS cat. 1220.0)",
            "vc_standard":      "W3C Verifiable Credentials Data Model 1.1",
            "rsd_standard":     "IMS Global Open Skills Network RSD v1",
            "evidence_standard": "Blocksure.com.au Verifiable Assessment Record",
        },
    }

    return record


def records_to_jsonl(engine: Engine, uoc_codes: list[str]) -> str:
    """Produce JSONL (one JSON object per line) for a list of UOC codes."""
    lines = []
    for code in uoc_codes:
        rec = build_uoc_record(engine, code, include_all_links=True)
        lines.append(json.dumps(rec, ensure_ascii=False, default=str))
    return "\n".join(lines)
