"""
core/tga_ingestor.py

TGA National Training Register ingestion via SOAP web services.

Key operations:
  - Authenticate with TGA credentials
  - Fetch all current qualifications (filtered — no superseded)
  - Parse ContentBundle: Classifications + PackagingRules
  - Detect imported units (owned by different TP)
  - Upsert into uoc_registry, qual_registry, qual_taxonomy_links,
    uoc_qual_memberships, uoc_classifications
"""
from __future__ import annotations
import time
import re
import logging
from typing import Optional
from sqlalchemy import text, Engine

log = logging.getLogger(__name__)

# Confidence weights — used by linkage engine
CONF = {
    "direct":            1.00,
    "core_native":       0.85,
    "core_imported":     0.70,
    "elective_native":   0.50,
    "elective_imported": 0.40,
    "asc_specialist":    0.30,
}

SOAP_WSDL   = "https://ws.training.gov.au/Dws/dws.asmx?WSDL"
REST_SEARCH = "https://training.gov.au/api/training/search"
RATE_LIMIT  = 0.5   # seconds between SOAP calls


class TGAIngestor:
    """
    Connects to the TGA SOAP API and ingests qualification/UOC data.

    Usage:
        ingestor = TGAIngestor(engine, tga_user, tga_pass)
        ingestor.run(tp_codes=["MSL","BSB"])   # or None for all TPs
    """

    def __init__(self, engine: Engine, username: str, password: str,
                 pipeline_run_id: int | None = None):
        self.engine          = engine
        self.username        = username
        self.password        = password
        self.pipeline_run_id = pipeline_run_id
        self._client         = None

    # ── SOAP client ──────────────────────────────────────────────────────────
    def _get_client(self):
        """Lazy-initialise zeep SOAP client."""
        if self._client is None:
            try:
                import zeep
                self._client = zeep.Client(SOAP_WSDL)
                self._client.service.Authenticate(
                    username=self.username,
                    password=self.password,
                )
                log.info("TGA SOAP authenticated")
            except Exception as e:
                raise RuntimeError(f"TGA SOAP connection failed: {e}") from e
        return self._client

    # ── REST fallback helpers ─────────────────────────────────────────────────
    def _rest_search_qualifications(self, tp_code: str | None = None) -> list[dict]:
        """
        Use TGA REST search to get current qualification codes.
        Fallback when SOAP is unavailable.
        Returns list of {code, title, tp_code, status}
        """
        import urllib.request, urllib.parse, json

        params = {
            "filter[type]":   "Qualification",
            "filter[status]": "Current",
            "page[size]":     "200",
        }
        if tp_code:
            params["filter[trainingPackage]"] = tp_code

        url = REST_SEARCH + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            results = []
            for item in data.get("data", []):
                attrs = item.get("attributes", {})
                results.append({
                    "code":   attrs.get("code", ""),
                    "title":  attrs.get("title", ""),
                    "tp_code": attrs.get("code", "")[:3] if attrs.get("code") else "",
                    "status": attrs.get("usageRecommendation", "Current"),
                })
            return results
        except Exception as e:
            log.warning(f"TGA REST search failed: {e}")
            return []

    def _rest_get_details(self, code: str) -> dict:
        """Fetch basic details for a qualification or UOC via REST."""
        import urllib.request, json

        url = f"https://training.gov.au/api/training/details/{code}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            time.sleep(0.3)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            log.warning(f"REST details failed for {code}: {e}")
            return {}

    # ── Main ingestion entry point ─────────────────────────────────────────────
    def run(self,
            tp_codes: list[str] | None = None,
            use_soap: bool = True,
            progress_callback=None) -> dict:
        """
        Full ingestion pipeline.
        If use_soap=False (or SOAP unavailable), falls back to REST-only mode
        which gives qual/UOC metadata but not packaging rules.

        Returns dict with counts: quals, uocs, links
        """
        counts = {"quals": 0, "uocs": 0, "memberships": 0}

        if use_soap:
            try:
                return self._run_soap(tp_codes, progress_callback, counts)
            except Exception as e:
                log.warning(f"SOAP ingestion failed ({e}), falling back to REST")

        return self._run_rest(tp_codes, progress_callback, counts)

    # ── SOAP ingestion ─────────────────────────────────────────────────────────
    def _run_soap(self, tp_codes, progress_callback, counts) -> dict:
        client = self._get_client()

        if progress_callback:
            progress_callback(0.05, "Fetching current qualifications from TGA…")

        # Get current qualifications
        search_result = client.service.Search(
            InformationRequest={
                "SearchCriteria": {
                    "UsageRecommendation": "Current",
                    "ClassificationScheme": tp_codes or [],
                },
                "IncludeData": "TrainingPackage,Qualifications",
            }
        )

        quals = [q for q in (search_result.Qualifications or [])
                 if q.UsageRecommendation == "Current"]

        if tp_codes:
            quals = [q for q in quals
                     if any(q.Code.startswith(tp) for tp in tp_codes)]

        total = len(quals)
        log.info(f"Found {total} current qualifications")

        for i, qual in enumerate(quals):
            if progress_callback:
                progress_callback(0.05 + 0.85 * (i / max(total, 1)),
                                  f"Processing {qual.Code} ({i+1}/{total})…")
            try:
                self._ingest_qual_soap(client, qual)
                counts["quals"] += 1
            except Exception as e:
                log.error(f"Failed to ingest {qual.Code}: {e}")

        if progress_callback:
            progress_callback(1.0, f"Ingested {counts['quals']} qualifications")
        return counts

    def _ingest_qual_soap(self, client, qual_summary):
        """Fetch full ContentBundle for a qualification and persist."""
        time.sleep(RATE_LIMIT)
        bundle = client.service.GetQualification(
            Code=qual_summary.Code,
            InformationRequest={"IncludeData": "ContentBundle"},
        )

        # Persist qual
        self._upsert_qual(
            code=qual_summary.Code,
            title=qual_summary.Title,
            tp_code=qual_summary.Code[:3],
            aqf_level=getattr(bundle, "AQFLevel", None),
            status="Current",
        )

        # Extract and persist classifications
        for cls in getattr(bundle, "Classifications", None) or []:
            self._upsert_qual_taxonomy(
                qual_code=qual_summary.Code,
                scheme=cls.Scheme,
                code=getattr(cls, "Code", None),
                value=getattr(cls, "Value", "") or "",
            )

        # Extract packaging rules
        rules = getattr(bundle, "PackagingRules", None)
        if rules:
            for u in getattr(rules, "CoreUnits", None) or []:
                uoc_code = u.Code
                is_imported = not uoc_code.startswith(qual_summary.Code[:3])
                self._upsert_uoc(uoc_code, getattr(u, "Title", ""),
                                 uoc_code[:3])
                self._upsert_membership(uoc_code, qual_summary.Code,
                                        "core", None, uoc_code[:3],
                                        is_imported)
                counts = getattr(self, "_counts", None)

            for grp in getattr(rules, "ElectiveGroups", None) or []:
                grp_name = getattr(grp, "GroupName", None)
                for u in getattr(grp, "ElectiveUnits", None) or []:
                    uoc_code = u.Code
                    is_imported = not uoc_code.startswith(qual_summary.Code[:3])
                    self._upsert_uoc(uoc_code, getattr(u, "Title", ""),
                                     uoc_code[:3])
                    self._upsert_membership(uoc_code, qual_summary.Code,
                                            "elective", grp_name,
                                            uoc_code[:3], is_imported)

    # ── REST ingestion (fallback) ──────────────────────────────────────────────
    def _run_rest(self, tp_codes, progress_callback, counts) -> dict:
        """
        REST-only mode — no packaging rules, but registers quals and UOCs.
        Suitable for initial setup without SOAP credentials.
        """
        if progress_callback:
            progress_callback(0.05, "Fetching qualifications via REST API…")

        tp_list = tp_codes or [None]
        all_quals = []
        for tp in tp_list:
            all_quals.extend(self._rest_search_qualifications(tp))

        # Deduplicate
        seen = set()
        quals = [q for q in all_quals
                 if q["code"] not in seen and not seen.add(q["code"])]

        total = len(quals)
        log.info(f"Found {total} qualifications via REST")

        for i, q in enumerate(quals):
            if progress_callback:
                progress_callback(0.05 + 0.90 * (i / max(total, 1)),
                                  f"Ingesting {q['code']} ({i+1}/{total})…")
            try:
                detail = self._rest_get_details(q["code"])
                self._ingest_qual_rest(q, detail)
                counts["quals"] += 1
            except Exception as e:
                log.error(f"Failed REST ingest for {q['code']}: {e}")

        if progress_callback:
            progress_callback(1.0, f"Ingested {counts['quals']} qualifications (REST mode)")
        return counts

    def _ingest_qual_rest(self, qual_summary: dict, detail: dict):
        """Persist a qualification from REST response."""
        code  = qual_summary["code"]
        title = qual_summary.get("title", "")
        tp    = code[:3]

        self._upsert_qual(code, title, tp,
                          aqf_level=detail.get("aqfLevel"),
                          status="Current")

        # Classifications from REST (partial — not all fields available)
        for cls in detail.get("classifications", []):
            self._upsert_qual_taxonomy(
                qual_code=code,
                scheme=cls.get("scheme", ""),
                code=cls.get("code"),
                value=cls.get("value", ""),
            )

        # Extract UOC codes from component list if available
        for comp in detail.get("components", []):
            uoc_code = comp.get("code", "")
            if not uoc_code:
                continue
            mtype = "core" if comp.get("type") == "Core" else "elective"
            is_imported = not uoc_code.startswith(tp)
            self._upsert_uoc(uoc_code, comp.get("title", ""), uoc_code[:3])
            self._upsert_membership(uoc_code, code, mtype, None,
                                    uoc_code[:3], is_imported)

    # ── DB helpers ─────────────────────────────────────────────────────────────
    def _upsert_qual(self, code, title, tp_code, aqf_level=None,
                     status="Current"):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO qual_registry
                    (qual_code, qual_title, tp_code, aqf_level, status)
                VALUES (:code, :title, :tp, :aqf, :status)
                ON CONFLICT (qual_code) DO UPDATE SET
                    qual_title = EXCLUDED.qual_title,
                    status     = EXCLUDED.status,
                    aqf_level  = COALESCE(EXCLUDED.aqf_level,
                                          qual_registry.aqf_level)
            """), {"code": code, "title": title, "tp": tp_code,
                   "aqf": aqf_level, "status": status})

    def _upsert_uoc(self, code, title, tp_code):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO uoc_registry
                    (uoc_code, uoc_title, tp_code, usage_recommendation)
                VALUES (:code, :title, :tp, 'Current')
                ON CONFLICT (uoc_code) DO UPDATE SET
                    uoc_title = EXCLUDED.uoc_title
            """), {"code": code, "title": title or "", "tp": tp_code})

    def _upsert_qual_taxonomy(self, qual_code, scheme, code, value):
        if not scheme or not value:
            return
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO qual_taxonomy_links
                    (qual_code, scheme, code, value, pipeline_run_id)
                VALUES (:qc, :sc, :co, :val, :rid)
                ON CONFLICT (qual_code, scheme, COALESCE(code, ''))
                DO UPDATE SET value = EXCLUDED.value
            """), {"qc": qual_code, "sc": scheme, "co": code,
                   "val": value, "rid": self.pipeline_run_id})

    def _upsert_membership(self, uoc_code, qual_code, mtype,
                           group, owner_tp, is_imported):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO uoc_qual_memberships
                    (uoc_code, qual_code, membership_type, elective_group,
                     owner_tp_code, is_imported, pipeline_run_id)
                VALUES (:uc, :qc, :mt, :grp, :otp, :imp, :rid)
                ON CONFLICT (uoc_code, qual_code, membership_type) DO NOTHING
            """), {"uc": uoc_code, "qc": qual_code, "mt": mtype,
                   "grp": group, "otp": owner_tp, "imp": is_imported,
                   "rid": self.pipeline_run_id})

    # ── Seed from existing rsd_skill_records ───────────────────────────────────
    def seed_from_rsd_records(self) -> int:
        """
        Bootstrap uoc_registry and qual_registry from the existing
        rsd_skill_records table — useful before TGA ingestion runs.
        Returns number of UOCs seeded.
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT unit_code, unit_title, tp_code, tp_title
                FROM rsd_skill_records
                WHERE unit_code IS NOT NULL AND unit_code != ''
            """)).fetchall()

        count = 0
        for row in rows:
            uoc_code = str(row[0]).strip()
            uoc_title = str(row[1] or "").strip()
            tp_code   = str(row[2] or uoc_code[:3]).strip() or uoc_code[:3]
            tp_title  = str(row[3] or "").strip()

            if not uoc_code:
                continue

            # Ensure qual placeholder if not exists
            self._upsert_uoc(uoc_code, uoc_title, tp_code)
            count += 1

        log.info(f"Seeded {count} UOCs from rsd_skill_records")
        return count
