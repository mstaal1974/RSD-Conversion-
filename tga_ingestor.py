"""
tga_ingestor.py  (root-level, used by linkage_engine and admin pages)

TGA National Training Register ingestion via SOAP web services.

Key operations:
  - Authenticate with TGA credentials
  - Fetch all current qualifications (filtered — no superseded)
  - Parse ContentBundle: Classifications + PackagingRules
  - Detect imported units (owned by different TP)
  - Upsert into uoc_registry, qual_registry, qual_taxonomy_links,
    uoc_qual_memberships, uoc_classifications

Endpoint reference
------------------
  Sandbox  (TGA_ENV=sandbox):
    TrainingComponent : https://ws.sandbox.training.gov.au/Deewr.Tga.Webservices/TrainingComponentService.svc
    Organisation      : https://ws.sandbox.training.gov.au/Deewr.Tga.WebServices/OrganisationService.svc
    Classification    : https://ws.sandbox.training.gov.au/Deewr.Tga.Webservices/ClassificationService.svc

  Production (TGA_ENV=production):
    TrainingComponent : https://ws.training.gov.au/Deewr.Tga.Webservices/TrainingComponentService.svc
    Organisation      : https://ws.training.gov.au/Deewr.Tga.WebServices/OrganisationService.svc
    Classification    : https://ws.training.gov.au/Deewr.Tga.Webservices/ClassificationService.svc

  Credentials (read-only sandbox):
    Username: WebService.Read
    Password: Asdf098
"""
from __future__ import annotations
import os
import time
import logging
from sqlalchemy import text, Engine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence weights — used by linkage engine
# ---------------------------------------------------------------------------
CONF = {
    "direct":            1.00,
    "core_native":       0.85,
    "core_imported":     0.70,
    "elective_native":   0.50,
    "elective_imported": 0.40,
    "asc_specialist":    0.30,
}

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------
_ENDPOINTS = {
    "sandbox": {
        "training":       "https://ws.sandbox.training.gov.au/Deewr.Tga.Webservices/TrainingComponentService.svc?wsdl",
        "organisation":   "https://ws.sandbox.training.gov.au/Deewr.Tga.WebServices/OrganisationService.svc?wsdl",
        "classification": "https://ws.sandbox.training.gov.au/Deewr.Tga.Webservices/ClassificationService.svc?wsdl",
    },
    "production": {
        "training":       "https://ws.training.gov.au/Deewr.Tga.Webservices/TrainingComponentService.svc?wsdl",
        "organisation":   "https://ws.training.gov.au/Deewr.Tga.WebServices/OrganisationService.svc?wsdl",
        "classification": "https://ws.training.gov.au/Deewr.Tga.Webservices/ClassificationService.svc?wsdl",
    },
}

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
                 pipeline_run_id: int | None = None,
                 env: str | None = None):
        self.engine          = engine
        self.username        = username or os.getenv("TGA_USERNAME", "WebService.Read")
        self.password        = password or os.getenv("TGA_PASSWORD", "Asdf098")
        self.pipeline_run_id = pipeline_run_id
        self.env             = (env or os.getenv("TGA_ENV", "sandbox")).lower()
        if self.env not in _ENDPOINTS:
            raise ValueError(f"TGA_ENV must be 'sandbox' or 'production', got {self.env!r}")
        self._clients: dict[str, object] = {}

    # ── SOAP clients ──────────────────────────────────────────────────────────
    def _get_client(self, service: str = "training"):
        """Lazy-initialise zeep SOAP client for the requested service."""
        if service not in self._clients:
            try:
                from zeep import Client
                from zeep.wsse.username import UsernameToken
                wsdl = _ENDPOINTS[self.env][service]
                log.info("Connecting to TGA %s (%s): %s", service, self.env, wsdl)
                self._clients[service] = Client(
                    wsdl,
                    wsse=UsernameToken(self.username, self.password),
                )
                log.info("TGA %s client ready", service)
            except Exception as e:
                raise RuntimeError(f"TGA SOAP connection failed ({service}): {e}") from e
        return self._clients[service]

    @property
    def _training_client(self):
        return self._get_client("training")

    @property
    def _org_client(self):
        return self._get_client("organisation")

    @property
    def _class_client(self):
        return self._get_client("classification")

    # ── REST fallback helpers ─────────────────────────────────────────────────
    def _rest_search_qualifications(self, tp_code: str | None = None) -> list[dict]:
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
                    "code":    attrs.get("code", ""),
                    "title":   attrs.get("title", ""),
                    "tp_code": attrs.get("code", "")[:3] if attrs.get("code") else "",
                    "status":  attrs.get("usageRecommendation", "Current"),
                })
            return results
        except Exception as e:
            log.warning("TGA REST search failed: %s", e)
            return []

    def _rest_get_details(self, code: str) -> dict:
        import urllib.request, json

        url = f"https://training.gov.au/api/training/details/{code}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            time.sleep(0.3)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            log.warning("REST details failed for %s: %s", code, e)
            return {}

    # ── Main ingestion entry point ────────────────────────────────────────────
    def run(self,
            tp_codes: list[str] | None = None,
            use_soap: bool = True,
            progress_callback=None) -> dict:
        counts = {"quals": 0, "uocs": 0, "memberships": 0}

          if use_soap:
            return self._run_soap(tp_codes, progress_callback, counts)

        return self._run_rest(tp_codes, progress_callback, counts)
              
    # ── SOAP ingestion ────────────────────────────────────────────────────────
    def _run_soap(self, tp_codes, progress_callback, counts) -> dict:
        client = self._training_client

        if progress_callback:
            progress_callback(0.05, "Fetching current qualifications from TGA…")

        # Build filter — TPComponentTypes must use the proper zeep type,
        # NOT a raw Python dict with a list value
        filter_obj = {
            "ClassificationFilters":   None,
            "FieldOfEducationFilters": None,
            "IncludeSuperseded":       False,
            "TrainingPackageCode":     tp_codes[0] if tp_codes and len(tp_codes) == 1 else None,
        }
        try:
            tc_type = client.get_type("ns0:ArrayOfTrainingComponentTypeFilter")
            filter_obj["TPComponentTypes"] = tc_type(
                TrainingComponentTypeFilter=["Qualification"]
            )
        except Exception:
            pass  # Skip type filter — API returns all types, we filter below

        search_result = client.service.Search({
            "Filter":   filter_obj,
            "StartRow": 1,
            "RowCount": 500,
            "OrderBy":  "Code",
        })

        quals = _safe_list(search_result, "Results", "TrainingComponentSummary")

        # Keep only Qualifications and filter to requested TPs
        quals = [q for q in quals
                 if _v(q, "TrainingComponentType") in ("Qualification", "")]
        if tp_codes and len(tp_codes) > 1:
            quals = [q for q in quals
                     if any(_v(q, "Code").startswith(tp) for tp in tp_codes)]

        total = len(quals)
        log.info("Found %d current qualifications", total)

        for i, qual in enumerate(quals):
            if progress_callback:
                progress_callback(
                    0.05 + 0.85 * (i / max(total, 1)),
                    f"Processing {_v(qual, 'Code')} ({i+1}/{total})…"
                )
            try:
                self._ingest_qual_soap(client, qual, counts)
                counts["quals"] += 1
            except Exception as e:
                log.error("Failed to ingest %s: %s", _v(qual, "Code"), e)

        if progress_callback:
            progress_callback(1.0, f"Ingested {counts['quals']} qualifications")
        return counts

    def _ingest_qual_soap(self, client, qual_summary, counts):
        """Fetch full details for a qualification and persist."""
        code = _v(qual_summary, "Code")
        time.sleep(RATE_LIMIT)

        detail = client.service.GetDetails({"Code": code, "ShowReleases": False})

        self._upsert_qual(
            code=code,
            title=_v(qual_summary, "Title"),
            tp_code=code[:3],
            aqf_level=_v(detail, "AQFLevel") or None,
            status="Current",
        )

        # Classifications
        for cls in _safe_list(detail, "Classifications", "Classification"):
            self._upsert_qual_taxonomy(
                qual_code=code,
                scheme=_v(cls, "Scheme"),
                code=_v(cls, "Code") or None,
                value=_v(cls, "Value"),
            )

        # Packaging rules — CoreUnits
        for u in _safe_list(detail, "CoreUnits", "Unit"):
            uoc_code = _v(u, "Code")
            if not uoc_code:
                continue
            is_imported = not uoc_code.startswith(code[:3])
            self._upsert_uoc(uoc_code, _v(u, "Title"), uoc_code[:3])
            self._upsert_membership(uoc_code, code, "core", None,
                                    uoc_code[:3], is_imported)
            counts["uocs"] += 1
            counts["memberships"] += 1

        # Packaging rules — ElectiveGroups
        for grp in _safe_list(detail, "ElectiveGroups", "ElectiveGroup"):
            grp_name = _v(grp, "GroupName") or None
            for u in _safe_list(grp, "ElectiveUnits", "Unit"):
                uoc_code = _v(u, "Code")
                if not uoc_code:
                    continue
                is_imported = not uoc_code.startswith(code[:3])
                self._upsert_uoc(uoc_code, _v(u, "Title"), uoc_code[:3])
                self._upsert_membership(uoc_code, code, "elective", grp_name,
                                        uoc_code[:3], is_imported)
                counts["uocs"] += 1
                counts["memberships"] += 1

    # ── REST ingestion (fallback) ─────────────────────────────────────────────
    def _run_rest(self, tp_codes, progress_callback, counts) -> dict:
        if progress_callback:
            progress_callback(0.05, "Fetching qualifications via REST API…")

        tp_list = tp_codes or [None]
        all_quals = []
        for tp in tp_list:
            all_quals.extend(self._rest_search_qualifications(tp))

        seen: set[str] = set()
        quals = [q for q in all_quals
                 if q["code"] not in seen and not seen.add(q["code"])]

        total = len(quals)
        log.info("Found %d qualifications via REST", total)

        for i, q in enumerate(quals):
            if progress_callback:
                progress_callback(
                    0.05 + 0.90 * (i / max(total, 1)),
                    f"Ingesting {q['code']} ({i+1}/{total})…"
                )
            try:
                detail = self._rest_get_details(q["code"])
                self._ingest_qual_rest(q, detail, counts)
                counts["quals"] += 1
            except Exception as e:
                log.error("Failed REST ingest for %s: %s", q["code"], e)

        if progress_callback:
            progress_callback(1.0, f"Ingested {counts['quals']} qualifications (REST mode)")
        return counts

    def _ingest_qual_rest(self, qual_summary: dict, detail: dict, counts: dict):
        code  = qual_summary["code"]
        title = qual_summary.get("title", "")
        tp    = code[:3]

        self._upsert_qual(code, title, tp,
                          aqf_level=detail.get("aqfLevel"),
                          status="Current")

        for cls in detail.get("classifications", []):
            self._upsert_qual_taxonomy(
                qual_code=code,
                scheme=cls.get("scheme", ""),
                code=cls.get("code"),
                value=cls.get("value", ""),
            )

        for comp in detail.get("components", []):
            uoc_code = comp.get("code", "")
            if not uoc_code:
                continue
            mtype = "core" if comp.get("type") == "Core" else "elective"
            is_imported = not uoc_code.startswith(tp)
            self._upsert_uoc(uoc_code, comp.get("title", ""), uoc_code[:3])
            self._upsert_membership(uoc_code, code, mtype, None,
                                    uoc_code[:3], is_imported)
            counts["uocs"] += 1
            counts["memberships"] += 1

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _upsert_qual(self, code, title, tp_code, aqf_level=None, status="Current"):
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

    # ── Seed from existing rsd_skill_records ──────────────────────────────────
    def seed_from_rsd_records(self) -> int:
        """
        Bootstrap uoc_registry from the existing rsd_skill_records table.
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
            uoc_code  = str(row[0]).strip()
            uoc_title = str(row[1] or "").strip()
            tp_code   = str(row[2] or uoc_code[:3]).strip() or uoc_code[:3]
            if not uoc_code:
                continue
            self._upsert_uoc(uoc_code, uoc_title, tp_code)
            count += 1

        log.info("Seeded %d UOCs from rsd_skill_records", count)
        return count


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _v(obj, key: str, default: str = "") -> str:
    if obj is None:
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return str(val) if val is not None else default


def _safe_list(response, *keys: str) -> list:
    obj = response
    for k in keys:
        if obj is None:
            return []
        obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]
