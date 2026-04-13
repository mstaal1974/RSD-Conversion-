"""
core/tga_client.py
~~~~~~~~~~~~~~~~~~
Thin wrapper around the three TGA (training.gov.au) SOAP web services:
  - TrainingComponentService  – unit/qualification/skill-set lookups
  - OrganisationService       – RTO details
  - ClassificationService     – ANZSCO / ANZSIC codes

Authentication uses WS-Security UsernameToken (PasswordText).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------
_BASE = {
    "sandbox":    "https://ws.sandbox.training.gov.au/Deewr.Tga.Webservices",
    "production": "https://ws.training.gov.au/Deewr.Tga.Webservices",
}

_SERVICES = {
    "training":       "TrainingComponentService.svc",
    "organisation":   "OrganisationService.svc",
    "classification": "ClassificationService.svc",
}


def _wsdl(env: str, service: str) -> str:
    return f"{_BASE[env]}/{_SERVICES[service]}?wsdl"


# ---------------------------------------------------------------------------
# TGA Client
# ---------------------------------------------------------------------------
class TGAClient:
    """Lazy-initialising client for all three TGA web services."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        env: str | None = None,
    ) -> None:
        self.username = username or os.getenv("TGA_USERNAME", "WebService.Read")
        self.password = password or os.getenv("TGA_PASSWORD", "Asdf098")
        self.env = (env or os.getenv("TGA_ENV", "sandbox")).lower()
        if self.env not in _BASE:
            raise ValueError(f"TGA_ENV must be 'sandbox' or 'production', got {self.env!r}")
        self._clients: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_client(self, service: str):
        """Return (and cache) a zeep Client for the requested service."""
        if service not in self._clients:
            try:
                from zeep import Client
                from zeep.wsse.username import UsernameToken
            except ImportError as exc:
                raise ImportError("zeep is required: pip install zeep") from exc

            wsdl = _wsdl(self.env, service)
            log.debug("Connecting to TGA %s: %s", service, wsdl)
            self._clients[service] = Client(
                wsdl,
                wsse=UsernameToken(self.username, self.password),
            )
        return self._clients[service]

    @property
    def _training(self):
        return self._get_client("training")

    @property
    def _organisation(self):
        return self._get_client("organisation")

    @property
    def _classification(self):
        return self._get_client("classification")

    # ------------------------------------------------------------------
    # Training Component Service
    # ------------------------------------------------------------------
    def search_units(
        self,
        title: str = "",
        code: str = "",
        training_package_code: str = "",
        include_superseded: bool = False,
        page_size: int = 50,
    ) -> list[dict]:
        """Search for Units of Competency."""
        svc = self._training

        # Build filter — TPComponentTypes uses a nested zeep type,
        # NOT a plain Python list
        filter_obj = {
            "Code":                   code or None,
            "Title":                  title or None,
            "TrainingPackageCode":    training_package_code or None,
            "IncludeSuperseded":      include_superseded,
            "ClassificationFilters":  None,
            "FieldOfEducationFilters": None,
        }

        # Only add TPComponentTypes if the WSDL type supports it
        try:
            tc_type = svc.get_type("ns0:ArrayOfTrainingComponentTypeFilter")
            filter_obj["TPComponentTypes"] = tc_type(
                TrainingComponentTypeFilter=["UnitOfCompetency"]
            )
        except Exception:
            pass  # Skip if type not available — API will return all types

        request = {
            "Filter":   filter_obj,
            "StartRow": 1,
            "RowCount": page_size,
            "OrderBy":  "Code",
        }

        try:
            response = svc.service.Search(request)
        except Exception as exc:
            log.error("TGA search_units failed: %s", exc)
            raise

        results = []
        for item in _safe_list(response, "Results", "TrainingComponentSummary"):
            results.append({
                "code":                  _v(item, "Code"),
                "title":                 _v(item, "Title"),
                "training_package_code": _v(item, "TrainingPackageCode"),
                "status":                _v(item, "Status"),
                "release_date":          _v(item, "ReleaseDate"),
                "type":                  _v(item, "TrainingComponentType"),
            })
        return results

    def get_unit(self, code: str) -> dict:
        """Fetch full detail for a single Unit of Competency."""
        svc = self._training
        try:
            response = svc.service.GetDetails(
                {"Code": code, "ShowReleases": False}
            )
        except Exception as exc:
            log.error("TGA get_unit(%s) failed: %s", code, exc)
            raise
        return _parse_unit_detail(response)

    def unit_to_dataframe(self, code: str) -> pd.DataFrame:
        """Fetch a unit and return a normalised DataFrame."""
        detail = self.get_unit(code)
        rows = []
        for elem in detail.get("elements", []):
            for pc in elem.get("performance_criteria", []):
                rows.append({
                    "unit_code":     detail["code"],
                    "unit_title":    detail["title"],
                    "element_num":   elem["number"],
                    "element_title": elem["title"],
                    "pc_num":        pc["number"],
                    "pc_text":       pc["text"],
                })
        if not rows:
            log.warning("unit_to_dataframe: no PCs found for %s", code)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Organisation Service
    # ------------------------------------------------------------------
    def get_organisation(self, rto_code: str) -> dict:
        svc = self._organisation
        try:
            response = svc.service.Get({"OrganisationCode": rto_code})
        except Exception as exc:
            log.error("TGA get_organisation(%s) failed: %s", rto_code, exc)
            raise
        return _parse_organisation(response)

    def get_rto_scope(self, rto_code: str) -> list[dict]:
        svc = self._organisation
        try:
            response = svc.service.GetScopeByOrganisationCode(
                {"OrganisationCode": rto_code}
            )
        except Exception as exc:
            log.error("TGA get_rto_scope(%s) failed: %s", rto_code, exc)
            raise
        results = []
        for item in _safe_list(response, "OrganisationScopeItems", "OrganisationScopeItem"):
            results.append({
                "code":   _v(item, "ComponentCode"),
                "title":  _v(item, "ComponentTitle"),
                "type":   _v(item, "TrainingComponentType"),
                "status": _v(item, "RegistrationStatus"),
            })
        return results

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        """Quick connectivity check — just verify the WSDL loads."""
        try:
            client = self._get_client("training")
            return client is not None
        except Exception as exc:
            log.warning("TGA ping failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _v(obj: Any, key: str, default: str = "") -> str:
    if obj is None:
        return default
    val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return str(val) if val is not None else default


def _safe_list(response: Any, *keys: str) -> list:
    obj = response
    for k in keys:
        if obj is None:
            return []
        obj = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]


def _parse_unit_detail(response: Any) -> dict:
    if response is None:
        return {}
    unit: dict[str, Any] = {
        "code":                  _v(response, "Code"),
        "title":                 _v(response, "Title"),
        "status":                _v(response, "Status"),
        "release_date":          _v(response, "ReleaseDate"),
        "description":           _v(response, "ApplicationOfUnit"),
        "training_package_code": _v(response, "TrainingPackageCode"),
        "training_package_title":_v(response, "TrainingPackageTitle"),
        "unit_descriptor":       _v(response, "UnitDescriptor"),
        "elements":              [],
        "knowledge_evidence":    _v(response, "KnowledgeEvidence"),
        "performance_evidence":  _v(response, "PerformanceEvidence"),
        "assessment_conditions": _v(response, "AssessmentConditions"),
        "foundation_skills":     _v(response, "FoundationSkills"),
    }
    for elem in _safe_list(response, "Elements", "Element"):
        element: dict[str, Any] = {
            "number": _v(elem, "Num"),
            "title":  _v(elem, "Title"),
            "performance_criteria": [],
        }
        for pc in _safe_list(elem, "PerformanceCriteria", "PerformanceCriterion"):
            element["performance_criteria"].append({
                "number": _v(pc, "Num"),
                "text":   _v(pc, "Description"),
            })
        unit["elements"].append(element)
    return unit


def _parse_organisation(response: Any) -> dict:
    if response is None:
        return {}
    return {
        "code":         _v(response, "Code"),
        "legal_name":   _v(response, "LegalName"),
        "trading_name": _v(response, "TradingName"),
        "status":       _v(response, "Status"),
        "state":        _v(response, "AddressState"),
        "postcode":     _v(response, "AddressPostcode"),
    }
