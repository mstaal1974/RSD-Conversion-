"""
tests/test_tga_client.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests for TGA client parsing helpers — no live network required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from unittest.mock import MagicMock

from core.tga_client import (
    TGAClient,
    _v,
    _safe_list,
    _parse_unit_detail,
    _parse_organisation,
)


# ---------------------------------------------------------------------------
# _v helper
# ---------------------------------------------------------------------------
class TestV:
    def test_dict_hit(self):
        assert _v({"Code": "BSB123"}, "Code") == "BSB123"

    def test_dict_miss_returns_default(self):
        assert _v({}, "Code") == ""
        assert _v({}, "Code", "N/A") == "N/A"

    def test_object_attr(self):
        obj = MagicMock()
        obj.Title = "My Unit"
        assert _v(obj, "Title") == "My Unit"

    def test_none_returns_default(self):
        assert _v(None, "anything") == ""

    def test_none_value_returns_default(self):
        assert _v({"Code": None}, "Code") == ""


# ---------------------------------------------------------------------------
# _safe_list helper
# ---------------------------------------------------------------------------
class TestSafeList:
    def test_nested_dict(self):
        data = {"Elements": {"Element": [{"Num": "1"}, {"Num": "2"}]}}
        assert _safe_list(data, "Elements", "Element") == [{"Num": "1"}, {"Num": "2"}]

    def test_missing_key_returns_empty(self):
        assert _safe_list({}, "Elements", "Element") == []

    def test_none_response_returns_empty(self):
        assert _safe_list(None, "Results") == []

    def test_single_item_wrapped(self):
        # SOAP sometimes returns a single object instead of a list
        data = {"Elements": {"Element": {"Num": "1"}}}
        result = _safe_list(data, "Elements", "Element")
        assert result == [{"Num": "1"}]


# ---------------------------------------------------------------------------
# _parse_unit_detail
# ---------------------------------------------------------------------------
def _mock_unit_response():
    """Build a MagicMock that mimics the zeep GetDetails response."""
    pc1 = MagicMock(); pc1.Num = "1.1"; pc1.Description = "Identify communication needs"
    pc2 = MagicMock(); pc2.Num = "1.2"; pc2.Description = "Select appropriate method"
    pc3 = MagicMock(); pc3.Num = "2.1"; pc3.Description = "Draft the message"

    elem1 = MagicMock(); elem1.Num = "1"; elem1.Title = "Plan communication"
    elem1.PerformanceCriteria.PerformanceCriterion = [pc1, pc2]

    elem2 = MagicMock(); elem2.Num = "2"; elem2.Title = "Deliver communication"
    elem2.PerformanceCriteria.PerformanceCriterion = [pc3]

    resp = MagicMock()
    resp.Code = "BSBCMM411"
    resp.Title = "Make presentations"
    resp.Status = "Current"
    resp.ReleaseDate = "2020-01-01"
    resp.TrainingPackageCode = "BSB"
    resp.TrainingPackageTitle = "Business Services Training Package"
    resp.Elements.Element = [elem1, elem2]
    resp.KnowledgeEvidence = "Knowledge of communication theory"
    resp.PerformanceEvidence = "Must demonstrate ability to communicate"
    resp.AssessmentConditions = "Workplace or simulated environment"
    resp.FoundationSkills = "Reading: Interprets workplace documents"
    return resp


class TestParseUnitDetail:
    def test_basic_fields(self):
        detail = _parse_unit_detail(_mock_unit_response())
        assert detail["code"] == "BSBCMM411"
        assert detail["title"] == "Make presentations"
        assert detail["status"] == "Current"
        assert detail["training_package_code"] == "BSB"

    def test_elements_parsed(self):
        detail = _parse_unit_detail(_mock_unit_response())
        assert len(detail["elements"]) == 2
        assert detail["elements"][0]["number"] == "1"
        assert detail["elements"][0]["title"] == "Plan communication"

    def test_performance_criteria_parsed(self):
        detail = _parse_unit_detail(_mock_unit_response())
        pcs = detail["elements"][0]["performance_criteria"]
        assert len(pcs) == 2
        assert pcs[0]["number"] == "1.1"
        assert pcs[0]["text"] == "Identify communication needs"

    def test_none_response_returns_empty_dict(self):
        assert _parse_unit_detail(None) == {}

    def test_knowledge_evidence(self):
        detail = _parse_unit_detail(_mock_unit_response())
        assert "communication theory" in detail["knowledge_evidence"]


# ---------------------------------------------------------------------------
# unit_to_dataframe (via mocked client)
# ---------------------------------------------------------------------------
class TestUnitToDataframe:
    def _client_with_mock(self):
        client = TGAClient.__new__(TGAClient)
        client.username = "test"
        client.password = "test"
        client.env = "sandbox"
        client._clients = {}
        client.get_unit = lambda code: _parse_unit_detail(_mock_unit_response())
        return client

    def test_returns_dataframe(self):
        client = self._client_with_mock()
        df = client.unit_to_dataframe("BSBCMM411")
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        client = self._client_with_mock()
        df = client.unit_to_dataframe("BSBCMM411")
        expected = {"unit_code", "unit_title", "element_num", "element_title", "pc_num", "pc_text"}
        assert set(df.columns) == expected

    def test_row_count(self):
        # 2 PCs in elem1 + 1 PC in elem2 = 3 rows
        client = self._client_with_mock()
        df = client.unit_to_dataframe("BSBCMM411")
        assert len(df) == 3

    def test_unit_code_propagated(self):
        client = self._client_with_mock()
        df = client.unit_to_dataframe("BSBCMM411")
        assert all(df["unit_code"] == "BSBCMM411")


# ---------------------------------------------------------------------------
# _parse_organisation
# ---------------------------------------------------------------------------
class TestParseOrganisation:
    def test_basic(self):
        resp = MagicMock()
        resp.Code = "1234"
        resp.LegalName = "TAFE NSW"
        resp.TradingName = "TAFE NSW"
        resp.Status = "Active"
        resp.AddressState = "NSW"
        resp.AddressPostcode = "2000"
        org = _parse_organisation(resp)
        assert org["code"] == "1234"
        assert org["legal_name"] == "TAFE NSW"
        assert org["state"] == "NSW"

    def test_none_returns_empty(self):
        assert _parse_organisation(None) == {}


# ---------------------------------------------------------------------------
# TGAClient init
# ---------------------------------------------------------------------------
class TestTGAClientInit:
    def test_defaults(self):
        c = TGAClient()
        assert c.username == "WebService.Read"
        assert c.password == "Asdf098"
        assert c.env == "sandbox"

    def test_explicit_args(self):
        c = TGAClient(username="u", password="p", env="production")
        assert c.env == "production"

    def test_invalid_env_raises(self):
        with pytest.raises(ValueError):
            TGAClient(env="staging")
