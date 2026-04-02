"""Tests for exporter modules."""
import pandas as pd
import pytest
from core.exporters import to_rsd_rows, to_traceability

SAMPLE_DF = pd.DataFrame([{
    "unit_code":       "BSBWHS201",
    "unit_title":      "Contribute to health and safety",
    "element_title":   "Prepare to work safely",
    "pcs_text":        "1.1 Identify hazards\n1.2 Follow WHS",
    "skill_statement": "Apply WHS procedures using PPE to ensure a safe work environment.",
    "bart_prompt":     "Write one BART skill statement...",
    "bart_model":      "gpt-4.1-mini",
    "bart_temperature": 0.2,
    "qa_one_sentence": True,
    "qa_word_count":   12,
    "qa_has_method":   True,
    "qa_has_outcome":  True,
    "qa_passes":       True,
    "rewrite_count":   0,
    "keywords":        "whs; ppe; hazard identification",
    "error_message":   "",
}])


def test_rsd_rows_columns():
    out = to_rsd_rows(SAMPLE_DF)
    expected = {"Unit Code", "Unit Title", "Element Title", "Skill Statement", "Keywords", "QA Pass"}
    assert expected.issubset(set(out.columns))

def test_rsd_rows_count():
    out = to_rsd_rows(SAMPLE_DF)
    assert len(out) == 1

def test_rsd_rows_values():
    out = to_rsd_rows(SAMPLE_DF)
    assert out.iloc[0]["Unit Code"] == "BSBWHS201.1"
    assert out.iloc[0]["RSD Name"] == "Prepare to work safely"
    assert out.iloc[0]["Skill Statement"].startswith("Apply WHS")

def test_traceability_columns():
    out = to_traceability(SAMPLE_DF)
    expected = {"Unit Code", "Element", "Performance Criteria", "Skill Statement", "QA: Passes", "Rewrites"}
    assert expected.issubset(set(out.columns))

def test_traceability_count():
    out = to_traceability(SAMPLE_DF)
    assert len(out) == 1

def test_traceability_includes_prompt():
    out = to_traceability(SAMPLE_DF)
    assert "Prompt" in out.columns

def test_handles_missing_columns_gracefully():
    sparse = pd.DataFrame([{"unit_code": "X", "skill_statement": "Test statement."}])
    out = to_rsd_rows(sparse)
    assert len(out) == 1
    # Missing columns should be filled with empty strings not raise
    assert "Unit Title" in out.columns
