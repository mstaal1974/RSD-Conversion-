"""Tests for extractor modules."""
import pytest
import pandas as pd
from core.extractors.blob_extractor import BlobExtractor, _parse_blob
from core.extractors.row_per_pc_extractor import RowPerPCExtractor
from core.extractor import normalize_training_package_csv, content_fingerprint


# ── Fixtures ──────────────────────────────────────────────────────────────────

BLOB_SAMPLE = """\
1 Prepare to work safely
Performance criteria
1.1 Identify hazards in the workplace and report them to the supervisor
1.2 Follow WHS procedures and policies at all times
1.3 Use personal protective equipment as required

2 Complete work tasks
Performance criteria
2.1 Use correct tools and equipment for each task
2.2 Follow supervisor instructions and workplace guidelines
"""

BLOB_DF = pd.DataFrame({
    "unit_code": ["BSBWHS201"],
    "unit_title": ["Contribute to health and safety of self and others"],
    "elements_and_pcs": [BLOB_SAMPLE],
})

ROW_PC_DF = pd.DataFrame({
    "unit_code":      ["BSBWHS201"] * 5,
    "unit_title":     ["Contribute to health and safety"] * 5,
    "element_title":  ["Prepare to work safely"] * 3 + ["Complete work tasks"] * 2,
    "performance_criteria": [
        "1.1 Identify hazards in the workplace",
        "1.2 Follow WHS procedures",
        "1.3 Use PPE as required",
        "2.1 Use correct tools",
        "2.2 Follow supervisor instructions",
    ],
})


# ── _parse_blob ───────────────────────────────────────────────────────────────

def test_parse_blob_element_count():
    elements = _parse_blob(BLOB_SAMPLE)
    assert len(elements) == 2

def test_parse_blob_element_titles():
    elements = _parse_blob(BLOB_SAMPLE)
    assert elements[0][1] == "Prepare to work safely"
    assert elements[1][1] == "Complete work tasks"

def test_parse_blob_pc_count():
    elements = _parse_blob(BLOB_SAMPLE)
    assert len(elements[0][2]) == 3
    assert len(elements[1][2]) == 2

def test_parse_blob_pc_numbering():
    elements = _parse_blob(BLOB_SAMPLE)
    assert elements[0][2][0].startswith("1.1")

def test_parse_blob_empty():
    assert _parse_blob("") == []

def test_parse_blob_no_pcs():
    blob = "1 Some element\nNo performance criteria here\n"
    elements = _parse_blob(blob)
    # Element found but no PCs
    assert len(elements) == 1
    assert elements[0][2] == []


# ── BlobExtractor ─────────────────────────────────────────────────────────────

def test_blob_extractor_scores_high():
    ext = BlobExtractor()
    score = ext.score(BLOB_DF)
    assert score >= 0.3

def test_blob_extractor_normalises_shape():
    ext = BlobExtractor()
    result = ext.extract(BLOB_DF)
    assert list(result.columns) == [
        "unit_code", "unit_title", "element_title", "pcs_text",
        "tp_code", "tp_title",
    ]

def test_blob_extractor_row_count():
    ext = BlobExtractor()
    result = ext.extract(BLOB_DF)
    assert len(result) == 2

def test_blob_extractor_unit_code_propagated():
    ext = BlobExtractor()
    result = ext.extract(BLOB_DF)
    assert (result["unit_code"] == "BSBWHS201").all()

def test_blob_extractor_no_pc_tokens_in_element_title():
    import re
    ext = BlobExtractor()
    result = ext.extract(BLOB_DF)
    for title in result["element_title"]:
        assert not re.search(r"\b\d+\.\d+\b", title), f"PC token in element title: {title!r}"


# ── RowPerPCExtractor ─────────────────────────────────────────────────────────

def test_row_pc_extractor_scores_high():
    ext = RowPerPCExtractor()
    score = ext.score(ROW_PC_DF)
    assert score >= 0.5

def test_row_pc_extractor_normalises_shape():
    ext = RowPerPCExtractor()
    result = ext.extract(ROW_PC_DF)
    assert list(result.columns) == ["unit_code", "unit_title", "element_title", "pcs_text"]

def test_row_pc_extractor_row_count():
    ext = RowPerPCExtractor()
    result = ext.extract(ROW_PC_DF)
    assert len(result) == 2

def test_row_pc_extractor_pcs_joined():
    ext = RowPerPCExtractor()
    result = ext.extract(ROW_PC_DF)
    elem1 = result[result["element_title"] == "Prepare to work safely"].iloc[0]
    assert "1.1" in elem1["pcs_text"]
    assert "1.2" in elem1["pcs_text"]


# ── Auto-detection ─────────────────────────────────────────────────────────────

def test_auto_detects_blob_format():
    _, ext_name, scorecard = normalize_training_package_csv(BLOB_DF)
    assert ext_name == "training_gov_blob"
    assert scorecard is not None

def test_auto_detects_row_pc_format():
    _, ext_name, _ = normalize_training_package_csv(ROW_PC_DF)
    assert ext_name == "row_per_pc"

def test_forced_extractor():
    result, ext_name, scorecard = normalize_training_package_csv(BLOB_DF, "training_gov_blob")
    assert ext_name == "training_gov_blob"
    assert scorecard is None  # no scorecard when forced

def test_bad_csv_raises():
    bad_df = pd.DataFrame({"col_a": ["hello", "world"], "col_b": [1, 2]})
    with pytest.raises(ValueError, match="No extractor scored"):
        normalize_training_package_csv(bad_df)


# ── Fingerprinting ────────────────────────────────────────────────────────────

def test_same_df_same_fingerprint():
    fp1 = content_fingerprint(BLOB_DF)
    fp2 = content_fingerprint(BLOB_DF.copy())
    assert fp1 == fp2

def test_different_data_different_fingerprint():
    df2 = BLOB_DF.copy()
    df2.loc[0, "unit_code"] = "CHANGED999"
    assert content_fingerprint(BLOB_DF) != content_fingerprint(df2)

def test_fingerprint_is_hex_string():
    fp = content_fingerprint(BLOB_DF)
    assert len(fp) == 32
    int(fp, 16)  # should not raise
