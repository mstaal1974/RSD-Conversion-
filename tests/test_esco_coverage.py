"""
tests/test_esco_coverage.py

Unit tests for the three-tier ESCO coverage classifier. Pure stdlib — no
numpy/pandas/torch required, so this runs anywhere.
"""
from core.esco_coverage import (
    CLEAN, PARTIAL, NONE,
    CoverageThresholds, MatchRecord,
    classify_match, lexical_agreement, summarize,
)


# ── lexical_agreement ────────────────────────────────────────────────────────

def test_lexical_exact_label_terms_present():
    # Every content word of the label appears in the statement.
    stmt = "Provide emergency first aid to an injured worker on site."
    assert lexical_agreement(stmt, "provide first aid") == 1.0


def test_lexical_partial_label_terms():
    # 'aid' present, 'first' present, 'cardiopulmonary' absent → 2/3.
    stmt = "Give first aid support to patients."
    assert abs(lexical_agreement(stmt, "first aid cardiopulmonary") - (2 / 3)) < 1e-9


def test_lexical_stems_inflections():
    # manage/managing/management collapse via light stemming.
    assert lexical_agreement("managing a small team", "manage teams") == 1.0


def test_lexical_empty_label_is_zero():
    assert lexical_agreement("anything at all", "") == 0.0
    assert lexical_agreement("", "welding") == 0.0


# ── classify_match ───────────────────────────────────────────────────────────

def test_clean_needs_both_signals():
    assert classify_match(semantic=0.72, lexical=0.6) == CLEAN


def test_high_semantic_low_lexical_is_partial():
    # Strong paraphrase, but the ESCO label's words aren't present → granularity
    # / scope mismatch, not a clean match.
    assert classify_match(semantic=0.72, lexical=0.1) == PARTIAL


def test_mid_semantic_is_partial():
    assert classify_match(semantic=0.50, lexical=0.9) == PARTIAL


def test_below_floor_is_none():
    assert classify_match(semantic=0.30, lexical=0.9) == NONE


def test_margin_gate_downgrades_ambiguous_clean():
    # Would be clean on semantic+lexical, but it sits between two ESCO skills.
    assert classify_match(semantic=0.70, lexical=0.6, margin=0.005) == PARTIAL
    assert classify_match(semantic=0.70, lexical=0.6, margin=0.10) == CLEAN


def test_margin_ignored_when_none():
    assert classify_match(semantic=0.70, lexical=0.6, margin=None) == CLEAN


# ── summarize ────────────────────────────────────────────────────────────────

def test_summary_percentages_sum_to_100():
    recs = [
        MatchRecord("provide first aid to a casualty", "provide first aid", 0.80),
        MatchRecord("weld steel using MIG techniques", "weld metal", 0.62),
        MatchRecord("paraphrased broad capability", "manage stakeholders", 0.66,
                    lexical=0.0),                                   # partial
        MatchRecord("mid confidence neighbour", "operate machinery", 0.50),  # partial
        MatchRecord("uniquely australian regulatory capability", "x", 0.20),  # none
    ]
    s = summarize(recs)
    assert s.total == 5
    assert s.counts[CLEAN] == 2
    assert s.counts[PARTIAL] == 2
    assert s.counts[NONE] == 1
    assert abs(s.pct(CLEAN) + s.pct(PARTIAL) + s.pct(NONE) - 100.0) < 1e-9


def test_summary_render_shape():
    recs = [MatchRecord("provide first aid", "provide first aid", 0.80)]
    text = summarize(recs).render()
    assert "Across the canonical corpus (1 statements):" in text
    assert "matched cleanly" in text
    assert "found no credible match" in text


def test_reverse_summary_labels_and_unmatched():
    # 1 covered, 1 partial, 2 with no Australian source.
    recs = [
        MatchRecord("provide first aid to a casualty", "provide first aid", 0.80),
        MatchRecord("loosely related statement", "operate machinery", 0.50),
        MatchRecord("weak", "operate nuclear reactor", 0.20),
        MatchRecord("weak", "trade carbon derivatives", 0.15),
    ]
    s = summarize(recs, direction="reverse")
    assert s.direction == "reverse"
    assert s.counts[NONE] == 2
    assert abs(s.unmatched_pct() - 50.0) < 1e-9
    text = s.render()
    assert "Across the ESCO skill taxonomy (4 ESCO skills):" in text
    assert "no Australian source" in text
    assert "found 50.0% of ESCO unmatched" in text


def test_forward_render_unchanged_by_direction_default():
    recs = [MatchRecord("provide first aid", "provide first aid", 0.80)]
    text = summarize(recs).render()  # default forward
    assert "Across the canonical corpus" in text
    assert "matched cleanly" in text


def test_empty_corpus_is_zero_not_crash():
    s = summarize([])
    assert s.total == 0
    assert s.pct(CLEAN) == 0.0


def test_thresholds_validate():
    import pytest
    with pytest.raises(ValueError):
        CoverageThresholds(sem_none=0.9, sem_clean=0.5)
