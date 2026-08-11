"""
tests/test_alignment_consensus.py

Pure-stdlib tests for the cross-engine (ours vs LAiSER) ESCO consensus. No
LAiSER, torch, API key, or DB required.
"""
from core.alignment_consensus import (
    VALIDATED, CONFLICT, SINGLE, NONE,
    ConsensusThresholds, consensus_for_statement, labels_match,
    summarize_consensus,
)

ESCO = "http://data.europa.eu/esco/skill/abc-123"
ESCO2 = "http://data.europa.eu/esco/skill/xyz-999"


def _laiser(uri, title, score):
    return {"source_url": uri, "taxonomy_concept": title, "score": score,
            "taxonomy_source": "esco"}


# ── labels_match ─────────────────────────────────────────────────────────────

def test_labels_match_same_concept():
    assert labels_match("provide first aid", "provide first aid") is True
    # word-order / phrasing tolerant
    assert labels_match("welding of carbon steel", "carbon steel welding") is True


def test_labels_match_different_concepts():
    assert labels_match("weld metal", "operate machinery") is False


# ── consensus verdicts ───────────────────────────────────────────────────────

def test_validated_on_same_uri():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="provide first aid", our_score=0.80,
        laiser_esco_rows=[_laiser(ESCO, "administer first aid", 0.9)],
    )
    assert c.verdict == VALIDATED   # URIs match even though labels differ


def test_validated_on_label_when_uris_differ():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="provide first aid", our_score=0.80,
        laiser_esco_rows=[_laiser("", "provide first aid", 0.7)],
    )
    assert c.verdict == VALIDATED


def test_conflict_when_both_fire_but_disagree():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="weld metal", our_score=0.70,
        laiser_esco_rows=[_laiser(ESCO2, "operate machinery", 0.8)],
    )
    assert c.verdict == CONFLICT


def test_single_source_when_only_ours_fires():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="weld metal", our_score=0.70,
        laiser_esco_rows=[],
    )
    assert c.verdict == SINGLE


def test_single_source_when_laiser_below_threshold():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="weld metal", our_score=0.70,
        laiser_esco_rows=[_laiser(ESCO, "weld metal", 0.10)],  # below laiser_min
    )
    assert c.verdict == SINGLE


def test_none_when_neither_fires():
    c = consensus_for_statement(
        id="1", our_uri="", our_title="", our_score=0.10,
        laiser_esco_rows=[],
    )
    assert c.verdict == NONE


def test_our_below_threshold_is_not_credible():
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="x", our_score=0.30,  # below our_min
        laiser_esco_rows=[_laiser(ESCO, "x", 0.9)],
    )
    assert c.verdict == SINGLE   # only LAiSER credible


# ── summary ──────────────────────────────────────────────────────────────────

def test_summary_and_agreement_rate():
    rows = [
        consensus_for_statement(id="1", our_uri=ESCO, our_title="a", our_score=0.8,
                                laiser_esco_rows=[_laiser(ESCO, "a", 0.9)]),   # validated
        consensus_for_statement(id="2", our_uri=ESCO, our_title="a", our_score=0.8,
                                laiser_esco_rows=[_laiser(ESCO2, "b", 0.9)]),  # conflict
        consensus_for_statement(id="3", our_uri=ESCO, our_title="a", our_score=0.8,
                                laiser_esco_rows=[]),                          # single
        consensus_for_statement(id="4", our_uri="", our_title="", our_score=0.1,
                                laiser_esco_rows=[]),                          # none
    ]
    s = summarize_consensus(rows)
    assert s.total == 4
    assert s.counts[VALIDATED] == 1 and s.counts[CONFLICT] == 1
    # both fired for 2 statements (validated+conflict) → 50% agreement
    assert abs(s.agreement_rate() - 50.0) < 1e-9
    assert "Cross-engine ESCO validation (4 statements):" in s.render()


def test_thresholds_override():
    t = ConsensusThresholds(laiser_min=0.05)
    c = consensus_for_statement(
        id="1", our_uri=ESCO, our_title="weld metal", our_score=0.70,
        laiser_esco_rows=[_laiser(ESCO, "weld metal", 0.10)],
        thresholds=t,
    )
    assert c.verdict == VALIDATED   # now LAiSER's 0.10 clears the lowered floor
