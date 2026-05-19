"""
Tests for core.esco_local.

Skipped automatically if the ESCO CSV release isn't present on disk.
Drop the v1.2.1 release into data/esco/ (see data/esco/README.md) to run.
"""
import pytest

from core import esco_local


pytestmark = pytest.mark.skipif(
    not esco_local.is_available(),
    reason="ESCO CSVs not installed in data/esco/",
)


@pytest.fixture(scope="module")
def matcher():
    return esco_local.get_matcher()


def test_index_loaded(matcher):
    assert matcher.vectorizer is not None
    assert matcher.matrix is not None
    assert matcher.matrix.shape[0] > 10_000
    assert matcher.matrix.shape[0] == len(matcher.skills_df)


def test_match_returns_expected_shape(matcher):
    out = matcher.match_statement("write Python code to analyse data")
    assert set(out.keys()) == {
        "top_skill_uri",
        "top_skill_title",
        "top_skill_score",
        "all_occupation_titles",
        "all_occupation_uris",
    }
    assert isinstance(out["top_skill_score"], float)
    assert out["top_skill_uri"].startswith("http")


def test_empty_statement_returns_empty(matcher):
    out = matcher.match_statement("")
    assert out["top_skill_uri"] == ""
    assert out["top_skill_score"] == 0.0


def test_min_score_filter(matcher):
    out = matcher.match_statement("xyzzy plugh nonsense gibberish", min_score=0.99)
    assert out["top_skill_uri"] == ""


def test_batch_match_matches_single(matcher):
    statements = [
        "write Python code to analyse data",
        "weld steel components",
        "",
        "prepare financial reports",
    ]
    batch = matcher.batch_match(statements)
    assert len(batch) == len(statements)

    # Empty statement gets empty result
    assert batch[2]["top_skill_uri"] == ""

    # Non-empty rows match the single-call equivalent
    for i in (0, 1, 3):
        single = matcher.match_statement(statements[i])
        assert batch[i]["top_skill_uri"] == single["top_skill_uri"]
        assert batch[i]["top_skill_score"] == pytest.approx(single["top_skill_score"], abs=1e-3)
