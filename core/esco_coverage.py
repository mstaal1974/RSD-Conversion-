"""
core/esco_coverage.py

Three-tier ESCO coverage classifier for the canonical RSD corpus.

The ESCO Alignment pass (pages/3) stores a single semantic cosine score
(`esco_skill_score`) per skill statement. That number alone answers "how
close is the nearest ESCO concept" but not the question curators actually
ask:

    Across the canonical corpus:
      [XX]% matched cleanly    — high semantic AND lexical agreement to a
                                 single ESCO skill
      [XX]% matched partially  — a defensible ESCO neighbour exists, but at
                                 the wrong granularity or with meaningful
                                 scope differences
      [XX]% found no credible  — Australian certified capability with no
             match              European equivalent

This module turns per-statement match evidence into that breakdown. It is
pure standard library (no numpy / pandas / torch) so it can be unit-tested
and imported anywhere cheaply. The heavy semantic scoring lives in
`core.esco_local`; here we only *classify* the evidence it produces.

Evidence used per statement:
  • semantic — top-1 cosine similarity of the statement against the nearest
    ESCO skill label+description embedding (0..1).
  • lexical  — directional token overlap: how much of the ESCO concept's own
    wording actually appears in the Australian statement (0..1). This is what
    separates a genuine lexical match ("provide first aid") from a semantic
    paraphrase at a different granularity.
  • margin   — (optional) top-1 minus top-2 cosine. A large margin means the
    statement points at a *single* ESCO skill rather than sitting ambiguously
    between several. Only applied to the "clean" test when supplied.

Thresholds are calibration defaults for `all-MiniLM-L6-v2` cosine over the
ESCO v1.2.1 English release. They are deliberately conservative and fully
overridable — see `CoverageThresholds`. The report script prints the
thresholds it used so any number is reproducible and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Lexical agreement
# ─────────────────────────────────────────────────────────────────────────────

# Terms that carry no discriminating meaning for a skill concept. Kept small
# and domain-neutral on purpose — over-aggressive stopping hides real overlap.
_STOPWORDS = frozenset("""
a an the and or of to for in on at by with from into over under as is are be
been being this that these those it its their your our his her they them we you
i he she who whom which what when where why how not no nor so than then too very
can could should would may might must will shall do does did done have has had
using use used within across per via about above below between during such other
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'managing'/'manage'/'management' align.

    This is intentionally cruder than a real stemmer: we only need adjacent
    inflections of the same VET/ESCO vocabulary to collapse, and we must never
    depend on an external NLP dependency at import time.
    """
    for suf in ("ational", "iveness", "ization", "isation", "ements", "ement",
                "ing", "ies", "er", "ed", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            base = tok[: -len(suf)]
            if suf == "ies":
                base += "y"
            return _drop_final_e(base)
    return _drop_final_e(tok)


def _drop_final_e(tok: str) -> str:
    # Collapse the silent-'e' remnant so 'manage'/'managing' → 'manag'.
    if len(tok) > 3 and tok.endswith("e"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if len(raw) < 3 or raw in _STOPWORDS:
            continue
        out.add(_stem(raw))
    return out


def lexical_agreement(statement: str, esco_label: str) -> float:
    """Fraction of the ESCO concept's own content words present in the statement.

    Directional on purpose: RSD statements are long (30–60 words) and ESCO
    preferred labels are short (2–6 words). Symmetric Jaccard would be dominated
    by the statement's length and read near-zero even for exact matches. What we
    want is "does the Australian statement actually say what the European concept
    is called" — i.e. recall of the label's content terms.

    Returns 0.0 when the label has no content tokens (nothing to agree with).
    """
    label_toks = _tokens(esco_label)
    if not label_toks:
        return 0.0
    stmt_toks = _tokens(statement)
    if not stmt_toks:
        return 0.0
    overlap = label_toks & stmt_toks
    return len(overlap) / len(label_toks)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

CLEAN = "clean"
PARTIAL = "partial"
NONE = "none"

# Forward (Australia → ESCO): "does this AU statement land on an ESCO skill".
_LABELS = {
    CLEAN: "matched cleanly",
    PARTIAL: "matched partially",
    NONE: "found no credible match",
}

_BLURBS = {
    CLEAN: "high semantic and lexical agreement to a single ESCO skill",
    PARTIAL: ("a defensible ESCO neighbour exists, but at the wrong "
              "granularity or with meaningful scope differences"),
    NONE: "Australian certified capability with no European equivalent",
}

# Reverse (ESCO → Australia): "does any AU statement express this ESCO skill".
_LABELS_REVERSE = {
    CLEAN: "covered cleanly",
    PARTIAL: "covered partially",
    NONE: "no Australian source",
}

_BLURBS_REVERSE = {
    CLEAN: "an Australian statement expresses this ESCO skill closely",
    PARTIAL: ("an Australian statement is in the neighbourhood, but at a "
              "different granularity or scope"),
    NONE: "European skill with no Australian training source — a coverage gap",
}

_HEADERS = {
    "forward": "Across the canonical corpus",
    "reverse": "Across the ESCO skill taxonomy",
}
_UNITS = {"forward": "statements", "reverse": "ESCO skills"}


@dataclass(frozen=True)
class CoverageThresholds:
    """Cut points for the three-tier classifier.

    Defaults are calibrated for `all-MiniLM-L6-v2` cosine over the ESCO v1.2.1
    English release (statement embedded against label + altLabels + truncated
    description). Tune per model / release; the report prints whatever it used.
    """

    sem_none: float = 0.42     # below this top-1 cosine → no credible match
    sem_clean: float = 0.58    # at/above this (with lexical support) → clean
    lex_clean: float = 0.34    # min share of ESCO label terms present for clean
    margin_clean: float = 0.03  # min top1−top2 separation for "single" skill
                                # (only enforced when a margin is supplied)

    def __post_init__(self) -> None:
        if not (0.0 <= self.sem_none <= self.sem_clean <= 1.0):
            raise ValueError("require 0 <= sem_none <= sem_clean <= 1")
        if not (0.0 <= self.lex_clean <= 1.0):
            raise ValueError("lex_clean must be in [0, 1]")
        if self.margin_clean < 0.0:
            raise ValueError("margin_clean must be >= 0")


def classify_match(
    semantic: float,
    lexical: float,
    margin: float | None = None,
    thresholds: CoverageThresholds | None = None,
) -> str:
    """Return one of CLEAN / PARTIAL / NONE for a single statement's evidence.

    Decision order:
      1. semantic below the credibility floor            → NONE
      2. semantic strong AND lexical present AND (if a    → CLEAN
         margin was supplied) the top skill is distinct
      3. otherwise a defensible neighbour exists          → PARTIAL
    """
    t = thresholds or CoverageThresholds()
    sem = float(semantic or 0.0)
    lex = float(lexical or 0.0)

    if sem < t.sem_none:
        return NONE

    margin_ok = True if margin is None else (float(margin) >= t.margin_clean)
    if sem >= t.sem_clean and lex >= t.lex_clean and margin_ok:
        return CLEAN

    return PARTIAL


@dataclass
class CoverageSummary:
    """Aggregate three-tier coverage over a corpus.

    `direction` selects the vocabulary: 'forward' = Australia → ESCO (how well
    AU statements land on ESCO), 'reverse' = ESCO → Australia (how much of ESCO
    has an Australian training source).
    """

    total: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {CLEAN: 0, PARTIAL: 0, NONE: 0}
    )
    thresholds: CoverageThresholds = field(default_factory=CoverageThresholds)
    direction: str = "forward"

    def pct(self, tier: str) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.counts.get(tier, 0) / self.total

    def unmatched_pct(self) -> float:
        """Percentage in the NONE tier — the headline coverage-gap number."""
        return self.pct(NONE)

    def _labels(self) -> dict:
        return _LABELS_REVERSE if self.direction == "reverse" else _LABELS

    def _blurbs(self) -> dict:
        return _BLURBS_REVERSE if self.direction == "reverse" else _BLURBS

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "total": self.total,
            "counts": dict(self.counts),
            "percent": {tier: round(self.pct(tier), 1)
                        for tier in (CLEAN, PARTIAL, NONE)},
            "unmatched_pct": round(self.unmatched_pct(), 1),
            "thresholds": self.thresholds.__dict__,
        }

    def render(self) -> str:
        """Human-facing block in the exact shape curators asked for."""
        labels, blurbs = self._labels(), self._blurbs()
        header = _HEADERS.get(self.direction, _HEADERS["forward"])
        unit = _UNITS.get(self.direction, _UNITS["forward"])
        lines = [f"{header} ({self.total:,} {unit}):", ""]
        width = max(len(labels[t]) for t in (CLEAN, PARTIAL, NONE))
        for tier in (CLEAN, PARTIAL, NONE):
            label = labels[tier].ljust(width)
            lines.append(
                f"  {self.pct(tier):5.1f}%  {label}  — {blurbs[tier]} "
                f"({self.counts.get(tier, 0):,})"
            )
        if self.direction == "reverse":
            lines.append("")
            lines.append(
                f"  → found {self.unmatched_pct():.1f}% of ESCO unmatched — "
                "no Australian training source."
            )
        return "\n".join(lines)


@dataclass
class MatchRecord:
    """Minimal per-statement evidence the classifier needs."""

    statement: str
    esco_label: str
    semantic: float
    margin: float | None = None
    lexical: float | None = None  # computed from statement+label when None

    def resolved_lexical(self) -> float:
        if self.lexical is not None:
            return float(self.lexical)
        return lexical_agreement(self.statement, self.esco_label)


def classify_records(
    records: Iterable[MatchRecord],
    thresholds: CoverageThresholds | None = None,
) -> list[tuple[MatchRecord, str, float]]:
    """Classify each record → list of (record, tier, lexical_used)."""
    t = thresholds or CoverageThresholds()
    out: list[tuple[MatchRecord, str, float]] = []
    for rec in records:
        lex = rec.resolved_lexical()
        tier = classify_match(rec.semantic, lex, rec.margin, t)
        out.append((rec, tier, lex))
    return out


def summarize(
    records: Sequence[MatchRecord],
    thresholds: CoverageThresholds | None = None,
    direction: str = "forward",
) -> CoverageSummary:
    """Aggregate a corpus of match records into a CoverageSummary.

    direction: 'forward' (Australia → ESCO) or 'reverse' (ESCO → Australia).
    The classifier logic is identical either way — only the reporting
    vocabulary changes — because a weak best-similarity means "no credible
    counterpart" in both directions.
    """
    t = thresholds or CoverageThresholds()
    summary = CoverageSummary(thresholds=t, direction=direction)
    for _rec, tier, _lex in classify_records(records, t):
        summary.total += 1
        summary.counts[tier] = summary.counts.get(tier, 0) + 1
    return summary
