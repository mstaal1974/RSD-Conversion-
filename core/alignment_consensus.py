"""
core/alignment_consensus.py

Cross-validate our in-house ESCO alignment against LAiSER's independent
alignment, per skill statement, and aggregate the agreement into a validation
summary.

Two independent aligners that land on the *same* ESCO concept is a far
stronger signal than either engine's score alone — it turns the forward
coverage classifier from single-model into consensus:

    validated     — both engines produced a credible ESCO match AND they agree
                    (same URI, or the same concept by label)
    conflict      — both produced a credible match but on different concepts
                    (a review queue: one of them is wrong, or they're at
                    different granularity)
    single_source — only one engine produced a credible match
    none          — neither did

This module is pure standard library: it consumes plain dicts/lists so it can
be unit-tested without LAiSER, torch, an API key, or a database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from core.esco_coverage import _tokens  # shared tokeniser for label matching

VALIDATED = "validated"
CONFLICT = "conflict"
SINGLE = "single_source"
NONE = "none"

_LABELS = {
    VALIDATED: "validated (both engines agree)",
    CONFLICT: "conflict (engines disagree)",
    SINGLE: "single-source (one engine only)",
    NONE: "no credible match from either",
}


def _norm_uri(uri: str) -> str:
    """Normalise an ESCO URI for comparison (trim, lowercase, drop scheme)."""
    u = (uri or "").strip().lower().rstrip("/")
    u = re.sub(r"^https?://", "", u)
    return u


def labels_match(a: str, b: str, min_jaccard: float = 0.6) -> bool:
    """True if two ESCO concept labels denote the same skill.

    Symmetric token Jaccard (unlike the directional lexical measure used for
    statement↔label), because here both sides are short concept labels and we
    want mutual agreement, tolerant of alt-labels and word order.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and (inter / union) >= min_jaccard


@dataclass(frozen=True)
class ConsensusThresholds:
    our_min: float = 0.42       # our MiniLM cosine floor for a credible match
    laiser_min: float = 0.50    # LAiSER correlation-coefficient floor
    label_jaccard: float = 0.60  # token-Jaccard to call two labels "the same"


@dataclass
class StatementConsensus:
    id: str
    our_uri: str
    our_title: str
    our_score: float
    laiser_uri: str
    laiser_title: str
    laiser_score: float
    verdict: str


def consensus_for_statement(
    *,
    id: str,
    our_uri: str,
    our_title: str,
    our_score: float,
    laiser_esco_rows: Sequence[dict],
    thresholds: ConsensusThresholds | None = None,
) -> StatementConsensus:
    """Classify one statement's cross-engine agreement.

    `laiser_esco_rows` are LAiSER's ESCO rows for this statement (normalised
    schema from core.laiser_align — keys: source_url, taxonomy_concept, score).
    Only the top-scoring ESCO row is compared.
    """
    t = thresholds or ConsensusThresholds()

    our_ok = bool(our_uri) and float(our_score or 0.0) >= t.our_min

    top = None
    for r in laiser_esco_rows:
        if top is None or float(r.get("score", 0.0) or 0.0) > float(top.get("score", 0.0) or 0.0):
            top = r
    l_uri = str((top or {}).get("source_url", "") or "")
    l_title = str((top or {}).get("taxonomy_concept", "") or "")
    l_score = float((top or {}).get("score", 0.0) or 0.0)
    laiser_ok = bool(l_uri or l_title) and l_score >= t.laiser_min

    if not our_ok and not laiser_ok:
        verdict = NONE
    elif our_ok != laiser_ok:
        verdict = SINGLE
    else:
        agree = (
            (_norm_uri(our_uri) and _norm_uri(our_uri) == _norm_uri(l_uri))
            or labels_match(our_title, l_title, t.label_jaccard)
        )
        verdict = VALIDATED if agree else CONFLICT

    return StatementConsensus(
        id=str(id),
        our_uri=our_uri, our_title=our_title, our_score=float(our_score or 0.0),
        laiser_uri=l_uri, laiser_title=l_title, laiser_score=l_score,
        verdict=verdict,
    )


@dataclass
class ValidationSummary:
    total: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {VALIDATED: 0, CONFLICT: 0, SINGLE: 0, NONE: 0}
    )
    thresholds: ConsensusThresholds = field(default_factory=ConsensusThresholds)

    def pct(self, verdict: str) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.counts.get(verdict, 0) / self.total

    def agreement_rate(self) -> float:
        """Of statements where BOTH engines fired, the share that agree."""
        both = self.counts.get(VALIDATED, 0) + self.counts.get(CONFLICT, 0)
        return 0.0 if both == 0 else 100.0 * self.counts.get(VALIDATED, 0) / both

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "counts": dict(self.counts),
            "percent": {v: round(self.pct(v), 1)
                        for v in (VALIDATED, CONFLICT, SINGLE, NONE)},
            "agreement_rate_both_fired": round(self.agreement_rate(), 1),
            "thresholds": self.thresholds.__dict__,
        }

    def render(self) -> str:
        lines = [f"Cross-engine ESCO validation ({self.total:,} statements):", ""]
        width = max(len(_LABELS[v]) for v in _LABELS)
        for v in (VALIDATED, CONFLICT, SINGLE, NONE):
            lines.append(
                f"  {self.pct(v):5.1f}%  {_LABELS[v].ljust(width)}  "
                f"({self.counts.get(v, 0):,})"
            )
        lines.append("")
        lines.append(
            f"  → where both engines produced a match, they agree "
            f"{self.agreement_rate():.1f}% of the time."
        )
        return "\n".join(lines)


def summarize_consensus(
    rows: Iterable[StatementConsensus],
    thresholds: ConsensusThresholds | None = None,
) -> ValidationSummary:
    s = ValidationSummary(thresholds=thresholds or ConsensusThresholds())
    for r in rows:
        s.total += 1
        s.counts[r.verdict] = s.counts.get(r.verdict, 0) + 1
    return s
