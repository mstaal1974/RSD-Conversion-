#!/usr/bin/env python3
"""
scripts/laiser_validate.py

Cross-validate our in-house ESCO alignment against LAiSER (an independent
aligner), and optionally enrich statements with multi-taxonomy IDs.

For a sample of skill statements that already carry our stored ESCO match
(from pages/3 ESCO Alignment), this:
  1. runs LAiSER over the same statements (API-backed: OpenAI or Gemini),
  2. compares LAiSER's top ESCO concept to ours per statement,
  3. prints a validation summary (validated / conflict / single-source / none)
     and the agreement rate where both engines fired,
  4. optionally writes a per-statement consensus CSV and a multi-taxonomy
     enrichment CSV (ESCO + O*NET + OSN).

Requires the optional `laiser` package and an LLM key:
    pip install -r requirements-laiser.txt
    export OPENAI_API_KEY=...        # or GEMINI_API_KEY=...

Examples:
    python scripts/laiser_validate.py --limit 200
    python scripts/laiser_validate.py --limit 500 \
        --allowed-sources esco,onet,osn --out-enrichment enriched.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.alignment_consensus import (  # noqa: E402
    VALIDATED, CONFLICT, SINGLE, NONE,
    ConsensusThresholds, consensus_for_statement, summarize_consensus,
)


def _load_sample(limit: int) -> list[dict]:
    """Statements with our stored ESCO match, most-confident first."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    load_dotenv()
    url = os.getenv("DATABASE_URL", "")
    if not url:
        sys.exit("DATABASE_URL not set.")
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, skill_statement, esco_skill_uri, esco_skill_title, "
            "esco_skill_score FROM rsd_skill_records "
            "WHERE skill_statement IS NOT NULL AND skill_statement <> '' "
            "AND esco_skill_uri IS NOT NULL AND esco_skill_uri <> '' "
            "ORDER BY esco_skill_score DESC NULLS LAST "
            "LIMIT :lim"
        ), {"lim": limit}).mappings().all()
    return [dict(r) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=200,
                   help="how many statements to validate (LAiSER calls cost money)")
    p.add_argument("--allowed-sources", default="esco",
                   help="comma list: esco,onet,osn (esco required for validation)")
    p.add_argument("--backend", help="openai | gemini (inferred from keys if unset)")
    p.add_argument("--model-id", help="override LLM model id")
    p.add_argument("--our-min", type=float)
    p.add_argument("--laiser-min", type=float)
    p.add_argument("--label-jaccard", type=float)
    p.add_argument("--out-consensus", help="write per-statement consensus CSV")
    p.add_argument("--out-enrichment", help="write all LAiSER taxonomy rows CSV")
    args = p.parse_args()

    from core import laiser_align
    if not laiser_align.is_available():
        sys.exit("The `laiser` package isn't installed. "
                 "pip install -r requirements-laiser.txt")

    sources = [s.strip().lower() for s in args.allowed_sources.split(",") if s.strip()]
    if "esco" not in sources:
        sys.exit("--allowed-sources must include 'esco' for validation.")

    overrides = {k: v for k, v in {
        "our_min": args.our_min, "laiser_min": args.laiser_min,
        "label_jaccard": args.label_jaccard,
    }.items() if v is not None}
    thresholds = ConsensusThresholds(**{**ConsensusThresholds().__dict__, **overrides})

    sample = _load_sample(args.limit)
    if not sample:
        sys.exit("No statements with a stored ESCO match found. Run the forward "
                 "ESCO Alignment pass first.")

    print(f"Running LAiSER ({args.backend or 'auto'} backend) over "
          f"{len(sample):,} statements across sources {sources}…", file=sys.stderr)
    aligned = laiser_align.align_statements(
        [{"id": str(r["id"]), "skill_statement": r["skill_statement"]} for r in sample],
        allowed_sources=sources,
        backend=args.backend,
        model_id=args.model_id,
    )

    # Group LAiSER ESCO rows by statement id.
    by_id_esco: dict[str, list[dict]] = {}
    for _, row in aligned.iterrows():
        src = str(row["taxonomy_source"]).lower()
        if src.startswith("esco"):
            by_id_esco.setdefault(str(row["id"]), []).append(dict(row))

    results = [
        consensus_for_statement(
            id=str(r["id"]),
            our_uri=str(r.get("esco_skill_uri", "") or ""),
            our_title=str(r.get("esco_skill_title", "") or ""),
            our_score=float(r.get("esco_skill_score", 0.0) or 0.0),
            laiser_esco_rows=by_id_esco.get(str(r["id"]), []),
            thresholds=thresholds,
        )
        for r in sample
    ]
    summary = summarize_consensus(results, thresholds)

    print()
    print(summary.render())
    print()
    print(f"Thresholds: our≥{thresholds.our_min}, LAiSER≥{thresholds.laiser_min}, "
          f"label-Jaccard≥{thresholds.label_jaccard}.")

    if args.out_consensus:
        _write_consensus(args.out_consensus, results)
        print(f"\nPer-statement consensus → {args.out_consensus}")
    if args.out_enrichment:
        aligned.to_csv(args.out_enrichment, index=False)
        print(f"Multi-taxonomy enrichment ({len(aligned):,} rows) → {args.out_enrichment}")


def _write_consensus(path: str, results) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "verdict", "our_esco_title", "our_score",
                    "laiser_esco_title", "laiser_score", "our_uri", "laiser_uri"])
        for r in results:
            w.writerow([r.id, r.verdict, r.our_title, f"{r.our_score:.4f}",
                        r.laiser_title, f"{r.laiser_score:.4f}",
                        r.our_uri, r.laiser_uri])


if __name__ == "__main__":
    main()
