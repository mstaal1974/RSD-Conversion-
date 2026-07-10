#!/usr/bin/env python3
"""
scripts/esco_coverage_report.py

Produce the three-tier ESCO coverage breakdown across the canonical corpus:

    Across the canonical corpus (N statements):

       XX.X%  matched cleanly           — high semantic and lexical agreement …
       XX.X%  matched partially         — a defensible ESCO neighbour exists …
       XX.X%  found no credible match   — Australian certified capability …

Sources
───────
  --source db     (default) read matched rows from `rsd_skill_records`
                  (needs DATABASE_URL; uses the ESCO scores already stored by
                  pages/3 ESCO Alignment).
  --source csv    read a CSV with a `skill_statement` column (and optionally
                  `esco_skill_title` / `esco_skill_score`).
  --source xlsx   read a sheet with a `skill_statement` (or `Contents`) column.

If the source has no ESCO scores, or you pass --rematch, statements are scored
fresh with the bundled local matcher (`core.esco_local`), which also yields the
top-1/top-2 margin used for the "single ESCO skill" test.

Examples
────────
  python scripts/esco_coverage_report.py                       # DB, stored scores
  python scripts/esco_coverage_report.py --rematch             # DB rows, re-scored locally
  python scripts/esco_coverage_report.py --source csv --path corpus.csv
  python scripts/esco_coverage_report.py --out-csv classified.csv --out-json summary.json

Thresholds are calibration defaults for all-MiniLM-L6-v2 (see
core.esco_coverage.CoverageThresholds); override any of them on the CLI and the
report prints exactly what it used so the numbers are reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root on path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.esco_coverage import (  # noqa: E402
    CLEAN, PARTIAL, NONE,
    CoverageThresholds, MatchRecord, classify_records, summarize,
)


def _load_dataframe(args) -> "object":
    import pandas as pd

    if args.source == "db":
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text
        load_dotenv()
        url = os.getenv("DATABASE_URL", "")
        if not url:
            sys.exit("DATABASE_URL not set — needed for --source db.")
        engine = create_engine(url, pool_pre_ping=True)
        where = "skill_statement IS NOT NULL AND skill_statement <> ''"
        if not args.rematch:
            # Only rows that actually carry a stored ESCO match.
            where += " AND esco_skill_uri IS NOT NULL AND esco_skill_uri <> ''"
        cols = ("unit_code, element_title, skill_statement, "
                "esco_skill_title, esco_skill_score")
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT {cols} FROM rsd_skill_records WHERE {where}")
            ).mappings().all()
        return pd.DataFrame([dict(r) for r in rows])

    if args.source == "csv":
        if not args.path:
            sys.exit("--path is required for --source csv")
        return pd.read_csv(args.path)

    if args.source == "xlsx":
        if not args.path:
            sys.exit("--path is required for --source xlsx")
        df = pd.read_excel(args.path, sheet_name=args.sheet)
        if "skill_statement" not in df.columns and "Contents" in df.columns:
            df = df.rename(columns={"Contents": "skill_statement"})
        return df

    sys.exit(f"unknown source: {args.source}")


def _str(v) -> str:
    # pandas reads blank cells as NaN; NaN is truthy so `v or ""` keeps it.
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _num(v) -> float:
    if v is None or (isinstance(v, float) and v != v):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _records_from_stored(df) -> list[MatchRecord]:
    recs: list[MatchRecord] = []
    for _, r in df.iterrows():
        recs.append(MatchRecord(
            statement=_str(r.get("skill_statement", "")),
            esco_label=_str(r.get("esco_skill_title", "")),
            semantic=_num(r.get("esco_skill_score", 0.0)),
            margin=None,  # stored pass keeps only the top skill
        ))
    return recs


def _records_from_matcher(df) -> list[MatchRecord]:
    from core import esco_local
    if not esco_local.is_available():
        sys.exit("ESCO CSVs not found in data/esco/ — cannot --rematch. "
                 "See data/esco/README.md.")
    statements = [_str(s) for s in df["skill_statement"].tolist()]
    print(f"Scoring {len(statements):,} statements with the local ESCO "
          f"matcher (first run embeds ~14k skills; cached after)…",
          file=sys.stderr)
    matcher = esco_local.get_matcher()
    scored = matcher.coverage_scan(statements)
    recs: list[MatchRecord] = []
    for stmt, sc in zip(statements, scored):
        recs.append(MatchRecord(
            statement=stmt,
            esco_label=sc["esco_title"],
            semantic=sc["semantic"],
            margin=sc["margin"],
        ))
    return recs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["db", "csv", "xlsx"], default="db")
    p.add_argument("--path", help="file path for csv/xlsx sources")
    p.add_argument("--sheet", default=0, help="sheet name/index for xlsx")
    p.add_argument("--rematch", action="store_true",
                   help="ignore stored scores; re-score with the local matcher "
                        "(adds the top1−top2 margin signal)")
    p.add_argument("--sem-none", type=float)
    p.add_argument("--sem-clean", type=float)
    p.add_argument("--lex-clean", type=float)
    p.add_argument("--margin-clean", type=float)
    p.add_argument("--out-csv", help="write per-statement classification here")
    p.add_argument("--out-json", help="write the summary dict here")
    args = p.parse_args()

    overrides = {k: v for k, v in {
        "sem_none": args.sem_none, "sem_clean": args.sem_clean,
        "lex_clean": args.lex_clean, "margin_clean": args.margin_clean,
    }.items() if v is not None}
    thresholds = CoverageThresholds(**{**CoverageThresholds().__dict__, **overrides})

    df = _load_dataframe(args)
    if df is None or len(df) == 0:
        sys.exit("No statements found for the chosen source/filter.")
    if "skill_statement" not in df.columns:
        sys.exit(f"source has no 'skill_statement' column (columns: "
                 f"{list(df.columns)}).")

    have_scores = ("esco_skill_score" in df.columns
                   and "esco_skill_title" in df.columns)
    if args.rematch or not have_scores:
        records = _records_from_matcher(df)
    else:
        records = _records_from_stored(df)

    summary = summarize(records, thresholds)

    print()
    print(summary.render())
    print()
    print("Thresholds (all-MiniLM-L6-v2 cosine): "
          f"none<{thresholds.sem_none}, "
          f"clean≥{thresholds.sem_clean} & lexical≥{thresholds.lex_clean}"
          + (f" & margin≥{thresholds.margin_clean}"
             if any(r.margin is not None for r in records) else "")
          + ".")
    if not any(r.margin is not None for r in records):
        print("Note: stored scores carry no top-2 margin; run --rematch to add "
              "the 'single ESCO skill' distinctiveness test.")

    if args.out_csv:
        _write_csv(args.out_csv, records, thresholds)
        print(f"\nPer-statement classification → {args.out_csv}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary.to_dict(), indent=2))
        print(f"Summary → {args.out_json}")


def _write_csv(path: str, records: list[MatchRecord],
               thresholds: CoverageThresholds) -> None:
    import csv
    tier_label = {CLEAN: "matched cleanly", PARTIAL: "matched partially",
                  NONE: "no credible match"}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["skill_statement", "esco_skill_title", "semantic",
                    "lexical", "margin", "tier", "tier_label"])
        for rec, tier, lex in classify_records(records, thresholds):
            w.writerow([rec.statement, rec.esco_label,
                        f"{rec.semantic:.4f}", f"{lex:.4f}",
                        "" if rec.margin is None else f"{rec.margin:.4f}",
                        tier, tier_label[tier]])


if __name__ == "__main__":
    main()
