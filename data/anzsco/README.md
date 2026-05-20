# ANZSCO ↔ ISCO-08 crosswalk

The Occupational Profiles page uses this directory to translate ESCO's
ISCO-08 occupation codes into Australian occupation codes.

## What's here

`isco_to_anzsco.csv` — **authoritative ABS-sourced ISCO-08 → ANZSCO
v1.3 correspondence**, derived from the OSCA 2024 v1.0 Correspondence
Tables datacube (ABS, released 6 December 2024).

- **1,544 (ISCO, ANZSCO) pairs** across all 431 ISCO-08 unit groups
  (complete coverage of ESCO's ISCO-tagging).
- 6-digit ANZSCO v1.3 codes.
- `match_quality` flag: `exact` (648 rows, 1:1 mapping) or `partial`
  (896 rows, approximate / many-to-many).

## How it was built

ABS does not publish a direct ISCO-08 ↔ ANZSCO file in the OSCA era.
The crosswalk is the chain of two published correspondences:

```
   ISCO-08 ──(Table 7)──► OSCA 2024 v1.0 ──(Table 2)──► ANZSCO v1.3
```

Both tables ship in the ABS workbook
*OSCA_correspondence_tables_v2.xlsx*. The chain logic (in this commit's
build script) forward-fills the sparse ISCO column, drops "No
Correspondence" rows, and computes `match_quality` as `exact` only when
**both** legs are exact — any `partial` in the chain demotes the
result.

The OSCA pivot is invisible to the platform: the CSV the loader reads
contains only ISCO and ANZSCO columns.

## Regenerating

If a newer OSCA correspondence is released (or a future ANZSCO version
ships):

1. Download the latest **OSCA Correspondence Tables** XLSX from
   https://www.abs.gov.au/statistics/classifications/osca-occupation-standard-classification-australia/2024-version-10
2. Re-run the chain logic — the salient extract is in the commit that
   introduced this file; `git log -p data/anzsco/isco_to_anzsco.csv`.
3. Save the regenerated CSV over `isco_to_anzsco.csv`. The loader
   tolerates `#`-prefixed comment lines and blank lines.

## File format

| column          | type   | notes                                              |
|-----------------|--------|----------------------------------------------------|
| `isco_code`     | string | 4-digit ISCO-08 unit group, e.g. `7212`            |
| `anzsco_code`   | string | 6-digit ANZSCO v1.3 code                           |
| `anzsco_title`  | string | ANZSCO occupation title                            |
| `match_quality` | enum   | `exact`, `partial`, `broader`, or `narrower`       |

`match_quality` weights the per-row score via the table in
`core/anzsco_crosswalk.py` (1.0 / 0.7 / 0.6 / 0.5).

## Override location

If the file lives outside the repo (e.g. mounted volume), set the
`ANZSCO_CROSSWALK_PATH` env var to its absolute path.

## Future: OSCA migration

OSCA superseded ANZSCO in December 2024. The platform still uses
ANZSCO codes (column name `anzsco_code` throughout the schema). When
the platform migrates to OSCA, this file should be regenerated
directly from Table 7 (ISCO → OSCA) with no chaining, gaining ~100 net
correspondence rows and removing the partial-flag amplification from
the chain.
