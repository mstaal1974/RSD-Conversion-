# ANZSCO ↔ ISCO-08 crosswalk

The Occupational Profiles page uses this directory to translate ESCO's
ISCO-08 occupation codes into Australian ANZSCO occupations.

## Get the source data

The Australian Bureau of Statistics publishes the official ANZSCO ↔ ISCO-08
correspondence as part of the ANZSCO release. Find it at:

> https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations

Download the *Correspondence Table* (XLSX). It typically has one row per
ANZSCO unit group with the corresponding ISCO-08 code and a quality flag.

## Convert to the expected schema

Save the cleaned table as `data/anzsco/isco_to_anzsco.csv` with this header:

```
isco_code,anzsco_code,anzsco_title,match_quality
```

Field details:

| column          | type   | notes                                                      |
|-----------------|--------|------------------------------------------------------------|
| `isco_code`     | string | 4-digit ISCO-08 unit group, e.g. `7212`                    |
| `anzsco_code`   | string | ANZSCO 6-digit code, or 4-digit minor group as fallback    |
| `anzsco_title`  | string | Human-readable ANZSCO title                                |
| `match_quality` | enum   | `exact`, `partial`, `broader`, or `narrower`               |

`match_quality` is folded into the per-row score via the weight table
in `core/anzsco_crosswalk.py` (1.0 / 0.7 / 0.6 / 0.5 respectively), so
the more confident the mapping the louder it speaks in aggregate scores.

## Alternative location

If the file lives outside the repo (e.g. mounted volume), set the
`ANZSCO_CROSSWALK_PATH` env var to its absolute path.

## Why this isn't bundled

The ABS publishes ANZSCO under a Creative Commons licence with attribution,
but the structure changes between editions. We avoid baking a specific
edition into the repo so you can pick the one that matches your existing
`anzsco_codes` table contents.
