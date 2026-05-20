# ANZSCO ↔ ISCO-08 crosswalk

The Occupational Profiles page uses this directory to translate ESCO's
ISCO-08 occupation codes into Australian ANZSCO occupations.

## What's already here

`isco_to_anzsco.csv` ships with a **starter crosswalk** (~60 rows)
covering the ISCO unit groups most often hit by Australian VET training
packages — trades, services, care, education, drivers, common
professionals. Enough to demonstrate the page on a typical UOC corpus.

**This is not authoritative.** Rows are marked `match_quality=exact`
where the correspondence is 1:1 and `partial` where the mapping is
approximate (e.g. ISCO bundles two occupations that ANZSCO splits
across two unit groups). Replace it with the full ABS file before
relying on the profile output for any external reporting.

## Upgrading to the authoritative ABS file

The Australian Bureau of Statistics publishes the official ANZSCO ↔
ISCO-08 correspondence as part of the ANZSCO release. The current
edition lives here:

> https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations

### Steps (in VS Code on a machine with internet)

1. From the ABS page above, download the *Correspondence Table*
   (usually an XLSX of ANZSCO unit groups with the corresponding
   ISCO-08 codes and quality flags).
2. Open the XLSX. The relevant sheet typically has columns like
   *ANZSCO Code*, *ANZSCO Title*, *ISCO Code*, *Correspondence
   Quality*.
3. Re-arrange / rename to match the four columns the loader expects:

   ```
   isco_code,anzsco_code,anzsco_title,match_quality
   ```

   Field details:

   | column          | type   | notes                                                      |
   |-----------------|--------|------------------------------------------------------------|
   | `isco_code`     | string | 4-digit ISCO-08 unit group, e.g. `7212`                    |
   | `anzsco_code`   | string | ANZSCO 4 or 6-digit code                                   |
   | `anzsco_title`  | string | Human-readable ANZSCO title                                |
   | `match_quality` | enum   | `exact`, `partial`, `broader`, or `narrower`               |

4. Save over `data/anzsco/isco_to_anzsco.csv`.
5. Commit and push from VS Code's Source Control panel (the loader
   tolerates `#`-prefixed comment lines and blank lines, so you can
   keep header notes if you want).

`match_quality` is folded into the per-row score via the weight table
in `core/anzsco_crosswalk.py` (1.0 / 0.7 / 0.6 / 0.5 for
exact/partial/broader/narrower).

## Override location

If the file lives outside the repo (e.g. mounted volume), set the
`ANZSCO_CROSSWALK_PATH` env var to its absolute path.

## Why this isn't bundled in full

The ABS publishes ANZSCO under a Creative Commons licence with
attribution, but the structure changes between editions. Bundling
a specific edition would risk falling out of sync with your
`anzsco_codes` table. The starter is intentionally small so you have
something runnable today without committing to a stale snapshot.
