# ESCO offline dataset

Drop the **ESCO v1.2.1 classification CSV release** in this directory to enable
local (offline) ESCO matching in the ESCO Alignment page.

## Get the files

1. Visit https://esco.ec.europa.eu/en/use-esco/download
2. Download the *ESCO dataset – classification – v1.2.1 – CSV* archive
3. Unzip into this directory. At minimum the matcher needs:
   - `skills_en.csv`
   - `occupationSkillRelations_en.csv`

The matcher will build a TF-IDF index on first use and cache it to
`index.joblib` here (~50 MB). Subsequent app starts load in ~2 seconds.

## Override location

Set the `ESCO_DATA_DIR` env var to point somewhere else (e.g. a mounted volume).
