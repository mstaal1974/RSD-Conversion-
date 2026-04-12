-- migrations/002_create_taxonomy_tables.sql
-- Run this once against your Postgres database before using the
-- Occupational Taxonomy page or the LinkageEngine.
--
-- Tables created:
--   uoc_occupation_links    — core output of LinkageEngine
--   uoc_classifications     — direct ANZSCO/taxonomy tags on UOCs
--   asc_specialist_tasks    — ASC keyword matching source data (Priority 4)
--
-- Assumes 001 (or equivalent) already created:
--   rsd_skill_records, uoc_registry, qual_registry,
--   qual_taxonomy_links, uoc_qual_memberships

-- ---------------------------------------------------------------------------
-- 1. uoc_occupation_links
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uoc_occupation_links (
    id                  BIGSERIAL PRIMARY KEY,

    -- Core linkage
    uoc_code            TEXT        NOT NULL,
    anzsco_code         TEXT        NOT NULL DEFAULT '',
    anzsco_title        TEXT        NOT NULL DEFAULT '',
    anzsco_major_group  TEXT        NOT NULL DEFAULT '',

    -- ASCED / industry enrichment
    asced_code          TEXT,
    asced_title         TEXT,
    industry_sector     TEXT,
    occupation_titles   TEXT,

    -- Confidence & provenance
    confidence          NUMERIC(5,3) NOT NULL,
    mapping_source      TEXT        NOT NULL,   -- e.g. core_native, direct_uoc_classification
    source_qual_code    TEXT,
    source_asc_task_id  TEXT,

    -- Flags
    is_primary          BOOLEAN     NOT NULL DEFAULT FALSE,
    is_imported         BOOLEAN     NOT NULL DEFAULT FALSE,

    -- W3C VC fields (Ref 1)
    anzsco_uri          TEXT        NOT NULL DEFAULT '',
    vc_context          TEXT        NOT NULL DEFAULT 'https://www.w3.org/2018/credentials/v1',
    vc_type             TEXT        NOT NULL DEFAULT 'TaxonomicAlignment',

    -- AQF / skill level (Ref 2)
    aqf_level           TEXT,
    skill_level_label   TEXT,

    -- Ownership (Ref 4)
    owner_tp_code       TEXT,
    home_tp_title       TEXT,

    -- Soft-delete / SCD2
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to            TIMESTAMPTZ,

    -- Pipeline traceability
    pipeline_run_id     BIGINT
);

-- Unique constraint used by ON CONFLICT in linkage engine
CREATE UNIQUE INDEX IF NOT EXISTS uoc_occupation_links_uq
    ON uoc_occupation_links (uoc_code, anzsco_code, pipeline_run_id)
    WHERE valid_to IS NULL;

-- Query performance
CREATE INDEX IF NOT EXISTS idx_uol_uoc_code
    ON uoc_occupation_links (uoc_code);
CREATE INDEX IF NOT EXISTS idx_uol_primary
    ON uoc_occupation_links (uoc_code, is_primary, valid_to)
    WHERE is_primary = TRUE AND valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_uol_anzsco
    ON uoc_occupation_links (anzsco_code);
CREATE INDEX IF NOT EXISTS idx_uol_confidence
    ON uoc_occupation_links (confidence);


-- ---------------------------------------------------------------------------
-- 2. uoc_classifications
--    Direct ANZSCO / taxonomy tags stored against a UOC
--    (used by LinkageEngine Priority 1)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS uoc_classifications (
    id          BIGSERIAL PRIMARY KEY,
    uoc_code    TEXT    NOT NULL,
    scheme      TEXT    NOT NULL,   -- e.g. 'ANZSCO Identifier', 'Taxonomy-Occupation'
    code        TEXT,
    value       TEXT,
    UNIQUE (uoc_code, scheme, COALESCE(code, ''))
);

CREATE INDEX IF NOT EXISTS idx_uoc_class_uoc
    ON uoc_classifications (uoc_code);


-- ---------------------------------------------------------------------------
-- 3. asc_specialist_tasks
--    Source data for TF-IDF keyword matching (LinkageEngine Priority 4)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asc_specialist_tasks (
    id                BIGSERIAL PRIMARY KEY,
    task_id           TEXT        NOT NULL UNIQUE,
    task_description  TEXT        NOT NULL,
    anzsco_code       TEXT        NOT NULL,
    anzsco_title      TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_asc_anzsco
    ON asc_specialist_tasks (anzsco_code);
