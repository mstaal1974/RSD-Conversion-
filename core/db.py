"""
Postgres persistence layer.

Run isolation: every run is scoped by (source_fingerprint + session_token).
Users can only resume their own runs via the session_token they were issued.
"""
from __future__ import annotations
import json
import uuid
import pandas as pd
from sqlalchemy import create_engine, text, Engine


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def get_engine(db_url: str) -> Engine:
    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 10},
    )


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS rsd_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token    TEXT NOT NULL,
    source_filename  TEXT,
    source_fingerprint TEXT,
    extractor_name   TEXT,
    extractor_version TEXT,
    sil_version      TEXT,
    model            TEXT,
    provider         TEXT,
    settings         JSONB,
    status           TEXT NOT NULL DEFAULT 'created',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rsd_skill_records (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id           UUID NOT NULL REFERENCES rsd_runs(id) ON DELETE CASCADE,
    row_index        INTEGER NOT NULL,
    unit_code        TEXT,
    unit_title       TEXT,
    element_title    TEXT,
    pcs_text         TEXT,
    tp_code          TEXT,
    tp_title         TEXT,
    skill_statement  TEXT,
    bart_prompt      TEXT,
    qa_one_sentence  BOOLEAN,
    qa_word_count    INTEGER,
    qa_has_method    BOOLEAN,
    qa_has_outcome   BOOLEAN,
    qa_passes        BOOLEAN,
    rewrite_count    INTEGER,
    bart_model       TEXT,
    bart_temperature REAL,
    keywords         TEXT,
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, row_index)
);

CREATE INDEX IF NOT EXISTS idx_rsd_runs_session ON rsd_runs(session_token);
CREATE INDEX IF NOT EXISTS idx_rsd_records_run  ON rsd_skill_records(run_id, row_index);
"""


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in _DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

        # Migrations — safe to run on existing databases
        for col, defn in [("tp_code", "TEXT"), ("tp_title", "TEXT")]:
            conn.execute(text(f"""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='rsd_skill_records' AND column_name='{col}'
              ) THEN
                ALTER TABLE rsd_skill_records ADD COLUMN {col} {defn};
              END IF;
            END $$;
            """))


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(
    engine: Engine,
    session_token: str,
    source_filename: str,
    source_fingerprint: str,
    extractor_name: str,
    extractor_version: str,
    sil_version: str,
    model: str,
    provider: str,
    settings: dict,
) -> str:
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO rsd_runs
                    (id, session_token, source_filename, source_fingerprint,
                     extractor_name, extractor_version, sil_version,
                     model, provider, settings, status)
                VALUES
                    (:id, :session_token, :source_filename, :source_fingerprint,
                     :extractor_name, :extractor_version, :sil_version,
                     :model, :provider, :settings::jsonb, 'created')
            """),
            dict(
                id=run_id,
                session_token=session_token,
                source_filename=source_filename,
                source_fingerprint=source_fingerprint,
                extractor_name=extractor_name,
                extractor_version=extractor_version,
                sil_version=sil_version,
                model=model,
                provider=provider,
                settings=json.dumps(settings),
            ),
        )
    return run_id


def validate_run_owner(engine: Engine, run_id: str, session_token: str) -> bool:
    """Return True only if the session_token matches the run's owner."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM rsd_runs WHERE id=:id AND session_token=:tok"),
            {"id": run_id, "tok": session_token},
        ).fetchone()
    return row is not None


def update_run_status(engine: Engine, run_id: str, status: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE rsd_runs SET status=:s, updated_at=NOW() WHERE id=:id"),
            {"s": status, "id": run_id},
        )


def upsert_skill_records(
    engine: Engine,
    run_id: str,
    batch_df: pd.DataFrame,
    row_index_start: int,
) -> None:
    rows = []
    for offset, (_, r) in enumerate(batch_df.iterrows()):
        rows.append(
            dict(
                run_id=run_id,
                row_index=row_index_start + offset,
                unit_code=str(r.get("unit_code", "") or ""),
                unit_title=str(r.get("unit_title", "") or ""),
                element_title=str(r.get("element_title", "") or ""),
                pcs_text=str(r.get("pcs_text", "") or ""),
                tp_code=str(r.get("tp_code", "") or ""),
                tp_title=str(r.get("tp_title", "") or ""),
                skill_statement=str(r.get("skill_statement", "") or ""),
                bart_prompt=str(r.get("bart_prompt", "") or ""),
                qa_one_sentence=bool(r.get("qa_one_sentence", False)),
                qa_word_count=int(r.get("qa_word_count", 0)),
                qa_has_method=bool(r.get("qa_has_method", False)),
                qa_has_outcome=bool(r.get("qa_has_outcome", False)),
                qa_passes=bool(r.get("qa_passes", False)),
                rewrite_count=int(r.get("rewrite_count", 0)),
                bart_model=str(r.get("bart_model", "") or ""),
                bart_temperature=float(r.get("bart_temperature", 0.2)),
                keywords=str(r.get("keywords", "") or ""),
                error_message=str(r.get("error_message", "") or ""),
            )
        )

    with engine.begin() as conn:
        for row in rows:
            conn.execute(
                text("""
                    INSERT INTO rsd_skill_records
                        (run_id, row_index, unit_code, unit_title, element_title,
                         pcs_text, tp_code, tp_title, skill_statement, bart_prompt,
                         qa_one_sentence, qa_word_count, qa_has_method, qa_has_outcome,
                         qa_passes, rewrite_count, bart_model, bart_temperature,
                         keywords, error_message)
                    VALUES
                        (:run_id, :row_index, :unit_code, :unit_title, :element_title,
                         :pcs_text, :tp_code, :tp_title, :skill_statement, :bart_prompt,
                         :qa_one_sentence, :qa_word_count, :qa_has_method, :qa_has_outcome,
                         :qa_passes, :rewrite_count, :bart_model, :bart_temperature,
                         :keywords, :error_message)
                    ON CONFLICT (run_id, row_index) DO UPDATE SET
                        skill_statement  = EXCLUDED.skill_statement,
                        bart_prompt      = EXCLUDED.bart_prompt,
                        qa_one_sentence  = EXCLUDED.qa_one_sentence,
                        qa_word_count    = EXCLUDED.qa_word_count,
                        qa_has_method    = EXCLUDED.qa_has_method,
                        qa_has_outcome   = EXCLUDED.qa_has_outcome,
                        qa_passes        = EXCLUDED.qa_passes,
                        rewrite_count    = EXCLUDED.rewrite_count,
                        bart_model       = EXCLUDED.bart_model,
                        bart_temperature = EXCLUDED.bart_temperature,
                        keywords         = EXCLUDED.keywords,
                        error_message    = EXCLUDED.error_message
                """),
                row,
            )


def get_next_index(engine: Engine, run_id: str) -> int:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT COALESCE(MAX(row_index), -1) + 1 FROM rsd_skill_records WHERE run_id=:id"),
            {"id": run_id},
        ).fetchone()
    return int(row[0]) if row else 0


def fetch_run_records(engine: Engine, run_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM rsd_skill_records WHERE run_id=:id ORDER BY row_index"),
            {"id": run_id},
        )
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=list(cols))


# ---------------------------------------------------------------------------
# Taxonomy schema migration  (called from init_db and the taxonomy page)
# ---------------------------------------------------------------------------

_TAXONOMY_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               SERIAL       PRIMARY KEY,
    run_type         VARCHAR(20)  NOT NULL DEFAULT 'full',
    tp_scope         TEXT,
    started_at       TIMESTAMPTZ  DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    status           VARCHAR(20)  DEFAULT 'running',
    quals_processed  INTEGER      DEFAULT 0,
    uocs_processed   INTEGER      DEFAULT 0,
    links_created    INTEGER      DEFAULT 0,
    links_updated    INTEGER      DEFAULT 0,
    error_message    TEXT
);

CREATE TABLE IF NOT EXISTS anzsco_codes (
    anzsco_code       VARCHAR(10)  PRIMARY KEY,
    anzsco_title      TEXT         NOT NULL,
    anzsco_level      VARCHAR(20)  NOT NULL DEFAULT 'unit',
    major_group_code  VARCHAR(3),
    major_group_title TEXT,
    created_at        TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS uoc_registry (
    uoc_code              VARCHAR(20)  PRIMARY KEY,
    uoc_title             TEXT         NOT NULL,
    tp_code               VARCHAR(10)  NOT NULL,
    tp_title              TEXT,
    usage_recommendation  VARCHAR(20)  DEFAULT 'Current',
    release_number        INTEGER      DEFAULT 1,
    tga_last_updated      DATE,
    ingested_at           TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_uoc_tp     ON uoc_registry(tp_code);
CREATE INDEX IF NOT EXISTS idx_uoc_status ON uoc_registry(usage_recommendation);

CREATE TABLE IF NOT EXISTS qual_registry (
    qual_code            VARCHAR(20)  PRIMARY KEY,
    qual_title           TEXT         NOT NULL,
    tp_code              VARCHAR(10)  NOT NULL,
    aqf_level            VARCHAR(40),
    status               VARCHAR(20)  NOT NULL DEFAULT 'Current',
    superseded_by        VARCHAR(20),
    tga_last_updated     DATE,
    ingested_at          TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_qual_tp     ON qual_registry(tp_code);
CREATE INDEX IF NOT EXISTS idx_qual_status ON qual_registry(status);

CREATE TABLE IF NOT EXISTS qual_taxonomy_links (
    id              SERIAL       PRIMARY KEY,
    qual_code       VARCHAR(20)  NOT NULL REFERENCES qual_registry(qual_code) ON DELETE CASCADE,
    scheme          VARCHAR(100) NOT NULL,
    code            VARCHAR(20),
    value           TEXT         NOT NULL,
    pipeline_run_id INTEGER      REFERENCES pipeline_runs(id),
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (qual_code, scheme, COALESCE(code, ''))
);
CREATE INDEX IF NOT EXISTS idx_qtl_qual   ON qual_taxonomy_links(qual_code);
CREATE INDEX IF NOT EXISTS idx_qtl_scheme ON qual_taxonomy_links(scheme);

CREATE TABLE IF NOT EXISTS uoc_classifications (
    id              SERIAL       PRIMARY KEY,
    uoc_code        VARCHAR(20)  NOT NULL REFERENCES uoc_registry(uoc_code) ON DELETE CASCADE,
    scheme          VARCHAR(100) NOT NULL,
    code            VARCHAR(20),
    value           TEXT,
    pipeline_run_id INTEGER      REFERENCES pipeline_runs(id),
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (uoc_code, scheme, COALESCE(code, ''))
);

CREATE TABLE IF NOT EXISTS uoc_qual_memberships (
    id              SERIAL       PRIMARY KEY,
    uoc_code        VARCHAR(20)  NOT NULL REFERENCES uoc_registry(uoc_code) ON DELETE CASCADE,
    qual_code       VARCHAR(20)  NOT NULL REFERENCES qual_registry(qual_code) ON DELETE CASCADE,
    membership_type VARCHAR(10)  NOT NULL CHECK (membership_type IN ('core','elective')),
    elective_group  TEXT,
    owner_tp_code   VARCHAR(10)  NOT NULL,
    is_imported     BOOLEAN      NOT NULL DEFAULT FALSE,
    pipeline_run_id INTEGER      REFERENCES pipeline_runs(id),
    created_at      TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (uoc_code, qual_code, membership_type)
);
CREATE INDEX IF NOT EXISTS idx_uqm_uoc      ON uoc_qual_memberships(uoc_code);
CREATE INDEX IF NOT EXISTS idx_uqm_qual     ON uoc_qual_memberships(qual_code);
CREATE INDEX IF NOT EXISTS idx_uqm_type     ON uoc_qual_memberships(membership_type);
CREATE INDEX IF NOT EXISTS idx_uqm_imported ON uoc_qual_memberships(is_imported);

CREATE TABLE IF NOT EXISTS asc_specialist_tasks (
    task_id           VARCHAR(20)  PRIMARY KEY,
    task_description  TEXT         NOT NULL,
    anzsco_code       VARCHAR(10)  NOT NULL,
    anzsco_title      TEXT,
    skill_cluster     TEXT,
    technology_level  VARCHAR(20),
    created_at        TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS uoc_occupation_links (
    id                  SERIAL        PRIMARY KEY,
    uoc_code            VARCHAR(20)   NOT NULL REFERENCES uoc_registry(uoc_code) ON DELETE CASCADE,
    anzsco_code         VARCHAR(10)   NOT NULL,
    anzsco_title        TEXT          NOT NULL,
    anzsco_major_group  TEXT,
    asced_code          VARCHAR(10),
    asced_title         TEXT,
    industry_sector     TEXT,
    occupation_titles   TEXT,
    confidence          NUMERIC(4,3)  NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    mapping_source      VARCHAR(50)   NOT NULL CHECK (mapping_source IN (
                            'direct_uoc_classification',
                            'core_native','core_imported',
                            'elective_native','elective_imported',
                            'asc_specialist_task'
                        )),
    source_qual_code    VARCHAR(20)   REFERENCES qual_registry(qual_code),
    source_asc_task_id  VARCHAR(20)   REFERENCES asc_specialist_tasks(task_id),
    is_primary          BOOLEAN       NOT NULL DEFAULT FALSE,
    pipeline_run_id     INTEGER       NOT NULL REFERENCES pipeline_runs(id),
    valid_from          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    valid_to            TIMESTAMPTZ,
    created_at          TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (uoc_code, anzsco_code, pipeline_run_id)
);
CREATE INDEX IF NOT EXISTS idx_uol_uoc        ON uoc_occupation_links(uoc_code);
CREATE INDEX IF NOT EXISTS idx_uol_anzsco     ON uoc_occupation_links(anzsco_code);
CREATE INDEX IF NOT EXISTS idx_uol_confidence ON uoc_occupation_links(confidence);
CREATE INDEX IF NOT EXISTS idx_uol_primary    ON uoc_occupation_links(uoc_code) WHERE is_primary = TRUE;
CREATE INDEX IF NOT EXISTS idx_uol_current    ON uoc_occupation_links(uoc_code) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_uol_source     ON uoc_occupation_links(mapping_source);
"""


def init_taxonomy_db(engine: Engine) -> None:
    """Create taxonomy tables. Safe to call on existing DBs."""
    with engine.begin() as conn:
        for stmt in _TAXONOMY_DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass  # index/table already exists


def start_pipeline_run(engine: Engine, run_type: str = "full",
                       tp_scope: str | None = None) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO pipeline_runs (run_type, tp_scope, status)
            VALUES (:rt, :scope, 'running')
            RETURNING id
        """), {"rt": run_type, "scope": tp_scope}).fetchone()
    return int(row[0])


def finish_pipeline_run(engine: Engine, run_id: int, status: str,
                        quals: int = 0, uocs: int = 0,
                        created: int = 0, updated: int = 0,
                        error: str | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE pipeline_runs
            SET status=:s, completed_at=NOW(),
                quals_processed=:q, uocs_processed=:u,
                links_created=:c, links_updated=:up,
                error_message=:e
            WHERE id=:id
        """), {"s": status, "q": quals, "u": uocs,
               "c": created, "up": updated, "e": error, "id": run_id})


# ---------------------------------------------------------------------------
# Refinement migrations (Ref 1-4) — safe to run on existing DBs
# ---------------------------------------------------------------------------

_REFINEMENT_DDL = """
-- Ref 1 & 3: Add W3C VC alignment fields + evidence to uoc_occupation_links
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_name='uoc_occupation_links' AND column_name='anzsco_uri')
  THEN ALTER TABLE uoc_occupation_links
    ADD COLUMN anzsco_uri        TEXT,
    ADD COLUMN vc_context        TEXT DEFAULT 'https://www.w3.org/2018/credentials/v1',
    ADD COLUMN vc_type           TEXT DEFAULT 'TaxonomicAlignment';
  END IF;
END $$;

-- Ref 2: Propagate aqf_level + skill_level to occupation links
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_name='uoc_occupation_links' AND column_name='aqf_level')
  THEN ALTER TABLE uoc_occupation_links
    ADD COLUMN aqf_level         TEXT,
    ADD COLUMN skill_level_label TEXT;
  END IF;
END $$;

-- Ref 4: Surface is_imported + owner_tp on occupation links
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_name='uoc_occupation_links' AND column_name='is_imported')
  THEN ALTER TABLE uoc_occupation_links
    ADD COLUMN is_imported       BOOLEAN DEFAULT FALSE,
    ADD COLUMN owner_tp_code     TEXT,
    ADD COLUMN home_tp_title     TEXT;
  END IF;
END $$;

-- Ref 3: Evidence fields on rsd_skill_records
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_name='rsd_skill_records' AND column_name='evidence_hash')
  THEN ALTER TABLE rsd_skill_records
    ADD COLUMN evidence_hash     TEXT,
    ADD COLUMN evidence_type     TEXT DEFAULT 'Simulation Performance Data',
    ADD COLUMN evidence_uri      TEXT,
    ADD COLUMN element_id        TEXT;
  END IF;
END $$;
"""


def init_refinements(engine) -> None:
    """Apply Ref 1-4 column additions. Safe on existing DBs."""
    with engine.begin() as conn:
        for stmt in _REFINEMENT_DDL.strip().split("$$;"):
            s = stmt.strip()
            if s:
                try:
                    conn.execute(text(s + "$$;"))
                except Exception:
                    pass  # column already exists
