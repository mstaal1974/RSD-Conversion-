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
             :model, :provider, CAST(:settings AS jsonb), 'created')
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
                         pcs_text, skill_statement, bart_prompt,
                         qa_one_sentence, qa_word_count, qa_has_method, qa_has_outcome,
                         qa_passes, rewrite_count, bart_model, bart_temperature,
                         keywords, error_message)
                    VALUES
                        (:run_id, :row_index, :unit_code, :unit_title, :element_title,
                         :pcs_text, :skill_statement, :bart_prompt,
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
