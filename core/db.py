# core/db.py
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================
# Helpers
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_record_id(unit_code: str, element_title: str, pcs_text: str = "") -> str:
    raw = f"{(unit_code or '').strip()}||{(element_title or '').strip()}||{(pcs_text or '').strip()}".encode(
        "utf-8", errors="ignore"
    )
    return hashlib.sha256(raw).hexdigest()[:24]


def get_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set.")
    return create_engine(url, pool_pre_ping=True)


# ============================================================
# Schema / Migrations
# ============================================================

DDL_EXT = "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
  status             TEXT NOT NULL DEFAULT 'running',

  session_token      TEXT,
  source_filename    TEXT,
  source_fingerprint TEXT,
  training_package   TEXT,

  extractor_name     TEXT,
  extractor_version  TEXT,
  sil_version        TEXT,

  model              TEXT,
  provider           TEXT,
  settings_json      JSONB
);

CREATE TABLE IF NOT EXISTS skill_records (
  run_id             UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  record_id          TEXT NOT NULL,

  created_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),

  row_index          INT,

  unit_code          TEXT,
  unit_title         TEXT,
  element_title      TEXT,
  pcs_text           TEXT,

  asced6_name        TEXT,

  skill_statement    TEXT,
  keywords_semicolon TEXT,
  synonyms_semicolon TEXT,

  qa_passes          BOOLEAN,
  qa_one_sentence    BOOLEAN,
  qa_word_count      INT,
  qa_has_method      BOOLEAN,
  qa_has_outcome     BOOLEAN,
  rewrite_count      INT,

  bart_model         TEXT,
  bart_temperature   FLOAT,
  bart_prompt        TEXT,

  error_message      TEXT,
  sil_json           JSONB,

  PRIMARY KEY (run_id, record_id)
);

CREATE INDEX IF NOT EXISTS idx_skill_records_run_id  ON skill_records(run_id);
CREATE INDEX IF NOT EXISTS idx_skill_records_unit    ON skill_records(unit_code);
CREATE INDEX IF NOT EXISTS idx_skill_records_rid     ON skill_records(record_id);
CREATE INDEX IF NOT EXISTS idx_skill_records_run_row ON skill_records(run_id, row_index);
CREATE INDEX IF NOT EXISTS idx_runs_session          ON runs(session_token);
"""


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        try:
            conn.execute(text(DDL_EXT))
        except Exception:
            pass

        for stmt in DDL.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

        # Migrations — safe to run on existing databases
        for col, table, defn in [
            ("session_token", "runs",          "TEXT"),
            ("provider",      "runs",          "TEXT"),
            ("error_message", "skill_records", "TEXT"),
        ]:
            conn.execute(text(f"""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='{table}' AND column_name='{col}'
              ) THEN
                ALTER TABLE {table} ADD COLUMN {col} {defn};
              END IF;
            END $$;
            """))

        # Migration: fix PK to composite (run_id, record_id) if needed
        pk_cols = conn.execute(text("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.table_name = 'skill_records'
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
        """)).fetchall()
        pk_cols = [r[0] for r in pk_cols] if pk_cols else []

        if pk_cols != ["run_id", "record_id"]:
            conn.execute(text("""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE schemaname='public'
                  AND tablename='skill_records'
                  AND indexname='uq_skill_records_run_record'
              ) THEN
                CREATE UNIQUE INDEX uq_skill_records_run_record
                ON skill_records (run_id, record_id);
              END IF;
            END $$;
            """))
            if pk_cols:
                pk_name = conn.execute(text("""
                    SELECT conname FROM pg_constraint
                    WHERE conrelid='skill_records'::regclass AND contype='p'
                """)).scalar()
                if pk_name:
                    conn.execute(text(f'ALTER TABLE skill_records DROP CONSTRAINT "{pk_name}";'))

            conn.execute(text("""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid='skill_records'::regclass AND contype='p'
              ) THEN
                ALTER TABLE skill_records
                  ADD CONSTRAINT skill_records_pk PRIMARY KEY (run_id, record_id);
              END IF;
            END $$;
            """))


# ============================================================
# Run operations
# ============================================================

def create_run(
    engine: Engine,
    *,
    session_token: str = "",
    source_filename: str,
    source_fingerprint: str,
    extractor_name: str,
    extractor_version: str = "unknown",
    sil_version: str = "1.0.0",
    model: str,
    provider: str = "",
    settings: Dict[str, Any],
    training_package: Optional[str] = None,
) -> str:
    with engine.begin() as conn:
        res = conn.execute(
            text("""
            INSERT INTO runs (
              session_token,
              source_filename, source_fingerprint, training_package,
              extractor_name, extractor_version, sil_version,
              model, provider, settings_json, status, updated_at_utc
            )
            VALUES (
              :session_token,
              :source_filename, :source_fingerprint, :training_package,
              :extractor_name, :extractor_version, :sil_version,
              :model, :provider, CAST(:settings_json AS JSONB), 'running', now()
            )
            RETURNING run_id
            """),
            dict(
                session_token=session_token,
                source_filename=source_filename,
                source_fingerprint=source_fingerprint,
                training_package=training_package,
                extractor_name=extractor_name,
                extractor_version=extractor_version,
                sil_version=sil_version,
                model=model,
                provider=provider,
                settings_json=json.dumps(settings),
            ),
        )
        return str(res.scalar())


def validate_run_owner(engine: Engine, run_id: str, session_token: str) -> bool:
    """Return True only if the session_token matches the run's owner."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM runs WHERE run_id=:id AND session_token=:tok"),
            {"id": run_id, "tok": session_token},
        ).fetchone()
    return row is not None


def update_run_status(engine: Engine, run_id: str, status: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            UPDATE runs SET status=:status, updated_at_utc=now()
            WHERE run_id=:run_id
            """),
            dict(status=status, run_id=run_id),
        )


def get_run(engine: Engine, run_id: str) -> Optional[Dict[str, Any]]:
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT * FROM runs WHERE run_id=:run_id"),
            dict(run_id=run_id),
        ).mappings().first()
        return dict(res) if res else None


# ============================================================
# Record operations
# ============================================================

def upsert_skill_records(
    engine: Engine,
    run_id: str,
    batch_df: pd.DataFrame,
    *,
    row_index_start: int,
) -> None:
    df = batch_df.copy()

    # Stable record_id
    if "record_id" not in df.columns:
        df["record_id"] = [
            stable_record_id(str(u), str(e), str(p))
            for u, e, p in zip(
                df.get("unit_code",     pd.Series([""] * len(df))),
                df.get("element_title", pd.Series([""] * len(df))),
                df.get("pcs_text",      pd.Series([""] * len(df))),
            )
        ]

    df["row_index"] = range(row_index_start, row_index_start + len(df))

    # Normalise keyword column name
    if "keywords" in df.columns and "keywords_semicolon" not in df.columns:
        df["keywords_semicolon"] = df["keywords"]

    # Safe defaults for all expected columns
    defaults = {
        "asced6_name":        "",
        "keywords_semicolon": "",
        "synonyms_semicolon": "",
        "rewrite_count":      0,
        "bart_temperature":   0.2,
        "bart_model":         "",
        "bart_prompt":        "",
        "error_message":      "",
        "qa_passes":          False,
        "qa_one_sentence":    False,
        "qa_word_count":      0,
        "qa_has_method":      False,
        "qa_has_outcome":     False,
        "skill_statement":    "",
        "unit_code":          "",
        "unit_title":         "",
        "element_title":      "",
        "pcs_text":           "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # sil_json from extra columns
    base_cols = set(defaults.keys()) | {
        "record_id", "row_index", "keywords", "sil_json",
    }
    sil_payloads = []
    for _, r in df.iterrows():
        extra = {c: r[c] for c in df.columns if c not in base_cols and not pd.isna(r[c])}
        sil_payloads.append(json.dumps(extra, ensure_ascii=False) if extra else None)
    df["sil_json"] = sil_payloads

    insert_sql = text("""
    INSERT INTO skill_records (
      run_id, record_id, row_index,
      unit_code, unit_title, element_title, pcs_text,
      asced6_name, skill_statement,
      keywords_semicolon, synonyms_semicolon,
      qa_passes, qa_one_sentence, qa_word_count,
      qa_has_method, qa_has_outcome, rewrite_count,
      bart_model, bart_temperature, bart_prompt,
      error_message,
      sil_json,
      created_at_utc, updated_at_utc
    )
    VALUES (
      :run_id, :record_id, :row_index,
      :unit_code, :unit_title, :element_title, :pcs_text,
      :asced6_name, :skill_statement,
      :keywords_semicolon, :synonyms_semicolon,
      :qa_passes, :qa_one_sentence, :qa_word_count,
      :qa_has_method, :qa_has_outcome, :rewrite_count,
      :bart_model, :bart_temperature, :bart_prompt,
      :error_message,
      CAST(:sil_json AS JSONB),
      now(), now()
    )
    ON CONFLICT (run_id, record_id) DO UPDATE SET
      row_index          = EXCLUDED.row_index,
      unit_code          = EXCLUDED.unit_code,
      unit_title         = EXCLUDED.unit_title,
      element_title      = EXCLUDED.element_title,
      pcs_text           = EXCLUDED.pcs_text,
      asced6_name        = EXCLUDED.asced6_name,
      skill_statement    = EXCLUDED.skill_statement,
      keywords_semicolon = EXCLUDED.keywords_semicolon,
      synonyms_semicolon = EXCLUDED.synonyms_semicolon,
      qa_passes          = EXCLUDED.qa_passes,
      qa_one_sentence    = EXCLUDED.qa_one_sentence,
      qa_word_count      = EXCLUDED.qa_word_count,
      qa_has_method      = EXCLUDED.qa_has_method,
      qa_has_outcome     = EXCLUDED.qa_has_outcome,
      rewrite_count      = EXCLUDED.rewrite_count,
      bart_model         = EXCLUDED.bart_model,
      bart_temperature   = EXCLUDED.bart_temperature,
      bart_prompt        = EXCLUDED.bart_prompt,
      error_message      = EXCLUDED.error_message,
      sil_json           = EXCLUDED.sil_json,
      updated_at_utc     = now()
    """)

    records = df.to_dict(orient="records")
    for rec in records:
        rec["run_id"] = run_id

    with engine.begin() as conn:
        conn.execute(insert_sql, records)


def fetch_run_records(engine: Engine, run_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        res = conn.execute(
            text("""
            SELECT * FROM skill_records
            WHERE run_id=:run_id
            ORDER BY row_index ASC
            """),
            dict(run_id=run_id),
        ).mappings().all()
    return pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()


def get_next_index(engine: Engine, run_id: str) -> int:
    with engine.connect() as conn:
        res = conn.execute(
            text("""
            SELECT COALESCE(MAX(row_index), -1) AS mx
            FROM skill_records WHERE run_id=:run_id
            """),
            dict(run_id=run_id),
        ).mappings().first()
    mx = int(res["mx"]) if res and res["mx"] is not None else -1
    return mx + 1
