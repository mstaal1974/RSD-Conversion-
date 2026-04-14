"""
core/linkage_engine.py

Four-priority linkage algorithm assigning ANZSCO occupations to UOCs.

Priority 1 (1.00) — Direct classification on the UOC itself
Priority 2 (0.85/0.70) — Inherited from Core unit in Current qualification
Priority 3 (0.50/0.40) — Inherited from Elective unit in Current qualification
Priority 4 (0.30) — ASC Specialist Task keyword match against skill statements

Results written to uoc_occupation_links with confidence scores,
mapping_source, and is_primary flag.
"""
from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from sqlalchemy import text, Engine

log = logging.getLogger(__name__)

CONF = {
    "direct_uoc_classification": 1.00,
    "core_native":               0.85,
    "core_imported":             0.70,
    "elective_native":           0.50,
    "elective_imported":         0.40,
    "asc_specialist_task":       0.30,
}


class LinkageEngine:

    def __init__(self, engine: Engine, pipeline_run_id: int):
        self.engine         = engine
        self.pipeline_run_id = pipeline_run_id

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, uoc_codes: list[str] | None = None,
            run_asc: bool = True,
            progress_callback=None) -> dict:
        """
        Run full linkage for all (or specified) UOCs.
        Returns {links_created, links_updated}
        """
        counts = {"links_created": 0, "links_updated": 0}

        uocs = self._get_uoc_list(uoc_codes)
        total = len(uocs)
        log.info(f"Running linkage for {total} UOCs")

        # Preload ASC data if needed
        asc_vectorizer = asc_matrix = asc_df = None
        if run_asc:
            asc_vectorizer, asc_matrix, asc_df = self._load_asc_data()

        for i, uoc_code in enumerate(uocs):
            if progress_callback and i % 10 == 0:
                progress_callback(i / max(total, 1),
                                  f"Linking {uoc_code} ({i+1}/{total})…")
            try:
                n = self._link_uoc(uoc_code, asc_vectorizer,
                                   asc_matrix, asc_df)
                counts["links_created"] += n
            except Exception as e:
                log.error(f"Linkage failed for {uoc_code}: {e}")

        # Mark is_primary for all UOCs
        self._mark_primary()

        if progress_callback:
            progress_callback(1.0, f"Linked {counts['links_created']} occupation pairs")
        return counts

    # ── Per-UOC linkage ───────────────────────────────────────────────────────
    def _link_uoc(self, uoc_code: str, asc_vectorizer, asc_matrix,
                  asc_df) -> int:
        links = []

        # ── Priority 1: Direct UOC classifications ────────────────────────────
        with self.engine.connect() as conn:
            direct = conn.execute(text("""
                SELECT scheme, code, value FROM uoc_classifications
                WHERE uoc_code = :uc
                AND scheme IN ('ANZSCO Identifier', 'Taxonomy-Occupation', '01')
            """), {"uc": uoc_code}).fetchall()

        for row in direct:
            scheme, code, value = row
            if code or value:
                links.append(self._make_link(
                    uoc_code=uoc_code,
                    anzsco_code=code or "",
                    anzsco_title=value or "",
                    confidence=CONF["direct_uoc_classification"],
                    source="direct_uoc_classification",
                    source_qual=None,
                ))

        # ── Priority 2 & 3: Inheritance from qualifications ───────────────────
        with self.engine.connect() as conn:
            memberships = conn.execute(text("""
                SELECT m.qual_code, m.membership_type,
                       m.is_imported, m.owner_tp_code,
                       t.code  AS anzsco_code,
                       t.value AS anzsco_title,
                       t2.code  AS asced_code,
                       t2.value AS asced_title,
                       t3.value AS industry_sector,
                       t4.value AS occupation_titles
                FROM uoc_qual_memberships m
                JOIN qual_registry q ON q.qual_code = m.qual_code
                    AND q.status = 'Current'
                LEFT JOIN qual_taxonomy_links t ON t.qual_code = m.qual_code
                   AND t.scheme IN ('ANZSCO Identifier', '01')
                LEFT JOIN qual_taxonomy_links t2 ON t2.qual_code = m.qual_code
                    AND t2.scheme LIKE 'ASCED%'
                LEFT JOIN qual_taxonomy_links t3 ON t3.qual_code = m.qual_code
                    AND t3.scheme = 'Taxonomy-Industry Sector'
                LEFT JOIN qual_taxonomy_links t4 ON t4.qual_code = m.qual_code
                    AND t4.scheme IN ('Taxonomy-Occupation', '01')
                WHERE m.uoc_code = :uc
            """), {"uc": uoc_code}).fetchall()

        for row in memberships:
            (qual_code, mtype, is_imported, owner_tp,
             anzsco_code, anzsco_title,
             asced_code, asced_title,
             industry_sector, occupation_titles) = row

            if not anzsco_code and not anzsco_title:
                continue  # Qual has no ANZSCO — skip

            if mtype == "core":
                src = "core_native" if not is_imported else "core_imported"
            else:
                src = "elective_native" if not is_imported else "elective_imported"

            links.append(self._make_link(
                uoc_code=uoc_code,
                anzsco_code=anzsco_code or "",
                anzsco_title=anzsco_title or "",
                confidence=CONF[src],
                source=src,
                source_qual=qual_code,
                asced_code=asced_code,
                asced_title=asced_title,
                industry_sector=industry_sector,
                occupation_titles=occupation_titles,
            ))

        # ── Priority 4: ASC matching — only if no high-confidence links ───────
        has_strong = any(lk["confidence"] >= 0.70 for lk in links)
        if not has_strong and asc_df is not None and len(asc_df) > 0:
            stmts = self._get_skill_statements(uoc_code)
            for stmt in stmts:
                for match in self._asc_match(
                        stmt, asc_vectorizer, asc_matrix, asc_df):
                    links.append(self._make_link(
                        uoc_code=uoc_code,
                        anzsco_code=match["anzsco_code"],
                        anzsco_title=match["anzsco_title"],
                        confidence=CONF["asc_specialist_task"],
                        source="asc_specialist_task",
                        source_qual=None,
                        asc_task_id=match["task_id"],
                    ))

        if not links:
            return 0

        # Deduplicate — keep highest confidence per UOC × ANZSCO
        best: dict[tuple, dict] = {}
        for lk in links:
            key = (lk["uoc_code"], lk["anzsco_code"])
            if key not in best or lk["confidence"] > best[key]["confidence"]:
                best[key] = lk

        # Expire old links
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE uoc_occupation_links
                SET valid_to = NOW()
                WHERE uoc_code = :uc AND valid_to IS NULL
            """), {"uc": uoc_code})

        # Insert new — includes Ref 1-4 enriched columns
        with self.engine.begin() as conn:
            for lk in best.values():
                conn.execute(text("""
                    INSERT INTO uoc_occupation_links (
                        uoc_code, anzsco_code, anzsco_title, anzsco_major_group,
                        asced_code, asced_title, industry_sector, occupation_titles,
                        confidence, mapping_source, source_qual_code, source_asc_task_id,
                        is_primary, pipeline_run_id,
                        anzsco_uri, vc_context, vc_type,
                        aqf_level, skill_level_label,
                        is_imported, owner_tp_code, home_tp_title
                    ) VALUES (
                        :uc, :ac, :at, :amg,
                        :asced_c, :asced_t, :ind, :occ_t,
                        :conf, :src, :sq, :ast,
                        FALSE, :rid,
                        :anzsco_uri, :vc_ctx, :vc_type,
                        :aqf_level, :skill_label,
                        :is_imported, :owner_tp, :home_tp_title
                    )
                    ON CONFLICT (uoc_code, anzsco_code, pipeline_run_id) DO UPDATE SET
                        confidence       = GREATEST(EXCLUDED.confidence,
                                                    uoc_occupation_links.confidence),
                        mapping_source   = EXCLUDED.mapping_source,
                        aqf_level        = COALESCE(EXCLUDED.aqf_level,
                                                    uoc_occupation_links.aqf_level),
                        skill_level_label = COALESCE(EXCLUDED.skill_level_label,
                                                     uoc_occupation_links.skill_level_label),
                        is_imported      = EXCLUDED.is_imported,
                        owner_tp_code    = EXCLUDED.owner_tp_code
                """), {**lk, "rid": self.pipeline_run_id,
                        "anzsco_uri": lk.get("anzsco_uri",""),
                        "vc_ctx": "https://www.w3.org/2018/credentials/v1",
                        "vc_type": "TaxonomicAlignment",
                        "aqf_level": lk.get("aqf_level",""),
                        "skill_label": lk.get("skill_label",""),
                        "is_imported": lk.get("is_imported", False),
                        "owner_tp": lk.get("owner_tp",""),
                        "home_tp_title": lk.get("home_tp_title",""),
                })

        return len(best)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_link(self, uoc_code, anzsco_code, anzsco_title,
                   confidence, source, source_qual,
                   asced_code=None, asced_title=None,
                   industry_sector=None, occupation_titles=None,
                   asc_task_id=None,
                   aqf_level=None, is_imported=False,
                   owner_tp=None, home_tp_title=None) -> dict:
        """Build a link dict including all Ref 1-4 enriched fields."""
        from core.rsd_record import anzsco_uri, aqf_to_skill_label
        major = self._anzsco_major(anzsco_code, anzsco_title)
        return {
            # Core fields
            "uoc_code":         uoc_code,
            "anzsco_code":      anzsco_code or "",
            "anzsco_title":     anzsco_title or "",
            "anzsco_major_group": major,
            "asced_c":          asced_code or "",
            "asced_t":          asced_title or "",
            "ind":              industry_sector or "",
            "occ_t":            occupation_titles or "",
            "conf":             float(confidence),
            "src":              source,
            "sq":               source_qual,
            "ast":              asc_task_id,
            "confidence":       float(confidence),
            "ac":               anzsco_code or "",
            "at":               anzsco_title or "",
            "amg":              major,
            # Ref 1 — W3C VC URI
            "anzsco_uri":      anzsco_uri(anzsco_code),
            # Ref 2 — AQF level
            "aqf_level":       aqf_level or "",
            "skill_label":     aqf_to_skill_label(aqf_level),
            # Ref 4 — imported flag
            "is_imported":     is_imported,
            "owner_tp":        owner_tp or (uoc_code[:3] if uoc_code else ""),
            "home_tp_title":   home_tp_title or "",
        }

    def _anzsco_major(self, code: str, title: str) -> str:
        major_map = {
            "1": "Managers",
            "2": "Professionals",
            "3": "Technicians and Trades Workers",
            "4": "Community and Personal Service Workers",
            "5": "Clerical and Administrative Workers",
            "6": "Sales Workers",
            "7": "Machinery Operators and Drivers",
            "8": "Labourers",
        }
        if code and len(code) >= 1:
            return major_map.get(code[0], "")
        return ""

    def _get_uoc_list(self, uoc_codes: list | None) -> list[str]:
        with self.engine.connect() as conn:
            if uoc_codes:
                rows = conn.execute(text("""
                    SELECT uoc_code FROM uoc_registry
                    WHERE uoc_code = ANY(:codes)
                    AND usage_recommendation = 'Current'
                """), {"codes": uoc_codes}).fetchall()
            else:
                rows = conn.execute(text("""
                    SELECT uoc_code FROM uoc_registry
                    WHERE usage_recommendation = 'Current'
                    ORDER BY uoc_code
                """)).fetchall()
        return [r[0] for r in rows]

    def _get_skill_statements(self, uoc_code: str) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT skill_statement FROM rsd_skill_records
                WHERE unit_code = :uc AND skill_statement IS NOT NULL
                AND skill_statement != ''
                LIMIT 20
            """), {"uc": uoc_code}).fetchall()
        return [r[0] for r in rows]

    def _load_asc_data(self):
        """Load ASC specialist tasks from DB into sklearn TF-IDF matrix."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT task_id, task_description, anzsco_code, anzsco_title
                    FROM asc_specialist_tasks
                """)).fetchall()

            if not rows:
                return None, None, None

            asc_df = pd.DataFrame(rows,
                columns=["task_id","task_description","anzsco_code","anzsco_title"])

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), stop_words="english", max_features=10000)
            matrix = vectorizer.fit_transform(asc_df["task_description"])
            return vectorizer, matrix, asc_df

        except Exception as e:
            log.warning(f"Could not load ASC data: {e}")
            return None, None, None

    def _asc_match(self, statement: str, vectorizer, matrix,
                   asc_df, threshold: float = 0.25) -> list[dict]:
        """TF-IDF cosine similarity match against ASC specialist tasks."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            vec = vectorizer.transform([statement])
            scores = cosine_similarity(vec, matrix)[0]
            matches = []
            for i, score in enumerate(scores):
                if score >= threshold:
                    row = asc_df.iloc[i]
                    matches.append({
                        "task_id":     row["task_id"],
                        "anzsco_code": row["anzsco_code"],
                        "anzsco_title": row["anzsco_title"],
                        "score":       round(float(score), 3),
                    })
            return sorted(matches, key=lambda x: -x["score"])[:3]
        except Exception:
            return []

    def _mark_primary(self):
        """Set is_primary=True on the highest-confidence current link per UOC."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE uoc_occupation_links SET is_primary = FALSE
                WHERE valid_to IS NULL
                AND pipeline_run_id = :rid
            """), {"rid": self.pipeline_run_id})

            conn.execute(text("""
                UPDATE uoc_occupation_links SET is_primary = TRUE
                WHERE id IN (
                    SELECT DISTINCT ON (uoc_code) id
                    FROM uoc_occupation_links
                    WHERE valid_to IS NULL
                    AND pipeline_run_id = :rid
                    ORDER BY uoc_code, confidence DESC
                )
            """), {"rid": self.pipeline_run_id})

    # ── Query helpers for the Streamlit page ──────────────────────────────────
    @staticmethod
    def get_linked_records(engine: Engine,
                           unit_code: str | None = None,
                           min_confidence: float = 0.0,
                           limit: int = 500) -> pd.DataFrame:
        """
        Join rsd_skill_records with uoc_occupation_links.
        Returns enriched DataFrame ready for display/export.
        """
        filter_sql = "AND s.unit_code = :uc" if unit_code else ""
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    s.unit_code, s.unit_title, s.element_title,
                    s.skill_statement, s.qa_passes, s.keywords,
                    o.anzsco_code, o.anzsco_title, o.anzsco_major_group,
                    o.asced_code, o.asced_title,
                    o.industry_sector, o.occupation_titles,
                    o.confidence, o.mapping_source,
                    o.source_qual_code, o.is_primary
                FROM rsd_skill_records s
                LEFT JOIN uoc_occupation_links o
                    ON o.uoc_code = s.unit_code
                    AND o.is_primary = TRUE
                    AND o.valid_to IS NULL
                    AND o.confidence >= :minc
                WHERE s.skill_statement IS NOT NULL {filter_sql}
                ORDER BY s.unit_code, o.confidence DESC NULLS LAST
                LIMIT :lim
            """), {"minc": min_confidence, "uc": unit_code, "lim": limit}
            ).mappings().all()
        return pd.DataFrame([dict(r) for r in rows])

    @staticmethod
    def coverage_stats(engine: Engine) -> dict:
        """Return coverage statistics for the taxonomy page dashboard."""
        with engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT
                    COUNT(DISTINCT s.unit_code) AS total_uocs,
                    COUNT(DISTINCT CASE WHEN o.uoc_code IS NOT NULL
                          THEN s.unit_code END) AS linked_uocs,
                    COUNT(DISTINCT CASE WHEN o.confidence >= 0.70
                          THEN s.unit_code END) AS high_conf_uocs,
                    COUNT(DISTINCT o.anzsco_code) AS unique_anzsco,
                    COUNT(DISTINCT o.anzsco_major_group) AS major_groups,
                    ROUND(AVG(o.confidence)::numeric, 3) AS avg_confidence
                FROM rsd_skill_records s
                LEFT JOIN uoc_occupation_links o
                    ON o.uoc_code = s.unit_code
                    AND o.is_primary = TRUE
                    AND o.valid_to IS NULL
            """)).fetchone()
        if stats:
            return dict(zip(
                ["total_uocs","linked_uocs","high_conf_uocs",
                 "unique_anzsco","major_groups","avg_confidence"],
                stats
            ))
        return {}
