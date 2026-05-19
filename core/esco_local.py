"""
core/esco_local.py

Local ESCO matcher using the offline ESCO v1.2.1 CSV dataset and TF-IDF
cosine similarity. Drop-in replacement for the REST API client in
`core/esco.py` — same return shape, ~1000x faster, no rate limits.

Expected files in ESCO_DATA_DIR (default: ./data/esco/):
  • skills_en.csv
  • occupationSkillRelations_en.csv

Download the official ESCO classification CSV release from
https://esco.ec.europa.eu/en/use-esco/download and unzip into that directory.

The first call builds a TF-IDF index over ~14k ESCO skills and caches it
to ESCO_DATA_DIR/index.joblib. Subsequent loads take ~1s.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _data_dir() -> Path:
    return Path(os.getenv("ESCO_DATA_DIR", "data/esco")).resolve()


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_skill_text(row: pd.Series) -> str:
    # Repeat preferredLabel and altLabels to give them more TF-IDF weight
    # than the long description text.
    pref = str(row.get("preferredLabel", "") or "")
    alt = str(row.get("altLabels", "") or "").replace("\n", " ")
    desc = str(row.get("description", "") or "")[:500]
    parts = [pref, pref, pref, alt, alt, desc]
    return _norm(" ".join(parts))


class ESCOLocalMatcher:
    """
    Loads ESCO CSVs, builds TF-IDF index over skill labels + descriptions,
    and answers nearest-neighbour queries.

    Thread-safe after construction.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else _data_dir()
        self.skills_df: pd.DataFrame = pd.DataFrame()
        self.relations_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None  # sparse TF-IDF matrix
        self._occ_by_skill: dict[str, list[dict]] = {}

    # ───────────────────────────────────────────────────────────────────
    # Loading / indexing
    # ───────────────────────────────────────────────────────────────────

    def _paths(self) -> tuple[Path, Path, Path]:
        return (
            self.data_dir / "skills_en.csv",
            self.data_dir / "occupationSkillRelations_en.csv",
            self.data_dir / "index.joblib",
        )

    def load(self, rebuild: bool = False) -> "ESCOLocalMatcher":
        skills_csv, rel_csv, index_cache = self._paths()
        if not skills_csv.exists() or not rel_csv.exists():
            raise FileNotFoundError(
                f"ESCO CSVs not found in {self.data_dir}. "
                "Download the v1.2.1 classification CSV release from "
                "https://esco.ec.europa.eu/en/use-esco/download and unzip into that directory."
            )

        if index_cache.exists() and not rebuild:
            import joblib
            payload = joblib.load(index_cache)
            self.skills_df = payload["skills_df"]
            self.relations_df = payload["relations_df"]
            self.vectorizer = payload["vectorizer"]
            self.matrix = payload["matrix"]
            self._occ_by_skill = payload["occ_by_skill"]
            return self

        # Load skills — keep only skill concepts
        skills = pd.read_csv(skills_csv, low_memory=False)
        skills = skills[skills["conceptType"].str.contains("Skill", case=False, na=False)].copy()
        skills["_text"] = skills.apply(_build_skill_text, axis=1)
        skills = skills[skills["_text"].str.len() > 0].reset_index(drop=True)

        # Build TF-IDF index over skill text
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(skills["_text"].values)

        # Load occupation-skill relations and group by skillUri
        relations = pd.read_csv(rel_csv, low_memory=False)
        occ_by_skill: dict[str, list[dict]] = {}
        for skill_uri, group in relations.groupby("skillUri"):
            occ_by_skill[skill_uri] = [
                {
                    "uri": r["occupationUri"],
                    "title": r["occupationLabel"],
                    "relation": r["relationType"],  # 'essential' or 'optional'
                }
                for _, r in group.iterrows()
            ]

        self.skills_df = skills
        self.relations_df = relations
        self.vectorizer = vectorizer
        self.matrix = matrix
        self._occ_by_skill = occ_by_skill

        # Cache to disk
        try:
            import joblib
            joblib.dump(
                {
                    "skills_df": skills,
                    "relations_df": relations,
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                    "occ_by_skill": occ_by_skill,
                },
                index_cache,
                compress=3,
            )
        except Exception:
            pass  # cache write is best-effort

        return self

    # ───────────────────────────────────────────────────────────────────
    # Querying
    # ───────────────────────────────────────────────────────────────────

    def _topk(self, query: str, k: int) -> list[tuple[int, float]]:
        assert self.vectorizer is not None and self.matrix is not None
        q = self.vectorizer.transform([_norm(query)])
        sims = linear_kernel(q, self.matrix).ravel()
        if k >= len(sims):
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            idx = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in idx[:k]]

    def search_skills(self, text: str, limit: int = 5) -> list[dict]:
        if not text or not text.strip() or self.vectorizer is None:
            return []
        hits = self._topk(text, limit)
        out: list[dict] = []
        for i, score in hits:
            row = self.skills_df.iloc[i]
            out.append({
                "uri": str(row.get("conceptUri", "")),
                "title": str(row.get("preferredLabel", "")),
                "score": round(score, 4),
                "description": str(row.get("description", "") or "")[:500],
            })
        return out

    def occupations_for_skill(self, skill_uri: str, limit: int = 10) -> tuple[list[dict], list[dict]]:
        """Returns (essential, optional) occupation lists for a skill URI."""
        rels = self._occ_by_skill.get(skill_uri, [])
        essential, optional = [], []
        for r in rels:
            entry = {"uri": r["uri"], "title": r["title"]}
            if r["relation"] == "essential":
                if len(essential) < limit:
                    essential.append(entry)
            else:
                if len(optional) < limit:
                    optional.append(entry)
            if len(essential) >= limit and len(optional) >= limit:
                break
        return essential, optional

    def match_statement(
        self,
        statement: str,
        top_n_skills: int = 3,
        top_n_occupations: int = 8,
        min_score: float = 0.0,
    ) -> dict:
        """
        Match shape mirrors `pages/3_🌐_ESCO_Alignment.py::match_statement`:
          top_skill_uri, top_skill_title, top_skill_score,
          all_occupation_titles (pipe-separated), all_occupation_uris (pipe-separated)
        """
        empty = dict(
            top_skill_uri="",
            top_skill_title="",
            top_skill_score=0.0,
            all_occupation_titles="",
            all_occupation_uris="",
        )
        skills = self.search_skills(statement, limit=top_n_skills)
        if not skills or skills[0]["score"] < min_score:
            return empty

        top = skills[0]
        essential, optional = self.occupations_for_skill(top["uri"], limit=top_n_occupations)

        occ_titles: list[str] = []
        occ_uris: list[str] = []
        for occ in essential + optional:
            if occ["uri"] and occ["uri"] not in occ_uris:
                occ_uris.append(occ["uri"])
                occ_titles.append(occ["title"])

        return dict(
            top_skill_uri=top["uri"],
            top_skill_title=top["title"],
            top_skill_score=top["score"],
            all_occupation_titles=" | ".join(occ_titles[:top_n_occupations]),
            all_occupation_uris=" | ".join(occ_uris[:top_n_occupations]),
        )

    def batch_match(
        self,
        statements: list[str],
        top_n_skills: int = 3,
        top_n_occupations: int = 8,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Vectorised batch match — scores all statements against the full
        index in one sparse matmul. ~100x faster than calling
        match_statement in a loop for large batches.
        """
        assert self.vectorizer is not None and self.matrix is not None
        empty = dict(
            top_skill_uri="",
            top_skill_title="",
            top_skill_score=0.0,
            all_occupation_titles="",
            all_occupation_uris="",
        )

        n = len(statements)
        if n == 0:
            return []

        normed = [_norm(s) if s else "" for s in statements]
        mask = np.array([bool(s) for s in normed])
        if not mask.any():
            return [empty.copy() for _ in range(n)]

        Q = self.vectorizer.transform(normed)
        sims = linear_kernel(Q, self.matrix)  # dense (n × n_skills)

        results: list[dict] = []
        for row_idx in range(n):
            if not mask[row_idx]:
                results.append(empty.copy())
                continue

            row = sims[row_idx]
            if top_n_skills >= len(row):
                top_idx = np.argsort(-row)[:top_n_skills]
            else:
                part = np.argpartition(-row, top_n_skills)[:top_n_skills]
                top_idx = part[np.argsort(-row[part])]

            top_i = int(top_idx[0])
            top_score = float(row[top_i])
            if top_score < min_score:
                results.append(empty.copy())
                continue

            srow = self.skills_df.iloc[top_i]
            top_uri = str(srow.get("conceptUri", ""))
            top_title = str(srow.get("preferredLabel", ""))

            essential, optional = self.occupations_for_skill(top_uri, limit=top_n_occupations)
            occ_titles: list[str] = []
            occ_uris: list[str] = []
            for occ in essential + optional:
                if occ["uri"] and occ["uri"] not in occ_uris:
                    occ_uris.append(occ["uri"])
                    occ_titles.append(occ["title"])

            results.append(dict(
                top_skill_uri=top_uri,
                top_skill_title=top_title,
                top_skill_score=round(top_score, 4),
                all_occupation_titles=" | ".join(occ_titles[:top_n_occupations]),
                all_occupation_uris=" | ".join(occ_uris[:top_n_occupations]),
            ))

        return results


# ───────────────────────────────────────────────────────────────────────
# Module-level singleton (cheap after first load thanks to joblib cache)
# ───────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_matcher() -> ESCOLocalMatcher:
    return ESCOLocalMatcher().load()


def is_available() -> bool:
    """Cheap check that ESCO CSVs are present without building the index."""
    d = _data_dir()
    return (d / "skills_en.csv").exists() and (d / "occupationSkillRelations_en.csv").exists()
