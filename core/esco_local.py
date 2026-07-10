"""
core/esco_local.py

Local ESCO matcher using offline ESCO v1.2.1 CSVs and sentence-transformer
embeddings (model: sentence-transformers/all-MiniLM-L6-v2, 384 dim, English).

Drop-in replacement for the REST API client in `core/esco.py` — same return
shape, true semantic matching (handles paraphrases, synonyms), no rate limits.

Expected files in ESCO_DATA_DIR (default: ./data/esco/):
  • skills_en.csv
  • occupationSkillRelations_en.csv

Download the official ESCO classification CSV release from
https://esco.ec.europa.eu/en/use-esco/download and unzip into that directory.

First-run cost: embedding ~14k skills on CPU takes a few minutes. The result
is cached to ESCO_DATA_DIR/embeddings_<model>.npz so subsequent loads take ~2s.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def _data_dir() -> Path:
    return Path(os.getenv("ESCO_DATA_DIR", "data/esco")).resolve()


def _resolve_model_source() -> str:
    """
    Resolve the sentence-transformer model source, in priority order:

      1. $ESCO_MODEL_PATH — explicit local directory
      2. <ESCO_DATA_DIR>/model/ — bundled local copy (preferred for offline)
      3. The Hugging Face Hub id (downloads on first use)
    """
    override = os.getenv("ESCO_MODEL_PATH", "").strip()
    if override and Path(override).is_dir():
        return str(Path(override).resolve())

    bundled = _data_dir() / "model"
    if bundled.is_dir() and (bundled / "config.json").exists():
        return str(bundled)

    return MODEL_NAME


def _norm(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)


def _build_skill_text(row: pd.Series) -> str:
    """
    Compose the text we embed for each ESCO skill. We include the preferred
    label, altLabels (newline-separated in ESCO CSVs), and a truncated
    description. all-MiniLM-L6-v2 has a 256-token cap so we stay terse.
    """
    pref = str(row.get("preferredLabel", "") or "")
    alt = str(row.get("altLabels", "") or "").replace("\n", ", ")
    desc = str(row.get("description", "") or "")[:400]
    return _norm(f"{pref}. {alt}. {desc}".strip(". "))


class ESCOLocalMatcher:
    """
    Loads ESCO CSVs, embeds skill labels + descriptions with a sentence
    transformer, and answers nearest-neighbour queries via cosine similarity.

    Thread-safe for queries after construction.
    """

    def __init__(self, data_dir: Path | None = None, model_name: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else _data_dir()
        self.model_name = model_name or _resolve_model_source()
        self.skills_df: pd.DataFrame = pd.DataFrame()
        self.relations_df: pd.DataFrame = pd.DataFrame()
        self.embeddings: np.ndarray | None = None  # (n_skills, dim) L2-normalized
        self._model = None  # lazy SentenceTransformer
        self._occ_by_skill: dict[str, list[dict]] = {}
        self._occ_meta: dict[str, dict] = {}

    # ───────────────────────────────────────────────────────────────────
    # Loading / indexing
    # ───────────────────────────────────────────────────────────────────

    def _paths(self) -> tuple[Path, Path, Path]:
        # Use a stable slug so the same model loaded from a local path or
        # from the Hub hits the same cache. Allow override for swapping
        # to other models in future.
        if "MiniLM-L6-v2" in self.model_name:
            slug = "sentence-transformers_all-MiniLM-L6-v2"
        else:
            slug = Path(self.model_name).name.replace("/", "_") or "model"
        return (
            self.data_dir / "skills_en.csv",
            self.data_dir / "occupationSkillRelations_en.csv",
            self.data_dir / f"embeddings_{slug}.npz",
        )

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def load(self, rebuild: bool = False, progress_callback=None) -> "ESCOLocalMatcher":
        skills_csv, rel_csv, embed_cache = self._paths()
        if not skills_csv.exists() or not rel_csv.exists():
            raise FileNotFoundError(
                f"ESCO CSVs not found in {self.data_dir}. "
                "Download the v1.2.1 classification CSV release from "
                "https://esco.ec.europa.eu/en/use-esco/download and unzip into that directory."
            )

        # Load skills — keep only skill concepts and stable ordering
        skills = pd.read_csv(skills_csv, low_memory=False)
        skills = skills[skills["conceptType"].str.contains("Skill", case=False, na=False)].copy()
        skills["_text"] = skills.apply(_build_skill_text, axis=1)
        skills = skills[skills["_text"].str.len() > 0].reset_index(drop=True)

        # Try to load cached embeddings (keyed by model + n_skills)
        embeddings: np.ndarray | None = None
        if embed_cache.exists() and not rebuild:
            try:
                with np.load(embed_cache, allow_pickle=False) as npz:
                    cached = npz["embeddings"]
                if cached.shape[0] == len(skills) and cached.shape[1] == EMBED_DIM:
                    embeddings = cached.astype(np.float32, copy=False)
            except Exception:
                embeddings = None

        if embeddings is None:
            if progress_callback:
                progress_callback(0.0, f"Encoding {len(skills)} ESCO skills with {self.model_name}…")
            embeddings = self.model.encode(
                skills["_text"].tolist(),
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            try:
                np.savez_compressed(embed_cache, embeddings=embeddings)
            except Exception:
                pass  # cache write is best-effort

        # Load occupation-skill relations grouped by skillUri
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

        # Load occupations metadata (URI → ISCO group) — optional; needed by
        # the ANZSCO crosswalk pipeline.
        occ_meta: dict[str, dict] = {}
        occ_csv = self.data_dir / "occupations_en.csv"
        if occ_csv.exists():
            occ_df = pd.read_csv(occ_csv, low_memory=False)
            for _, r in occ_df.iterrows():
                uri = str(r.get("conceptUri", "") or "")
                if uri:
                    occ_meta[uri] = {
                        "title":      str(r.get("preferredLabel", "") or ""),
                        "isco_group": str(r.get("iscoGroup", "") or "").strip(),
                        "code":       str(r.get("code", "") or ""),
                    }

        self.skills_df = skills
        self.relations_df = relations
        self.embeddings = embeddings
        self._occ_by_skill = occ_by_skill
        self._occ_meta = occ_meta
        return self

    def occupation_meta(self, occupation_uri: str) -> dict | None:
        """Return {title, isco_group, code} for an ESCO occupation URI, if loaded."""
        return self._occ_meta.get(occupation_uri)

    def occupations_for_isco(self, isco_group: str) -> list[dict]:
        """All ESCO occupations under a given ISCO-08 unit group."""
        code = str(isco_group or "").strip()
        if not code:
            return []
        return [
            {"uri": uri, "title": meta.get("title", ""), "code": meta.get("code", "")}
            for uri, meta in self._occ_meta.items()
            if meta.get("isco_group") == code
        ]

    # ───────────────────────────────────────────────────────────────────
    # Querying
    # ───────────────────────────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def _topk(self, query: str, k: int) -> list[tuple[int, float]]:
        assert self.embeddings is not None
        q = self._encode([_norm(query)])  # (1, dim)
        sims = (self.embeddings @ q[0])   # (n_skills,)
        if k >= len(sims):
            idx = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k)[:k]
            idx = part[np.argsort(-sims[part])]
        return [(int(i), float(sims[i])) for i in idx[:k]]

    def search_skills(self, text: str, limit: int = 5) -> list[dict]:
        if not text or not text.strip() or self.embeddings is None:
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

    def coverage_scan(self, statements: list[str]) -> list[dict]:
        """Score statements for three-tier coverage classification.

        Unlike `batch_match` (which resolves occupations for the top skill),
        this returns just the evidence `core.esco_coverage` needs, including
        the top-2 skills so the caller can measure the top-1/top-2 *margin*
        (how distinctly the statement points at a single ESCO concept).

        Returns one dict per input with keys:
          esco_uri, esco_title, semantic (top-1 cosine), margin (top1−top2).
        Empty statements yield zeros and empty labels.
        """
        assert self.embeddings is not None
        empty = dict(esco_uri="", esco_title="", semantic=0.0, margin=0.0)
        n = len(statements)
        if n == 0:
            return []

        normed = [_norm(s) if s else "" for s in statements]
        mask = np.array([bool(s) for s in normed])
        results: list[dict] = [empty.copy() for _ in range(n)]
        if not mask.any():
            return results

        non_empty_idx = np.where(mask)[0]
        Q = self._encode([normed[i] for i in non_empty_idx])  # (m, dim)
        sims = Q @ self.embeddings.T  # (m, n_skills)

        for j, row_idx in enumerate(non_empty_idx):
            row = sims[j]
            # top-2 indices, descending
            if len(row) >= 2:
                part = np.argpartition(-row, 2)[:2]
                top2 = part[np.argsort(-row[part])]
            else:
                top2 = np.argsort(-row)[:2]
            top_i = int(top2[0])
            top_score = float(row[top_i])
            second = float(row[int(top2[1])]) if len(top2) > 1 else 0.0
            srow = self.skills_df.iloc[top_i]
            results[int(row_idx)] = dict(
                esco_uri=str(srow.get("conceptUri", "")),
                esco_title=str(srow.get("preferredLabel", "")),
                semantic=round(top_score, 4),
                margin=round(top_score - second, 4),
            )
        return results

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
        Vectorised batch match — encodes all statements in one forward pass,
        then a single dense matmul against the skill matrix.
        """
        assert self.embeddings is not None
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

        # Only encode non-empty rows; pad results back into full output.
        non_empty_idx = np.where(mask)[0]
        Q = self._encode([normed[i] for i in non_empty_idx])  # (m, dim)
        sims_partial = Q @ self.embeddings.T  # (m, n_skills)

        results: list[dict] = [empty.copy() for _ in range(n)]
        for j, row_idx in enumerate(non_empty_idx):
            row = sims_partial[j]
            if top_n_skills >= len(row):
                top_idx = np.argsort(-row)[:top_n_skills]
            else:
                part = np.argpartition(-row, top_n_skills)[:top_n_skills]
                top_idx = part[np.argsort(-row[part])]

            top_i = int(top_idx[0])
            top_score = float(row[top_i])
            if top_score < min_score:
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

            results[int(row_idx)] = dict(
                top_skill_uri=top_uri,
                top_skill_title=top_title,
                top_skill_score=round(top_score, 4),
                all_occupation_titles=" | ".join(occ_titles[:top_n_occupations]),
                all_occupation_uris=" | ".join(occ_uris[:top_n_occupations]),
            )

        return results


# ───────────────────────────────────────────────────────────────────────
# Module-level singleton (cheap after first load thanks to .npz cache)
# ───────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_matcher() -> ESCOLocalMatcher:
    return ESCOLocalMatcher().load()


def is_available() -> bool:
    """Cheap check that ESCO CSVs are present without loading the model."""
    d = _data_dir()
    return (d / "skills_en.csv").exists() and (d / "occupationSkillRelations_en.csv").exists()
