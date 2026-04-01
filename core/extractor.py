"""Extractor registry — scores and dispatches to the right extractor."""
from __future__ import annotations
import hashlib
import pandas as pd
from .extractors.blob_extractor import BlobExtractor
from .extractors.row_per_pc_extractor import RowPerPCExtractor


class ExtractorRegistry:
    def __init__(self) -> None:
        self._extractors = [BlobExtractor(), RowPerPCExtractor()]

    def list_names(self) -> list[str]:
        return [e.name for e in self._extractors]

    def get(self, name: str):
        for e in self._extractors:
            if e.name == name:
                return e
        raise KeyError(f"Unknown extractor: {name!r}")

    def best_extractor(self, df: pd.DataFrame):
        """Return the highest-scoring extractor and its score."""
        scored = [(e.score(df), e) for e in self._extractors]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0]


def build_registry() -> ExtractorRegistry:
    return ExtractorRegistry()


def content_fingerprint(df: pd.DataFrame) -> str:
    """MD5 of the actual cell data — detects re-uploads with same shape but changed content."""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def normalize_training_package_csv(
    df: pd.DataFrame,
    forced_extractor: str | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """
    Returns:
        norm_df      — normalised Unit/Element/PCs dataframe
        extractor_name — name of extractor used
        scorecard    — DataFrame showing all extractor scores, or None if forced
    """
    reg = build_registry()

    if forced_extractor:
        ext = reg.get(forced_extractor)
        norm_df = ext.extract(df)
        return norm_df, ext.name, None

    scores = [(e.score(df), e) for e in reg._extractors]
    scores.sort(key=lambda t: t[0], reverse=True)
    scorecard = pd.DataFrame(
        [{"extractor": e.name, "score": round(s, 3)} for s, e in scores]
    )

    best_score, best_ext = scores[0]
    if best_score < 0.1:
        raise ValueError(
            "No extractor scored above 0.1 — check your CSV format. "
            f"Scores: {scorecard.to_dict('records')}"
        )

    norm_df = best_ext.extract(df)
    return norm_df, best_ext.name, scorecard
