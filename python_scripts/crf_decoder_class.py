# crf_decoder.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class CRFReferenceDecoder:

    def __init__(self, bank_path: str | Path, top_k: int = 5) -> None:
        bank_path = Path(bank_path)
        if not bank_path.exists():
            raise FileNotFoundError(f"Reference bank not found: {bank_path}")

        data       = np.load(bank_path, allow_pickle=True)
        self.bank  = data["reference_bank"].astype(np.float32)  # (R, L, C)
        self.names = np.asarray(data["reference_names"])         # (R,)
        self.R, self.L, self.C = self.bank.shape

        self.top_k     = top_k
        self.crf_types = [self._parse_crf_type(n) for n in self.names]

        print(
            f"[CRFReferenceDecoder] {self.R} refs | "
            f"{len(set(self.crf_types))} CRF types | "
            f"profile ({self.L} × {self.C})"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_crf_type(full_id: str) -> str:
        parts = str(full_id).split(".")
        return parts[1] if len(parts) > 1 else str(full_id)

    @staticmethod
    def _aggregate_by_type(
        crf_types: List[str],
        scores: np.ndarray,
        top_k: int,
    ) -> List[Dict]:
        type_scores: Dict[str, List[float]] = {}
        for crf, s in zip(crf_types, scores.tolist()):
            type_scores.setdefault(crf, []).append(s)

        rows = [
            {
                "crf_type"    : crf,
                "max_score": float(max(vals)),
                "mean_score": float(np.mean(vals)),
                "n_refs"      : len(vals),
            }
            for crf, vals in type_scores.items()
        ]
        rows.sort(key=lambda r: r["max_score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows[:top_k]

    # ------------------------------------------------------------------
    # core distance computation
    # ------------------------------------------------------------------

    def _intersection_scores(
        self,
        probs: np.ndarray,   # (L, C) normalized
        mask: np.ndarray,    # (L,)
    ) -> np.ndarray:         # (R,)
        n_valid = float(mask.sum())
        if n_valid == 0.0:
            return np.zeros(self.R, dtype=np.float32)

        intersection = np.minimum(self.bank, probs[None, :, :])  # (R, L, C)
        score_per_position = intersection.sum(axis=-1)            # (R, L)
        scores = (score_per_position * mask[None, :]).sum(axis=-1) / n_valid  # (R,)

        return scores.astype(np.float32)
    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def query(
        self,
        probs: np.ndarray,
        query_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        probs      : (L, C) per-position subtype probability profile
        query_mask : (L,) optional; derived from non-zero rows if not given
        """
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (self.L, self.C):
            raise ValueError(
                f"Expected ({self.L}, {self.C}), got {probs.shape}."
            )

        mask = (
            (np.linalg.norm(probs, axis=-1) > 0).astype(np.float32)
            if query_mask is None
            else np.asarray(query_mask, dtype=np.float32)
        )

        scores = self._intersection_scores(probs, mask)
        top_idx = np.argsort(scores)[::-1][:self.top_k]

        top_sequences = [
            {
                "rank"    : rank + 1,
                "name"    : str(self.names[i]),
                "crf_type": self.crf_types[i],
                "score": float(scores[i]),
            }
            for rank, i in enumerate(top_idx)
        ]

        return {
            "top_sequences": top_sequences,
            "top_crf_types": self._aggregate_by_type(
                self.crf_types, scores, self.top_k
            ),
        }

    def best_crf_type(
        self,
        probs: np.ndarray,
        query_mask: Optional[np.ndarray] = None,
    ) -> str:
        """Return the single closest CRF type label."""
        return self.query(probs, query_mask)["top_crf_types"][0]["crf_type"]