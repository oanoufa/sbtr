"""Decode subtype predictions against a bank of CRF references."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter1d

import numpy as np
from . import config

ST_TO_ID_DICT = config.ST_TO_ID_DICT
ID_TO_ST_DICT = {v: k for k, v in ST_TO_ID_DICT.items()}
SLIDING_WINDOW_SIZE = config.SLIDING_WINDOW_SIZE
TOP_K = config.TOP_K
PROB_ZERO_THRESHOLD = config.PROB_ZERO_THRESHOLD
CRF_MATCH_MARGIN = config.CRF_MATCH_MARGIN
CRF_ASSIGN_THR = config.CRF_ASSIGN_THR
PARTIAL_THR = config.PARTIAL_THR
START_5LTR = config.START_5LTR
NEF_3LTR = config.NEF_3LTR
LTR_LABELS = {"5'LTR", "3'LTR"}

class CRFReferenceDecoder:

    def __init__(
        self,
        bank_path: str | Path,
    ) -> None:

        bank_path = Path(bank_path)
        if not bank_path.exists():
            raise FileNotFoundError(f"Reference bank not found: {bank_path}")

        data       = np.load(bank_path, allow_pickle=True)
        self.bank  = data["reference_bank"].astype(np.float32)   # (R, L, C)
        self.names = np.asarray(data["reference_names"])          # (R,)
        self.R, self.L, self.C = self.bank.shape

        self.top_k            = TOP_K
        self.prob_threshold   = PROB_ZERO_THRESHOLD
        self.crf_types        = [self._parse_crf_type(n) for n in self.names]
        self.window_size      = SLIDING_WINDOW_SIZE
        self.crf_match_margin    = CRF_MATCH_MARGIN
        self.partial_thr = PARTIAL_THR
        self.crf_assign_thr = CRF_ASSIGN_THR

        print(
            f"[CRFReferenceDecoder] {self.R} refs | "
            f"{len(set(self.crf_types))} CRF types | "
            f"profile ({self.L} x {self.C})"
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_crf_type(full_id: str) -> str:
        parts = str(full_id).split(".")
        return parts[1] if "Ref." in parts else parts[0]

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
                "crf_type"  : crf,
                "max_score" : float(max(vals)),
                "mean_score": float(np.mean(vals)),
                "n_refs"    : len(vals),
            }
            for crf, vals in type_scores.items()
        ]
        rows.sort(key=lambda r: r["max_score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows[:top_k]

    @staticmethod
    def _extract_breakpoints(
        label_names: np.ndarray,
        skip_label: str = "U",
    ) -> List[Tuple[int, int, str]]:
        """
        Extract contiguous subtype regions from a 1-D label array.

        Parameters
        ----------
        label_names : (L,) string array — skip_label positions are ignored.
        skip_label  : sentinel for invalid / gap positions.

        Returns
        -------
        List of (start, end_included, subtype) in 1-based coordinates relative
        to the non-skip subsequence.
        """
        valid = label_names[label_names != skip_label]
        if len(valid) == 0:
            return []

        regions: List[Tuple[int, int, str]] = []
        start, current = 0, label_names[0]  # 1-based coordinates
        for i in range(1, len(label_names)):
            if label_names[i] != current:
                regions.append((start + 1, i, str(current)))
                start, current = i, label_names[i]
        regions.append((start + 1, len(label_names), str(current)))
        return regions

    def _purity_stats(
        self,
        label_names: np.ndarray,
        skip_label: str = "U",
    ) -> Tuple[str, float, bool]:
        """Return (dominant_subtype, dominant_fraction, is_candidate_pure)."""
        label_names = np.asarray(label_names)
        excluded = {skip_label, *LTR_LABELS}
        non_skip = label_names[~np.isin(label_names, list(excluded))]
        if len(non_skip) == 0:
            return ("", 0.0)
        unique, counts = np.unique(non_skip, return_counts=True)
        best      = counts.argmax()
        dominant  = str(unique[best])
        fraction  = float(counts[best] / len(non_skip))
        if dominant == 'U':
            print(unique, counts, non_skip, label_names, flush=True)
            assert dominant != 'U', "Dominant subtype should not be 'U'"
        return dominant, fraction

    def _composition_from_regions(
        self,
        regions: List[Tuple[int, int, str]],
        min_region_fraction: float = 0.02,
        total_len: Optional[int] = None,
        skip_label: str = "U",
    ) -> List[str]:
        """
        Ordered, deduplicated subtype list from breakpoint regions.
        Regions shorter than min_region_fraction of total_len are dropped.
        LTR labels are excluded (they're not real subtype calls).
        """
        if not regions:
            return []
        if total_len is None:
            total_len = regions[-1][1]
        if total_len == 0:
            return []

        seen: set = set()
        composition: List[str] = []
        for start, end, subtype in regions:
            if subtype == skip_label or subtype in LTR_LABELS:
                continue
            if (end - start) / total_len < min_region_fraction:
                continue
            if subtype not in seen:
                seen.add(subtype)
                composition.append(subtype)
        return composition

    # core computations

    def _intersection_scores(
        self,
        probs: np.ndarray,   # (L, C)
        mask: np.ndarray,    # (L,)
    ) -> np.ndarray:         # (R,)
        n_valid = float(mask.sum())
        if n_valid == 0.0:
            return np.zeros(self.R, dtype=np.float32)

        # Apply mask directly onto probability matrix to zero out gaps
        probs_masked = probs * mask[:, None]  # (L, C)

        # Take minimum across third dimension directly
        intersection = np.minimum(self.bank, probs_masked[None, :, :])
        
        # Sum along L and C simultaneously
        scores = intersection.sum(axis=(-2, -1)) / n_valid
        return scores.astype(np.float32)

    def _sliding_window(
        self,
        probs: np.ndarray,   # (L, C)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smooths subtype probabilities across the full aligned sequence without masking gaps.
        
        Returns
        -------
        num_path : (L,) int32 — subtype index (0 to C-1), or -1 where 'U' is predicted
        str_path : (L,) str   — subtype name, or 'U' where no subtype is confident
        """
        L, C = probs.shape

        # 1. Zero out low-confidence subtype probabilities
        probs_thr = np.where(probs >= self.prob_threshold, probs, 0.0)

        # 2. Virtual 'U' channel: active (1.0) where no subtype cleared the threshold
        # Note: If no subtype cleared threshold, this position votes 'U'
        uninformative = (probs_thr.sum(axis=-1) == 0.0).astype(np.float32)
        probs_aug = np.concatenate([probs_thr, uninformative[:, None]], axis=1)  # (L, C+1)

        # 3. Continuous uniform window filter over the whole aligned sequence
        # uniform_filter1d computes moving average smoothly without needing mask division
        smoothed = uniform_filter1d(
            probs_aug, size=self.window_size, axis=0, mode="nearest"
        )  # (L, C+1)

        # 4. Predict winning channel across sequence
        winning_channel = smoothed.argmax(axis=1)  # (L,)

        # 5. Build output arrays
        num_path = np.full(L, -1, dtype=np.int32)
        str_path = np.full(L, "U", dtype=object)

        # Known subtypes are winning indices strictly less than C (index C is 'U')
        is_subtype = winning_channel < C
        num_path[is_subtype] = winning_channel[is_subtype]

        # Map subtype integer IDs to strings
        if np.any(is_subtype):
            str_path[is_subtype] = np.array(
                [ID_TO_ST_DICT[i] for i in num_path[is_subtype]], dtype=object
            )

        return num_path, str_path

    @staticmethod
    def _trim_flanking_to_skip(
        label_names: np.ndarray,
        mask: np.ndarray,
        skip_label: str = "U",
    ) -> np.ndarray:
        """
        Force positions before the first valid (mask==1) index and after the
        last valid index to skip_label. Interior mask==0 gaps (alignment gaps
        within the real sequence span) are left untouched.
        """
        valid_idx = np.flatnonzero(mask)
        label_names = np.asarray(label_names).copy()
        if len(valid_idx) == 0:
            label_names[:] = skip_label
            return label_names

        first, last = valid_idx[0], valid_idx[-1]
        label_names[:first] = skip_label
        label_names[last + 1:] = skip_label
        return label_names

    def _gen_labels_dealigned(
        self,
        mask: np.ndarray,
        compactmapout_entry: List[Tuple[int, int, int]],
        label_names_aligned: List[str],
    ) -> List[str]:
        """
        Generate label_names_dealigned by inserting the removed insertions in the labels as the previous label, then applying the mask to get the final label_names_dealigned.

        Parameters
        ----------
        mask : np.ndarray, boolean array indicating valid positions in the aligned sequence.
        compactmapout_entry : List[Tuple[int, int, int]], the mapping of removed insertions to ata positions.
        label_names_aligned : List[str], the labels corresponding to the aligned (ata) sequence.

        Returns
        -------
        List[str], dealigned label names that match the length of the initial_seq.
        """
        labels = list(label_names_aligned)
        mask_list = list(mask)

        offset = 0
        for ata_pos, start_pos, end_pos in sorted(compactmapout_entry, key=lambda t: t[0]):
            insert_idx = ata_pos + offset  # insertion point right after ref column `ata_pos`
            n_insert = end_pos - start_pos + 1
            prev_label = labels[insert_idx - 1] if insert_idx > 0 else None
            next_label = labels[insert_idx] if insert_idx < len(labels) else None
            if prev_label and prev_label != "U":
                label_to_insert = prev_label
            elif next_label and next_label != "U":
                label_to_insert = next_label
            else:
                label_to_insert = "U"

            labels[insert_idx:insert_idx] = [label_to_insert] * n_insert
            mask_list[insert_idx:insert_idx] = [True] * n_insert
            offset += n_insert

        label_names_dealigned = [lbl for lbl, keep in zip(labels, mask_list) if keep]

        return label_names_dealigned

    def _overwrite_LTR(
        self,
        label_names_aligned: List[str],
        hxb2_to_ata: dict,
    ) -> List[str]:
        """
        Overwrite the first and last positions with labels 5'LTR and 3'LTR.
        """
        START_5LTR_ATA = [hxb2_to_ata[i] for i in START_5LTR]
        NEF_3LTR_ATA = [hxb2_to_ata[i] for i in NEF_3LTR]
        max_5LTR = max(START_5LTR_ATA)
        min_3LTR = min(NEF_3LTR_ATA)

        for i in range(0, max_5LTR + 1):
            label_names_aligned[i] = "5'LTR"

        for i in range(min_3LTR, len(label_names_aligned)):
            label_names_aligned[i] = "3'LTR"

        return label_names_aligned

    def _final_decision(
        self,
        dominant: str,
        composition: List[str],
        top_crf_types: List[Dict],
        seq_len: int,
    ) -> str:
        """
        Collapse the decoder output into a single final call:
          "pure.<subtype>.full.<len>"                                                   confident full single-subtype sequence
          "pure.<subtype>.partial.<len>.<like/(un)assigned>.<close(st)_match(es)>"      partial single-subtype sequence that could also come from a recombinant
          "recombinant.<comp>.<partial/full>.<len>.like.<close_matches>"                close to several CRFs within crf_match_margin
          "recombinant.<comp>.<partial/full>.<len>.assigned.<closest_crf>"              recombinant with one CRF match on top of the rest
          "recombinant.<comp>.<partial/full>.<len>.unassigned"                          recombinant with no CRF match over crf_assign_thr
        """

        is_pure = len(composition) <= 1
        is_partial =  seq_len < self.partial_thr
        comp_str = "+".join(composition)
        crf_candidates = [
            row for row in top_crf_types
        ]
        best_score = crf_candidates[0]["max_score"]
        close_matches = [
            row for row in crf_candidates
            if best_score - row["max_score"] <= self.crf_match_margin
        ]

        if is_pure:
            decision = f"pure.{dominant}"
        else:
            decision = f"recombinant.{comp_str}"

        if is_partial:
            decision += f".partial"
        else:
            decision += f".full"
            if is_pure:
                return decision

        if best_score < self.crf_assign_thr:
            decision += ".unassigned"
            return decision
        elif len(close_matches) == 1:
            closest_crf = close_matches[0]["crf_type"]
            decision += f".assigned.{closest_crf}"
            return decision
        else:
            close_crfs = "+".join(row["crf_type"] for row in close_matches)
            decision += f".like.{close_crfs}"
            return decision

    def query(
        self,
        sample_name: str,
        probs: np.ndarray,
        compactmapout_entry: List[Tuple[int, int, int]],
        hxb2_to_ata: dict,
        query_mask: np.ndarray,
    ) -> Dict:
        """
        Parameters
        ----------
        sample_name : str, name of the sequence being queried
        probs      : (L, C) per-position subtype probability profile
        compactmapout_entry : List[Tuple[int, int, int]], the mapping of removed insertions to ata positions
        query_mask : (L,) bool; valid positions of the sequence

        Returns
        -------
        Dict with keys:
            top_sequences, top_crf_types         — global intersection ranking
            label_names_aligned                  — (L,) str, 'U' at invalid/unknown pos
            label_names_dealigned                — (n_valid,) str
            regions_aligned, regions_dealigned   — breakpoint region lists
            dominant_subtype, dominant_fraction  — purity stats
            is_candidate_pure                    — bool
            composition, composition_str         — abstract characterisation
        """
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (self.L, self.C):
            raise ValueError(f"Expected ({self.L}, {self.C}), got {probs.shape}.")

        mask = np.asarray(query_mask, dtype=np.float32)

        # global ranking
        scores  = self._intersection_scores(probs, mask)
        top_idx = np.argsort(scores)[::-1][: self.top_k]

        top_sequences = [
            {
                "rank"    : rank + 1,
                "name"    : str(self.names[i]),
                "crf_type": self.crf_types[i],
                "score"   : float(scores[i]),
            }
            for rank, i in enumerate(top_idx)
        ]

        # per-position label sequence
        _, str_path = self._sliding_window(probs)       # (L,)
        str_path = self._trim_flanking_to_skip(str_path, mask, skip_label="U")

        label_names_aligned = self._overwrite_LTR(
            label_names_aligned=str_path,
            hxb2_to_ata=hxb2_to_ata)

        label_names_dealigned = self._gen_labels_dealigned(
            sample_name=sample_name,
            mask=mask,
            compactmapout_entry=compactmapout_entry,
            label_names_aligned=label_names_aligned)

        # breakpoint regions
        regions_aligned   = self._extract_breakpoints(label_names_aligned,   skip_label="U")
        regions_dealigned = self._extract_breakpoints(label_names_dealigned, skip_label="U")

        # purity
        dominant, fraction = self._purity_stats(label_names_dealigned)

        # composition
        total_real  = int(mask.sum())
        composition = self._composition_from_regions(regions_dealigned, total_len=total_real)
        if not composition:
            composition = [dominant] if dominant else []
        composition_str = (
            "_".join(composition) if len(composition) > 1
            else (composition[0] if composition else "")
        )

        top_crf_types = self._aggregate_by_type(self.crf_types, scores, self.top_k)

        final_decision = self._final_decision(
            dominant=dominant,
            composition=composition,
            top_crf_types=top_crf_types,
            seq_len=len(label_names_dealigned),
        )

        return {
            "top_sequences"         : top_sequences,
            "top_crf_types"         : top_crf_types,
            "label_names_aligned"   : label_names_aligned,
            "label_names_dealigned" : label_names_dealigned,
            "regions_aligned"       : regions_aligned,
            "regions_dealigned"     : regions_dealigned,
            "dominant_subtype"      : dominant,
            "dominant_fraction"     : fraction,
            "composition"           : composition,
            "composition_str"       : composition_str,
            "final_decision"        : final_decision,
        }