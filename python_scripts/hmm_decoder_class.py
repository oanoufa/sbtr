"""
HIV Subtype Decoder
===================

Unified decoder that wraps the Viterbi HMM and LANL CRF template matching
into a single class with a clean public API.

Pipeline (per sequence)
-----------------------
  1. HMM decodes per-position sigmoid scores → label sequence (pure subtype
     names, one per position).
  2. Purity check: if ≥ `purity_threshold` of non-gap positions share the
     same subtype label the sequence is flagged as *candidate-pure*.
  3. CRF matching: the label sequence is soft-scored against every LANL CRF
     template loaded at init time.
  4. Classification decision:
       - candidate-pure  → call it pure; *also* report the best CRF match so
                           the caller can decide whether a high CRF score
                           should override the pure call.
       - not pure        → call it recombinant.
  5. Abstract characterisation: ordered list of dominant subtypes extracted
     directly from the HMM breakpoint regions (e.g. ['A2', 'C', 'G', 'K']),
     independent of the CRF catalogue.

Usage
-----
    decoder = HIVDecoder(
        id_to_subtype   = {0: "A", 1: "B", ...},
        crf_labels_path = "lanl_crf_label_seqs.npz",
        epsilon         = 1e-6,
        purity_threshold= 0.95,
    )
    result = decoder.decode(probs, gap_mask=is_real)
    print(result)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecoderResult:
    """
    All outputs for a single sequence.

    Attributes
    ----------
    sample_name : str
        Passed through from the call site for convenience.

    # ── HMM outputs ────────────────────────────────────────────────────────
    label_names_aligned : np.ndarray[str]
        Per-position subtype label in ATA-alignment coordinates.
        Gap positions retain their HMM label (use gap_mask to filter).
    label_names_dealigned : np.ndarray[str]
        Per-position subtype label with gap positions removed.
    regions_aligned : List[Tuple[int, int, str]]
        Breakpoint regions in ATA coordinates: (start, end_excl, subtype).
    regions_dealigned : List[Tuple[int, int, str]]
        Breakpoint regions in raw-sequence coordinates.

    # ── Purity ─────────────────────────────────────────────────────────────
    dominant_subtype : str
        The most frequent HMM label among non-gap positions.
    dominant_fraction : float
        Fraction of non-gap positions carrying `dominant_subtype`.
    is_candidate_pure : bool
        True when dominant_fraction >= purity_threshold.

    # ── Abstract characterisation ──────────────────────────────────────────
    composition : List[str]
        Ordered, deduplicated list of subtypes present in the breakpoint
        regions of the *dealigned* sequence (e.g. ['A2', 'C', 'G']).
        Length-1 list for pure sequences.
    composition_str : str
        Human-readable form, e.g. 'A2_C_G' or 'B' for pure.

    # ── CRF matching ───────────────────────────────────────────────────────
    best_crf : str
        Name of the highest-scoring LANL CRF template.
    best_crf_score : float
        Soft-match score of best_crf (in [0, 1]).
    top_crf_matches : Dict[str, float]
        Full {crf_name: score} dict, sorted descending.

    # ── Classification ─────────────────────────────────────────────────────
    classification : str
        'pure' or 'recombinant'.
        For candidate-pure sequences this is always 'pure'; the caller can
        inspect `best_crf_score` and `best_crf` to decide whether to
        override it.
    """
    sample_name: str = ""

    # HMM
    label_names_aligned:   np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    label_names_dealigned: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    regions_aligned:   List[Tuple[int, int, str]] = field(default_factory=list)
    regions_dealigned: List[Tuple[int, int, str]] = field(default_factory=list)

    # Purity
    dominant_subtype:   str   = ""
    dominant_fraction:  float = 0.0
    is_candidate_pure:  bool  = False

    # Abstract characterisation
    composition:     List[str] = field(default_factory=list)
    composition_str: str       = ""

    # CRF
    best_crf:        str              = ""
    best_crf_score:  float            = 0.0
    top_crf_matches: Dict[str, float] = field(default_factory=dict)

    # Classification
    classification: str = ""

    def __str__(self) -> str:
        lines = [
            f"Sample          : {self.sample_name}",
            f"Classification  : {self.classification}",
            f"Composition     : {self.composition_str}",
            f"Dominant subtype: {self.dominant_subtype} ({self.dominant_fraction:.1%})",
            f"Candidate pure  : {self.is_candidate_pure}",
            f"Best CRF        : {self.best_crf}  (score={self.best_crf_score:.4f})",
        ]
        if self.top_crf_matches:
            top5 = list(self.top_crf_matches.items())[:5]
            lines.append("Top-5 CRF       : " +
                         "  ".join(f"{k}={v:.4f}" for k, v in top5))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Label-comparison helpers  (unchanged logic, moved here from inference.py)
# ---------------------------------------------------------------------------

_COMP_RE: re.Pattern = re.compile(r'^([A-Z])(\d+)$')


def _parse_component(label: str) -> Tuple[str, Optional[str]]:
    """Split one label atom into (base_letter, sub_index | None).

    'A1' → ('A', '1')   |   'A' → ('A', None)
    """
    m = _COMP_RE.match(label.strip().upper())
    return (m.group(1), m.group(2)) if m else (label.strip().upper(), None)


def _score_atoms(
    p_base: str, p_sub: Optional[str],
    c_base: str, c_sub: Optional[str],
) -> float:
    """Score two parsed label atoms.

    p_sub / c_sub    score   reason
    ─────────────────────────────────────────────
    None  / None     1.0     A  vs A   – exact
    '1'   / '1'      1.0     A1 vs A1  – exact
    None  / '1'      1.0     A  vs A1  – parent matches child
    '1'   / None     1.0     A1 vs A   – child matches parent
    '1'   / '2'      0.75    A1 vs A2  – siblings (same parent)
    ─────────────────────────────────────────────
    Different base letter → 0.0
    """
    if p_base != c_base:
        return 0.0
    if p_sub == c_sub:          # both None, or identical numbers
        return 1.0
    if p_sub is None or c_sub is None:   # one is the bare parent — symmetric
        return 1.0
    return 0.75                 # siblings, e.g. A1 vs A2


def _score_label_pair(pred: str, crf: str) -> float:
    """Soft-match score for one alignment position.

    Compound labels (e.g. 'A/E') are split on '/'; for each predicted
    component the best score against any CRF component is taken, and
    per-component scores are averaged.
    """
    if pred == crf:
        return 1.0
    pred_atoms = [_parse_component(p) for p in pred.split('/')]
    crf_atoms  = [_parse_component(c) for c in crf.split('/')]
    total = sum(
        max(_score_atoms(*pa, *ca) for ca in crf_atoms)
        for pa in pred_atoms
    )
    return total / len(pred_atoms)


def _crf_match_score(pred_labels: np.ndarray, crf_labels: np.ndarray) -> float:
    """Soft-match score in [0, 1] between a predicted label sequence and one
    CRF template.  'X' positions in the CRF are skipped."""
    if not isinstance(pred_labels, np.ndarray):
        pred_labels = np.array(pred_labels, dtype=object)
    if not isinstance(crf_labels, np.ndarray):
        crf_labels = np.array(crf_labels, dtype=object)
    assert pred_labels.shape == crf_labels.shape

    valid = crf_labels != 'X'
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0

    p = pred_labels[valid]
    c = crf_labels[valid]

    exact  = (p == c)
    scores = exact.astype(np.float64)
    for idx in np.where(~exact)[0]:
        scores[idx] = _score_label_pair(str(p[idx]), str(c[idx]))

    return float(scores.sum() / n_valid)


# ---------------------------------------------------------------------------
# HMM (Viterbi) — extracted from hmm_decoder_class.py, lightly refactored
# ---------------------------------------------------------------------------

class _HMMDecoder:
    """Internal Viterbi decoder.  Use HIVDecoder for the public API."""

    def __init__(
        self,
        id_to_subtype: Dict[int, str],
        epsilon:  float = 1e-6,
        min_prob: float = 1e-9,
    ):
        self.id_to_subtype  = id_to_subtype
        self.subtype_to_id  = {v: k for k, v in id_to_subtype.items()}
        self.n_states       = len(id_to_subtype)
        self.min_prob       = min_prob
        self.epsilon        = None
        self.log_transition = None
        self.set_epsilon(epsilon)

    # ── Transition matrix ───────────────────────────────────────────────────

    def set_epsilon(self, epsilon: float) -> None:
        assert 0 < epsilon < 1, "epsilon must be in (0, 1)"
        self.epsilon = epsilon
        K = self.n_states
        T = np.full((K, K), epsilon / (K - 1))
        np.fill_diagonal(T, 1.0 - epsilon)
        self.log_transition = np.log(T)

    # ── Core Viterbi ────────────────────────────────────────────────────────

    def _viterbi_single(self, log_emission: np.ndarray) -> np.ndarray:
        T_len, K = log_emission.shape
        viterbi  = np.full((T_len, K), -np.inf)
        backptr  = np.zeros((T_len, K), dtype=np.int32)
        viterbi[0] = log_emission[0] - np.log(K)
        for t in range(1, T_len):
            candidates    = viterbi[t - 1, :, None] + self.log_transition
            best_prev     = candidates.max(axis=0)
            backptr[t]    = candidates.argmax(axis=0)
            viterbi[t]    = best_prev + log_emission[t]
        path      = np.empty(T_len, dtype=np.int32)
        path[-1]  = viterbi[-1].argmax()
        for t in range(T_len - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]
        return path

    # ── Emission prep ───────────────────────────────────────────────────────

    def _to_log_emission(
        self,
        probs: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        return np.log(np.clip(probs, self.min_prob, None))

    # ── Decode ──────────────────────────────────────────────────────────────

    def decode_indices(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        gap_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return integer indices (seq_len,); -1 at gap positions."""
        log_em = self._to_log_emission(probs)                # (L, K)
        L      = log_em.shape[0]
        labels = np.full(L, -1, dtype=np.int32)
        valid  = gap_mask.astype(bool) if gap_mask is not None else np.ones(L, bool)
        labels[valid] = self._viterbi_single(log_em[valid])
        return labels

    def decode_names(
        self,
        probs: Union[np.ndarray, torch.Tensor],
        gap_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return string labels (seq_len,); gap positions get 'GAP'."""
        indices = self.decode_indices(probs, gap_mask)
        names   = np.where(
            indices >= 0,
            np.array([self.id_to_subtype.get(i, 'GAP') for i in indices], dtype=object),
            'GAP',
        )
        return names

    # ── Breakpoints ─────────────────────────────────────────────────────────

    def extract_breakpoints(
        self,
        label_names: np.ndarray,
        skip_label: str = 'GAP',
    ) -> List[Tuple[int, int, str]]:
        """Extract contiguous regions from a 1-D string label array.

        Returns list of (start, end_excl, subtype), 1-based, gap positions
        excluded.  The start/end refer to positions within the *non-skip*
        subsequence.
        """
        valid = label_names[label_names != skip_label]
        if len(valid) == 0:
            return []
        regions = []
        start, current = 1, valid[0]
        for i in range(1, len(valid)):
            if valid[i] != current:
                regions.append((start, i, str(current)))
                start, current = i, valid[i]
        regions.append((start, len(valid), str(current)))
        return regions

    # ── Serialisation ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(
            path,
            epsilon   = np.array([self.epsilon]),
            min_prob  = np.array([self.min_prob]),
            id_to_subtype_keys = np.array(list(self.id_to_subtype.keys())),
            id_to_subtype_vals = np.array(list(self.id_to_subtype.values())),
        )

    @classmethod
    def load(cls, path: str) -> "_HMMDecoder":
        data = np.load(path, allow_pickle=False)
        id_to_subtype = dict(zip(
            data["id_to_subtype_keys"].tolist(),
            data["id_to_subtype_vals"].tolist(),
        ))
        return cls(
            id_to_subtype = id_to_subtype,
            epsilon       = float(data["epsilon"][0]),
            min_prob      = float(data["min_prob"][0]),
        )

    def fit_epsilon(
        self,
        probs_list:       List[Union[np.ndarray, torch.Tensor]],
        true_labels_list: List[np.ndarray],
        epsilon_grid:     Optional[np.ndarray] = None,
        gap_masks:        Optional[List[np.ndarray]] = None,
        verbose:          bool = True,
    ) -> float:
        if epsilon_grid is None:
            epsilon_grid = np.logspace(-6, -1, 30)
        best_eps, best_err = None, np.inf
        for eps in epsilon_grid:
            self.set_epsilon(eps)
            errors = []
            for i, probs in enumerate(probs_list):
                mask = gap_masks[i] if gap_masks else None
                pred = self.decode_indices(probs, mask)
                true = true_labels_list[i]
                valid = pred >= 0
                errors.append((pred[valid] != true[valid]).mean())
            mean_err = float(np.mean(errors))
            if verbose:
                print(f"  ε={eps:.2e}  error={mean_err:.4f}")
            if mean_err < best_err:
                best_err, best_eps = mean_err, eps
        self.set_epsilon(best_eps)
        if verbose:
            print(f"\nBest ε = {best_eps:.2e}  (error={best_err:.4f})")
        return best_eps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HIVDecoder:
    """
    Unified HIV subtype decoder: HMM + CRF template matching.

    Parameters
    ----------
    id_to_subtype : dict
        {int_index: subtype_string} — must match the model's output order.
    crf_labels_path : str or Path
        Path to a .npz file produced by the LANL CRF pipeline.
        Loaded once at init; keys are CRF names, values are object arrays of
        per-position subtype strings.
    epsilon : float
        HMM transition probability (smaller = fewer breakpoints).
    min_prob : float
        Emission floor before log; avoids log(0).
    purity_threshold : float
        Fraction of non-gap positions that must share one subtype for a
        sequence to be flagged as candidate-pure. Default 0.95.
    """

    def __init__(
        self,
        id_to_subtype:    Dict[int, str],
        crf_labels_path:  Union[str, Path],
        epsilon:          float = 1e-6,
        min_prob:         float = 1e-9,
        purity_threshold: float = 0.95,
    ):
        self._hmm = _HMMDecoder(
            id_to_subtype = id_to_subtype,
            epsilon       = epsilon,
            min_prob      = min_prob,
        )
        self.purity_threshold = purity_threshold

        crf_data = np.load(str(crf_labels_path), allow_pickle=True)
        # Store as plain dict so the file handle can be closed
        self._crf_templates: Dict[str, np.ndarray] = {
            k: crf_data[k] for k in crf_data.files
        }
        print(f"Loaded {len(self._crf_templates)} CRF templates from {crf_labels_path}")

    # ── Pass-through HMM controls ───────────────────────────────────────────

    def set_epsilon(self, epsilon: float) -> None:
        """Update the HMM stickiness parameter."""
        self._hmm.set_epsilon(epsilon)

    def fit_epsilon(self, *args, **kwargs) -> float:
        """Delegate to _HMMDecoder.fit_epsilon(); see its docstring."""
        return self._hmm.fit_epsilon(*args, **kwargs)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _purity_stats(
        self,
        label_names: np.ndarray,
        gap_label: str = 'GAP',
    ) -> Tuple[str, float, bool]:
        """Return (dominant_subtype, dominant_fraction, is_candidate_pure)."""
        non_gap = label_names[label_names != gap_label]
        if len(non_gap) == 0:
            return ('', 0.0, False)
        unique, counts = np.unique(non_gap, return_counts=True)
        best_idx  = counts.argmax()
        dominant  = str(unique[best_idx])
        fraction  = float(counts[best_idx] / len(non_gap))
        is_pure   = fraction >= self.purity_threshold
        return dominant, fraction, is_pure

    def _composition_from_regions(
        self,
        regions: List[Tuple[int, int, str]],
        min_region_fraction: float = 0.02,
        total_len: Optional[int] = None,
    ) -> List[str]:
        """
        Derive an ordered, deduplicated list of subtypes from HMM breakpoint
        regions, filtering out very short spurious regions.

        Parameters
        ----------
        regions : output of _HMMDecoder.extract_breakpoints()
        min_region_fraction : regions shorter than this fraction of the total
            sequence are dropped (noise filter). Default 2%.
        total_len : total non-gap sequence length; inferred from regions if None.
        """
        if not regions:
            return []
        if total_len is None:
            total_len = regions[-1][1]          # last end position
        if total_len == 0:
            return []

        seen: set = set()
        composition: List[str] = []
        for start, end, subtype in regions:
            region_len = end - start
            if region_len / total_len < min_region_fraction:
                continue                         # skip tiny spurious fragments
            if subtype not in seen:
                seen.add(subtype)
                composition.append(subtype)

        return composition

    def _run_crf_matching(
        self,
        label_names: np.ndarray,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Score label_names against every loaded CRF template."""
        scores = {
            crf_name: _crf_match_score(label_names, crf_labels)
            for crf_name, crf_labels in self._crf_templates.items()
        }
        scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        best_crf   = next(iter(scores))
        best_score = scores[best_crf]
        return best_crf, best_score, scores

    # ── Public decode ───────────────────────────────────────────────────────

    def decode(
        self,
        probs:       Union[np.ndarray, torch.Tensor],
        gap_mask:    Optional[np.ndarray] = None,
        sample_name: str = "",
        ata_to_hxb2: Optional[Dict[int, int]] = None,
    ) -> DecoderResult:
        """
        Full decoding pipeline for a single sequence.

        Parameters
        ----------
        probs : (ata_len, n_subtypes) — sigmoid outputs from the NT model,
            in ATA-alignment coordinates (gaps already included).
        gap_mask : (ata_len,) bool — True = real nucleotide, False = gap.
            If None, all positions are treated as real.
        sample_name : optional label propagated into DecoderResult.
        ata_to_hxb2 : optional {ata_pos: hxb2_pos} mapping used to populate
            HXB2-coordinate region annotations (stored in DecoderResult
            regions_aligned with hxb2 coords when supplied).

        Returns
        -------
        DecoderResult
        """
        ata_len = probs.shape[0] if isinstance(probs, np.ndarray) else probs.shape[0]
        if gap_mask is None:
            gap_mask = np.ones(ata_len, dtype=bool)

        # ── 1. HMM decode ───────────────────────────────────────────────────
        label_names_aligned = self._hmm.decode_names(probs, gap_mask)   # (ata_len,)

        # Dealigned: only real (non-gap) positions
        label_names_dealigned = label_names_aligned[gap_mask]

        # ── 2. Breakpoint regions ───────────────────────────────────────────
        regions_aligned   = self._hmm.extract_breakpoints(label_names_aligned,   skip_label='GAP')
        regions_dealigned = self._hmm.extract_breakpoints(label_names_dealigned, skip_label='GAP')

        # ── 3. Purity check ─────────────────────────────────────────────────
        dominant, fraction, is_pure = self._purity_stats(label_names_dealigned)

        # ── 4. Abstract composition (from dealigned regions) ────────────────
        total_real = int(gap_mask.sum())
        composition = self._composition_from_regions(regions_dealigned, total_len=total_real)
        if not composition:
            composition = [dominant] if dominant else []
        composition_str = '_'.join(composition) if len(composition) > 1 else (composition[0] if composition else '')

        # ── 5. CRF template matching ─────────────────────────────────────────
        # Match against the aligned label sequence so positions correspond
        best_crf, best_score, all_scores = self._run_crf_matching(label_names_aligned)

        # ── 6. Classification ────────────────────────────────────────────────
        classification = 'pure' if is_pure else 'recombinant'

        return DecoderResult(
            sample_name            = sample_name,
            label_names_aligned    = label_names_aligned,
            label_names_dealigned  = label_names_dealigned,
            regions_aligned        = regions_aligned,
            regions_dealigned      = regions_dealigned,
            dominant_subtype       = dominant,
            dominant_fraction      = fraction,
            is_candidate_pure      = is_pure,
            composition            = composition,
            composition_str        = composition_str,
            best_crf               = best_crf,
            best_crf_score         = best_score,
            top_crf_matches        = all_scores,
            classification         = classification,
        )

    def __repr__(self) -> str:
        return (
            f"HIVDecoder("
            f"n_subtypes={self._hmm.n_states}, "
            f"epsilon={self._hmm.epsilon:.2e}, "
            f"purity_threshold={self.purity_threshold:.0%}, "
            f"n_crf_templates={len(self._crf_templates)})"
        )