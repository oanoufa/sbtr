"""
mutator.py
----------
GTR-based HIV sequence mutation utilities.

Provides:
  - Module-level functions (safe to call from parallel workers):
        parse_iqtree_rates, parse_iqtree_Q, build_substitution_probs,
        compute_n_muts, mutate_sequence_gtr

  - SequenceMutator class (for single-threaded / high-level use):
        Loads IQ-TREE files once, caches normalized rate arrays, and
        exposes site_rates_dict / sub_probs_dict for direct use in
        parallel workers, plus mutate() / augment_to_target() for
        callers that want a simpler interface.
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import config

CLOCK_RATES = config.CLOCK_RATES
MAX_YEAR    = config.MAX_YEAR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ACGT_BYTES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)
_BASE_TO_ACGT_IDX = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(_ACGT_BYTES):
    _BASE_TO_ACGT_IDX[_b] = _i

_REF_PREFIX            = re.compile(r"^Ref\.")
_SUBTYPES_WITH_FILES   = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# ---------------------------------------------------------------------------
# Module-level functions  (import these in parallel workers)
# ---------------------------------------------------------------------------

def parse_iqtree_rates(rate_path: str, ata_len: int) -> np.ndarray:
    """
    Parse per-site posterior mean rates from an IQ-TREE .rate file.
    Returns a float32 array of length ata_len normalized to sum to 1.
    """
    rates = np.zeros(ata_len, dtype=np.float32)
    with open(rate_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Site'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            site_idx = int(parts[0]) - 1          # 1-based → 0-based
            if 0 <= site_idx < ata_len:
                rates[site_idx] = float(parts[1])
    total = rates.sum()
    if total > 0.0:
        rates /= total
    return rates


def parse_iqtree_Q(iqtree_path: str) -> np.ndarray:
    """
    Extract the 4×4 GTR rate matrix Q (ACGT row/column order)
    from an IQ-TREE .iqtree file.
    """
    _ROW       = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    Q          = np.zeros((4, 4), dtype=np.float64)
    rows_found = 0
    in_q_block = False

    with open(iqtree_path) as fh:
        for line in fh:
            if 'Rate matrix Q:' in line:
                in_q_block = True
                continue
            if not in_q_block:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            first = stripped[0]
            if first in _ROW:
                parts = stripped.split()
                Q[_ROW[first]] = [float(v) for v in parts[1:5]]
                rows_found += 1
                if rows_found == 4:
                    break
            elif rows_found > 0:
                break                              # past the Q block

    if rows_found != 4:
        raise ValueError(
            f"Could not parse Q matrix from {iqtree_path} "
            f"(found {rows_found}/4 rows)"
        )
    return Q


def build_substitution_probs(Q: np.ndarray) -> np.ndarray:
    """
    Convert instantaneous rate matrix Q to per-event substitution probs.
    P[i, j] = Q[i, j] / (-Q[i, i])  for i ≠ j,  P[i, i] = 0.
    Each row sums to 1.
    """
    sub_probs = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        rate_away = -Q[i, i]
        for j in range(4):
            if i != j:
                sub_probs[i, j] = Q[i, j] / rate_away
    return sub_probs


def compute_n_muts(
    source_year: int,
    target_year: int,
    rate_arr:    np.ndarray,
    seg_start:   int,
    seg_len:     int,
    subtype:     str = 'avg',
    hxb2_len:    int = 9719,
) -> int:
    """
    Expected number of GTR substitutions for a genomic segment.

    Parameters
    ----------
    source_year : year the sequence was sampled.
    target_year : year to evolve towards.
    rate_arr    : per-site rates normalized to sum to 1 over the full genome.
    seg_start   : first ATA position of the segment.
    seg_len     : length of the segment.
    subtype     : key for clock-rate lookup (e.g. 'A1', 'B', 'avg').
    hxb2_len    : reference genome length for scaling.
    """
    year_delta = max(0, target_year - source_year)
    clock_rate = CLOCK_RATES.get(subtype, CLOCK_RATES["avg"])
    total_muts = clock_rate * hxb2_len * year_delta
    seg_weight = rate_arr[seg_start:seg_start + seg_len].sum()
    return int(round(total_muts * seg_weight))


def mutate_sequence_gtr(
    seg_arr:   np.ndarray,
    seg_start: int,
    rate_arr:  np.ndarray,
    rng:       np.random.Generator,
    n_muts:    int,
    sub_probs: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Apply n_muts GTR substitutions to seg_arr (in place).

    Parameters
    ----------
    seg_arr   : mutable uint8 array for the segment.
    seg_start : position of seg_arr[0] within the full-genome rate_arr.
    rate_arr  : per-site rates normalized to sum to 1.
    rng       : caller-owned numpy Generator.
    n_muts    : number of substitution events.
    sub_probs : (4, 4) ACGT-order matrix; sub_probs[i,j] = P(i→j | subst.).

    Returns
    -------
    (mutated_seg_arr, n_muts)
    """
    n = len(seg_arr)

    weights = rate_arr[seg_start:seg_start + n].copy()
    weights[~np.isin(seg_arr, _ACGT_BYTES)] = 0.0

    total = weights.sum()
    if total == 0.0 or n_muts == 0:
        return seg_arr, 0
    weights /= total

    sites    = rng.choice(n, size=n_muts, replace=True, p=weights)
    cur_idxs = _BASE_TO_ACGT_IDX[seg_arr[sites]]
    valid    = cur_idxs >= 0
    if not valid.any():
        return seg_arr, 0

    v_sites    = sites[valid]
    v_cur_idxs = cur_idxs[valid]
    n_valid    = int(valid.sum())

    cumprobs      = np.cumsum(sub_probs, axis=1)        # (4, 4)
    site_cumprobs = cumprobs[v_cur_idxs]                # (n_valid, 4)
    rand_vals     = rng.random(n_valid)

    new_idxs = (site_cumprobs <= rand_vals[:, None]).sum(axis=1).astype(np.int8)
    new_idxs = np.clip(new_idxs, 0, 3)

    seg_arr[v_sites] = _ACGT_BYTES[new_idxs]
    return seg_arr, n_muts


# ---------------------------------------------------------------------------
# SequenceMutator
# ---------------------------------------------------------------------------

class SequenceMutator:
    """
    High-level GTR mutator.

    Loads IQ-TREE per-site rates and Q matrices for subtypes A–G;
    computes averages for all others.  The public attributes
    ``site_rates_dict`` and ``sub_probs_dict`` can be passed directly
    to parallel workers (e.g. in seq_gen.py) so that workers call the
    module-level ``compute_n_muts`` / ``mutate_sequence_gtr`` with
    their own RNG.

    For single-threaded callers (e.g. crf_ref_bank.py) the instance
    methods ``mutate()`` and ``augment_to_target()`` use an internal RNG.

    Parameters
    ----------
    iqtree_dir : str
        Directory that contains ``HIV1_{st}_ALIGNED.fasta.rate`` and
        ``HIV1_{st}_ALIGNED.fasta.iqtree`` for st in A–G.
    ata_len : int
        Length of the ATA alignment (number of sites).
    seed : int
        Seed for the internal numpy and Python RNGs.
    cache_dir : str, optional
        Directory for caching pre-computed ``.npy`` rate arrays.
        Defaults to ``iqtree_dir``.
    """

    def __init__(
        self,
        iqtree_dir: str,
        ata_len:    int,
        seed:       int = 42,
        cache_dir:  Optional[str] = None,
    ):
        self.iqtree_dir = Path(iqtree_dir)
        self.ata_len    = ata_len
        self._rng       = np.random.default_rng(seed)
        self._py_rng    = random.Random(seed)

        _cache = Path(cache_dir) if cache_dir else self.iqtree_dir
        _cache.mkdir(parents=True, exist_ok=True)

        self.site_rates_dict: Dict[str, np.ndarray] = {}
        self.sub_probs_dict:  Dict[str, np.ndarray] = {}
        _Q_matrices:          Dict[str, np.ndarray] = {}

        # ---- per-subtype loading ------------------------------------
        for st in _SUBTYPES_WITH_FILES:
            rate_file   = self.iqtree_dir / f"HIV1_{st}_ALIGNED.fasta.rate"
            iqtree_file = self.iqtree_dir / f"HIV1_{st}_ALIGNED.fasta.iqtree"
            cache_path  = _cache / f"site_rates_{st}.npy"

            if cache_path.exists():
                rates = np.load(cache_path)
            else:
                print(f"  [SequenceMutator] Parsing IQ-TREE rates for subtype {st} …")
                rates = parse_iqtree_rates(str(rate_file), ata_len)
                np.save(cache_path, rates)

            self.site_rates_dict[st] = rates

            Q = parse_iqtree_Q(str(iqtree_file))
            _Q_matrices[st]         = Q
            self.sub_probs_dict[st] = build_substitution_probs(Q)

        # ---- average over A-G ---------------------------------------
        avg_cache = _cache / "site_rates_avg.npy"
        if avg_cache.exists():
            avg_rates = np.load(avg_cache)
        else:
            stacked   = np.stack(
                [self.site_rates_dict[s] for s in _SUBTYPES_WITH_FILES]
            )
            avg_rates = stacked.mean(axis=0)
            avg_rates /= avg_rates.sum()
            np.save(avg_cache, avg_rates)

        self.site_rates_dict['avg'] = avg_rates

        avg_Q = np.mean(
            [_Q_matrices[s] for s in _SUBTYPES_WITH_FILES], axis=0
        )
        self.sub_probs_dict['avg'] = build_substitution_probs(avg_Q)

        print(
            f"  [SequenceMutator] Ready — "
            f"{len(self.site_rates_dict)} rate arrays, "
            f"{len(self.sub_probs_dict)} Q matrices."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year(record_id: str) -> int:
        """
        Extract the sampling year from a sequence ID of the form
        ``[Ref.]TYPE.COUNTRY.YEAR.NAME.ACCESSION``.

        Two-digit rule: year < 40 → 2000 + year, year >= 40 → 1900 + year.
        Falls back to 2000 with a warning on any parsing failure.
        """
        clean = _REF_PREFIX.sub("", record_id)
        parts = clean.split(".")
        try:
            year_int = int(parts[2])
        except (IndexError, ValueError):
            print(
                f"  WARNING: could not extract year from '{record_id}', "
                f"defaulting to 2000."
            )
            return 2000

        if year_int < 100:
            return 2000 + year_int if year_int < 40 else 1900 + year_int
        return year_int

    def _resolve_rate_key(self, subtype_key: str) -> str:
        """
        Map any subtype string to a valid key in site_rates_dict.

        Priority order:
          1. Direct match (e.g. 'avg', 'A', 'B').
          2. First character of a multi-character name (e.g. 'A1' → 'A').
          3. 'avg' as final fallback.
        """
        if subtype_key in self.site_rates_dict:
            return subtype_key
        first = subtype_key[0] if subtype_key else ''
        if first in self.site_rates_dict:
            return first
        return 'avg'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mutate(
        self,
        arr:         np.ndarray,
        source_year: int,
        target_year: int,
        seg_start:   int = 0,
        subtype_key: str = 'avg',
    ) -> Tuple[np.ndarray, int]:
        """
        Apply GTR mutations to a copy of ``arr`` using the internal RNG.

        Parameters
        ----------
        arr         : uint8 array of the full-genome (or segment) sequence.
        source_year : year the parent sequence was sampled.
        target_year : year to evolve towards.
        seg_start   : offset of ``arr[0]`` within the full-genome rate array.
        subtype_key : subtype for rate / clock-rate lookup (e.g. 'A1', 'avg').

        Returns
        -------
        (mutated_copy, n_muts_applied)
        """
        rk        = self._resolve_rate_key(subtype_key)
        rate_arr  = self.site_rates_dict[rk]
        sub_probs = self.sub_probs_dict[rk]

        n_muts = compute_n_muts(
            source_year, target_year,
            rate_arr, seg_start, len(arr),
            subtype=subtype_key,
        )
        return mutate_sequence_gtr(
            arr.copy(), seg_start, rate_arr, self._rng, n_muts, sub_probs
        )

    def augment_to_target(
        self,
        records:      List[SeqRecord],
        target_count: int = 10,
        subtype_key:  str = 'avg',
    ) -> List[SeqRecord]:
        """
        Pad a list of SeqRecords to ``target_count`` by mutating existing ones.

        For each synthetic record:
          1. A parent is chosen at random from ``records``.
          2. The parent's sampling year is extracted from its ID.
          3. A target year is sampled uniformly in [parent_year, MAX_YEAR].
          4. GTR mutations are applied.

        Synthetic record IDs have the form
        ``{parent_id}_syn{k}_yr{target_year}``.

        Returns the original records plus any synthetic ones;
        length = max(len(records), target_count).
        """
        if len(records) >= target_count:
            return list(records)

        n_to_gen  = target_count - len(records)
        synthetic: List[SeqRecord] = []

        for k in range(1, n_to_gen + 1):
            parent   = self._py_rng.choice(records)
            src_year = self._extract_year(parent.id)
            tgt_year = self._py_rng.randint(src_year, MAX_YEAR)

            arr = np.frombuffer(
                str(parent.seq).upper().encode(), dtype=np.uint8
            ).copy()

            mutated, _ = self.mutate(
                arr, src_year, tgt_year,
                seg_start=0, subtype_key=subtype_key,
            )
            mutated[mutated == ord('-')] = ord('N')

            synthetic.append(SeqRecord(
                Seq(mutated.tobytes().decode('ascii')),
                id=f"{parent.id}_syn{k}_yr{tgt_year}",
                description="",
            ))

        return list(records) + synthetic