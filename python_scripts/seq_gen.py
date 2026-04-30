"""
Unified HIV recombinant sequence generator.

Flow per recombinant:
  1. Sample n_subtypes, n_breakpoints from distributions inferred from real CRFs
  2. Pick subtypes (divergence-group-aware weighting)
  3. Pick one parent sequence per subtype
  4. Compute windowed divergence profile for each adjacent-subtype transition
  5. Place breakpoints at positions weighted by divergence (≥ threshold)
  6. Build chimeric sequence, mutate, write

Eliminates the blueprint pre-generation step entirely.
"""

import random
import json
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys
import gzip
import config
from pathlib import Path

from utils import build_hxb2_ata_maps


workspace_path = config.WORKSPACE_PATH
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--n_seq",             type=int,   default=100)
parser.add_argument("--rp",               type=float, default=0.8)
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--n_workers",        type=int,   default=None)
parser.add_argument("--window_half",      type=int,   default=50,
                    help="Half-window (each side) for divergence profile.")
parser.add_argument("--min_div",          type=int,   default=15,
                    help="Min Hamming differences in the window for eligibility.")
parser.add_argument("--max_retries",      type=int,   default=50)
parser.add_argument("--realistic",      action="store_true",
                    help="Whether to generate more realistic recombinants or not")
parser.add_argument("--force_divergent", action="store_true",
                    help="If true, force breakpoints to occur in divergent regions. If false, random placement.")
args = parser.parse_args()

N_SEQ         = args.n_seq
RP            = args.rp
SEED          = args.seed
N_WORKERS     = args.n_workers or mp.cpu_count()
WINDOW_HALF   = args.window_half
MIN_DIV       = args.min_div
MAX_RETRIES   = args.max_retries
REALISTIC     = args.realistic
FORCE_DIV     = args.force_divergent

out_dir    = f"{workspace_path}/data/output/{N_SEQ}_{RP}"
out_seqs   = f"{out_dir}/sequences_{N_SEQ}_{RP}.npy"
out_labels = f"{out_dir}/labels_{N_SEQ}_{RP}.npy"
out_masks  = f"{out_dir}/loss_masks_{N_SEQ}_{RP}.npy"
out_meta   = f"{out_dir}/metadata_{N_SEQ}_{RP}.tsv"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Subtype divergence groups
# ---------------------------------------------------------------------------
SUBTYPE_GROUPS = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A6": 0, "A7": 0, "A8": 0,
    "F1": 1, "F2": 1,
    "B": 2, "C": 2, "D": 2, "G": 2, "H": 2, "J": 2,
    "K": 2, "L": 2, "E": 2, "O": 2, "N": 2, "P": 2, "CPZ": 2, "GOR": 2,
}
DIVERGENCE_WEIGHTS = [
    [0.05, 1.00, 1.00],
    [1.00, 0.05, 1.00],
    [1.00, 1.00, 1.00],
]

_ATCG = np.array([ord("A"), ord("T"), ord("C"), ord("G")], dtype=np.uint8)
_BASE_TO_IDX = np.zeros(256, dtype=np.int8)
for _i, _b in enumerate(_ATCG):
    _BASE_TO_IDX[_b] = _i

_TRANSITIONS = np.array([3, 2, 1, 0], dtype=np.int8)  # A->G(3), T->C(2), C->T(1), G->A(0)
# Transversions (2 options per base)
_TRANSVERSIONS_1 = np.array([1, 0, 3, 2], dtype=np.int8) # A->T, T->A, C->G, G->C
_TRANSVERSIONS_2 = np.array([2, 3, 0, 1], dtype=np.int8) # A->C, T->G, C->A, G->T

# ---------------------------------------------------------------------------
# Parameter inference
# ---------------------------------------------------------------------------
def infer_params(df_seg: pd.DataFrame, ata_len: int, hxb2_len: int = 9719) -> dict:
    """
    Infer generative parameters from real CRF segment annotations.
    Returns dict with keys: n_breakpoints, n_subtypes, min_seg_len (in ATA coords).
    """
    params = {}
    is_compound = df_seg["subtype"].str.contains("/", regex=False)
    df_pure = df_seg[~is_compound]

    # --- breakpoints / CRF  →  negative binomial ---
    n_pure_per_crf = df_pure.groupby("crf").size()
    n_bp = (n_pure_per_crf - 1).clip(lower=0)
    mu, var = float(n_bp.mean()), float(n_bp.var())
    if var > mu:
        r, p = mu ** 2 / (var - mu), mu / var
    else:
        r, p = 1e6, mu / (mu + 1)
    params["n_breakpoints"] = {"r": r, "p": p}

    # --- distinct subtypes / CRF  →  empirical PMF ---
    pmf = df_pure.groupby("crf")["subtype"].nunique().value_counts(normalize=True).sort_index()
    params["n_subtypes"] = {"values": pmf.index.tolist(), "probs": pmf.values.tolist()}

    # --- minimum segment length  (scaled HXB2 → ATA) ---
    pure_len = df_pure["length"].dropna().astype(float)
    min_hxb2 = int(pure_len[pure_len > 0].min())
    params["min_seg_len"] = max(1, int(min_hxb2 * ata_len / hxb2_len))

    print(f"  n_bp/CRF  mean={mu:.1f} var={var:.1f}  NB(r={r:.2f}, p={p:.3f})")
    print(f"  subtypes  {dict(zip(pmf.index, (pmf.values*100).round(1)))}")
    print(f"  min_seg   {min_hxb2} HXB2 → {params['min_seg_len']} ATA")
    return params

def compare_generated_vs_real(names, out_labels, n_subtypes, n_packed, ata_len,
                               pure_st_list, df_seg, ata_to_hxb2):
    """
    Read back generated labels, extract per-recombinant statistics,
    and compare against real CRF segment data.
    
    Segment lengths are converted to HXB2 coordinates via ata_to_hxb2
    for a fair comparison against the real data (which is annotated in HXB2).
    """
    lbl_mm = np.load(out_labels, mmap_mode="r")

    gen_n_bp      = []
    gen_n_st      = []
    gen_seg_lens_ata  = []
    gen_seg_lens_hxb2 = []

    rec_indices = [i for i, n in enumerate(names) if n.startswith("recombinant")]

    for idx in tqdm(rec_indices, desc="Analyzing generated recombinants", mininterval=10):
        lbl = np.unpackbits(lbl_mm[idx], axis=-1)[:, :n_subtypes]

        # Active subtype at each position (-1 where all labels are False, i.e. N)
        any_active = lbl.any(axis=1)
        active = np.full(ata_len, -1, dtype=np.int32)
        active[any_active] = lbl[any_active].argmax(axis=1)

        labeled_pos = np.where(active >= 0)[0]
        if len(labeled_pos) < 2:
            continue

        labeled_st = active[labeled_pos]

        # Detect transitions between adjacent labeled positions
        change_idx = np.where(np.diff(labeled_st) != 0)[0]

        gen_n_bp.append(len(change_idx))
        gen_n_st.append(len(np.unique(labeled_st)))

        # Segment boundaries in ATA, then convert to HXB2 lengths
        bp_ata       = labeled_pos[change_idx + 1]
        seg_starts   = np.concatenate([[labeled_pos[0]],      bp_ata])
        seg_ends     = np.concatenate([bp_ata, [labeled_pos[-1] + 1]])

        for s, e in zip(seg_starts, seg_ends):
            gen_seg_lens_ata.append(int(e) - int(s))
            e_clamped = min(int(e) - 1, ata_len - 1)
            hxb2_len  = int(ata_to_hxb2[e_clamped]) - int(ata_to_hxb2[int(s)])
            if hxb2_len > 0:
                gen_seg_lens_hxb2.append(hxb2_len)
    # ------------------------------------------------------------------
    # Real CRF statistics
    # ------------------------------------------------------------------
    is_compound   = df_seg["subtype"].str.contains("/", regex=False)
    df_pure       = df_seg[~is_compound]

    real_n_bp     = (df_pure.groupby("crf").size() - 1).clip(lower=0)
    real_n_st     = df_pure.groupby("crf")["subtype"].nunique()
    real_seg_lens = df_pure["length"].dropna().astype(float)
    real_seg_lens = real_seg_lens[real_seg_lens > 0]

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------
    W_L, W_C = 30, 12

    def _header(title):
        print(f"\n  -- {title} --")
        print(f"  {'':>{W_L}}  {'real':>{W_C}}  {'generated':>{W_C}}")
        print(f"  {'-' * (W_L + 2 * W_C + 4)}")

    def _row(label, rv, gv, fmt=".1f"):
        def _f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a".rjust(W_C)
            return format(v, fmt).rjust(W_C)
        print(f"  {label:>{W_L}}  {_f(rv)}  {_f(gv)}")

    def _discrete(title, real_s, gen_list):
        _header(title)
        r = pd.Series(real_s)
        g = pd.Series(gen_list, dtype=int)
        _row("n",    len(r),    len(g),    fmt="d")
        _row("mean", r.mean(),  g.mean())
        _row("std",  r.std(),   g.std())
        _row("min",  r.min(),   g.min(),   fmt=".0f")
        _row("max",  r.max(),   g.max(),   fmt=".0f")
        # Full PMF comparison
        all_vals = sorted(set(r.tolist()) | set(g.tolist()))
        rpmf = r.value_counts(normalize=True)
        gpmf = g.value_counts(normalize=True)
        print(f"\n  {'value':>{W_L}}  {'real %':>{W_C}}  {'gen %':>{W_C}}")
        for v in all_vals:
            _row(str(v), rpmf.get(v, 0.0) * 100, gpmf.get(v, 0.0) * 100)

    def _continuous(title, real_s, gen_list):
        _header(title)
        r = pd.Series(real_s, dtype=float)
        g = pd.Series(gen_list, dtype=float)
        g = g[g > 0]
        _row("n",      len(r),      len(g),      fmt="d")
        _row("min",    r.min(),     g.min()    if len(g) else float("nan"))
        _row("median", r.median(),  g.median() if len(g) else float("nan"))
        _row("mean",   r.mean(),    g.mean()   if len(g) else float("nan"))
        _row("std",    r.std(),     g.std()    if len(g) else float("nan"))
        _row("max",    r.max(),     g.max()    if len(g) else float("nan"))
        if len(r) > 1 and len(g) > 1:
            _row("log-normal μ", np.log(r).mean(), np.log(g).mean(), fmt=".3f")
            _row("log-normal σ", np.log(r).std(),  np.log(g).std(),  fmt=".3f")

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(f"\n  === Distribution comparison: {len(real_n_bp)} real CRFs  vs  "
          f"{len(rec_indices)} generated recombinants ===")
    _discrete("n_breakpoints / sequence",  real_n_bp,     gen_n_bp)
    _discrete("n_subtypes / sequence",     real_n_st,     gen_n_st)
    _continuous("segment lengths (HXB2 bp)", real_seg_lens, gen_seg_lens_hxb2)
    _continuous("segment lengths (ATA pos)", pd.Series(dtype=float), gen_seg_lens_ata)

# ---------------------------------------------------------------------------
# Empirical mutation rates
# ---------------------------------------------------------------------------
def compute_empirical_mutation_rates(
    alignment_path:   str,
    ata_len:          int,
    target_mean_rate: float = 0.04,
) -> np.ndarray:
    """
    Derives site-specific mutation rates from the diversity of a reference
    alignment.  Fully vectorised — no Python loop over columns.

    Uses a large within-subtype alignment for best results (see notes above).
    """
    print("\nComputing empirical per-position mutation rates from alignment...")

    # ── 1. Load all sequences into a (N, ata_len) uint8 array ─────────────
    rows = []
    opener = gzip.open if alignment_path.endswith(".gz") else open
    with opener(alignment_path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            seq = np.frombuffer(str(rec.seq).upper().encode(), dtype=np.uint8)
            if seq.shape[0] == ata_len:
                rows.append(seq)

    if not rows:
        raise ValueError("No sequences of expected length found.")

    arr = np.stack(rows)                         # (N, ata_len)  uint8

    # ── 2. Gap mask  ──────────────────────────────────────────────────────
    GAP   = ord("-")
    DOT   = ord(".")
    is_gap = (arr == GAP) | (arr == DOT)         # (N, ata_len) bool

    # ── 3. Vectorised per-column diversity ────────────────────────────────
    # For each column we need  1 - (count of modal base / valid base count).
    # Strategy: iterate over the 4 nucleotide bytes; for each base compute
    # its frequency, then take the column-wise maximum.

    BASES = [ord("A"), ord("C"), ord("G"), ord("T")]

    valid_counts = (~is_gap).sum(axis=0).astype(np.float32)   # (ata_len,)
    max_counts   = np.zeros(ata_len, dtype=np.float32)

    for base in BASES:
        counts = (arr == base).sum(axis=0).astype(np.float32) # (ata_len,)
        np.maximum(max_counts, counts, out=max_counts)

    # Avoid division by zero at fully-gapped columns
    safe_valid  = np.where(valid_counts > 0, valid_counts, 1.0)
    max_freq    = max_counts / safe_valid                      # (ata_len,)
    diversity   = np.where(valid_counts > 0, 1.0 - max_freq, 0.0)

    # ── 4. Scale to target mean rate ──────────────────────────────────────
    mean_div = diversity.mean()
    if mean_div > 0:
        diversity *= target_mean_rate / mean_div

    # ── 5. Clip extremes ──────────────────────────────────────────────────
    return np.clip(diversity, 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------
def mutate_sequence_vec(seg_arr, seg_start, rate_array, rng):
    is_base = np.isin(seg_arr, _ATCG)
    n = len(seg_arr)
    mutate_mask = is_base & (rng.random(n, dtype=np.float32) < rate_array[seg_start:seg_start + n])
    
    if mutate_mask.any():
        idx = np.where(mutate_mask)[0]
        cur = _BASE_TO_IDX[seg_arr[idx]]
        
        # 80% Transition, 20% Transversion
        rand_ti_tv = rng.random(len(idx), dtype=np.float32)
        
        # Default to Transition
        new = _TRANSITIONS[cur]
        
        # Overwrite with Transversions where applicable
        tv1_mask = (rand_ti_tv > 0.8) & (rand_ti_tv <= 0.9)
        new[tv1_mask] = _TRANSVERSIONS_1[cur[tv1_mask]]
        
        tv2_mask = rand_ti_tv > 0.9
        new[tv2_mask] = _TRANSVERSIONS_2[cur[tv2_mask]]
        
        seg_arr[idx] = _ATCG[new]
    return seg_arr

# ---------------------------------------------------------------------------
# Subtype sampling
# ---------------------------------------------------------------------------
def sample_subtypes(n, pool, py_rng):
    """Pick n distinct subtypes with divergence-aware weighting."""
    chosen, remain = [], list(pool)
    for _ in range(n):
        if not chosen:
            st = py_rng.choice(remain)
        else:
            weights = []
            for st_cand in remain:
                g = SUBTYPE_GROUPS.get(st_cand, 2)
                w = max(DIVERGENCE_WEIGHTS[g][SUBTYPE_GROUPS.get(a, 2)] for a in chosen)
                weights.append(w)
            total = sum(weights)
            r = py_rng.random() * total
            cum, st = 0.0, remain[-1]
            for st_cand, w in zip(remain, weights):
                cum += w
                if r <= cum:
                    st = st_cand
                    break
        chosen.append(st)
        remain.remove(st)
    return chosen

# ---------------------------------------------------------------------------
# Divergence-aware breakpoint placement
# ---------------------------------------------------------------------------
def divergence_profile(seq1, seq2, window_half):
    """
    Windowed Hamming distance (each side = window_half positions).
    Returns float32 array, same length as input.  ~50 µs for 10,475 positions.
    """
    v1 = np.isin(seq1, _ATCG)
    v2 = np.isin(seq2, _ATCG)
    diff = ((seq1 != seq2) & v1 & v2).astype(np.float32)
    kernel = np.ones(2 * window_half + 1, dtype=np.float32)
    return np.convolve(diff, kernel, mode="same")


def assign_subtypes_to_segments(n_seg, subtype_pool, py_rng):
    """
    Assign subtypes so that every subtype appears ≥ 1 and no two adjacent
    segments share the same subtype.

    Since the pool is a list of *distinct* subtypes, a random permutation is
    already free of adjacent duplicates.  Extension appends random picks that
    differ from the last element.
    """
    pool = list(subtype_pool)
    py_rng.shuffle(pool)          # distinct elements → no adjacent dups
    result = list(pool)
    while len(result) < n_seg:
        cands = [s for s in pool if s != result[-1]]
        result.append(py_rng.choice(cands))
    return result[:n_seg]


def place_breakpoints(seg_subtypes, parents, ata_len,
                      window_half, min_div, min_seg_len, rng, force_div=False):
    """
    Place one breakpoint between each pair of adjacent segments.

    For each transition (different subtypes), the breakpoint is sampled from
    positions where the divergence profile ≥ min_div, **weighted** by
    divergence (prefers the most distinguishable spots).

    Spacing constraint: every segment ≥ min_seg_len positions.

    Returns: list[int] of ATA breakpoint positions, or None on failure.
    """
    n_bp = len(seg_subtypes) - 1
    
    # ML Robustness: Force segments to be at least 200bp to survive loss masking
    min_seg_len = max(min_seg_len, 200)

    if force_div:
        # --- Divergence-weighted placement (unchanged) ---
        breakpoints = []
        prev = 0
        _cache = {}
        for i in range(n_bp):
            st_L = seg_subtypes[i]
            st_R = seg_subtypes[i + 1]

            remaining_bps = n_bp - i - 1
            right_limit   = ata_len - (remaining_bps + 1) * min_seg_len

            key = (id(parents[st_L]), id(parents[st_R]))
            if key not in _cache:
                _cache[key] = divergence_profile(parents[st_L], parents[st_R], window_half)
            profile = _cache[key]
            eligible = profile >= min_div
            eligible[: prev + min_seg_len] = False
            eligible[right_limit:] = False
            cands = np.where(eligible)[0]
            if cands.size == 0:
                return None
            w = profile[cands].astype(np.float64)
            w /= w.sum()
            bp = int(rng.choice(cands, p=w))
            breakpoints.append(bp)
            prev = bp
        return breakpoints

    else:
        # --- True Uniform Random Partitioning ---
        # Calculate how much "free" space we have left after reserving min_seg_len for every segment
        L_free = ata_len - (n_bp + 1) * min_seg_len
        if L_free < 0:
            return None  # Too many breakpoints requested for this sequence length

        # Pick n_bp random points in the free space, and sort them
        raw_bps = np.sort(rng.choice(L_free + 1, size=n_bp, replace=True))

        # Re-add the minimum segment lengths to get actual ATA coordinates
        breakpoints = []
        current_shift = min_seg_len
        for rb in raw_bps:
            bp = rb + current_shift
            breakpoints.append(int(bp))
            current_shift += min_seg_len

        return breakpoints

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(cfg):
    row_start = cfg["row_start"]
    row_end   = cfg["row_end"]
    chunk_sz  = row_end - row_start

    params     = cfg["params"]
    nb         = params["n_breakpoints"]
    ns         = params["n_subtypes"]
    min_seg    = params["min_seg_len"]
    st_id      = cfg["pure_st_to_id_dict"]
    n_st_total = len(st_id)
    ata_len    = cfg["ata_len"]
    n_packed   = cfg["n_packed"]
    rate_arrays   = cfg["rate_arrays"]
    st_list    = cfg["pure_st_list"]
    whalf      = cfg["window_half"]
    mdiv       = cfg["min_div"]
    max_ret    = cfg["max_retries"]
    force_div  = cfg["force_divergent"]
    realistic  = cfg["realistic"]

    seq_mm  = np.load(cfg["out_seqs"],   mmap_mode="r+")
    lbl_mm  = np.load(cfg["out_labels"], mmap_mode="r+")
    mask_mm = np.load(cfg["out_masks"],  mmap_mode="r+")

    st_arr = {
        st: [np.frombuffer(s.encode(), dtype=np.uint8) for s in seqs]
        for st, seqs in cfg["st_to_seq_dict"].items()
    }

    rng    = np.random.default_rng(cfg["worker_seed"])
    py_rng = random.Random(cfg["worker_seed"])
    records  = []

    for li in range(chunk_sz):
        is_rec = py_rng.random() <= RP
        ok = False

        if not is_rec:
            n_bp = 0
            st  = py_rng.choice(st_list)
            src = py_rng.choice(st_arr[st]).copy()
            rate_arr = rate_arrays.get(st[0], rate_arrays['avg'])
            seq_row = mutate_sequence_vec(src, 0, rate_arr, rng)
            lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
            lbl_row[:, st_id[st]] = True
            mask_row = np.ones(ata_len, dtype=bool)  # No ambiguity
            name = f"pure_{st}"
            ok = True
        else:
            for _attempt in range(max_ret):
                if realistic:
                    n_sub = max(2, int(rng.choice(ns["values"], p=ns["probs"])))
                    n_sub = min(n_sub, len(st_list))
                    n_bp  = max(1, int(rng.negative_binomial(nb["r"], nb["p"])))
                    n_seg = n_bp + 1
                    n_sub = min(n_sub, n_seg)
                    if n_sub < 2: n_sub, n_seg, n_bp = 2, 2, 1
                else:
                    n_sub = rng.integers(2, 7)
                    n_bp  = rng.integers(1, 17)
                    n_seg = n_bp + 1
                    n_sub = min(n_sub, n_seg)

                chosen = sample_subtypes(n_sub, st_list, py_rng)
                seg_st = assign_subtypes_to_segments(n_seg, chosen, py_rng)
                parents = {st: py_rng.choice(st_arr[st]) for st in chosen}

                bps = place_breakpoints(seg_st, parents, ata_len,
                                        whalf, mdiv, min_seg, rng, force_div)
                if bps is None:
                    continue

                bounds  = [0] + bps + [ata_len]
                seq_row = np.empty(ata_len, dtype=np.uint8)
                lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
                mask_row = np.ones(ata_len, dtype=bool)

                # 1) Build chunks and labels
                for si in range(n_seg):
                    s, e = bounds[si], bounds[si + 1]
                    if e <= s: continue
                    chunk = parents[seg_st[si]][s:e].copy()
                    rate_arr = rate_arrays.get(seg_st[si][0], rate_arrays['avg'])
                    seq_row[s:e] = mutate_sequence_vec(chunk, s, rate_arr, rng)
                    lbl_row[s:e, st_id[seg_st[si]]] = True

                # 2) Build Ambiguity Mask (Loss Mask)
                # Find regions where Parent A == Parent B across the breakpoint
                for si in range(n_seg - 1):
                    p1 = parents[seg_st[si]]
                    p2 = parents[seg_st[si+1]]
                    bp = bps[si]
                    
                    l = bp - 1
                    while l >= 0 and p1[l] == p2[l]: l -= 1
                    r = bp
                    while r < ata_len and p1[r] == p2[r]: r += 1
                    
                    # Set mask to False in identical/ambiguous regions
                    mask_row[l+1:r] = False

                name = f"recombinant_{'+'.join(dict.fromkeys(seg_st))}"
                ok = True
                break

            if not ok:
                n_bp = 0
                st  = py_rng.choice(st_list)
                src = py_rng.choice(st_arr[st]).copy()
                rate_arr = rate_arrays.get(st[0], rate_arrays['avg'])
                seq_row = mutate_sequence_vec(src, 0, rate_arr, rng)
                lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
                lbl_row[:, st_id[st]] = True
                mask_row = np.ones(ata_len, dtype=bool)
                name = f"pure_{st}"

        # ---- cleanup & write ---------------------------------------------
        seq_row[seq_row == ord("-")] = ord("N")
        seq_mm[row_start + li] = seq_row
        lbl_mm[row_start + li] = np.packbits(lbl_row, axis=-1)
        mask_mm[row_start + li] = mask_row
        records.append((name, n_bp))

    seq_mm.flush()
    lbl_mm.flush()
    mask_mm.flush()
    return records

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    # ---- load pure-subtype alignment ------------------------------------
    st_to_seq_dict = defaultdict(list)
    fasta_path = (f"{workspace_path}/data/input/HIV1_PURE_REF.fasta")
    hxb2_ata_seq = ""
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            st_to_seq_dict[rec.id.split(".")[1]].append(str(rec.seq))

    pure_st_list   = list(st_to_seq_dict.keys())
    st_to_id       = {s: i for i, s in enumerate(pure_st_list)}
    n_subtypes     = len(st_to_id)
    n_packed       = int(np.ceil(n_subtypes / 8))
    ata_len        = len(hxb2_ata_seq)

    # ---- infer parameters from real CRFs --------------------------------
    df_seg = pd.read_csv(f"{workspace_path}/data/output/lanl_crf_segments.csv")
    print("Inferred parameters:")
    params = infer_params(df_seg, ata_len)

    # ---- rate arrays for mutation ----------------------------------------
    ata_to_hxb2 = build_hxb2_ata_maps(hxb2_ata_seq)
    # Retrieve rate arrays if they exist, otherwise build them
    rate_arrays = {}
    for st in ['A', 'B', 'C']:
        rate_array_path = Path(f"{workspace_path}/data/input/mutation_rates/empirical_mutation_rates_{st}.npy")
        if rate_array_path.exists():
            with open(rate_array_path, "rb") as f:
                rate_array = np.load(f)
        else:
            large_alignment_path = f"{workspace_path}/data/output/HIV1_{st}_ALIGNED.fasta"
            # This alignment corresponds to all complete genomes in LANL SEQ DB as of 23/04/2026, aligned to our pure alignment
            rate_array = compute_empirical_mutation_rates(large_alignment_path, ata_len, target_mean_rate=0.04)
            # Save the rate array to look at it after
            with open(rate_array_path, "wb") as f:
                np.save(f, rate_array)
        rate_arrays[st] = rate_array

    avg_rate_array_path = Path(f"{workspace_path}/data/input/mutation_rates/empirical_mutation_rates_avg.npy")
    if avg_rate_array_path.exists():
        with open(avg_rate_array_path, "rb") as f:
            avg_rate_array = np.load(f)
    else:
        avg_rate_array = np.mean([rate_arrays[st] for st in ['A', 'B', 'C']], axis=0)
        with open(avg_rate_array_path, "wb") as f:
            np.save(f, avg_rate_array)

    rate_arrays['avg'] = avg_rate_array

    print(f"Max mutation rate at any site: {avg_rate_array.max():.4f}")
    print(f"Min mutation rate at any site: {avg_rate_array.min():.4f}")
    print(f"Mean mutation rate across sites: {avg_rate_array.mean():.4f}")

    # ---- info -----------------------------------------------------------
    print(f"\nSubtypes       : {pure_st_list}")
    print(f"ATA length     : {ata_len}")
    print(f"Output shape   : seq ({N_SEQ},{ata_len})  "
          f"lbl ({N_SEQ},{ata_len},{n_packed})")
    print(f"Recombinant %  : {RP:.0%}")
    print(f"Window/thresh  : ±{WINDOW_HALF} pos, ≥{MIN_DIV} diffs")
    print(f"Workers        : {N_WORKERS}")
    print(f"Realistic      : {REALISTIC}")
    print(f"Force divergent: {FORCE_DIV}")

    # ---- allocate memmaps -----------------------------------------------
    seq_mm = np.lib.format.open_memmap(out_seqs, mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len))
    lbl_mm = np.lib.format.open_memmap(out_labels, mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len, n_packed))
    mask_mm = np.lib.format.open_memmap(out_masks, mode="w+", dtype=bool, shape=(N_SEQ, ata_len))
    del seq_mm, lbl_mm, mask_mm

    # ---- split work -----------------------------------------------------
    chunks = np.array_split(np.arange(N_SEQ), min(N_WORKERS, N_SEQ))
    seeds  = [int(s.generate_state(1)[0])
              for s in np.random.SeedSequence(SEED).spawn(len(chunks))]

    worker_args = [
        dict(worker_id=wid, row_start=int(c[0]), row_end=int(c[-1])+1,
             worker_seed=seeds[wid], out_seqs=out_seqs, out_labels=out_labels, out_masks=out_masks,
             st_to_seq_dict=dict(st_to_seq_dict), pure_st_list=pure_st_list,
             pure_st_to_id_dict=st_to_id, params=params,
             rate_arrays=rate_arrays, ata_len=ata_len, n_packed=n_packed,
             window_half=WINDOW_HALF, min_div=MIN_DIV, max_retries=MAX_RETRIES,
             force_divergent=FORCE_DIV, realistic=REALISTIC)
        for wid, c in enumerate(chunks) if len(c)
    ]

    print(f"\nLaunching {len(worker_args)} workers …")
    with mp.Pool(len(worker_args)) as pool:
        results = list(tqdm(pool.imap(_worker, worker_args),
                            total=len(worker_args), desc="Chunks"))

    records = [(n, nbp) for chunk in results for n, nbp in chunk]
    names   = [r[0] for r in records]
    n_bps   = [r[1] for r in records]

    # ---- stats ----------------------------------------------------------
    n_rec = sum(n.startswith("recombinant") for n in names)
    print(f"\nRecombinant proportion: {n_rec/len(names):.2%}")

    st_counts = defaultdict(int)
    for name in names:
        raw = name.split("_", 1)[1]
        for st in raw.split("+"):
            st_counts[st] += 1
    print("\nSubtype appearances:")
    for st in pure_st_list:
        print(f"  {st:6s}: {st_counts[st]:>6d}  ({st_counts[st]/len(names):6.2%})")

    # ---- metadata -------------------------------------------------------
    splits = np.random.default_rng(SEED).choice(
        ["train","val","test"], size=len(names), p=[0.9,0.05,0.05])
    with open(out_meta, "w") as f:
        f.write("sequence_id\tsequence_name\tpure_or_recombinant\tsubtypes\tn_subtypes\tn_breakpoints\tsplit\n")
        for i, name in enumerate(names):
            kind = "pure" if name.startswith("pure") else "recombinant"
            # Strip the "pure_" or "recombinant_" prefix
            raw = name.split("_", 1)[1]          # "A2+01_AE" or "B"
            st_list = raw.split("+")             # ["A2", "01_AE"] or ["B"]
            f.write(f"{i+1}\t{name}\t{kind}\t{'/'.join(st_list)}\t{len(st_list)}\t{n_bps[i]}\t{splits[i]}\n")

    print(f"\nFiles: {out_seqs}  {out_labels}  {out_meta}  {out_masks}")

    # ---- distribution comparison ----------------------------------------
    compare_generated_vs_real(
        names, out_labels, n_subtypes, n_packed, ata_len,
        pure_st_list, df_seg, ata_to_hxb2,
    )

    # ---- sanity check ---------------------------------------------------
    seq_mm = np.load(out_seqs,   mmap_mode="r")
    lbl_mm = np.load(out_labels, mmap_mode="r")
    for idx, name in enumerate(names):
        if name.startswith("recombinant"):
            lbl = np.unpackbits(lbl_mm[idx], axis=-1)[:, :n_subtypes]
            any_active = lbl.any(axis=1)   # True where at least one subtype is labeled
            present = [pure_st_list[j] for j in range(n_subtypes) if lbl[:, j].any()]

            print(f"\nExample #{idx+1}: {name}")
            print(f"  Sequence (first 2000 bases): {''.join(map(chr, seq_mm[idx][:2000]))} …")
            print(f"  Subtypes in labels: {', '.join(present)}")

            for st_name in present:
                col  = lbl[:, st_to_id[st_name]]
                runs = np.diff(np.concatenate([[0], col.astype(int), [0]]))
                raw_starts = np.where(runs ==  1)[0]
                raw_ends   = np.where(runs == -1)[0]

                if len(raw_starts) == 0:
                    continue

                # Merge runs whose gap contains only N positions (no subtype active)
                merged_starts = [raw_starts[0]]
                merged_ends   = []
                for i in range(len(raw_starts) - 1):
                    gap = any_active[raw_ends[i] : raw_starts[i + 1]]
                    if gap.any():
                        # Another subtype is active in the gap → real boundary
                        merged_ends.append(raw_ends[i])
                        merged_starts.append(raw_starts[i + 1])
                    # else: gap is all N's → skip, merging into current run
                merged_ends.append(raw_ends[-1])

                for s, e in zip(merged_starts, merged_ends):
                    print(f"    {st_name}: ATA [{s}, {e})  ({e-s} pos)")
            break