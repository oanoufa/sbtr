"""
HIV recombinant sequence generator
"""

import random
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from argparse import ArgumentParser
import os
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import uuid
import re

from src import config
from src.mutator_class import (
    SequenceMutator,
    compute_n_muts,
    mutate_sequence_gtr,
    _ACGT_BYTES,
)
from figs import (
    plot_fragment_length_distribution,
    plot_reference_distribution_with_year,
    visualize_generated_breakpoints,
    visualize_breakpoint_rank_profile,
    compare_generated_vs_real,
)

WORKSPACE_PATH = config.WORKSPACE_PATH

parser = ArgumentParser()
parser.add_argument("--n_workers",        type=int,   default=None)
parser.add_argument("--ref_alignment",    type=str,
                    default=f"{WORKSPACE_PATH}/data/input/HIV1_PURE_REF.fasta",
                    help="Reference alignment FASTA from which sequences are drawn")
parser.add_argument("--realistic",        action="store_true",
                    help="Whether to generate more realistic recombinants or not")
parser.add_argument("--force_divergent",  action="store_true",
                    help="Force breakpoints to occur in divergent regions")
args = parser.parse_args()

REF_ALIGNMENT = args.ref_alignment
REALISTIC     = args.realistic
FORCE_DIV     = args.force_divergent
N_WORKERS     = args.n_workers or mp.cpu_count()

N_SEQ         = config.N_SEQ
RP            = config.RP
SEED          = config.MODEL_CONFIG['seed']
DIV_WINDOW_SIZE   = config.DIV_WINDOW_SIZE
MIN_DIV       = config.MIN_DIV
MAX_RETRIES   = config.MAX_RETRIES
TEST_SET_SIZE = config.TEST_SET_SIZE
VERSION       = config.VERSION
GENES_RAW     = config.GENES_RAW

MIN_SEG_LEN = config.MIN_SEG_LEN
MAX_SUBTYPES = config.MAX_SUBTYPES
MAX_BREAKPOINTS = config.MAX_BREAKPOINTS
PARTIAL_FRAC  = config.PARTIAL_FRAC
MIN_FRAG_LEN  = config.MIN_FRAG_LEN
BP_PRIOR_LAMBDA = config.BP_PRIOR_LAMBDA
ST_TO_ID_DICT = config.ST_TO_ID_DICT
MAX_YEAR = config.MAX_YEAR

out_dir    = f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}_{PARTIAL_FRAC}"
out_seqs   = f"{out_dir}/sequences_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy"
out_labels = f"{out_dir}/labels_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy"
out_masks  = f"{out_dir}/loss_masks_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy"
out_meta   = f"{out_dir}/metadata_{N_SEQ}_{RP}_{PARTIAL_FRAC}.tsv"
os.makedirs(out_dir, exist_ok=True)

SUBTYPE_GROUPS = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0, "A6": 0, "A7": 0, "A8": 0,
    "F1": 1, "F2": 1,
    "E": 2, "C": 2, "G": 2, "H": 2, "J": 2, "K": 2, "L": 2,
    "N": 3, "O": 3, "P": 3,
    "B": 4, "D": 4,
}

# only recombine within M group
DIVERGENCE_WEIGHTS = [
    [0.00001, 1.00, 1.00, 0.000001, 1.00],
    [1.0000, 0.5, 1.0000, 0.0001, 1.0000],
    [1.00, 1.000, 1.000, 0.000001, 1.000],
    [0.0001, 0.0001, 0.0001, 1.0, 0.0001],
    [1.0000, 1.000, 1.00, 0.0001, 0.0001],
]

random.seed(SEED)

class Sequence:
    def __init__(self,
                 seq: str,
                 subtype: str,
                 country_code: str | None = None,
                 year: int | None = None):
        self.arr = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
        self.subtype = subtype
        self.country_code = country_code
        self.year = year

    def __len__(self):
        return len(self.arr)

def is_uncertain(label: str) -> bool:
    _MIX = re.compile(r"[/+,]")
    return label == "U" or bool(_MIX.search(label))

def load_mosaic_definitions(csv_path: str):
    # loads segments from csv, coordinates converted from 1 based inclusive to 0 based half open
    df = pd.read_csv(csv_path)
    mosaic_dict = defaultdict(list)
    for _, row in df.iterrows():
        crf_name = str(row["crf"]).strip()
        st_label = str(row["subtype"]).strip()
        start_0 = int(row["start"]) - 1
        end_0 = int(row["end"])

        # normalize segment label inside the sequence
        # 01_AE has E segments and A1 segments
        # A5U has A5 segments and U segments
        if st_label in ["AE", "01_AE"]:
            st_label = "E"
        elif st_label == "A5U":
            st_label = "A5"

        seg_entry = {
            "start": start_0,
            "end": end_0,
            "label": st_label
        }

        # map whatever crf name is in the csv to our parent names E and A5
        aliases = [crf_name]
        if crf_name in ["01_AE", "CRF01_AE", "AE", "E"]:
            aliases.extend(["E", "AE", "01_AE", "CRF01_AE"])
        if crf_name in ["A5U", "05_A5U", "A5"]:
            aliases.extend(["A5", "A5U", "05_A5U"])

        for a in set(aliases):
            mosaic_dict[a].append(seg_entry)

    for k in mosaic_dict:
        mosaic_dict[k].sort(key=lambda x: x["start"])

    return dict(mosaic_dict)


def label_parent_track(lbl_row, mask_row, s, e, st, st_id, mosaic_dict, rng=None, mask_bp_buf=30):
    # labels pure parents or mosaic parents like E and A5
    if st in mosaic_dict:
        for seg in mosaic_dict[st]:
            seg_s = max(s, seg["start"])
            seg_e = min(e, seg["end"])
            if seg_s >= seg_e:
                continue

            sub_st = seg["label"]

            # if segment is U or mixed like A1/G zero it out and mask it
            if is_uncertain(sub_st) or sub_st not in st_id:
                mask_row[seg_s:seg_e] = False
                lbl_row[seg_s:seg_e] = False
            else:
                lbl_row[seg_s:seg_e, st_id[sub_st]] = True

            # jitter loss mask across internal segment boundaries
            if seg["start"] > s:
                jit = int(rng.integers(mask_bp_buf // 3, mask_bp_buf * 2)) if rng else mask_bp_buf
                b_start = max(s, seg["start"] - jit)
                b_end = min(e, seg["start"] + jit)
                mask_row[b_start:b_end] = False

            if seg["end"] < e:
                jit = int(rng.integers(mask_bp_buf // 3, mask_bp_buf * 2)) if rng else mask_bp_buf
                b_start = max(s, seg["end"] - jit)
                b_end = min(e, seg["end"] + jit)
                mask_row[b_start:b_end] = False
    else:
        # regular pure subtype sequence
        if s < e and st in st_id:
            lbl_row[s:e, st_id[st]] = True

def labels_to_parts(lbl_row, id_to_st):
    pos = np.flatnonzero(lbl_row.any(axis=1))
    if len(pos) == 0:
        return ["unknown"], 0
    ids = lbl_row[pos].argmax(axis=1)
    cut = np.flatnonzero(np.diff(ids)) + 1
    sts = list(dict.fromkeys(id_to_st[int(i)] for i in ids[np.r_[0, cut]]))
    return sts, len(cut)

def extract_generated_breakpoints(names, out_labels, n_subtypes, ata_len, ata_to_hxb2):
    lbl_mm = np.load(out_labels, mmap_mode="r")
    rows = []
    for idx, n in enumerate(tqdm(names, desc="Extracting breakpoints", mininterval=10)):
        if not n.startswith("r"):
            continue
        lbl = np.unpackbits(lbl_mm[idx], axis=-1)[:, :n_subtypes]
        any_active = lbl.any(axis=1)
        active = np.full(ata_len, -1, dtype=np.int32)
        active[any_active] = lbl[any_active].argmax(axis=1)
        lp = np.where(active >= 0)[0]
        if len(lp) < 2:
            continue
        bp_ata = lp[np.where(np.diff(active[lp]) != 0)[0] + 1]
        for k, b in enumerate(bp_ata):
            rows.append((idx, k + 1, len(bp_ata), int(b), int(ata_to_hxb2[b])))
    return pd.DataFrame(rows, columns=["seq", "rank", "n_bp", "pos_ata", "pos"])

def segments_to_breakpoints(df_seg):
    rows = []
    for crf, g in df_seg.sort_values(["crf", "start"]).groupby("crf"):
        seg = g[~g["subtype"].map(is_uncertain)].reset_index(drop=True)
        for a, b in zip(seg.iloc[:-1].itertuples(), seg.iloc[1:].itertuples()):
            if a.subtype == b.subtype:
                continue
            lo, hi = a.end, b.start
            rows.append((crf, a.subtype, b.subtype, lo, hi, (lo + hi) / 2))
    return pd.DataFrame(rows, columns=["crf", "left", "right", "lo", "hi", "pos"])

def infer_params(
    df_seg: pd.DataFrame, ata_len: int,
    hxb2_len: int = config.HXB2_USED_LEN) -> dict:
    params = {}
    is_unc  = df_seg["subtype"].map(is_uncertain)
    df_pure = df_seg[~is_unc]
    print("Removed uncertain segments:", is_unc.sum(), "of", len(df_seg), "total segments")

    n_pure_per_crf = df_pure.groupby("crf").size()
    n_bp = (n_pure_per_crf - 1).clip(lower=0)
    n_bp_counts = n_bp.value_counts().sort_index()
    max_k = int(n_bp_counts.index.max())
    full_range = np.arange(1, max_k + 1)
    counts = n_bp_counts.reindex(full_range, fill_value=0).values.astype(float)

    def smooth_within_parity(counts, window=3):
        smoothed = counts.copy()
        for parity in (0, 1):
            idx = np.where((np.arange(len(counts)) % 2) == parity)[0]
            vals = counts[idx]
            kernel = np.ones(window) / window
            smoothed[idx] = np.convolve(vals, kernel, mode="same")
        return smoothed

    smoothed = smooth_within_parity(counts, window=3) + 0.5
    probs = smoothed / smoothed.sum()

    params["n_breakpoints"] = {"values": full_range.tolist(), "probs": probs.tolist()}

    pmf = df_pure.groupby("crf")["subtype"].nunique().value_counts(normalize=True).sort_index()
    params["n_subtypes"] = {"values": pmf.index.tolist(), "probs": pmf.values.tolist()}

    pure_len = df_pure["length"].dropna().astype(float)
    min_hxb2 = int(pure_len[pure_len > 0].min())
    params["min_seg_len"] = max(1, int(min_hxb2 * ata_len / hxb2_len))

    print(f"  n_bp/CRF  {params['n_breakpoints']}")
    print(f"  subtypes  {dict(zip(pmf.index, (pmf.values*100).round(1)))}")
    print(f"  min_seg   {min_hxb2} HXB2 to {params['min_seg_len']} ATA")
    return params

def weighted_subtype_probs(counts: dict, floor: float = 0.015) -> dict:
    w = {k: math.sqrt(v) for k, v in counts.items()}
    fixed, free = {}, dict(w)
    while True:
        remaining_mass = 1.0 - sum(fixed.values())
        wsum = sum(free.values())
        probs = {k: remaining_mass * (v / wsum) for k, v in free.items()}
        below = {k: p for k, p in probs.items() if p < floor}
        if not below:
            probs.update(fixed)
            return probs
        for k in below:
            fixed[k] = floor
            del free[k]

def sample_subtypes(n, pool, py_rng, st_probs=None):
    chosen, remain = [], list(pool)
    for _ in range(n):
        if not chosen:
            if st_probs:
                weights = [st_probs[s] for s in remain]
                st = py_rng.choices(remain, weights=weights)[0]
            else:
                st = py_rng.choice(remain)
        else:
            weights = []
            for st_cand in remain:
                g = SUBTYPE_GROUPS.get(st_cand, 2)
                w = min(DIVERGENCE_WEIGHTS[g][SUBTYPE_GROUPS[a]] for a in chosen)
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

def divergence_profile(seq1, seq2, DIV_WINDOW_SIZE):
    v1 = np.isin(seq1, _ACGT_BYTES)
    v2 = np.isin(seq2, _ACGT_BYTES)
    diff = ((seq1 != seq2) & v1 & v2).astype(np.float32)
    window_half = DIV_WINDOW_SIZE // 2
    kernel = np.ones(2 * window_half + 1, dtype=np.float32)
    return np.convolve(diff, kernel, mode="same")

def assign_subtypes_to_segments(n_seg, subtype_pool, py_rng):
    pool = list(subtype_pool)
    py_rng.shuffle(pool)
    result = list(pool)
    while len(result) < n_seg:
        cands = [s for s in pool if s != result[-1]]
        result.append(py_rng.choice(cands))
    return result[:n_seg]

def build_bp_prior(df_bp, hxb2_to_ata, ata_len, lam=0.5, bw=100):
    dens = np.zeros(ata_len)
    for lo, hi in zip(df_bp["lo"], df_bp["hi"]):
        a, b = sorted((int(hxb2_to_ata[int(lo)]), int(hxb2_to_ata[int(hi)])))
        a, b = max(a, 0), min(b, ata_len - 1)
        dens[a:b + 1] += 1.0 / (b - a + 1)
    k = np.exp(-0.5 * (np.arange(-4 * bw, 4 * bw + 1) / bw) ** 2)
    dens = np.convolve(dens, k / k.sum(), mode="same")
    dens /= dens.sum()
    return lam * dens + (1 - lam) / ata_len

def place_breakpoints(seg_subtypes, parents, ata_len,
                      DIV_WINDOW_SIZE, min_div, min_seg_len, rng,
                      force_div=False, prior=None, alpha=0.0):
    n_bp = len(seg_subtypes) - 1
    parents_seq = {st: seq.arr for st, seq in parents.items()}
    min_seg_len = max(min_seg_len, MIN_SEG_LEN)

    if force_div:
        L = ata_len
        _cache, w = {}, []
        for i in range(n_bp):
            key = (id(parents_seq[seg_subtypes[i]]), id(parents_seq[seg_subtypes[i + 1]]))
            if key not in _cache:
                _cache[key] = divergence_profile(
                    parents_seq[seg_subtypes[i]], parents_seq[seg_subtypes[i + 1]], DIV_WINDOW_SIZE)
            prof = _cache[key].astype(np.float64)
            wi = np.where(prof >= min_div, prof ** alpha, 0.0)
            if prior is not None:
                wi *= prior
            wi[: min_seg_len * (i + 1)] = 0.0
            wi[L - min_seg_len * (n_bp - i) + 1 :] = 0.0
            w.append(wi)

        F, prev_cum = [], None
        for i in range(n_bp):
            f = w[i].copy()
            if i > 0:
                shifted = np.zeros(L)
                shifted[min_seg_len:] = prev_cum[: L - min_seg_len]
                f *= shifted
            s = f.sum()
            if s == 0:
                return None
            f /= s
            F.append(f)
            prev_cum = np.cumsum(f)

        bps = [0] * n_bp
        bps[-1] = int(rng.choice(L, p=F[-1]))
        for i in range(n_bp - 2, -1, -1):
            p = F[i].copy()
            p[bps[i + 1] - min_seg_len + 1 :] = 0.0
            s = p.sum()
            if s == 0:
                return None
            bps[i] = int(rng.choice(L, p=p / s))
        return bps

    else:
        L_free = ata_len - (n_bp + 1) * min_seg_len
        if L_free < 0:
            return None

        raw_bps = np.sort(rng.choice(L_free + 1, size=n_bp, replace=True))
        breakpoints = []
        current_shift = min_seg_len
        for rb in raw_bps:
            bp = rb + current_shift
            breakpoints.append(int(bp))
            current_shift += min_seg_len

        return breakpoints

def sample_fragment_window(ata_len: int, min_frag_len: int, rng) -> tuple[int, int]:
    start = int(rng.integers(0, ata_len - min_frag_len + 1))
    max_len = ata_len - start
    log_len = rng.uniform(np.log(min_frag_len), np.log(max_len))
    length = int(np.clip(np.round(np.exp(log_len)), min_frag_len, max_len))
    return start, start + length

def _worker(cfg):
    row_start  = cfg["row_start"]
    row_end    = cfg["row_end"]
    chunk_sz   = row_end - row_start
    params     = cfg["params"]
    nb         = params["n_breakpoints"]
    ns         = params["n_subtypes"]
    min_seg    = params["min_seg_len"]
    bp_prior   = params.get("bp_prior")
    st_id      = cfg["pure_st_to_id_dict"]
    id_to_st   = {v: k for k, v in st_id.items()}
    n_st_total = len(st_id)
    ata_len    = cfg["ata_len"]
    site_rates_dict = cfg["site_rates_dict"]
    sub_probs_dict  = cfg["sub_probs_dict"]
    st_list    = cfg["pure_st_list"]
    div_window_size = cfg["div_window_size"]
    mdiv       = cfg["min_div"]
    max_ret    = cfg["max_retries"]
    force_div  = cfg["force_divergent"]
    realistic  = cfg["realistic"]
    st_to_seq_dict = cfg["st_to_seq_dict"]
    partial_frac   = cfg["partial_frac"]
    min_frag_len   = cfg["min_frag_len"]
    mosaic_dict    = cfg["mosaic_dict"]

    seq_mm  = np.load(cfg["out_seqs"],   mmap_mode="r+")
    lbl_mm  = np.load(cfg["out_labels"], mmap_mode="r+")
    mask_mm = np.load(cfg["out_masks"],  mmap_mode="r+")

    rng    = np.random.default_rng(cfg["worker_seed"])
    py_rng = random.Random(cfg["worker_seed"])
    records  = []
    st_probs_dict = weighted_subtype_probs({s: len(v) for s, v in st_to_seq_dict.items()})
    st_weights = [st_probs_dict[s] for s in st_list]

    for li in range(chunk_sz):
        is_rec = py_rng.random() <= RP
        is_pure = not is_rec
        ok = False
        total_muts = 0

        if is_pure:
            n_bp = 0
            st  = py_rng.choices(st_list, weights=st_weights)[0]
            seq_obj = py_rng.choice(st_to_seq_dict[st]) 
            target_year = py_rng.randint(seq_obj.year, MAX_YEAR)
            rate_arr  = site_rates_dict.get(st[0], site_rates_dict['avg'])
            sub_probs = sub_probs_dict.get(st[0],  sub_probs_dict['avg'])
            n_muts    = compute_n_muts(seq_obj.year, target_year, rate_arr, 0, len(seq_obj.arr), subtype=seq_obj.subtype)
            seq_row, total_muts = mutate_sequence_gtr(
                seq_obj.arr.copy(), 0, rate_arr, rng, n_muts, sub_probs)
            lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
            mask_row = np.ones(ata_len, dtype=bool)

            label_parent_track(lbl_row, mask_row, 0, ata_len, st, st_id, mosaic_dict, rng)
            name = f"p_{st}_{target_year}_{uuid.uuid4().hex[:8]}"
            ok = True
        else:
            for _attempt in range(max_ret):
                if realistic:
                    n_sub = max(2, int(rng.choice(ns["values"], p=ns["probs"])))
                    n_sub = min(n_sub, len(st_list))
                    n_bp = int(rng.choice(nb["values"], p=nb["probs"]))
                    n_seg = n_bp + 1
                    n_sub = min(n_sub, n_seg)
                else:
                    n_sub = rng.integers(2, MAX_SUBTYPES + 1)
                    n_bp  = rng.integers(1, MAX_BREAKPOINTS + 1)
                    n_seg = n_bp + 1
                    n_sub = min(n_sub, n_seg)

                bps = None
                for _placement_try in range(max_ret):
                    chosen_subtypes = sample_subtypes(n_sub, st_list, py_rng, st_probs_dict)
                    seg_st = assign_subtypes_to_segments(n_seg, chosen_subtypes, py_rng)
                    parents = {st: py_rng.choice(st_to_seq_dict[st]) for st in chosen_subtypes}
                    max_parent_year = max(p.year for p in parents.values())
                    target_year = py_rng.randint(max_parent_year, MAX_YEAR)
                    bps = place_breakpoints(seg_st, parents, ata_len, div_window_size,
                                            mdiv, min_seg, rng, force_div, bp_prior)
                    if bps is not None:
                        break
                if bps is None:
                    continue

                bounds  = [0] + bps + [ata_len]
                seq_row = np.empty(ata_len, dtype=np.uint8)
                lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
                mask_row = np.ones(ata_len, dtype=bool)

                # paste segments together and label
                for si in range(n_seg):
                    s, e = bounds[si], bounds[si + 1]
                    if e <= s: 
                        continue
                    chunk = parents[seg_st[si]].arr[s:e].copy()
                    rate_arr = site_rates_dict.get(seg_st[si][0], site_rates_dict['avg'])
                    sub_probs = sub_probs_dict.get(seg_st[si][0], sub_probs_dict['avg'])
                    chunk_len = e - s
                    n_muts = compute_n_muts(
                        parents[seg_st[si]].year, target_year, rate_arr,
                        s, chunk_len, subtype=parents[seg_st[si]].subtype
                    )
                    seq_row[s:e], muts = mutate_sequence_gtr(chunk, s, rate_arr, rng, n_muts, sub_probs)
                    total_muts += muts
                    label_parent_track(lbl_row, mask_row, s, e, seg_st[si], st_id, mosaic_dict, rng)

                # mask identical stretches around artificial recombination cuts
                for si in range(n_seg - 1):
                    p1 = parents[seg_st[si]].arr
                    p2 = parents[seg_st[si+1]].arr
                    bp = bps[si]
                    
                    l = bp - 1
                    while l >= 0 and p1[l] == p2[l]: 
                        l -= 1
                    r = bp
                    while r < ata_len and p1[r] == p2[r]: 
                        r += 1
                    
                    jit = int(rng.integers(5, 15))
                    l_bound = max(0, l + 1 - jit)
                    r_bound = min(ata_len, r + jit)
                    mask_row[l_bound:r_bound] = False

                name = f"r_{'+'.join(dict.fromkeys(seg_st))}_{target_year}_{uuid.uuid4().hex[:8]}"
                n_bp = len(bps)
                ok = True
                break

            if not ok:
                n_bp = 0
                st  = py_rng.choice(st_list)
                seq_obj = py_rng.choice(st_to_seq_dict[st])
                rate_arr = site_rates_dict.get(st[0], site_rates_dict['avg'])
                sub_probs = sub_probs_dict.get(st[0], sub_probs_dict['avg'])
                target_year = py_rng.randint(seq_obj.year, MAX_YEAR)
                n_muts = compute_n_muts(
                    seq_obj.year, target_year, rate_arr,
                    0, len(seq_obj.arr), subtype=seq_obj.subtype
                )
                seq_row, total_muts = mutate_sequence_gtr(seq_obj.arr.copy(), 0, rate_arr, rng, n_muts, sub_probs)
                lbl_row = np.zeros((ata_len, n_st_total), dtype=bool)
                mask_row = np.ones(ata_len, dtype=bool)
                label_parent_track(lbl_row, mask_row, 0, ata_len, st, st_id, mosaic_dict, rng)
                name = f"p_{st}_{target_year}_{uuid.uuid4().hex[:8]}"

        # partial sequence cropping
        frag_start, frag_end = 0, ata_len
        if partial_frac > 0.0 and py_rng.random() < partial_frac:
            for _attempt in range(max_ret):
                frag_start, frag_end = sample_fragment_window(ata_len, min_frag_len, rng)
                window_slice = seq_row[frag_start:frag_end]
                
                has_data = np.any(
                    (window_slice != ord('N')) & 
                    (window_slice != ord('-'))
                )
                if has_data:
                    break
            else:
                frag_start, frag_end = 0, ata_len

            seq_row[:frag_start]  = ord("-")
            seq_row[frag_end:]    = ord("-")
            lbl_row[:frag_start]  = False
            lbl_row[frag_end:]    = False
            mask_row[:frag_start] = False
            mask_row[frag_end:]   = False

        sts, n_bp_lbl = labels_to_parts(lbl_row, id_to_st)
        uid = name.rsplit("_", 1)[1]
        if name.startswith("r"):
            if len(sts) > 1:
                name, n_bp = f"r_{'+'.join(sts)}_{target_year}_{uid}", n_bp_lbl
            else:
                name, n_bp = f"p_{sts[0]}_{target_year}_{uid}", 0
        elif len(sts) == 1 and name.split("_")[1] != sts[0]:
            name = f"p_{sts[0]}_{target_year}_{uid}"

        seq_row[seq_row == ord("-")] = ord("N")
        seq_mm[row_start + li]  = seq_row
        lbl_mm[row_start + li]  = np.packbits(lbl_row, axis=-1)
        mask_mm[row_start + li] = mask_row
        records.append((name, n_bp, total_muts, target_year, frag_start, frag_end))

    seq_mm.flush()
    lbl_mm.flush()
    mask_mm.flush()
    return records

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    st_to_seq_dict: dict[str, list[Sequence]] = defaultdict(list)
    hxb2_ata_seq = ""

    for i, rec in enumerate(SeqIO.parse(REF_ALIGNMENT, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            parts = rec.id.split(".")
            subtype = parts[1]
            
            # map headers to E and A5
            if subtype in ["AE", "01_AE", "CRF01_AE"]:
                subtype = "E"
            elif subtype in ["A5U", "05_A5U"]:
                subtype = "A5"

            country_code = parts[2]
            two_digit_year = int(parts[3])
            year = 2000 + two_digit_year if two_digit_year < 40 else 1900 + two_digit_year
            st_to_seq_dict[subtype].append(Sequence(str(rec.seq),
                                                    country_code=country_code,
                                                    subtype=subtype,
                                                    year=year))

    pure_st_list = list(ST_TO_ID_DICT.keys())
    st_to_seq_dict_sorted = {st: st_to_seq_dict[st] for st in pure_st_list}
    st_to_seq_dict = st_to_seq_dict_sorted
    del st_to_seq_dict_sorted
    n_subtypes     = len(ST_TO_ID_DICT)
    n_packed       = int(np.ceil(n_subtypes / 8))
    ata_len        = len(hxb2_ata_seq)

    df_seg = pd.read_csv(f"{WORKSPACE_PATH}/data/output/lanl_crf_segments_hxb2.csv")
    lo = config.HXB2_START
    hi = lo + config.HXB2_USED_LEN - 1
    df_seg = df_seg[(df_seg["end"] >= lo) & (df_seg["start"] <= hi)].copy()
    df_seg["start"]  = df_seg["start"].clip(lower=lo)
    df_seg["end"]    = df_seg["end"].clip(upper=hi)
    df_seg["length"] = df_seg["end"] - df_seg["start"] + 1
    print("Inferred parameters:")
    params = infer_params(df_seg, ata_len)

    ata_to_hxb2, hxb2_to_ata = config.build_hxb2_ata_maps(hxb2_ata_seq)

    df_bp = segments_to_breakpoints(df_seg)
    params["bp_prior"] = build_bp_prior(df_bp, hxb2_to_ata, ata_len, lam=BP_PRIOR_LAMBDA)

    mosaic_dict = load_mosaic_definitions(config.CRF_SEGMENTS_BD_ATA)
    print(f"Loaded mosaic definitions for: {list(mosaic_dict.keys())}")

    mutator = SequenceMutator(
        iqtree_dir=f"{WORKSPACE_PATH}/data/output/rates/",
        ata_len=ata_len,
        seed=SEED,
    )
    site_rates_dict = mutator.site_rates_dict
    sub_probs_dict  = mutator.sub_probs_dict

    avg_rates = site_rates_dict['avg']
    print(f"Max site rate   (avg): {avg_rates.max():.6f}")
    print(f"Min site rate   (avg): {avg_rates.min():.6f}")
    print(f"Mean site rate  (avg): {avg_rates.mean():.6f}")

    print(f"\nSubtypes       : {pure_st_list}")
    print(f"ATA length     : {ata_len}")
    print(f"Output shape   : seq ({N_SEQ},{ata_len})  lbl ({N_SEQ},{ata_len},{n_packed})")
    print(f"Recombinant %  : {RP:.0%}")
    print(f"Window/thresh  : +-{DIV_WINDOW_SIZE} pos, >= {MIN_DIV} diffs")
    print(f"Workers        : {N_WORKERS}")
    print(f"Realistic      : {REALISTIC}")
    print(f"Force divergent: {FORCE_DIV}")
    print(f"Partial frac   : {PARTIAL_FRAC:.1%}  (min fragment {MIN_FRAG_LEN} ATA pos)")

    seq_mm  = np.lib.format.open_memmap(out_seqs,   mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len))
    lbl_mm  = np.lib.format.open_memmap(out_labels, mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len, n_packed))
    mask_mm = np.lib.format.open_memmap(out_masks,  mode="w+", dtype=bool,     shape=(N_SEQ, ata_len))
    del seq_mm, lbl_mm, mask_mm

    chunks = np.array_split(np.arange(N_SEQ), min(N_WORKERS, N_SEQ))
    seeds  = [int(s.generate_state(1)[0])
              for s in np.random.SeedSequence(SEED).spawn(len(chunks))]

    worker_args = [
        dict(worker_id=wid, row_start=int(c[0]), row_end=int(c[-1])+1,
             worker_seed=seeds[wid], out_seqs=out_seqs, out_labels=out_labels, out_masks=out_masks,
             st_to_seq_dict=dict(st_to_seq_dict), pure_st_list=pure_st_list,
             pure_st_to_id_dict=ST_TO_ID_DICT, hxb2_to_ata=hxb2_to_ata, params=params,
             site_rates_dict=site_rates_dict,
             sub_probs_dict=sub_probs_dict,
             ata_len=ata_len, mosaic_dict=mosaic_dict,
             div_window_size=DIV_WINDOW_SIZE, min_div=MIN_DIV, max_retries=MAX_RETRIES,
             force_divergent=FORCE_DIV, realistic=REALISTIC,
             partial_frac=PARTIAL_FRAC, min_frag_len=MIN_FRAG_LEN)
        for wid, c in enumerate(chunks) if len(c)
    ]

    print(f"\nLaunching {len(worker_args)} workers ...")
    with mp.Pool(len(worker_args)) as pool:
        results = list(tqdm(pool.imap(_worker, worker_args),
                            total=len(worker_args), desc="Chunks"))

    records = [
        (n, nbp, nmuts, ty, fs, fe)
        for chunk in results
        for n, nbp, nmuts, ty, fs, fe in chunk
    ]
    names        = [r[0] for r in records]
    n_bps        = [r[1] for r in records]
    n_muts       = [r[2] for r in records]
    target_years = [r[3] for r in records]
    frag_starts  = [r[4] for r in records]
    frag_ends    = [r[5] for r in records]

    n_rec     = sum(n.startswith("r") for n in names)
    n_partial = sum(fe - fs < ata_len for fs, fe in zip(frag_starts, frag_ends))
    print(f"\nRecombinant proportion : {n_rec/len(names):.2%}")
    print(f"Partial (cropped)      : {n_partial}/{len(names)}  ({n_partial/len(names):.2%})")

    st_counts = defaultdict(int)
    gen_subtype_data = defaultdict(list)
    for name, yr in zip(names, target_years):
        raw = name.split("_")[1]
        for st in raw.split("+"):
            st_counts[st] += 1
            gen_subtype_data[st].append(yr)
    print("\nSubtype appearances:")
    for st in pure_st_list:
        print(f"  {st:6s}: {st_counts[st]:>6d}  ({st_counts[st]/len(names):6.2%})")

    splits = np.random.default_rng(SEED).choice(
        ["train","val","test"], size=len(names), p=[0.9,0.05,0.05])
    with open(out_meta, "w") as f:
        f.write(
            "sequence_id\tsequence_name\tpure_or_recombinant\tsubtypes\t"
            "n_subtypes\tn_breakpoints\tn_mutations\ttarget_year\t"
            "frag_start\tfrag_end\tsplit\n"
        )
        for i, name in enumerate(names):
            kind    = "pure" if name.startswith("p") else "recombinant"
            raw     = name.split("_")[1]
            st_parts = raw.split("+")
            f.write(
                f"{i+1}\t{name}\t{kind}\t{'/'.join(st_parts)}\t{len(st_parts)}\t"
                f"{n_bps[i]}\t{n_muts[i]}\t{target_years[i]}\t"
                f"{frag_starts[i]}\t{frag_ends[i]}\t{splits[i]}\n"
            )

    print(f"\nFiles: {out_seqs}  {out_labels}  {out_meta}  {out_masks}")

    test_indices = [i for i, s in enumerate(splits) if s == "test"]
    test_seqs = np.load(out_seqs, mmap_mode="r")[test_indices]
    test_labels = np.load(out_labels, mmap_mode="r")[test_indices]
    test_dir = Path(WORKSPACE_PATH) / "data" / "output" / "test" / f"syn_v{VERSION}"
    os.makedirs(test_dir, exist_ok=True)
    out_test_fasta = f"{test_dir}/test_sequences_{N_SEQ}_{RP}_{PARTIAL_FRAC}.fasta"
    out_test_labels = f"{test_dir}/test_labels_{N_SEQ}_{RP}_{PARTIAL_FRAC}.csv"
    ID_TO_ST_DICT = {v: k for k, v in ST_TO_ID_DICT.items()}

    with open(out_test_fasta, "w") as f:
        with open(out_test_labels, "w") as lf:
            lf.write("sample_name,start,end,subtype,length\n")
            for i, seq in enumerate(test_seqs):
                if i == TEST_SET_SIZE:
                    break
                fs, fe = frag_starts[test_indices[i]], frag_ends[test_indices[i]]
                seq_frag = seq[fs:fe]
                labels_frag = np.unpackbits(test_labels[i][fs:fe], axis=-1)[:, :n_subtypes]

                keep = ~np.isin(seq_frag, [ord("-"), ord("N")])
                seq_clean = seq_frag[keep]
                labels_id = labels_frag[keep]

                label_names = []
                for j in range(len(labels_id)):
                    if labels_id[j].any():
                        label_names.append(ID_TO_ST_DICT[np.argmax(labels_id[j])])
                    else:
                        label_names.append("U")

                f.write(f">{names[test_indices[i]]}\n")
                f.write(f"{''.join(map(chr, seq_clean))}\n")

                if len(label_names) == 0:
                    continue

                start, current = 0, label_names[0]
                for j in range(1, len(label_names)):
                    if label_names[j] != current:
                        lf.write(f"{names[test_indices[i]]},{start+1},{j},{current},{j-start}\n")
                        start, current = j, label_names[j]
                lf.write(f"{names[test_indices[i]]},{start+1},{len(label_names)},{current},{len(label_names)-start}\n")

    print("\n  === Generated Mutations per Sequence ===")
    muts_s = pd.Series(n_muts)
    print(f"  Min   : {muts_s.min():.0f}")
    print(f"  Median: {muts_s.median():.0f}")
    print(f"  Mean  : {muts_s.mean():.1f}")
    print(f"  Max   : {muts_s.max():.0f}")
    print(f"  Std   : {muts_s.std():.1f}")

    if PARTIAL_FRAC > 0.0:
        frag_dist_path = f"{WORKSPACE_PATH}/figs/fragment_length_dist.html"
        plot_fragment_length_distribution(
            frag_starts, frag_ends, ata_len, MIN_FRAG_LEN, frag_dist_path,
        )

    gen_dist_path = f"{WORKSPACE_PATH}/figs/generated_subtype_distribution_with_year.html"
    plot_reference_distribution_with_year(gen_subtype_data, save_path=gen_dist_path)

    compare_generated_vs_real(
        names, out_labels, n_subtypes, ata_len, df_seg, ata_to_hxb2,
    )

    df_gen = extract_generated_breakpoints(names, out_labels, n_subtypes, ata_len, ata_to_hxb2)

    visualize_generated_breakpoints(df_gen, df_real=df_bp)
    n_mode = int(df_gen["n_bp"].mode()[0])
    visualize_breakpoint_rank_profile(df_gen, n_mode, ata_len)