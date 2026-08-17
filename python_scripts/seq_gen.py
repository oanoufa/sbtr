"""
HIV recombinant sequence generator.

Flow per recombinant:
  1. Sample n_subtypes, n_breakpoints from distributions inferred from real CRFs
  2. Pick subtypes (divergence-group-aware weighting)
  3. Pick one parent sequence per subtype
  4. Compute windowed divergence profile for each adjacent-subtype transition
  5. Place breakpoints at positions weighted by divergence (≥ threshold)
  6. Build chimeric sequence, mutate, optionally crop to a fragment, write
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

from src import config
from src.utils import build_hxb2_ata_maps
from src.mutator_class import (
    SequenceMutator,
    compute_n_muts,
    mutate_sequence_gtr,
    _ACGT_BYTES,
)

WORKSPACE_PATH = config.WORKSPACE_PATH
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--n_workers",        type=int,   default=None)
parser.add_argument("--ref_alignment",    type=str,
                    default=f"{WORKSPACE_PATH}/data/input/HIV1_PURE_REF.fasta",
                    help="Reference alignment (FASTA) from which sequences are drawn.")
parser.add_argument("--realistic",        action="store_true",
                    help="Whether to generate more realistic recombinants or not")
parser.add_argument("--force_divergent",  action="store_true",
                    help="If true, force breakpoints to occur in divergent regions. "
                         "If false, random placement.")
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

out_dir    = f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}"
out_seqs   = f"{out_dir}/sequences_{N_SEQ}_{RP}.npy"
out_labels = f"{out_dir}/labels_{N_SEQ}_{RP}.npy"
out_masks  = f"{out_dir}/loss_masks_{N_SEQ}_{RP}.npy"
out_meta   = f"{out_dir}/metadata_{N_SEQ}_{RP}.tsv"
os.makedirs(out_dir, exist_ok=True)

MIN_SEG_LEN = config.MIN_SEG_LEN
MAX_SUBTYPES = config.MAX_SUBTYPES
MAX_BREAKPOINTS = config.MAX_BREAKPOINTS
PARTIAL_FRAC  = config.PARTIAL_FRAC
MIN_FRAG_LEN  = config.MIN_FRAG_LEN

SUBTYPE_GROUPS = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0, "A6": 0, "A7": 0, "A8": 0, "AE": 0,
    "F1": 1, "F2": 1,
    "B": 4, "D": 4,
    "C": 2, "G": 2, "H": 2, "J": 2, "K": 2, "L": 2,
    "N": 3, "O": 3, "P": 3,
}

# Only recombine within M-group
DIVERGENCE_WEIGHTS = [
    [0.0001, 1.0000, 1.0000, 0.0001, 1.0000],
    [1.0000, 0.0001, 1.0000, 0.0001, 1.0000],
    [1.0000, 1.0000, 1.0000, 0.0001, 1.0000],
    [0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
    [1.0000, 0.0001, 1.0000, 0.0001, 0.0001],
]

ST_TO_ID_DICT = config.ST_TO_ID_DICT
MAX_YEAR = config.MAX_YEAR
GENES_HXB2 = config.GENES_RAW

# Initialize uuid with SEED
random.seed(SEED)

# Sequence Class used to store both the sequence and some metadata
class Sequence:
    def __init__(self,
                 seq: str,
                 subtype: str,
                 country_code: str | None = None,
                 year: int | None = None):
        self.arr = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
        self.subtype = subtype
        self.country_code = country_code  # None if unknown
        self.year = year  # None if unknown

    def __len__(self):
        return len(self.arr)


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

    # --- breakpoints / CRF  →  empirical, parity-aware ---
    n_pure_per_crf = df_pure.groupby("crf").size()
    n_bp = (n_pure_per_crf - 1).clip(lower=0)
    n_bp_counts = n_bp.value_counts().sort_index()  # observed k -> count
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

    smoothed = smooth_within_parity(counts, window=3) + 0.5  # +0.5 pseudo-count for zero bins
    probs = smoothed / smoothed.sum()

    params["n_breakpoints"] = {"values": full_range.tolist(), "probs": probs.tolist()}

    # --- distinct subtypes / CRF  →  empirical PMF ---
    pmf = df_pure.groupby("crf")["subtype"].nunique().value_counts(normalize=True).sort_index()
    params["n_subtypes"] = {"values": pmf.index.tolist(), "probs": pmf.values.tolist()}

    # --- minimum segment length  (scaled HXB2 → ATA) ---
    pure_len = df_pure["length"].dropna().astype(float)
    min_hxb2 = int(pure_len[pure_len > 0].min())
    params["min_seg_len"] = max(1, int(min_hxb2 * ata_len / hxb2_len))

    print(f"  n_bp/CRF  {params['n_breakpoints']}")
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

    rec_indices = [i for i, n in enumerate(names) if n.startswith("r")]

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

    def _plot_distribution_comparison(n_bp, n_st, seg_lens, path, title):

        color_scheme = ['#072C4B', '#F28089', '#71cddd']

        # -- n_breakpoints / sequence --
        n_bp = pd.Series(n_bp)
        n_st = pd.Series(n_st)
        seg_lens = pd.Series(seg_lens)
        bp_values = n_bp.value_counts().sort_index().index.tolist()
        bp_real_pct = n_bp.value_counts(normalize=True).sort_index().values.tolist()

        # -- n_subtypes / sequence --
        st_values = n_st.value_counts().sort_index().index.tolist()
        st_real_pct = n_st.value_counts(normalize=True).sort_index().values.tolist()

        # -- segment lengths --
        sl_mu = seg_lens.apply(lambda x: np.log(x)).mean()
        sl_sigma = seg_lens.apply(lambda x: np.log(x)).std()
        bin_width = 200

        # 1A. Data for the bars (binned percentages)
        bin_edges = np.arange(0, 9500 + bin_width, bin_width)
        bin_centers = bin_edges[:-1] + (bin_width / 2)
        sl_real_pct = []
        sl_hover_text = []

        def lognorm_cdf(x, mu, sigma):
            if x <= 0: return 0.0
            return 0.5 * (1 + math.erf((np.log(x) - mu) / (sigma * np.sqrt(2))))

        for i in range(len(bin_edges)-1):
            left_edge = bin_edges[i]
            right_edge = bin_edges[i+1]
            prob = lognorm_cdf(right_edge, sl_mu, sl_sigma) - lognorm_cdf(left_edge, sl_mu, sl_sigma)
            sl_real_pct.append(prob * 100)
            sl_hover_text.append(f"{int(left_edge)} - {int(right_edge)} bp<br>{prob*100:.1f}%")

        # 1B. Data for the continuous line (scaled PDF)
        sl_x_line = np.linspace(10, 9500, 500)
        sl_pdf = (1 / (sl_x_line * sl_sigma * np.sqrt(2 * np.pi))) * \
                np.exp(- (np.log(sl_x_line) - sl_mu)**2 / (2 * sl_sigma**2))
        sl_y_line = sl_pdf * bin_width * 100

        fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=(
                "Breakpoints per sequence", 
                "Subtypes per sequence", 
                "Segment lengths distribution"
            ),
            horizontal_spacing=0.08
        )

        fig.add_trace(
            go.Bar(x=bp_values, y=bp_real_pct, marker_color=color_scheme[0], hoverinfo='x+y', opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=st_values, y=st_real_pct, marker_color=color_scheme[1], hoverinfo='x+y', opacity=0.8),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=bin_centers, y=sl_real_pct, width=bin_width*0.9, 
                marker_color=color_scheme[0],
                hoverinfo="text", hovertext=sl_hover_text,
                opacity=0.8,
                name="Binned data"),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=sl_x_line, y=sl_y_line, mode='lines',
                    line=dict(color=color_scheme[1], width=3),
                    hoverinfo="none",
                    name="Continuous fit"),
            row=1, col=3
        )

        fig.update_xaxes(title_text="Number of breakpoints", tickmode='linear', tick0=1, dtick=2, row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_xaxes(title_text="Number of subtypes", tickmode='linear', tick0=2, dtick=1, row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig.update_xaxes(title_text="Segment length (HXB2 bp)", row=1, col=3)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=3)

        fig.update_layout(
            title=title,
            title_x=0.5,
            template="plotly_white",
            showlegend=False, 
            height=500,
            width=1200,
            bargap=0.15
        )
        fig.write_html(path)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(f"\n  === Distribution comparison: {len(real_n_bp)} real CRFs  vs  "
          f"{len(rec_indices)} generated recombinants ===")
    _discrete("n_breakpoints / sequence",  real_n_bp,     gen_n_bp)
    _discrete("n_subtypes / sequence",     real_n_st,     gen_n_st)
    _continuous("segment lengths (HXB2 bp)", real_seg_lens, gen_seg_lens_hxb2)
    _continuous("segment lengths (ATA pos)", pd.Series(dtype=float), gen_seg_lens_ata)

    crf_dist_path = f"{WORKSPACE_PATH}/figs/crf_dist.html"
    synthetic_dist_path = f"{WORKSPACE_PATH}/figs/synthetic_dist.html"
    _plot_distribution_comparison(gen_n_bp, gen_n_st, pd.Series(gen_seg_lens_hxb2), synthetic_dist_path,
                                  title=f'<b>Statistical distributions describing recombinant structure</b><br><sup style="color:gray">{len(gen_n_bp)} generated recombinants</sup>')
    _plot_distribution_comparison(real_n_bp, real_n_st, real_seg_lens, crf_dist_path,
                                  title=f'<b>Statistical distributions describing recombinant structure</b><br><sup style="color:gray">Data taken from LANL Sequence Database - {len(real_n_bp)} CRFs</sup>')

# ---------------------------------------------------------------------------
# Fragment-length distribution
# ---------------------------------------------------------------------------
def plot_fragment_length_distribution(frag_starts, frag_ends, ata_len,
                                      min_frag_len, path):
    """
    Print a summary and write an HTML plot for partial-sequence fragment
    lengths and start positions.

    Two panels:
      Left  - histogram of fragment lengths overlaid with the theoretical
               log-uniform density (the sampling distribution used in
               sample_fragment_window), so any systematic deviation is
               immediately visible.
      Right - histogram of fragment start positions (should be roughly
               uniform as a sanity-check).
    """
    partial_mask = [(fe - fs) < ata_len for fs, fe in zip(frag_starts, frag_ends)]
    frag_lens  = [frag_ends[i] - frag_starts[i] for i, m in enumerate(partial_mask) if m]
    starts_arr = [frag_starts[i]                for i, m in enumerate(partial_mask) if m]

    if not frag_lens:
        print("  No partial sequences — skipping fragment-length plot.")
        return

    fl = pd.Series(frag_lens, dtype=float)
    print(f"\n  === Partial Sequence Fragment Lengths  "
          f"({len(frag_lens)} / {len(frag_starts)} sequences) ===")
    print(f"  Min   : {fl.min():.0f}  ATA pos")
    print(f"  Median: {fl.median():.0f}  ATA pos")
    print(f"  Mean  : {fl.mean():.1f}  ATA pos")
    print(f"  Max   : {fl.max():.0f}  ATA pos")
    print(f"  Std   : {fl.std():.1f}  ATA pos")
    if len(fl) > 1:
        print(f"  log-uniform μ (expected): "
              f"{(np.log(min_frag_len) + np.log(ata_len)) / 2:.3f}")
        print(f"  log-normal μ (observed) : {np.log(fl).mean():.3f}")
        print(f"  log-uniform σ (expected): "
              f"{(np.log(ata_len) - np.log(min_frag_len)) / (2 * np.sqrt(3)):.3f}")
        print(f"  log-normal σ (observed) : {np.log(fl).std():.3f}")

    color_scheme = ['#072C4B', '#F28089', '#71cddd']

    # ---- panel 1: fragment lengths --------------------------------------
    n_bins      = min(60, max(10, (ata_len - min_frag_len) // 150))
    bin_edges   = np.linspace(min_frag_len, ata_len, n_bins + 1)
    bin_width   = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    counts, _ = np.histogram(frag_lens, bins=bin_edges)
    pcts      = counts / counts.sum() * 100

    hover_len = [
        f"{int(bin_edges[i])} – {int(bin_edges[i+1])} ATA pos<br>{pcts[i]:.1f}%"
        for i in range(len(bin_centers))
    ]

    # Theoretical log-uniform PDF:  f(x) = 1 / (x · ln(b/a)),  x ∈ [a, b]
    log_range  = np.log(ata_len / min_frag_len)
    x_theory   = np.linspace(min_frag_len, ata_len, 600)
    y_theory   = (1.0 / (x_theory * log_range)) * bin_width * 100  # scale to %

    # ---- panel 2: start positions ---------------------------------------
    n_bins_s     = min(60, max(10, ata_len // 150))
    s_edges      = np.linspace(0, ata_len, n_bins_s + 1)
    s_width      = s_edges[1] - s_edges[0]
    s_centers    = s_edges[:-1] + s_width / 2
    s_counts, _  = np.histogram(starts_arr, bins=s_edges)
    s_pcts       = s_counts / s_counts.sum() * 100 if s_counts.sum() > 0 else s_counts
    uniform_pct  = 100.0 / n_bins_s           # expected height if perfectly uniform
    hover_start  = [
        f"Start {int(s_edges[i])} – {int(s_edges[i+1])}<br>{s_pcts[i]:.1f}%"
        for i in range(len(s_centers))
    ]

    # ---- build figure ---------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fragment length distribution",
                        "Fragment start-position distribution"),
        horizontal_spacing=0.12,
    )

    # Left: observed bars
    fig.add_trace(
        go.Bar(x=bin_centers, y=pcts, width=bin_width * 0.9,
               marker_color=color_scheme[0], opacity=0.8,
               hoverinfo="text", hovertext=hover_len, name="Observed"),
        row=1, col=1,
    )
    # Left: theoretical log-uniform line
    fig.add_trace(
        go.Scatter(x=x_theory, y=y_theory, mode="lines",
                   line=dict(color=color_scheme[1], width=3),
                   name="Log-uniform (expected)", hoverinfo="none"),
        row=1, col=1,
    )

    # Right: start positions
    fig.add_trace(
        go.Bar(x=s_centers, y=s_pcts, width=s_width * 0.9,
               marker_color=color_scheme[2], opacity=0.8,
               hoverinfo="text", hovertext=hover_start, name="Start positions"),
        row=1, col=2,
    )
    # Right: uniform reference line
    fig.add_trace(
        go.Scatter(x=[0, ata_len], y=[uniform_pct, uniform_pct], mode="lines",
                   line=dict(color=color_scheme[1], width=2, dash="dash"),
                   name="Uniform (expected)", hoverinfo="none"),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Fragment length (ATA positions)", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_xaxes(title_text="Fragment start position (ATA)", row=1, col=2)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Partial-sequence fragment statistics</b><br>"
                f'<sup style="color:gray">'
                f"{len(frag_lens)} partial sequences  ·  "
                f"log-uniform sampling in [{min_frag_len}, {ata_len}] ATA positions"
                f"</sup>"
            ),
            x=0.5,
        ),
        template="plotly_white",
        showlegend=True,
        height=500,
        width=1100,
        bargap=0.1,
    )
    fig.write_html(path)
    print(f"  Fragment-length plot → {path}")

# ---------------------------------------------------------------------------
# Subtype sampling
# ---------------------------------------------------------------------------
def weighted_subtype_probs(counts: dict, floor: float = 0.015) -> dict:
    """
    sqrt(n)-weighted sampling probabilities per subtype, water-filled so that
    no subtype falls below `floor`. Subtypes too rare to reach `floor` under
    pure sqrt(n) weighting are pinned at `floor`; the remaining probability
    mass is redistributed by sqrt(n) among the rest.
    """
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
    """Pick n distinct subtypes with divergence-aware weighting.

    st_probs: optional dict {subtype: prob} used to weight the first pick
    (favoring subtypes with more reference diversity); later picks stay
    divergence-aware.
    """
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
                w = max(DIVERGENCE_WEIGHTS[g][SUBTYPE_GROUPS[a]] for a in chosen)
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
def divergence_profile(seq1, seq2, DIV_WINDOW_SIZE):
    """
    Windowed Hamming distance (DIV_WINDOW_SIZE centered on position).
    Returns float32 array, same length as input.
    """
    v1 = np.isin(seq1, _ACGT_BYTES)
    v2 = np.isin(seq2, _ACGT_BYTES)
    diff = ((seq1 != seq2) & v1 & v2).astype(np.float32)
    window_half = DIV_WINDOW_SIZE // 2
    kernel = np.ones(2 * window_half + 1, dtype=np.float32)
    return np.convolve(diff, kernel, mode="same")


def assign_subtypes_to_segments(n_seg, subtype_pool, py_rng):
    """
    Assign subtypes so that every subtype appears ≥ 1 and no two adjacent
    segments share the same subtype.
    """
    pool = list(subtype_pool)
    py_rng.shuffle(pool)
    result = list(pool)
    while len(result) < n_seg:
        cands = [s for s in pool if s != result[-1]]
        result.append(py_rng.choice(cands))
    return result[:n_seg]


def place_breakpoints(seg_subtypes, parents, ata_len,
                      DIV_WINDOW_SIZE, min_div, min_seg_len, rng, force_div=False):
    """
    Place one breakpoint between each pair of adjacent segments.
    Returns: list[int] of ATA breakpoint positions, or None on failure.
    """
    n_bp = len(seg_subtypes) - 1
    parents_seq = {st: seq.arr for st, seq in parents.items()}
    min_seg_len = max(min_seg_len, MIN_SEG_LEN)

    if force_div:
        breakpoints = []
        prev = 0
        _cache = {}
        for i in range(n_bp):
            st_L = seg_subtypes[i]
            st_R = seg_subtypes[i + 1]

            remaining_bps = n_bp - i - 1
            right_limit   = ata_len - (remaining_bps + 1) * min_seg_len

            key = (id(parents_seq[st_L]), id(parents_seq[st_R]))
            if key not in _cache:
                _cache[key] = divergence_profile(parents_seq[st_L], parents_seq[st_R], DIV_WINDOW_SIZE)
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

# Fragment cropping
def sample_fragment_window(ata_len: int, min_frag_len: int, rng) -> tuple[int, int]:
    """
    Sample a (start, end) fragment window.

    Start is sampled uniformly in [0, ata_len - min_frag_len].
    Length is then sampled log-uniformly in [min_frag_len, ata_len - start].

    Returns
    -------
    (start, end) : int, int
        Half-open interval [start, end) in ATA coordinates.
    """
    max_start = ata_len - min_frag_len
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    max_len = ata_len - start
    log_len = rng.uniform(
        np.log(min_frag_len),
        np.log(max_len),
    )
    length = int(
        np.clip(
            np.round(np.exp(log_len)),
            min_frag_len,
            max_len,
        )
    )
    return start, start + length


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _worker(cfg):
    row_start  = cfg["row_start"]
    row_end    = cfg["row_end"]
    chunk_sz   = row_end - row_start
    params     = cfg["params"]
    nb         = params["n_breakpoints"]
    ns         = params["n_subtypes"]
    min_seg    = params["min_seg_len"]
    st_id      = cfg["pure_st_to_id_dict"]
    n_st_total = len(st_id)
    ata_len    = cfg["ata_len"]
    n_packed   = cfg["n_packed"]
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
            lbl_row[:, st_id[st]] = True
            mask_row = np.ones(ata_len, dtype=bool)
            # name is of the form: p_<subtype>_<target_year>_<unique_id> to indicate a pure sequence
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
                    bps = place_breakpoints(seg_st, parents, ata_len, div_window_size, mdiv, min_seg, rng, force_div)
                    if bps is not None:
                        break
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
                    lbl_row[s:e, st_id[seg_st[si]]] = True

                # 2) Build Ambiguity Mask (Loss Mask)
                for si in range(n_seg - 1):
                    p1 = parents[seg_st[si]].arr
                    p2 = parents[seg_st[si+1]].arr
                    bp = bps[si]
                    
                    l = bp - 1
                    while l >= 0 and p1[l] == p2[l]: l -= 1
                    r = bp
                    while r < ata_len and p1[r] == p2[r]: r += 1
                    
                    mask_row[l+1:r] = False

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
                lbl_row[:, st_id[st]] = True
                mask_row = np.ones(ata_len, dtype=bool)
                name = f"p_{st}_{target_year}_{uuid.uuid4().hex[:8]}"

        # ---- fragment cropping -------------------------------------------
        # Applied branch-agnostically: pure and recombinant sequences are
        # cropped with the same probability, keeping the array shape fixed.
        # Out-of-window bytes are set to '-', which the cleanup line below
        # converts to 'N'; lbl_row and mask_row are zeroed so those positions
        # contribute nothing to the loss.
        frag_start, frag_end = 0, ata_len
        if partial_frac > 0.0 and py_rng.random() < partial_frac:
            frag_start, frag_end = sample_fragment_window(ata_len, min_frag_len, rng)

            # Zero out everything outside the window
            seq_row[:frag_start]  = ord("-")
            seq_row[frag_end:]    = ord("-")
            lbl_row[:frag_start]  = False
            lbl_row[frag_end:]    = False
            mask_row[:frag_start] = False
            mask_row[frag_end:]   = False

            # For recombinants: some subtypes may no longer have any labeled
            # positions inside the window → rebuild the name and recount n_bp.
            if name.startswith("r"):
                # Preserve the left-to-right order subtypes appeared in the
                # chimera; filter to those still visible after the crop.
                st_ordered = name.split("_")[1].split("+")
                remaining  = [s for s in st_ordered if lbl_row[:, st_id[s]].any()]

                if len(remaining) <= 1:
                    # Window fell inside a single segment → demote to pure.
                    # `remaining` always has ≥ 1 entry because the window is
                    # non-empty and segments cover the full genome before crop.
                    kept = remaining[0] if remaining else st_ordered[0]
                    name = f"p_{kept}_{target_year}_{uuid.uuid4().hex[:8]}"
                    n_bp = 0
                else:
                    name = f"r_{'+'.join(remaining)}_{target_year}_{uuid.uuid4().hex[:8]}"
                    # Recount breakpoints actually visible in the cropped window.
                    any_lbl     = lbl_row.any(axis=1)
                    labeled_pos = np.where(any_lbl)[0]
                    if len(labeled_pos) > 1:
                        labeled_st = lbl_row[labeled_pos].argmax(axis=1)
                        n_bp = int(np.sum(np.diff(labeled_st) != 0))
                    else:
                        n_bp = 0

        # ---- cleanup & write ---------------------------------------------
        seq_row[seq_row == ord("-")] = ord("N")  # gaps to N
        seq_mm[row_start + li]  = seq_row
        lbl_mm[row_start + li]  = np.packbits(lbl_row, axis=-1)
        mask_mm[row_start + li] = mask_row
        records.append((name, n_bp, total_muts, target_year, frag_start, frag_end))

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
    st_to_seq_dict: dict[str, list[Sequence]] = defaultdict(list)
    hxb2_ata_seq = ""

    for i, rec in enumerate(SeqIO.parse(REF_ALIGNMENT, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            parts = rec.id.split(".")  # Ref.A1.CD.87.PBS6126.MH705153
            subtype = parts[1]
            country_code = parts[2]
            two_digit_year = int(parts[3])
            year = 2000 + two_digit_year if two_digit_year < 40 else 1900 + two_digit_year
            st_to_seq_dict[subtype].append(Sequence(str(rec.seq),
                                                    country_code=country_code,
                                                    subtype=subtype,
                                                    year=year))

    # SORT st_to_seq_dict like ST_TO_ID_DICT
    pure_st_list   = list(ST_TO_ID_DICT.keys())
    st_to_seq_dict_sorted = {st: st_to_seq_dict[st] for st in pure_st_list}
    st_to_seq_dict = st_to_seq_dict_sorted
    del st_to_seq_dict_sorted
    n_subtypes     = len(ST_TO_ID_DICT)
    n_packed       = int(np.ceil(n_subtypes / 8))
    ata_len        = len(hxb2_ata_seq)

    # ---- infer parameters from real CRFs --------------------------------
    df_seg = pd.read_csv(f"{WORKSPACE_PATH}/data/output/lanl_crf_segments.csv")
    print("Inferred parameters:")
    params = infer_params(df_seg, ata_len)

    # ---- rate arrays and GTR substitution probabilities ----------------
    ata_to_hxb2, hxb2_to_ata = build_hxb2_ata_maps(hxb2_ata_seq)

    mutator = SequenceMutator(
        iqtree_dir=f"{WORKSPACE_PATH}/data/output/rates/",
        ata_len=ata_len,
        seed=SEED,
        cache_dir=f"{WORKSPACE_PATH}/data/input/diversity/",
    )
    site_rates_dict = mutator.site_rates_dict
    sub_probs_dict  = mutator.sub_probs_dict

    avg_rates = site_rates_dict['avg']
    print(f"Max site rate   (avg): {avg_rates.max():.6f}")
    print(f"Min site rate   (avg): {avg_rates.min():.6f}")
    print(f"Mean site rate  (avg): {avg_rates.mean():.6f}")

    # ---- info -----------------------------------------------------------
    print(f"\nSubtypes       : {pure_st_list}, \n{st_to_seq_dict.keys()}")
    print(f"ATA length     : {ata_len}")
    print(f"Output shape   : seq ({N_SEQ},{ata_len})  "
          f"lbl ({N_SEQ},{ata_len},{n_packed})")
    print(f"Recombinant %  : {RP:.0%}")
    print(f"Window/thresh  : ±{DIV_WINDOW_SIZE} pos, ≥{MIN_DIV} diffs")
    print(f"Workers        : {N_WORKERS}")
    print(f"Realistic      : {REALISTIC}")
    print(f"Force divergent: {FORCE_DIV}")
    print(f"Partial frac   : {PARTIAL_FRAC:.1%}  (min fragment {MIN_FRAG_LEN} ATA pos)")

    # ---- allocate memmaps -----------------------------------------------
    seq_mm  = np.lib.format.open_memmap(out_seqs,   mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len))
    lbl_mm  = np.lib.format.open_memmap(out_labels, mode="w+", dtype=np.uint8, shape=(N_SEQ, ata_len, n_packed))
    mask_mm = np.lib.format.open_memmap(out_masks,  mode="w+", dtype=bool,     shape=(N_SEQ, ata_len))
    del seq_mm, lbl_mm, mask_mm

    # ---- split work -----------------------------------------------------
    chunks = np.array_split(np.arange(N_SEQ), min(N_WORKERS, N_SEQ))
    seeds  = [int(s.generate_state(1)[0])
              for s in np.random.SeedSequence(SEED).spawn(len(chunks))]

    worker_args = [
        dict(worker_id=wid, row_start=int(c[0]), row_end=int(c[-1])+1,
             worker_seed=seeds[wid], out_seqs=out_seqs, out_labels=out_labels, out_masks=out_masks,
             st_to_seq_dict=dict(st_to_seq_dict), pure_st_list=pure_st_list,
             pure_st_to_id_dict=ST_TO_ID_DICT, params=params,
             site_rates_dict=site_rates_dict,
             sub_probs_dict=sub_probs_dict,
             ata_len=ata_len, n_packed=n_packed,
             div_window_size=DIV_WINDOW_SIZE, min_div=MIN_DIV, max_retries=MAX_RETRIES,
             force_divergent=FORCE_DIV, realistic=REALISTIC,
             partial_frac=PARTIAL_FRAC, min_frag_len=MIN_FRAG_LEN)
        for wid, c in enumerate(chunks) if len(c)
    ]

    print(f"\nLaunching {len(worker_args)} workers …")
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

    # ---- stats ----------------------------------------------------------
    n_rec     = sum(n.startswith("r") for n in names)
    n_partial = sum(fe - fs < ata_len for fs, fe in zip(frag_starts, frag_ends))
    print(f"\nRecombinant proportion : {n_rec/len(names):.2%}")
    print(f"Partial (cropped)      : {n_partial}/{len(names)}  ({n_partial/len(names):.2%})")

    st_counts = defaultdict(int)
    for name in names:
        raw = name.split("_")[1]
        for st in raw.split("+"):
            st_counts[st] += 1
    print("\nSubtype appearances:")
    for st in pure_st_list:
        print(f"  {st:6s}: {st_counts[st]:>6d}  ({st_counts[st]/len(names):6.2%})")

    # ---- metadata -------------------------------------------------------
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
            st_list = raw.split("+")
            f.write(
                f"{i+1}\t{name}\t{kind}\t{'/'.join(st_list)}\t{len(st_list)}\t"
                f"{n_bps[i]}\t{n_muts[i]}\t{target_years[i]}\t"
                f"{frag_starts[i]}\t{frag_ends[i]}\t{splits[i]}\n"
            )

    print(f"\nFiles: {out_seqs}  {out_labels}  {out_meta}  {out_masks}")

    # Output a FASTA file of the test sequences and a label file containing the labels (subtype per position) for each test sequence.
    test_indices = [i for i, s in enumerate(splits) if s == "test"]
    test_seqs = np.load(out_seqs, mmap_mode="r")[test_indices]
    test_labels = np.load(out_labels, mmap_mode="r")[test_indices]
    test_dir = Path(WORKSPACE_PATH) / "data" / "output" / "test" /f"syn_v{VERSION}"
    os.makedirs(test_dir, exist_ok=True)
    out_test_fasta = f"{test_dir}/test_sequences_{N_SEQ}_{RP}.fasta"
    out_test_labels = f"{test_dir}/test_labels_{N_SEQ}_{RP}.csv"
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

                f.write(f">{names[test_indices[i]]}\n")
                f.write(f"{''.join(map(chr, seq_clean))}\n")

                label_names = [ID_TO_ST_DICT[np.argmax(labels_id[j])] for j in range(len(labels_id))]
                regions = []
                start, current = 0, label_names[0]
                for j in range(1, len(label_names)):
                    if label_names[j] != current:
                        lf.write(f"{names[test_indices[i]]},{start+1},{j},{current},{j-start}\n")
                        start, current = j, label_names[j]
                lf.write(f"{names[test_indices[i]]},{start+1},{len(label_names)},{current},{len(label_names)-start}\n")

    # ---- Print Mutation Distribution Summary ----------------------------
    print("\n  === Generated Mutations per Sequence ===")
    muts_s = pd.Series(n_muts)
    print(f"  Min   : {muts_s.min():.0f}")
    print(f"  Median: {muts_s.median():.0f}")
    print(f"  Mean  : {muts_s.mean():.1f}")
    print(f"  Max   : {muts_s.max():.0f}")
    print(f"  Std   : {muts_s.std():.1f}")

    # ---- fragment length distribution -----------------------------------
    if PARTIAL_FRAC > 0.0:
        frag_dist_path = f"{WORKSPACE_PATH}/figs/fragment_length_dist.html"
        plot_fragment_length_distribution(
            frag_starts, frag_ends, ata_len, MIN_FRAG_LEN, frag_dist_path,
        )

    # ---- distribution comparison ----------------------------------------
    compare_generated_vs_real(
        names, out_labels, n_subtypes, n_packed, ata_len,
        pure_st_list, df_seg, ata_to_hxb2,
    )

    # ---- sanity check ---------------------------------------------------
    seq_mm = np.load(out_seqs,   mmap_mode="r")
    lbl_mm = np.load(out_labels, mmap_mode="r")
    for idx, name in enumerate(names):
        if name.startswith("r"):
            lbl = np.unpackbits(lbl_mm[idx], axis=-1)[:, :n_subtypes]
            any_active = lbl.any(axis=1)
            present = [pure_st_list[j] for j in range(n_subtypes) if lbl[:, j].any()]

            fs, fe = frag_starts[idx], frag_ends[idx]
            is_partial = fe - fs < ata_len
            print(f"\nExample #{idx+1}: {name}"
                  + (f"  [fragment {fs}:{fe}]" if is_partial else "  [full genome]"))
            print(f"  Sequence (first 2000 bases): {''.join(map(chr, seq_mm[idx][:2000]))} …")
            print(f"  Subtypes in labels: {', '.join(present)}")

            for st_name in present:
                col  = lbl[:, ST_TO_ID_DICT[st_name]]
                runs = np.diff(np.concatenate([[0], col.astype(int), [0]]))
                raw_starts = np.where(runs ==  1)[0]
                raw_ends   = np.where(runs == -1)[0]

                if len(raw_starts) == 0:
                    continue

                merged_starts = [raw_starts[0]]
                merged_ends   = []
                for i in range(len(raw_starts) - 1):
                    gap = any_active[raw_ends[i] : raw_starts[i + 1]]
                    if gap.any():
                        merged_ends.append(raw_ends[i])
                        merged_starts.append(raw_starts[i + 1])
                merged_ends.append(raw_ends[-1])

                for s, e in zip(merged_starts, merged_ends):
                    print(f"    {st_name}: ATA [{s}, {e})  ({e-s} pos)")
            break