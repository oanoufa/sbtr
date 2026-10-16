"""Create figures for sbtr predictions and evaluation results."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotly.subplots import make_subplots
from Bio import SeqIO
import pandas as pd
import sys
import re
import os
from collections import defaultdict
from itertools import groupby, product
import plotly.io as pio
import plotly.colors as pc
import matplotlib.patches as mpatches
pio.defaults.default_format = "png"
from collections import Counter
from pathlib import Path
import random
import warnings
import math
from matplotlib.colors import LinearSegmentedColormap
from html import escape
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from tqdm import tqdm


from python_scripts.src.mutator_class import parse_iqtree_rates
from python_scripts.src import config

warnings.filterwarnings("ignore", category=DeprecationWarning)

workspace_path = config.WORKSPACE_PATH
WORKSPACE_PATH = workspace_path

GENES_RAW = config.GENES_RAW
GENE_COLORS = config.GENE_COLORS
COLOR_SCHEME = config.COLOR_SCHEME
ST_COLORS = config.ST_COLORS
METRICS_DIR = config.MODEL_CONFIG["metrics_dir"]
VERSION = config.VERSION
ATA_LEN = config.ATA_LEN

JPHMM_TO_SBTR_LTR = {
    "5'-Insertion":"5'LTR",
    "3'-Insertion":"3'LTR",
}

FRAME_LANES = {1: (0.67, 1.0), 2: (0.33, 0.66), 3: (0.0, 0.33)}
PAD = 0.015
MULTI_EXON = ("tat", "rev")



def visualize_breakpoints(
    df_bp, 
    save_path=f"{workspace_path}/figs/breakpoint_distribution_with_genes.html"):
    # Initialize the figure with your histogram
    fig = px.histogram(df_bp, x="pos", nbins=200, 
                    color_discrete_sequence=["#072C4B"], opacity=0.8)

    # Define the vertical "lanes" for each frame (in paper coordinates 0-1)
    frame_lanes = {
        3: (0, 0.32),
        2: (0.33, 0.64),
        1: (0.65, 0.95)
    }

    for i, (gene, (start, end, frame)) in enumerate(GENES_RAW.items()):
        y0, y1 = frame_lanes[frame]
        color = GENE_COLORS.get(gene, "grey")
        
        # 1. Add the "Lane" segment for this gene
        fig.add_vrect(
            x0=start, x1=end,
            y0=y0, y1=y1,
            yref="paper",
            fillcolor=color,
            opacity=0.5,
            layer="below" if frame == 1 else "above",
            line_width=0
        )
        
        # 2. Add the Gene Label inside the lane
        # We only show labels for larger genes or use small font to avoid clutter
        y_pos = (y0 + y1) / 2
        if i % 2 == 0:
            y_pos += 0.03  # Shift up for some genes
        elif i % 2 == 1:
            y_pos -= 0.03  # Shift down for others

        fig.add_annotation(
            x=(start + end) / 2,
            y=y_pos,
            yref="paper",
            text=f"<b>{gene}</b>",
            showarrow=False,
            font=dict(size=8, color="black"),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor=color,
            borderwidth=0.5,
        )

    # Add "Frame" labels on the far left
    for frame, (y0, y1) in frame_lanes.items():
        fig.add_annotation(
            x=-0, y=(y0 + y1) / 2, xref="paper", yref="paper",
            text=f"F{frame}", showarrow=False, font=dict(size=12, color="gray")
        )

    fig.update_layout(
        title="Breakpoint distribution across HIV-1 genome, breakpoints taken from LANL Sequence DB",
        xaxis_title="HXB2 position",
        yaxis_title="Breakpoint count",
        template="plotly_white",
        xaxis=dict(range=[0, 9719]),
        # Ensure the Y-axis has enough room for the lanes at the bottom
        # yaxis=dict(range=[0, df_bp['pos'].value_counts().max() * 1.5]) 
    )

    # Set image length and width to 1200x800 for better quality
    fig.update_layout(width=1200, height=500)
    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))
    print(f"Breakpoints visualization saved to: {save_path}", flush=True)


def _add_gene_lanes(
    fig):
    """Gene lanes + frame labels."""
    frame_lanes = {3: (0.00, 0.09), 2: (0.10, 0.19), 1: (0.20, 0.29)}
    for i, (gene, (start, end, frame)) in enumerate(GENES_RAW.items()):
        y0, y1 = frame_lanes[frame]
        color = GENE_COLORS.get(gene, "grey")
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=start, x1=end, y0=y0, y1=y1,
            fillcolor=color, opacity=0.85,
            layer="below", line=dict(color="white", width=0.5),
        )
        shift = i % 2
        y_pos = (y0 + y1 - 0.05*shift) / 2
        fig.add_annotation(x=(start + end) / 2, y=y_pos, yref="paper",
                           text=f"<b>{gene}</b>", showarrow=False,
                           font=dict(size=8, color="black"),
                           bgcolor="rgba(255,255,255,0.6)", bordercolor=color, borderwidth=0.5)
    for frame, (y0, y1) in frame_lanes.items():
        fig.add_annotation(
            x=1.002, y=(y0 + y1 - 0.04) / 2, xref="paper", yref="paper",
            xanchor="left", text=f"F{frame}", showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.6)",
        )


def visualize_generated_breakpoints(
        df_gen,
        df_real=None,
        save_path=f"{workspace_path}/figs/generated_breakpoint_distribution.html"):
    """Generated breakpoint distribution along HXB2, optionally overlaid with the
    real CRF one. Both are shown in % of their own total so they are comparable."""
    xbins = dict(start=0, end=9719, size=50)
    fig = go.Figure()
    if df_real is not None:
        fig.add_histogram(x=df_real["pos"], xbins=xbins, histnorm="percent",
                          name="Real CRFs (LANL)", marker_color=COLOR_SCHEME[0], opacity=0.6)
    fig.add_histogram(x=df_gen["pos"], xbins=xbins, histnorm="percent",
                      name="Generated", marker_color=COLOR_SCHEME[1], opacity=0.7)
    _add_gene_lanes(fig)
    fig.update_layout(
        title="Breakpoint distribution across HIV-1 genome (generated recombinants)",
        xaxis_title="HXB2 position", yaxis_title="% of breakpoints",
        template="plotly_white", barmode="overlay",
        xaxis=dict(range=[0, 9719], anchor="free", position=0),
        yaxis=dict(domain=[0.33, 1.0]),
        width=1200, height=600,
        legend=dict(x=0.35, y=0.90, bgcolor="rgba(0,0,0,0.1)"),

    )
    fig.write_html(save_path)
    fig.write_image(Path(save_path).with_suffix(".svg"))
    print(f"Generated breakpoints figure saved to: {save_path}", flush=True)


def visualize_breakpoint_rank_profile(
        df_gen, n_bp_sel, ata_len,
        save_path=f"{workspace_path}/figs/breakpoint_rank_profile.html"):
    """For sequences with exactly n_bp_sel breakpoints, histogram of the relative
    position (0-1) of each breakpoint rank. Sequential sampling shows up as the
    last ranks stacked at 1; an unbiased sampler centres rank k near k/(n+1)."""
    d = df_gen[df_gen["n_bp"] == n_bp_sel].copy()
    d["rel"] = d["pos_ata"] / ata_len
    fig = go.Figure()
    for k in range(1, n_bp_sel + 1):
        fig.add_histogram(x=d.loc[d["rank"] == k, "rel"],
                          xbins=dict(start=0, end=1, size=0.01),
                          histnorm="percent", name=f"bp {k}", opacity=0.6)
    fig.update_layout(
        title=f"Relative position of each breakpoint rank (sequences with {n_bp_sel} breakpoints)",
        xaxis_title="Relative position in ATA alignment", yaxis_title="% of sequences",
        template="plotly_white", barmode="overlay", width=1200, height=450)
    fig.write_html(save_path)
    fig.write_image(Path(save_path).with_suffix(".svg"))
    print(f"Rank profile saved to: {save_path}", flush=True)

def compare_generated_vs_real(
        names, out_labels, n_subtypes, ata_len, df_seg, ata_to_hxb2,
        ):
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
    # Real CRF statistics
    is_compound   = df_seg["subtype"].str.contains("/", regex=False)
    df_pure       = df_seg[~is_compound]

    real_n_bp     = (df_pure.groupby("crf").size() - 1).clip(lower=0)
    real_n_st     = df_pure.groupby("crf")["subtype"].nunique()
    real_seg_lens = df_pure["length"].dropna().astype(float)
    real_seg_lens = real_seg_lens[real_seg_lens > 0]

    # Printing helpers
    W_L, W_C = 30, 12

    def _header(title):
        print(f"\n {title}")
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

        #n_breakpoints / sequence
        n_bp = pd.Series(n_bp)
        n_st = pd.Series(n_st)
        seg_lens = pd.Series(seg_lens)
        bp_values = n_bp.value_counts().sort_index().index.tolist()
        bp_real_pct = n_bp.value_counts(normalize=True).sort_index().values.tolist()

        #n_subtypes / sequence
        st_values = n_st.value_counts().sort_index().index.tolist()
        st_real_pct = n_st.value_counts(normalize=True).sort_index().values.tolist()

        #segment lengths
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
        path = Path(path)
        fig.write_image(path.with_suffix(".svg"))
    # Print
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


def visualize_diversity(
    diversity_arrays: dict,
    hxb2_to_ata: np.ndarray,
    st_to_print=['avg'],
    types_to_print=['smoothed'],
    window_size=100,
    save_path=f"{workspace_path}/figs/diversity_rate_profile.html"):
    """
    Visualizes empirical diversity arrays with HIV genes in the background.
    diversity_arrays: dict of {name: diversity_array}, all must have the same length.
    """
    # Use first array for reference length/mean
    first_array = next(iter(diversity_arrays.values()))
    ata_len = len(first_array)
    x_positions = np.arange(ata_len)
    y_max = max(arr.max() for arr in diversity_arrays.values()) * 1.1

    kernel = np.ones(window_size) / window_size

    fig = make_subplots(
        rows=1, cols=1,
        row_heights=[1.0],
        subplot_titles=("<b>Empirical diversity across the genome</b>",)
    )

    frame_lanes = {
        3: (0, y_max * 0.32),
        2: (y_max * 0.33, y_max * 0.64),
        1: (y_max * 0.65, y_max * 0.95)
    }
    
    frame_pos_axis = {
        3: y_max * 0.15,
        2: y_max * 0.485,
        1: y_max * 0.85,
    }

    for i, (gene, (start_hxb2, end_hxb2, frame)) in enumerate(GENES_RAW.items()):
        if 'LTR' in gene:
            # Only showing genes
            continue
        start_ata = hxb2_to_ata[start_hxb2]
        end_ata   = hxb2_to_ata[end_hxb2]
        y0, y1    = frame_lanes[frame]
        color     = GENE_COLORS.get(gene, "grey")

        fig.add_shape(
            type="rect", x0=start_ata, x1=end_ata, y0=y0, y1=y1,
            fillcolor=color, opacity=0.25, layer="below", line_width=0, row=1, col=1
        )

        y_pos    = (y0 + y1) / 2
        y_offset = (y_max * 0.04) if (i % 2 == 0) else -(y_max * 0.04)
        fig.add_annotation(
            x=(start_ata + end_ata) / 2, y=y_pos + y_offset,
            text=f"<b>{gene}</b>", showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="rgba(255,255,255,0.7)", bordercolor=color, borderwidth=1,
            row=1, col=1
        )

    for frame, (y0, y1) in frame_lanes.items():
        fig.add_annotation(
            x=1.02, y=frame_pos_axis[frame],
            xref="paper", yref="y",
            text=f"<b>F{frame}</b>", showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="right",
        )

    # ONE RAW + SMOOTHED TRACE PAIR PER RATE ARRAY
    # Plotly default color cycle
    trace_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for i, (name, diversity_array) in enumerate(diversity_arrays.items()):
        color          = trace_colors[i % len(trace_colors)]
        # Zero rates in LTR regions
        diversity_array /= diversity_array.sum()
        smoothed_rates = np.convolve(diversity_array, kernel, mode='same')
        mean_rate      = np.mean(diversity_array)
        if name in st_to_print:
            if 'raw' in types_to_print:
                fig.add_trace(go.Scatter(
                    x=x_positions, y=diversity_array,
                    mode='lines', line=dict(color=color, width=1),
                    opacity=0.3, name=f'{name} raw',
                ), row=1, col=1)
            if 'smoothed' in types_to_print:
                fig.add_trace(go.Scatter(
                    x=x_positions, y=smoothed_rates,
                    mode='lines', line=dict(color=color, width=2),
                    name=f'{name} smoothed',
                ), row=1, col=1)

    fig.add_hline(
        y=mean_rate, line_dash="dash", line_color='#000000', opacity=0.8,
        annotation_text=f"mean: {mean_rate:.1e}",
        annotation_position="top right",
        layer="above", row=1, col=1
    )

    fig.update_layout(
        template="plotly_white",
        height=500, width=1100,
        hovermode="x unified",
        showlegend=True,
        margin=dict(t=80, l=80),   # extra top margin so title doesn't touch legend
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="ATA Alignment Position (bp)", range=[0, ata_len], row=1, col=1)
    fig.update_yaxes(title_text="diversity", range=[0, y_max], row=1, col=1)

    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))
    
    print(f"Diversity visualization saved to: {save_path}", flush=True)


def _subtype_array(regions_aligned, n_total):
    arr = np.full(n_total, "", dtype=object)
    for r_start, r_end, st in regions_aligned:
        st = JPHMM_TO_SBTR_LTR.get(st, st)
        s, e = max(0, int(r_start)), min(n_total, int(r_end))
        if s < e:
            arr[s:e] = str(st)
    return arr


def _gene_entries(hxb2_to_ata, n_total):
    entries = []
    for name, (s_h, e_h, frame) in GENES_RAW.items():
        if 'LTR' in name:
            # Skip LTR
            continue
        s = max(0, int(hxb2_to_ata[s_h]))
        e = min(n_total, int(hxb2_to_ata[e_h]))
        if s >= e:
            continue
        base = next((b for b in MULTI_EXON if name.lower().startswith(b)), name)
        entries.append(dict(name=name, base=base, frame=frame, s=s, e=e))
    return entries


def _gene_runs(subtype_arr, s, e):
    pos = s
    for val, grp in groupby(subtype_arr[s:e]):
        n = sum(1 for _ in grp)
        yield val, pos, n
        pos += n


def _label_layout(entries, n_total, width, fs=7):
    """
    width: drawable width in the same unit as fs (pt for matplotlib, px for SVG).
    Returns labels [(x, y, text, ha, va)] and connectors [[(x, y), ...]],
    in (bp, lane-fraction) coordinates.
    """
    bp_per_u = n_total / width
    char_w = 0.55 * fs
    groups = defaultdict(list)
    lane_rects = defaultdict(list)
    for g in entries:
        groups[g["base"]].append(g)
        lane_rects[g["frame"]].append((g["s"], g["e"]))

    labels, connectors = [], []
    for base, exons in groups.items():
        exons = sorted(exons, key=lambda g: g["s"])
        text = base.replace("_", " ")
        # Multi-exon gene: connector line + one label centered on it
        if len(exons) > 1:
            for a, b in zip(exons[:-1], exons[1:]):
                xa, xb = (a["s"] + a["e"]) / 2, (b["s"] + b["e"]) / 2
                if a["frame"] != b["frame"]:
                    y_line = FRAME_LANES[min(a["frame"], b["frame"])][0]
                    def attach(g):  # edge of the exon facing the line
                        y0, y1 = FRAME_LANES[g["frame"]]
                        return (y1 - PAD) if y1 <= y_line + 1e-6 else (y0 + PAD)
                    pts = [(xa, attach(a)), (xa, y_line), (xb, y_line), (xb, attach(b))]
                else:
                    y0, y1 = FRAME_LANES[a["frame"]]
                    y_line = (y0 + y1) / 2
                    pts = [(a["e"], y_line), (b["s"], y_line)]
                connectors.append(pts)
                labels.append(((xa + xb) / 2, y_line + 0.01, text, "center", "bottom"))
            continue

        g = exons[0]
        y0, y1 = FRAME_LANES[g["frame"]]
        yc = (y0 + y1) / 2
        tw = len(text) * char_w * bp_per_u
        margin = 3 * bp_per_u
        if tw + margin <= g["e"] - g["s"]:
            labels.append(((g["s"] + g["e"]) / 2, yc, text, "center", "center"))
            continue

        # Too small: put the name next to the gene (right first, then left)
        gap = 2 * bp_per_u
        others = [r for r in lane_rects[g["frame"]] if r != (g["s"], g["e"])]
        placed = False
        for side in ("right", "left"):
            if side == "right":
                x0 = g["e"] + gap; x1 = x0 + tw; anchor, ha = x0, "left"
            else:
                x1 = g["s"] - gap; x0 = x1 - tw; anchor, ha = x1, "right"
            if x0 < 0 or x1 > n_total:
                continue
            if all(x1 <= s or x0 >= e for s, e in others):
                labels.append((anchor, yc, text, ha, "center"))
                placed = True
                break
        if not placed:
            xc = (g["s"] + g["e"]) / 2
            y_lab = y0 - 0.03
            connectors.append([(xc, y0 + PAD), (xc, y_lab + 0.005)])
            labels.append((xc, y_lab, text, "center", "top"))
    return labels, connectors

def visualize_sample_probs(
    preds_slice: np.ndarray,
    ploss_slice: np.ndarray,
    sample_name: str,
    regions_aligned: list,
    pure_st_to_id_dict: dict,
    hxb2_to_ata: np.ndarray,
    path: str,
    ) -> None:
    """
    Generate one sample figure directly from memory-mapped arrays.
    Parameters
    ----------
    preds_slice
        Slice of the predictions_*.npy memmap.
        Shape: (N, ata_len, n_subtypes)
    ploss_slice
        Slice of the post_loss_masks_*.npy memmap.
        Shape: (N, ata_len)
    sample_name
        Name used in the figure title/output identifier.
    regions_aligned
        CRF subtype regions for this sample.
    pure_st_to_id_dict
        Mapping subtype name -> subtype ID.
    hxb2_to_ata
        HXB2 -> ATA coordinate mapping.
    path
        Output PNG path.
    """

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_two_color",
        ["#072C4B", "#F28089"],  # couleur basse -> couleur haute
        N=256,
    )

    CMAP_CHOICES = [
        # "viridis", "plasma",
        "magma",
        # "cividis", "turbo", 
        # custom_cmap,
        ]

    loss_mask = np.asarray(ploss_slice)
    labels = np.asarray(preds_slice)

    real_loss_mask = loss_mask.astype(bool)
    n_real = int(real_loss_mask.sum())
    n_total = len(real_loss_mask)

    EXCLUDED_SUBTYPES = {"U", "5'LTR", "3'LTR"}
    id_to_st = {v: k for k, v in pure_st_to_id_dict.items()}
    keep_ids = [
        i for i in range(len(pure_st_to_id_dict))
        if id_to_st[i] not in EXCLUDED_SUBTYPES
    ]
    subtype_names = [id_to_st[i] for i in keep_ids]
    n_subtypes = len(subtype_names)

    # (n_subtypes, n_total)
    full_labels = labels.T[keep_ids, :]

    # Figure setup
    gene_track_h = 2.4
    fig = plt.figure(
        figsize=(14, 2 + gene_track_h + 0.3 * n_subtypes)
    )
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, n_subtypes, gene_track_h],
        width_ratios=[40, 1],
        hspace=0.10,
        wspace=0.03,
    )
    ax_mask = fig.add_subplot(gs[0, 0])
    ax_lab = fig.add_subplot(gs[1, 0], sharex=ax_mask)
    ax_st = fig.add_subplot(gs[2, 0], sharex=ax_mask)
    ax_cb = fig.add_subplot(gs[1, 1])
    frame_lanes = {
        1: (0.67, 1.0),
        2: (0.33, 0.66),
        3: (0.0, 0.33),
    }

    # Row 1: loss mask
    ax_mask.imshow(
        real_loss_mask[np.newaxis, :],
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_mask.set_yticks([0])
    ax_mask.set_yticklabels(["loss\nmask"], fontsize=8)
    ax_mask.set_title(
        f"Sample {sample_name}  —  "
        f"{n_real} real tokens / {n_total} total "
        f"({n_total - n_real} padding)",
        fontsize=10,
    )
    plt.setp(ax_mask.get_xticklabels(), visible=False)

    # Row 2: Subtype probability heatmap
    cmap_choice = random.choice(CMAP_CHOICES)
    im = ax_lab.imshow(
        full_labels,
        aspect="auto",
        cmap=cmap_choice,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_lab.set_yticks(range(n_subtypes))
    ax_lab.set_yticklabels(subtype_names, fontsize=7)
    plt.setp(ax_lab.get_xticklabels(), visible=False)

    # Row 3: Subtype region track
    ax_st.set_xlim(0, n_total)
    ax_st.set_ylim(0, 1)
    for spine in ("top", "right", "left"):
        ax_st.spines[spine].set_visible(False)
    ax_st.set_yticks([])
    ax_st.xaxis.set_major_locator(
        ticker.MultipleLocator(max(1, n_total // 10))
    )
    ax_st.tick_params(axis="x", labelsize=8)
    ax_st.set_xlabel(
        "ATA alignment position (bp)",
        fontsize=9,
    )

    subtype_arr = _subtype_array(regions_aligned, n_total)
    entries = _gene_entries(hxb2_to_ata, n_total)

    st_bars = defaultdict(list)
    for g in entries:
        y0, y1 = FRAME_LANES[g["frame"]]
        rect_h = (y1 - y0) - 2 * PAD
        for val, pos, n in _gene_runs(subtype_arr, g["s"], g["e"]):
            if val:
                st_bars[(ST_COLORS.get(val, "#CCCCCC"), y0 + PAD, rect_h)].append((pos, n))
        # thin black gene outline
        ax_st.add_patch(Rectangle(
            (g["s"], y0 + PAD), g["e"] - g["s"], rect_h,
            fill=False, ec="black", lw=0.5, zorder=3,
        ))

    for frame, (y0, y1) in FRAME_LANES.items():
        ax_st.text(-n_total * 0.005, (y0 + y1) / 2, f"F{frame}",
                   ha="right", va="center", fontsize=7, color="grey")

    width_pt = ax_st.get_position().width * fig.get_figwidth() * 72
    labels, connectors = _label_layout(entries, n_total, width_pt, fs=7)
    halo = [pe.withStroke(linewidth=1.2, foreground="white", alpha=0.7)]
    for pts in connectors:
        xs, ys = zip(*pts)
        ax_st.plot(xs, ys, color="black", lw=0.6, zorder=4)
    for x, y, txt, ha, va in labels:
        ax_st.text(x, y, txt, ha=ha, va=va, fontsize=7, zorder=5, path_effects=halo)

    for (st_color, y_bottom, height), xranges in st_bars.items():
        ax_st.broken_barh(
            xranges,
            (y_bottom, height),
            facecolors=st_color,
            alpha=0.88,
            linewidths=0,
        )

    LTR5 = "5'LTR"
    LTR3 = "3'LTR"
    seen_order = []
    for val in subtype_arr:
        if val != "" and val not in seen_order:
            seen_order.append(val)
    middle = [st for st in seen_order if st not in (LTR5, LTR3)]
    present_subtypes = (
        ([LTR5] if LTR5 in seen_order else [])
        + middle
        + ([LTR3] if LTR3 in seen_order else [])
    )

    if present_subtypes:
        legend_handles = [
            mpatches.Patch(
                fc=ST_COLORS.get(st, "#CCCCCC"),
                ec="#555555",
                lw=0.4,
                label=st,
            )
            for st in present_subtypes
        ]

        ax_st.legend(
            handles=legend_handles,
            title=None,
            fontsize=7,
            ncol=len(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            frameon=True,
            framealpha=0.85,
            borderpad=0.6,
            handlelength=1.2,
        )

    fig.colorbar(
        im,
        cax=ax_cb,
        label="score",
    )

    fig.patch.set_alpha(0.0)
    for ax in (ax_mask, ax_lab, ax_st, ax_cb):
        ax.patch.set_alpha(0.0)

    fig.savefig(
        path,
        dpi=100,
        transparent=True,
    )

    plt.close(fig)


_PATTERN_DEFS = (
    '<defs>'
    '<pattern id="p-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<line x1="0" y1="0" x2="0" y2="4" stroke="#000" stroke-width="1" stroke-opacity="0.6"/></pattern>'
    '<pattern id="p-plus" width="6" height="6" patternUnits="userSpaceOnUse">'
    '<path d="M3 1v4M1 3h4" stroke="#000" stroke-width="0.9" stroke-opacity="0.6" fill="none"/></pattern>'
    '<pattern id="p-cross" width="5" height="5" patternUnits="userSpaceOnUse">'
    '<path d="M0 0L5 5M5 0L0 5" stroke="#000" stroke-width="0.8" stroke-opacity="0.6" fill="none"/></pattern>'
    '<pattern id="p-dots" width="5" height="5" patternUnits="userSpaceOnUse">'
    '<circle cx="2.5" cy="2.5" r="0.9" fill="#000" fill-opacity="0.6"/></pattern>'
    '</defs>'
)
_PATTERN_CYCLE = [None, "hatch", "plus", "cross", "dots"]  # None = plain colour

def visualize_sample_probs_light(regions_aligned, hxb2_to_ata, n_total, path,
                    width=1100, lane_h=24, fs=9):
    ml, mr, top = 28, 20, 4
    pw = width - ml - mr
    sx = pw / n_total
    H3 = 3 * lane_h
    height = top + H3 + 46
    X = lambda x: ml + x * sx
    Y = lambda y: top + (1 - y) * H3

    arr = _subtype_array(regions_aligned, n_total)
    entries = _gene_entries(hxb2_to_ata, n_total)

    by_st, outlines, seen = defaultdict(list), [], []
    for g in entries:
        y0, y1 = FRAME_LANES[g["frame"]]
        yt, h = Y(y1 - PAD), (y1 - y0 - 2 * PAD) * H3
        w = (g["e"] - g["s"]) * sx
        outlines.append(f"M{X(g['s']):.1f} {yt:.1f}h{w:.1f}v{h:.1f}h-{w:.1f}z")
        for val, pos, n in _gene_runs(arr, g["s"], g["e"]):
            if val:
                if val not in seen:
                    seen.append(val)
                by_st[val].append(
                    f"M{X(pos):.1f} {yt:.1f}h{n * sx:.1f}v{h:.1f}h-{n * sx:.1f}z")

    order = list(ST_COLORS)
    present = sorted(seen, key=lambda s: order.index(s) if s in order else len(order))
    pat = {st: _PATTERN_CYCLE[i % len(_PATTERN_CYCLE)] for i, st in enumerate(present)}
    labels, connectors = _label_layout(entries, n_total, pw, fs=fs)
    anchor = {"left": "start", "center": "middle", "right": "end"}
    ay = top + H3 + 5

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        _PATTERN_DEFS]
    for st, parts in by_st.items():
        d = "".join(parts)
        out.append(f'<path fill="{ST_COLORS.get(st, "#CCCCCC")}" fill-opacity="0.88" d="{d}"/>')
        p = pat[st]
        if p:
            out.append(f'<path fill="url(#p-{p})" d="{d}"/>')
    out.append(f'<path fill="none" stroke="#000" stroke-width="0.5" d="{"".join(outlines)}"/>')
    for pts in connectors:
        p = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{p}" fill="none" stroke="#000" stroke-width="0.6"/>')
    for x, y, txt, ha, va in labels:
        base_attr = {"center": 'dominant-baseline="central"',
                     "top": 'dominant-baseline="hanging"'}.get(va, "")
        out.append(f'<text x="{X(x):.1f}" y="{Y(y):.1f}" font-size="{fs}" text-anchor="{anchor[ha]}" '
                   f'{base_attr} stroke="#fff" stroke-width="1.2" stroke-opacity="0.7" paint-order="stroke">{escape(txt)}</text>')
    for f, (y0, y1) in FRAME_LANES.items():
        out.append(f'<text x="{ml - 4}" y="{Y((y0 + y1) / 2):.1f}" font-size="{fs - 2}" fill="#888" '
                   f'text-anchor="end" dominant-baseline="central">F{f}</text>')

    # axis
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{ay}" y2="{ay}" stroke="#000" stroke-width="0.6"/>')
    step = max(1, n_total // 10)
    for t in range(0, n_total + 1, step):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{ay}" y2="{ay + 3}" stroke="#000" stroke-width="0.6"/>')
        out.append(f'<text x="{X(t):.1f}" y="{ay + 12}" font-size="{fs - 1}" text-anchor="middle">{t}</text>')

    # legend
    lx = ml
    for st in seen:
        out.append(f'<rect x="{lx}" y="{height - 13}" width="9" height="9" fill="{ST_COLORS.get(st, "#CCCCCC")}" stroke="#555" stroke-width="0.4"/>')
        p = pat[st]
        if p:
            out.append(f'<rect x="{lx}" y="{height - 13}" width="9" height="9" fill="url(#p-{p})"/>')
        out.append(f'<text x="{lx + 12}" y="{height - 5}" font-size="{fs - 1}">{escape(st)}</text>')
        lx += 22 + 6 * len(st)

    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))

def visualize_sample_probs_lines(preds_slice, pure_st_to_id_dict, n_total, path,
                                 width=1100, height=260, fs=9, min_peak=0.3,
                                 excluded=frozenset({"U", "5'LTR", "3'LTR"})):
    ml, mr, top, bot = 34, 20, 8, 46
    pw, ph = width - ml - mr, height - top - bot
    X = lambda x: ml + x * pw / n_total
    Y = lambda y: top + (1 - y) * ph

    P = np.asarray(preds_slice)                      # (ata_len, n_subtypes)
    step = max(1, n_total // pw)
    idx = np.arange(0, n_total, step)
    xs = idx + step / 2

    id_to_st = {v: k for k, v in pure_st_to_id_dict.items()}
    lines, seen = [], []
    for i in range(len(id_to_st)):
        st = id_to_st[i]
        if st in excluded or P[:, i].max() <= min_peak:
            continue
        ys = np.maximum.reduceat(P[:, i], idx)
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{ST_COLORS.get(st, "#CCCCCC")}" '
                     f'stroke-width="1.2" stroke-linejoin="round"/>')
        seen.append(st)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">']

    # y grid + labels
    for v in (0, 0.5, 1):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                   f'stroke="#ddd" stroke-width="0.5"/>')
        out.append(f'<text x="{ml - 4}" y="{Y(v):.1f}" font-size="{fs - 1}" fill="#555" '
                   f'text-anchor="end" dominant-baseline="central">{v:g}</text>')
    out.extend(lines)

    # x axis
    ay = top + ph + 2
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{ay}" y2="{ay}" stroke="#000" stroke-width="0.6"/>')
    tstep = max(1, n_total // 10)
    for t in range(0, n_total + 1, tstep):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{ay}" y2="{ay + 3}" stroke="#000" stroke-width="0.6"/>')
        out.append(f'<text x="{X(t):.1f}" y="{ay + 12}" font-size="{fs - 1}" text-anchor="middle">{t}</text>')

    # legend (colored line swatches)
    lx = ml
    for st in seen:
        c = ST_COLORS.get(st, "#CCCCCC")
        out.append(f'<line x1="{lx}" x2="{lx + 12}" y1="{height - 9}" y2="{height - 9}" stroke="{c}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 16}" y="{height - 5}" font-size="{fs - 1}">{escape(st)}</text>')
        lx += 26 + 6 * len(st)

    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))

def visualize_region_comparison(
    seq_id: str,
    regions_list: list[dict | list],
    labels_list: list[str],
    seq_length: int,
    path: str = None,
) -> None:
    """
    Compare N sets of (start, end, subtype) region calls for a single sequence.

    regions_list: List of iterables (or dicts) containing (start, end, subtype) tuples.
    labels_list: List of string labels corresponding to each region set in regions_list.
    """
    if len(regions_list) != len(labels_list):
        raise ValueError("regions_list and labels_list must have the same length.")

    n_tracks = len(regions_list)
    if n_tracks == 0:
        raise ValueError("At least one set of regions must be provided.")

    def _as_tuples(regions):
        return list(regions.values()) if isinstance(regions, dict) else list(regions)

    # Normalize all input region sets into lists of tuples
    regions_list = [_as_tuples(r) for r in regions_list]

    FALLBACK_COLOR = "#cccccc"  # grey used for missing/dash data

    def subtype_color(st):
        return ST_COLORS.get(st, FALLBACK_COLOR)

    # Layout: N rows, dynamic height
    track_h = 1.0
    fig = plt.figure(figsize=(14, n_tracks * track_h + 1.2))
    gs = fig.add_gridspec(
        n_tracks, 1,
        height_ratios=[track_h] * n_tracks,
        hspace=0.80 if n_tracks > 1 else 0.40,
    )

    axes = []
    for i in range(n_tracks):
        share = axes[0] if i > 0 else None
        axes.append(fig.add_subplot(gs[i, 0], sharex=share))

    # Helper function to draw an individual track
    def draw_track(ax, regions, label):
        ax.set_xlim(0, seq_length)
        ax.set_ylim(0, 1)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_yticks([0.5])
        ax.set_yticklabels([label], fontsize=8)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, seq_length // 10)))
        ax.tick_params(axis='x', labelsize=8)

        for start, end, subtype in regions:
            if subtype in JPHMM_TO_SBTR_LTR.keys():
                subtype = JPHMM_TO_SBTR_LTR[subtype]
            s = max(0, int(start))
            e = min(seq_length, int(end))
            if s >= e:
                continue
            ax.add_patch(plt.Rectangle(
                (s, 0.05), e - s, 0.9,
                fc=subtype_color(subtype),
                ec="#555555", linewidth=0.4, alpha=0.88,
            ))
            if (e - s) > seq_length * 0.02:
                ax.text(
                    s + (e - s) / 2, 0.5, str(subtype),
                    ha="center", va="center",
                    fontsize=6.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
                )

    # Draw all tracks dynamically
    for ax, regions, label in zip(axes, regions_list, labels_list):
        draw_track(ax, regions, label)

    # Hide x-axis tick labels for all but the bottom subplot
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    axes[-1].set_xlabel("Sequence position (bp)", fontsize=9)
    axes[0].set_title(f"Sequence {seq_id}  —  length {seq_length} bp", fontsize=10)

    # Shared legend across all N tracks
    present_subtypes = set()
    for regions in regions_list:
        for r in regions:
            subtype = str(r[2])
            # .get() simplifies dictionary lookup
            subtype = JPHMM_TO_SBTR_LTR.get(subtype, subtype)
            present_subtypes.add(subtype)

    present_subtypes = sorted(present_subtypes)
    
    if present_subtypes:
        legend_handles = [
            mpatches.Patch(
                fc=subtype_color(st), ec="#555555", lw=0.4, label=st,
            )
            for st in present_subtypes
        ]
        
        # Adjust legend vertical offset based on number of tracks
        y_bbox = -0.8 - (0.35 / n_tracks)
        axes[-1].legend(
            handles=legend_handles,
            title="Subtype",
            title_fontsize=8,
            fontsize=7,
            ncol=min(len(legend_handles), 6),
            loc="upper center",
            bbox_to_anchor=(0.5, y_bbox),
            frameon=True,
            framealpha=0.85,
            borderpad=0.3,
            handlelength=1.2,
        )

    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def visualize_metrics(save_path_loss,
                      save_path_evol):
    # Generate figures showing the evolution of the scores during training

    train_metrics_df = pd.read_csv(os.path.join(METRICS_DIR, f"train_metrics_v{VERSION}.tsv"), sep='\t')
    val_metrics_df = pd.read_csv(os.path.join(METRICS_DIR, f"val_metrics_v{VERSION}.tsv"), sep='\t')

    # f1/micro	precision/micro	recall/micro	loss	step
    # 0.07481227070093155	0.05566808208823204	0.11402550339698792	5.273273804485798	2000
    # 1. Data Preparation
    # Add a split column to distinguish data sources
    train_metrics_df['split'] = 'Train'
    val_metrics_df['split'] = 'Validation'

    # Combine for plotting
    df_combined = pd.concat([train_metrics_df, val_metrics_df], ignore_index=True)

    # Identify the metrics you want to track
    performance_metrics = ["f1/micro", "precision/micro", "recall/micro"]
    all_metrics = performance_metrics + ["loss"]

    # Get the final step for test marker placement
    final_step = df_combined['step'].max()

    # 2. Visualize Loss Evolution
    fig_loss = px.line(
        df_combined, 
        x="step", 
        y="loss", 
        color="split",
        title="Evolution of training and validation loss",
        labels={
            "loss": "Loss value", 
            "step": "Training step", 
            "split": "Dataset"
        },
        template="plotly_white"
    )


    fig_loss.update_layout(hovermode="x unified")
    fig_loss.update_layout(width=1200, height=600)
    fig_loss.write_html(save_path_loss)
    path = Path(save_path_loss)
    fig_loss.write_image(path.with_suffix(".svg"))

    # 3. Visualize Performance Metrics (F1, Precision, Recall)
    # Melt the dataframe to long format for metric-based coloring
    df_melted = df_combined.melt(
        id_vars=["step", "split"], 
        value_vars=performance_metrics, 
        var_name="metric", 
        value_name="score"
    )

    # Use line_dash for split and color for the metric type
    fig_perf = px.line(
        df_melted, 
        x="step", 
        y="score", 
        color="metric", 
        line_dash="split",
        title="Evolution of classification metrics (F1, precision, recall)",
        labels={
            "score": "Metric score", 
            "step": "Training step", 
            "metric": "Metric type"
        },
        template="plotly_white"
    )

    fig_perf.update_yaxes(range=[0, 1.05]) # Since metrics are usually [0, 1]
    fig_perf.update_layout(hovermode="x unified")
    fig_perf.update_layout(width=1200, height=600)
    fig_perf.write_html(save_path_evol)
    path = Path(save_path_evol)
    fig_perf.write_image(path.with_suffix(".svg"))

    print(f"Performance evolution saved to: {save_path_evol}", flush=True)

def visualize_confusion_matrix(
    metrics,
    save_path: str = None,
) -> None:
    """
    Two-panel interactive confusion matrix:
      - Left : FP co-occurrence heatmap (row = predicted subtype, col = true subtype)
               normalized by column (true count), i.e. false-positive rate per true label.
      - Right: Per-subtype performance bar chart (F1 / precision / recall).
    """
    m = metrics.compute_detailed()
    st_names   = [metrics.id_to_st[i] for i in range(metrics.num_subtypes)]
    n          = metrics.num_subtypes

    # FP matrix (n x n), zero diagonal
    fp_raw = m["fp_confusion"].numpy().copy()        # (n_pred, n_true)
    np.fill_diagonal(fp_raw, 0)

    # TP sits on the diagonal: predicted i AND true i
    tp_vec = m["tp_counts"].numpy()                  # (n,)
    fn_vec = m["fn_counts"].numpy()                  # (n,)

    # True positives on diagonal, FP off-diagonal → full "predicted × true" matrix
    full_raw = fp_raw.copy()
    for i in range(n):
        full_raw[i, i] = tp_vec[i]

    # Normalize by column (true label total = TP + FN)
    true_totals = tp_vec + fn_vec
    true_totals_safe = np.where(true_totals == 0, 1, true_totals)
    full_norm = full_raw / true_totals_safe[np.newaxis, :]

    # Log-transform for color mapping only (hover/text still use full_norm/full_raw)
    eps = 1e-4
    z_log = np.log10(full_norm + eps)
    z_min, z_max = np.log10(eps), 0  # 0 = log10(1)

    # Hover text: "Predicted X | True Y\nrate: 0.03\ncount: 1234"
    hover = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            label = "TP" if i == j else "FP"
            hover[i, j] = (
                f"Predicted: {st_names[i]}<br>"
                f"True: {st_names[j]}<br>"
                f"{label} rate: {full_norm[i, j]:.3f}<br>"
                f"Count: {int(full_raw[i, j]):,}"
            )

    # Figure
    fig = go.Figure()

    # Mask diagonal separately so it gets a different colorscale feel
    # We use a diverging-ish blue scale; diagonal TPs are visually distinct
    fig.add_trace(
        go.Heatmap(
            z=z_log,
            x=st_names,
            y=st_names,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale="Blues",
            zmin=z_min, zmax=z_max,
            colorbar=dict(
                title="Rate",
                tickvals=[np.log10(eps), -3, -2, -1, 0],
                ticktext=["0", "0.001", "0.01", "0.1", "1"],
                len=0.5, y=0.75,
                thickness=12,
            ),
            xgap=1, ygap=1,
        )
    )

    # Overlay diagonal boxes to highlight TPs visually
    for i, _ in enumerate(st_names):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color="#072C4B", width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )

    # Layout
    fig.update_layout(
        template="plotly_white",
        width=800,
        height=max(500, 30 * n + 150),
        margin=dict(t=80, l=100, r=40, b=80),
        title=dict(
            text=f"<b>Confusion matrix — {metrics.split.upper()} split (col-normalized by true label)</b>",
            font=dict(size=14),
        ),
    )

    fig.update_xaxes(title_text="True subtype", tickangle=45)
    fig.update_yaxes(title_text="Predicted subtype")
    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))

    print(f"Confusion matrix saved to: {save_path}", flush=True)

    return fig


def parse_fasta_headers(filepath):
    """
    Parse FASTA headers in format:
    >Ref.SUBTYPE.COUNTRY.YEAR.SAMPLE_NAME.ACCESSION
    e.g. >Ref.A.CH.03.HIV_CH_BID_V3538_2003.JQ403028
    Year can be 2-digit (03 → 2003) or 4-digit (2003).
    """
    subtype_data = defaultdict(list)
 
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                continue
 
            header = line[1:]
            parts = header.split(".")
 
            if len(parts) < 3:
                continue
 
            subtype = parts[1].strip()
            year_str = parts[3].strip()
 
            # Normalize 2-digit years
            if re.fullmatch(r"\d{2}", year_str):
                yy = int(year_str)
                # Heuristic: 00-26 → 2000s, 27-99 → 1900s
                year = 2000 + yy if yy <= 26 else 1900 + yy
            elif re.fullmatch(r"\d{4}", year_str):
                year = int(year_str)
            else:
                continue  # skip unparseable years
 
            subtype_data[subtype].append(year)
 
    return subtype_data


def plot_reference_distribution_with_year(subtype_data,
                                save_path=f"{workspace_path}/figs/subtype_distribution.html"):
    subtypes = sorted(subtype_data.keys())
    counts = [len(subtype_data[s]) for s in subtypes]

    BAR_COLOR = "#072C4B"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
    )

    # Top: total sequence counts, log scale
    fig.add_trace(
        go.Bar(
            x=subtypes,
            y=counts,
            marker_color=BAR_COLOR,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Sequences: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Bottom: dot per (subtype, year) present, sized by that year's count
    x_dots, y_dots, sizes, hover = [], [], [], []
    for s in subtypes:
        yr_counts = Counter(subtype_data[s])
        for yr, c in sorted(yr_counts.items()):
            x_dots.append(s)
            y_dots.append(yr)
            sizes.append(c)
            hover.append(f"<b>{s}</b><br>Year: {yr}<br>Sequences: {c}")

    max_c = max(sizes)
    marker_sizes = [4 + 10 * (c / max_c) ** 0.5 for c in sizes]

    fig.add_trace(
        go.Scatter(
            x=x_dots,
            y=y_dots,
            mode="markers",
            marker=dict(size=marker_sizes, color=BAR_COLOR, opacity=0.75, line=dict(width=0)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(t=30, b=60, l=70, r=30),
        height=650,
        width=1000,
    )

    fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,0.3)", linewidth=1, row=1, col=1)
    fig.update_xaxes(
        categoryorder="array", categoryarray=subtypes,
        showgrid=False, linecolor="rgba(0,0,0,0.3)", linewidth=1,
        title_text="Subtype", tickfont=dict(size=11, family="monospace"),
        row=2, col=1,
    )
    fig.update_yaxes(type="log", title_text="Sequences (log)",
                      gridcolor="rgba(0,0,0,0.07)", zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Year",
                      gridcolor="rgba(0,0,0,0.07)", zeroline=False, row=2, col=1)

    fig.write_html(save_path, include_plotlyjs="cdn")
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))
    print(f"\nPlot saved at {save_path} (.html/.svg)")

    return fig

def plot_fragment_length_distribution(
        frag_starts, frag_ends, ata_len, min_frag_len, path,
        ):
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

    # panel 1: fragment lengths
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

    # panel 2: start positions
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

    # build figure
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
    path = Path(path)
    fig.write_image(path.with_suffix(".svg"))

def plot_time_per_10k(csv_path, out_path=None):
    """
    Plot approximate processing time for 10,000 sequences per tool,
    extrapolated from measured time_sec / n_it.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: tool, time_sec, n_it
    out_path : str, optional
        If given, saves the figure (e.g. 'plot.pdf', 'plot.svg', 'plot.png').

    Returns
    -------
    plotly.graph_objects.Figure
    """
    bar_color = COLOR_SCHEME[0]
    df = pd.read_csv(csv_path)
    df["sec_per_seq"] = df["time_sec"] / df["n_it"]
    df["sec_per_10k"] = df["sec_per_seq"] * 10_000
    df = df.sort_values("sec_per_10k", ascending=True)

    # Human-readable labels for each bar
    def fmt_time(s):
        if s < 60:
            return f"{s:.1f} s"
        elif s < 3600:
            return f"{s/60:.1f} min"
        else:
            return f"{s/3600:.1f} h"

    labels = df["sec_per_10k"].apply(fmt_time)

    fig = go.Figure(
        go.Bar(
            x=df["sec_per_10k"],
            y=df["tool"],
            orientation="h",
            text=labels,
            textposition="outside",
            marker=dict(color=bar_color),
        )
    )

    fig.update_layout(
        template="simple_white",
        title="Approximate time to process 10,000 sequences",
        xaxis_title="Time (seconds, log scale)",
        yaxis_title="",
        xaxis_type="log",
        xaxis_range=[
            np.log10(df["sec_per_10k"].min()) - 0.3,
            np.log10(df["sec_per_10k"].max()) + 0.5,
        ],
        font=dict(size=12, color="black"),
        margin=dict(l=100, r=60, t=60, b=50),
        width=800,
        height=450,
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside")


    fig.write_html(out_path)
    path = Path(out_path)
    fig.write_image(path.with_suffix(".svg"))

def load_subtyping_csv(path, tool_name):
    """Load one tool's per-subtype confusion-matrix CSV into a tidy dataframe."""
    df = pd.read_csv(path, skiprows=1, header=None)
    df.columns = [
        "class", "n", "TP", "FP", "FN", "TN",
        "sensitivity", "sens_CI_lo", "sens_CI_hi",
        "specificity", "spec_CI_lo", "spec_CI_hi",
    ]
    df["Tool"] = tool_name
    return df


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion (in %). Not in the raw CSV, so we
    compute it ourselves for precision (PPV)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (centre - adj) / denom * 100, (centre + adj) / denom * 100


def add_precision(df):
    """Add PPV (precision, %) and its Wilson 95% CI to a tool dataframe."""
    df = df.copy()
    denom = df["TP"] + df["FP"]
    df["precision"] = np.where(denom > 0, df["TP"] / denom * 100, np.nan)
    ci = df.apply(lambda r: wilson_ci(r["TP"], r["TP"] + r["FP"]), axis=1)
    df["prec_CI_lo"] = [c[0] for c in ci]
    df["prec_CI_hi"] = [c[1] for c in ci]
    return df

def forest_grid(tool_dfs, tools, genes, save_path, metrics=("sensitivity", "precision"), order_by="n", subtypes=None):
    """
    tool_dfs : dict {f"{tool}_{gene}": dataframe}
    tools    : tool names, in display/legend/color order
    genes    : gene names, one column per gene
    metrics  : one row per metric
    subtypes : optional list of subtype labels to keep; None = plot every subtype found

    Returns rows=metrics x cols=genes, grouped bars (no CI) per tool per subtype.
    """

    n_colors = len(tools)
    palette = COLOR_SCHEME if n_colors <= len(COLOR_SCHEME) else pc.qualitative.Alphabet
    color = {tool: palette[i % len(palette)] for i, tool in enumerate(tools)}

    fig = make_subplots(
        rows=len(metrics), cols=len(genes), shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"{metric} - {gene}" for metric in metrics for gene in genes],
        vertical_spacing=0.1, horizontal_spacing=0.04,
    )

    for row, metric in enumerate(metrics, start=1):
        for col, gene in enumerate(genes, start=1):
            gene_tool_dfs = {tool: tool_dfs[f"{tool}_{gene}"] for tool in tools}
            combined = pd.concat(gene_tool_dfs.values(), ignore_index=True)
            n_per_subtype = combined.drop_duplicates("class").set_index("class")["n"]

            available = n_per_subtype.index.tolist()
            keep = [s for s in subtypes if s in available] if subtypes else available
            x_labels = [f"{s} - {n_per_subtype[s]}" for s in keep]

            for i, tool in enumerate(tools):
                df = gene_tool_dfs[tool].set_index("class")
                fig.add_trace(
                    go.Bar(
                        x=x_labels, y=df.loc[keep, metric].values,
                        marker_color=color[tool],
                        name=tool, legendgroup=tool, showlegend=(row == 1 and col == 1),
                    ),
                    row=row, col=col,
                )
            fig.update_yaxes(range=[0, 105], row=row, col=col)

    fig.update_layout(
            template="plotly_white",
            barmode="group",
            bargap=0.35,
            bargroupgap=0.05,
            height=350 * len(metrics), 
            width=450 * len(genes),
            legend_title_text="Tool",
            title="Subtyping tool comparison",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=80, t=70, b=40),
            legend=dict(
                x=0.13, y=0.01, xanchor="right", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
            ),
        )

    print(f"Grid metrics comparison saved at {save_path}")
    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))

def per_subtype_per_position_graph(tool_dfs, tools, save_path, metrics=("sensitivity", "precision"),
                 order_by=None, subtypes=None):
    """
    tool_dfs : dict {tool: dataframe}, one dataframe per tool, each with a "subtype"
               column and one column per metric (e.g. sensitivity, precision)
    tools    : tool names, in display/legend/color order
    metrics  : one row per metric
    order_by : optional column name in the dataframes to sort subtypes by (descending);
               None keeps the natural order of appearance
    subtypes : optional list of subtype labels to keep; None = plot every subtype found.
               Subtype "U" is always dropped, whether or not it's in this list.
 
    Returns rows=metrics x 1 col, grouped bars (no CI) per tool per subtype.
    Writes both an interactive HTML and a static SVG to save_path.
    """
    n_colors = len(tools)
    palette = COLOR_SCHEME if n_colors <= len(COLOR_SCHEME) else pc.qualitative.Alphabet
    color = {tool: palette[i % len(palette)] for i, tool in enumerate(tools)}
 
    combined = pd.concat(tool_dfs.values(), ignore_index=True)
    available = [s for s in combined["subtype"].unique() if s != "U"]
 
    keep = [s for s in subtypes if s in available] if subtypes else available
 
    if order_by and order_by in combined.columns:
        order_vals = combined.drop_duplicates("subtype").set_index("subtype")[order_by]
        keep = sorted(keep, key=lambda s: order_vals.get(s, 0), reverse=True)
 
    fig = make_subplots(
        rows=len(metrics), cols=1, shared_xaxes=True,
        subplot_titles=list(metrics),
        vertical_spacing=0.08,
    )
 
    for row, metric in enumerate(metrics, start=1):
        for tool in tools:
            df = tool_dfs[tool].set_index("subtype")
            fig.add_trace(
                go.Bar(
                    x=keep, y=df.reindex(keep)[metric].astype(float).values,
                    marker_color=color[tool],
                    name=tool, legendgroup=tool, showlegend=(row == 1),
                ),
                row=row, col=1,
            )
        fig.update_yaxes(range=[0, 1.05], row=row, col=1)
 
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        bargap=0.35,
        bargroupgap=0.05,
        height=350 * len(metrics),
        width=max(900, 45 * len(keep)),
        legend_title_text="Tool",
        title="Subtyping tool comparison",
        margin=dict(l=40, r=80, t=70, b=40),
        legend=dict(
            x=0.99, y=0.01, xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.5)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
        ),
    )
 
    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))
    print(f"Grid metrics comparison saved at {save_path}")

def pie_infections_sequences_dist(plot_df_path, xlsx_path, save_path):

    def get_global_2020_24_counts(xlsx_path, sheet_name="Table2"):
        """Pulls Table 2 / Global / 2020-24 and groups it like classify_subtype()."""
        t2 = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=2)
        t2['Region'] = t2['Region'].ffill()  # region name is only written on each block's first row
        row = t2[(t2['Region'] == 'Global') & (t2['Period'] == '2020-24')]
        if row.empty:
            raise ValueError("Could not find Global / 2020-24 in the Table2 sheet")
        row = row.iloc[0]

        direct_cols = {'A':'A','B':'B','C':'C','D':'D','F':'F','G':'G','H':'H','J':'J','K':'K','L':'L',
                    'CRF01_AE':'01_AE','CRF02_AG':'02_AG','CRF07_BC':'07_BC'}
        other_cols = ['Other CRFs', 'URFs', 'Unspecified recombinants']

        records = [{'Subtype Group': grp, 'Count': row[col]} for col, grp in direct_cols.items()]
        records.append({'Subtype Group': 'Other CRFs / URF', 'Count': sum(row[c] for c in other_cols)})
        counts_df = pd.DataFrame(records)

        base_order = ['A','B','C','D','F','G','H','J','K','L','O','N','P','01_AE','02_AG','07_BC']
        desired_order = ['Other CRFs / URF'] + base_order
        counts_df['Subtype Group'] = pd.Categorical(counts_df['Subtype Group'], categories=desired_order, ordered=True)
        return counts_df.sort_values('Subtype Group').dropna().reset_index(drop=True)

    def classify_subtype(st):
        st = str(st).strip().upper()
        sub_map = {'A1':'A','A2':'A','A3':'A','A4':'A','A6':'A','A7':'A','A8':'A','F1':'F','F2':'F'}
        if st in sub_map:
            st = sub_map[st]
        pure_clades = {'A','B','C','D','F','G','H','J','K','L','O','N','P','01_AE','02_AG','07_BC'}
        return st if st in pure_clades else 'Other CRFs / URF'

    # left panel: sequence-level data
    plot_df = pd.read_csv(plot_df_path, sep='\t')
    plot_df['pure_or_crf'] = plot_df['Subtype'].apply(classify_subtype)

    # Exclude 'O', 'N', 'P' from the dataframe
    plot_df = plot_df[~plot_df['pure_or_crf'].isin(['O', 'N', 'P'])]

    left_counts = plot_df['pure_or_crf'].value_counts().reset_index()
    left_counts.columns = ['Subtype Group', 'Count']

    # Updated base_order
    base_order = ['A','B','C','D','F','G','H','J','K','L','01_AE','02_AG','07_BC']
    desired_order = ['Other CRFs / URF'] + base_order

    left_counts['Subtype Group'] = pd.Categorical(left_counts['Subtype Group'], categories=desired_order, ordered=True)
    left_counts = left_counts.sort_values('Subtype Group').dropna().reset_index(drop=True)
    n_total_left = left_counts['Count'].sum()
    # right panel: Table2, Global, 2020-24, from Khalid et al 2026
    right_counts = get_global_2020_24_counts(xlsx_path)

    colour_map = ST_COLORS

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
        subplot_titles=(
            f"LANL HIV Database Sequences ({n_total_left:,} sequences)",
            "Global infections, 2020–24<br><sup>Khalid et al., 2026</sup>"
        )
    )

    fig.add_trace(go.Pie(
        labels=left_counts['Subtype Group'], values=left_counts['Count'], sort=False,
        marker=dict(colors=[colour_map.get(l, '#a6a6a6') for l in left_counts['Subtype Group']],
                    line=dict(color='#FFFFFF', width=0.5)),
        # texttemplate="<b>%{label}</b><br>%{value:,}<br>(%{percent})",
        texttemplate="<b>%{label}</b><br>(%{percent})",
        textposition='auto',
        hole=0.3,
        # insidetextorientation='auto',
        # rotation=140,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        legendgroup="subtype", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=right_counts['Subtype Group'], values=right_counts['Count'], sort=False,
        marker=dict(colors=[colour_map.get(l, '#a6a6a6') for l in right_counts['Subtype Group']],
                    line=dict(color='#FFFFFF', width=0.5)),
        texttemplate="<b>%{label}</b><br>%{percent}",
        textposition='auto',
        hole=0.3,
        # insidetextorientation='auto',
        # rotation=140,
        hovertemplate="<b>%{label}</b><br>Percentage: %{percent}<extra></extra>",
        legendgroup="subtype", showlegend=False,  # left panel's legend already covers every group
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title_text='<b>Distribution of HIV-1 subtypes</b>',
        legend=dict(title="<b>Sequence group</b>", orientation="v",
                    yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
        margin=dict(t=120, b=40, l=40, r=160),
        width=1500, height=650,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )

    fig.write_html(save_path)
    path = Path(save_path)
    fig.write_image(path.with_suffix(".svg"))
    print(f"Pie comparison saved at {save_path}")


def infection_map(xlsx_path, save_path, max_pie_diameter=0.28, min_pie_diameter=0.045):

    SUBTYPE_ORDER = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
                    '01_AE', '02_AG', '07_BC', 'Other CRFs / URF']

    # Region -> approximate centroid (lon, lat) used to place each pie.
    REGION_CENTROIDS = {
        'North America': (-100, 48),
        'Caribbean': (-64, 14),
        'Latin America': (-62, -18),
        'Western and central Europe': (10, 50),
        'Eastern Europe and central Asia': (60, 52),
        'South Asia': (78, 21),
        'Southeast Asia': (108, 8),
        'East Asia': (112, 33),
        'Oceania': (145, -27),
        'Middle East and north Africa': (25, 27),
        'West Africa': (-8, 10),
        'East Africa': (35, -1),
        'Ethiopia, Eritrea, and Djibouti': (42, 11),
        'Central Africa': (20, -3),
        'Southern Africa': (26, -27),
    }

    # Region -> countries (ISO-3) from Khalid et al appendix.

    REGION_COUNTRIES = {
        'North America': ['CAN', 'USA'],
        'Caribbean': ['ATG', 'ABW', 'BHS', 'BRB', 'BLZ', 'BMU', 'CYM', 'CUB',
                    'CUW', 'DMA', 'DOM', 'GRD', 'GLP', 'HTI', 'JAM', 'MTQ',
                    'PRI', 'SXM', 'KNA', 'LCA', 'VCT', 'TTO', 'TCA', 'VGB',
                    'VIR'],
        'Latin America': ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'CRI', 'ECU',
                        'SLV', 'GUF', 'GTM', 'GUY', 'HND', 'MEX', 'NIC',
                        'PAN', 'PRY', 'PER', 'SUR', 'URY', 'VEN'],
        'Western and central Europe': ['AND', 'AUT', 'BEL', 'BGR', 'HRV',
                                        'CYP', 'CZE', 'DNK', 'EST', 'FIN',
                                        'FRA', 'DEU', 'GIB', 'GRC', 'GRL',
                                        'HUN', 'ISL', 'IRL', 'IMN', 'ISR',
                                        'ITA', 'LVA', 'LIE', 'LTU', 'LUX',
                                        'MLT', 'MCO', 'NLD', 'NOR', 'POL',
                                        'PRT', 'ROU', 'SMR', 'SRB', 'SVK',
                                        'SVN', 'ESP', 'SWE', 'CHE', 'TUR',
                                        'GBR'],
        'Eastern Europe and central Asia': ['ALB', 'ARM', 'AZE', 'BLR', 'BIH',
                                            'GEO', 'KAZ', 'XKX', 'KGZ', 'MDA',
                                            'MNE', 'MKD', 'RUS', 'TJK', 'TKM',
                                            'UKR', 'UZB'],
        'South Asia': ['AFG', 'BGD', 'BTN', 'IND', 'IRN', 'MDV', 'NPL', 'PAK',
                        'LKA'],
        'Southeast Asia': ['BRN', 'KHM', 'IDN', 'LAO', 'MYS', 'MMR', 'PHL',
                            'SGP', 'THA', 'TLS', 'VNM'],
        'East Asia': ['CHN', 'HKG', 'JPN', 'MNG', 'PRK', 'KOR', 'TWN'],
        'Oceania': ['ASM', 'AUS', 'FJI', 'PYF', 'GUM', 'KIR', 'FSM', 'NRU',
                    'NCL', 'NZL', 'MNP', 'PLW', 'PNG', 'WSM', 'SLB', 'TON',
                    'TUV', 'VUT'],
        'Middle East and north Africa': ['DZA', 'BHR', 'EGY', 'IRQ', 'JOR',
                                        'KWT', 'LBN', 'LBY', 'MAR', 'OMN',
                                        'PSE', 'QAT', 'SAU', 'SDN', 'SYR',
                                        'TUN', 'ARE', 'YEM'],
        'West Africa': ['BEN', 'BFA', 'CPV', 'CMR', 'CIV', 'GMB', 'GHA',
                        'GIN', 'GNB', 'LBR', 'MLI', 'MRT', 'NER', 'NGA',
                        'STP', 'SEN', 'SLE', 'TGO'],
        'East Africa': ['BDI', 'COM', 'KEN', 'MDG', 'MUS', 'RWA', 'SYC',
                        'SOM', 'SSD', 'TZA', 'UGA'],
        'Ethiopia, Eritrea, and Djibouti': ['DJI', 'ERI', 'ETH'],
        'Central Africa': ['AGO', 'CAF', 'TCD', 'COD', 'GNQ', 'GAB', 'COG'],
        'Southern Africa': ['BWA', 'SWZ', 'LSO', 'MWI', 'MOZ', 'NAM', 'ZAF',
                            'ZMB', 'ZWE'],
    }

    LON_RANGE = (-170, 190)
    LAT_RANGE = (-58, 82)
    GEO_DOMAIN = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}


    def lonlat_to_domain(lon, lat):
        # Equirectangular: linear map (lon, lat) to the geo subplot's (x, y) domain fraction.
        fx = (lon - LON_RANGE[0]) / (LON_RANGE[1] - LON_RANGE[0])
        fy = (lat - LAT_RANGE[0]) / (LAT_RANGE[1] - LAT_RANGE[0])
        x0, x1 = GEO_DOMAIN['x']
        y0, y1 = GEO_DOMAIN['y']
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    def load_table1_plhiv_2020_24(xlsx_path):
        """{region: number of people living with HIV, 2020-24} from Table1."""
        t1 = pd.read_excel(xlsx_path, sheet_name='Table1', header=None)
        header_row = next(i for i, v in t1.iloc[:, 0].items()
                        if isinstance(v, str) and v.strip() == 'Period')
        regions = t1.iloc[header_row, 1:17].tolist()
        row_idx = next(i for i in t1.index
                        if i > header_row and t1.iloc[i, 0] == '2020-24')
        values = t1.iloc[row_idx, 1:17].tolist()
        return dict(zip(regions, values))


    def load_table2_region_period(xlsx_path, region, period):
        """{subtype_group: fraction} for one region/period from Table2."""
        t2 = pd.read_excel(xlsx_path, sheet_name='Table2', header=2)
        t2['Region'] = t2['Region'].ffill()
        row = t2[(t2['Region'] == region) & (t2['Period'] == period)].iloc[0]
        direct = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'F': 'F', 'G': 'G',
                'H': 'H', 'J': 'J', 'K': 'K', 'L': 'L',
                'CRF01_AE': '01_AE', 'CRF02_AG': '02_AG', 'CRF07_BC': '07_BC'}
        other_cols = ['Other CRFs', 'URFs', 'Unspecified recombinants']
        out = {grp: row[col] for col, grp in direct.items()}
        out['Other CRFs / URF'] = sum(row[c] for c in other_cols)
        return out

    plhiv = load_table1_plhiv_2020_24(xlsx_path)
    max_plhiv = max(plhiv[r] for r in REGION_CENTROIDS)

    def pie_diameter(region):
        frac = plhiv[region] / max_plhiv
        return min_pie_diameter + (max_pie_diameter - min_pie_diameter) * math.sqrt(frac)

    fig = go.Figure()
    labels = SUBTYPE_ORDER

    # world map, region shading
    region_names = list(REGION_COUNTRIES.keys())
    region_index = {r: i for i, r in enumerate(region_names)}
    locations, z = [], []
    for region, isos in REGION_COUNTRIES.items():
        for iso in isos:
            locations.append(iso)
            z.append(region_index[region])
    region_palette = [
        # Grey-blues
        '#9ea8b6', '#b0b8c4', '#c2c9d2', '#6b778d', '#8693a4',
        # Neutral greys
        '#adb5bd', '#cfd4da', '#8d99ae', '#6c757d', '#d6dbdf',
        # Grey-pinks
        # '#d8b4b8', '#c79d9e', '#e2c2c6', '#b5838d', '#e9d5d8'
    ]
    n = len(region_names)
    colorscale = []
    for i in range(n):
        colorscale.append([i / n, region_palette[i % len(region_palette)]])
        colorscale.append([(i + 1) / n, region_palette[i % len(region_palette)]])

    fig.add_trace(go.Choropleth(
            locations=locations, z=z, locationmode='ISO-3',
            colorscale=colorscale, zmin=0, zmax=n, showscale=False,
            marker_line_color='white', marker_line_width=0.3,
            hoverinfo='skip', geo='geo',
        ))

    # one pie per region, sized by PLHIV, placed on the map
    for region, (lon, lat) in REGION_CENTROIDS.items():
        x, y = lonlat_to_domain(lon, lat)
        d = pie_diameter(region)
        vals_dict = load_table2_region_period(xlsx_path, region, '2020-24')
        vals = [vals_dict[g] for g in labels]
        fig.add_trace(go.Pie(
            labels=labels, values=vals, sort=False,
            opacity=0.8,  # Sets transparency for pie charts (0.0 = fully transparent, 1.0 = fully opaque)
            domain={'x': [max(x - d / 2, 0), min(x + d / 2, 1)],
                    'y': [max(y - d / 2, 0), min(y + d / 2, 1)]},
            marker=dict(colors=[ST_COLORS.get(g, '#a6a6a6') for g in labels], line=dict(color='white', width=0.1)),
            textinfo='none',
            hovertemplate=f'<b>{region}</b><br>' + '%{label}: %{percent}<extra></extra>',
            legendgroup='subtype', showlegend=(region == 'Southern Africa'), name=region,
        ))
        fig.add_annotation(
            x=x, y=min(y + d / 2 + 0.015, 1.0),
            xref='paper', yref='paper',
            text=f'<b>{region}</b>',  # Made text bold for extra clarity
            showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=10, color='black'),  # Increased font size and explicitly forced pure black
        )

    fig.update_layout(
        template='plotly_white', width=1400, height=900,
        margin=dict(t=60, b=20, l=20, r=200),
        legend=dict(title='<b>HIV-1 variant</b>', x=1.02, y=0.5, yanchor='middle'),
        geo=dict(
            domain=GEO_DOMAIN, projection_type='equirectangular',
            lonaxis=dict(range=list(LON_RANGE)), lataxis=dict(range=list(LAT_RANGE)),
            showcountries=True, countrycolor='white',
            showframe=False, showcoastlines=False, bgcolor='rgba(0,0,0,0)',
        ),
        title=dict(text='<b>Regional distribution of HIV-1 variants, 2020-24</b>', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )

    fig.write_html(save_path)
    path = Path(save_path)
    # fig.write_image(path.with_suffix('.svg'))
    # Save as JSON format and convert to svg locally
    with open(path.with_suffix('.json'), 'w') as f:
        f.write(fig.to_json())
    print(f'Map saved at {save_path}')
















if __name__ == "__main__":
    breakpoints_path = f"{workspace_path}/data/output/lanl_crf_breakpoints_hxb2.csv"
    df_bp = pd.read_csv(breakpoints_path)

    df_bp.rename(columns={
        'position' : 'pos'
    }, inplace=True)
    df_bp.sort_values(
        by=['crf'],
        key=lambda s: s.str.extract(r'^(\d+)', expand=False).astype(int),
        inplace=True,
        ascending=False
    )
        
    visualize_breakpoints(
        df_bp,
        save_path=f"{workspace_path}/figs/breakpoint_distribution_with_genes.html")
    
    st_to_seq_dict = defaultdict(list)
    ref_fasta_path = (f"{workspace_path}/data/output/HIV1_PURE_REF.fasta")
    hxb2_ata_seq = ""
    for i, rec in enumerate(SeqIO.parse(ref_fasta_path, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            st_to_seq_dict[rec.id.split(".")[1]].append(str(rec.seq))

    print(f"Parsing {ref_fasta_path} ...")
    subtype_data = parse_fasta_headers(ref_fasta_path)
 
    if not subtype_data:
        print("No valid headers found. Check that headers follow the format:")
        print("  >Ref.SUBTYPE.COUNTRY.YEAR.SAMPLE.ACCESSION")
        sys.exit(1)
 
    total = sum(len(v) for v in subtype_data.values())
    print(f"Found {total} sequences across {len(subtype_data)} subtypes:")
    for s in sorted(subtype_data):
        yrs = subtype_data[s]
        print(f"  {s:12s}  n={len(yrs):4d}  [{min(yrs)} – {max(yrs)}]")

    save_path_ref_dist = f"{workspace_path}/figs/reference_subtype_distribution_with_year.html"

    fig = plot_reference_distribution_with_year(
        subtype_data,
        save_path=save_path_ref_dist)

    # rate array for diversity
    ata_to_hxb2, hxb2_to_ata = config.build_hxb2_ata_maps(hxb2_ata_seq)
    subtypes_with_data = [
        'A',
        'B', 
        'C', 'D', 'AE', 'F', 'G']
    names = subtypes_with_data + ['avg']
    diversity_arrays = {}
    
    for name in subtypes_with_data:
        rate_path = f"{workspace_path}/data/output/rates/HIV1_{name}_ALIGNED.fasta.rate"
        diversity_array = parse_iqtree_rates(rate_path, ATA_LEN, use_classes=True)
        diversity_arrays[name] = diversity_array

    # Calculate the element-wise average across all collected subtype arrays
    diversity_arrays['avg'] = np.mean(
        [diversity_arrays[name] for name in subtypes_with_data], 
        axis=0
    )

    visualize_diversity(
        diversity_arrays,
        hxb2_to_ata,
        st_to_print=names,
        types_to_print=['raw', 'smoothed'],
        window_size=100, 
        save_path=f"{workspace_path}/figs/empirical_diversity.html"
    )

    visualize_diversity(
        diversity_arrays,
        hxb2_to_ata,
        st_to_print=['avg'],
        types_to_print=['smoothed'],
        window_size=100, 
        save_path=f"{workspace_path}/figs/empirical_diversity_avg_smoothed.html"
    )
    
    save_path_loss = f"{workspace_path}/figs/loss_evolution.html"
    save_path_evol = f"{workspace_path}/figs/metrics_evolution.html"
    visualize_metrics(save_path_loss=save_path_loss,
                      save_path_evol=save_path_evol)

    processing_times = f"{workspace_path}/data/speed_test/processing_times.csv"
    save_path_time = f"{workspace_path}/figs/processing_time_per_10k.html"
    plot_time_per_10k(processing_times, save_path_time)

    # subtyping results comparison
    tools=['comet', 'regav3', 'sbtr']
    genes=['full', 'prot']
    base_path = f"{workspace_path}/data/input_sequences/regav3_testset"

    tool_paths = {
        f"{tool}_{gene}": f"{base_path}/{tool}_{gene}_results.csv" 
        for tool, gene in product(tools, genes)
    }
    subtypes=['A', 'B', 'C', '01_AE', '02_AG', '06_cpx', '07_BC', '12_BF', '14_BG']
    save_path = f"{workspace_path}/figs/metrics_comparison_grid.html"

    # add 'add_precision' if needed
    tool_dfs = {
        name: load_subtyping_csv(path, name)
        for name, path in tool_paths.items()
    }
    tool_dfs = {
        name: (df if "precision" in df.columns else add_precision(df))
        for name, df in tool_dfs.items()
    }

    for name, df in tool_dfs.items():
        if name == 'sbtr_prot':
            print(df)

    forest_grid(
        tool_dfs, tools=tools, genes=genes, save_path=save_path,
        metrics=["sensitivity", "precision"], subtypes=subtypes)


    path_to_results=f'/pasteur/helix/projects/mPath/oanoufa/sbtr/data/output/test/syn_v{VERSION}/matching_scores_per_subtype.csv'
    df = pd.read_csv(path_to_results)

    tool_dfs = {
        "sbtr": df[df["methods_compared"] == "sbtr_True"],
        "jpHMM": df[df["methods_compared"] == "jpHMM_True"],
    }

    save_path=f"{workspace_path}/figs/per_subtype_per_position_comparison.html"
    per_subtype_per_position_graph(
        tool_dfs,
        tools=["sbtr", "jpHMM"],
        save_path=save_path,
        metrics=("sensitivity", "precision"),
    )

    lanl_full_md_path='/pasteur/helix/projects/mPath/oanoufa/sbtr/data/input_sequences/lanl_full/metadata_FULL.tsv'
    khalid_et_al_data='/pasteur/helix/projects/mPath/oanoufa/sbtr/data/input_sequences/lanl_full/Khalid_2026_HIV1_Tables1_2.xlsx'
    save_path=f"{workspace_path}/figs/pie_subtype_infections.html"
    pie_infections_sequences_dist(
        plot_df_path=lanl_full_md_path,
        xlsx_path=khalid_et_al_data,
        save_path=save_path)

    save_path=f"{workspace_path}/figs/infection_map.html"
    infection_map(
        xlsx_path=khalid_et_al_data,
        save_path=save_path,
        max_pie_diameter=0.25,
        min_pie_diameter=0.03)
