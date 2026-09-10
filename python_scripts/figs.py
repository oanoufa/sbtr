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
from itertools import groupby
import plotly.io as pio
import matplotlib.patches as mpatches
pio.defaults.default_format = "png"
from src import config
from collections import Counter


workspace_path = config.WORKSPACE_PATH

GENES_RAW = config.GENES_RAW
GENE_COLORS = config.GENE_COLORS
COLOR_SCHEME = config.COLOR_SCHEME
ST_COLORS = config.ST_COLORS

JPHMM_TO_SBTR_LTR = {
    "5'-Insertion":"5'LTR",
    "3'-Insertion":"3'LTR",
}


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

    # Download the figure as a high quality PNG image
    # Set image length and width to 1200x800 for better quality
    fig.update_layout(width=1200, height=500)
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        elif save_path.endswith('.png'):
            fig.write_image(save_path, scale=2)
        else:
            png_path = save_path + '.png'
            html_path = save_path + '.html'
            fig.write_html(html_path)
            fig.write_image(png_path, scale=2)
        print(f"Breakpoints visualization saved to: {save_path}", flush=True)


def visualize_diversity(
    diversity_arrays: dict,
    hxb2_to_ata: np.ndarray,
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
            x=-0.02, y=frame_pos_axis[frame],
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

    # LTR boundaries
    max_5 = hxb2_to_ata[max(config.START_5LTR)]
    min_3 = hxb2_to_ata[min(config.NEF_3LTR)]
    max_3 = ata_len

    for i, (name, diversity_array) in enumerate(diversity_arrays.items()):
        color          = trace_colors[i % len(trace_colors)]
        # Zero rates in LTR regions
        diversity_array[0:max_5] = 1e-10
        diversity_array[min_3:max_3] = 1e-10
        diversity_array /= diversity_array.sum()
        smoothed_rates = np.convolve(diversity_array, kernel, mode='same')
        mean_rate      = np.mean(diversity_array)

        fig.add_trace(go.Scatter(
            x=x_positions, y=diversity_array,
            mode='lines', line=dict(color=color, width=1),
            opacity=0.3, name=f'{name} raw',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_positions, y=smoothed_rates,
            mode='lines', line=dict(color=color, width=2),
            name=f'{name} smoothed',
        ), row=1, col=1)

    fig.add_hline(
        y=mean_rate, line_dash="dash", line_color='#000000', opacity=0.8,
        annotation_text=f"mean: {mean_rate:.3f}",
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

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        elif save_path.endswith('.png'):
            fig.write_image(save_path, scale=2)
        else:
            fig.write_html(save_path + '.html')
            fig.write_image(save_path + '.png', scale=2)
        print(f"Diversity visualization saved to: {save_path}", flush=True)

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

    # Only load this one sample from disk.
    loss_mask = np.asarray(ploss_slice)
    labels = np.asarray(preds_slice)

    real_loss_mask = loss_mask.astype(bool)
    n_real = int(real_loss_mask.sum())
    n_total = len(real_loss_mask)

    id_to_st = {v: k for k, v in pure_st_to_id_dict.items()}
    subtype_names = [
        id_to_st[i]
        for i in range(len(pure_st_to_id_dict))
    ]
    n_subtypes = len(subtype_names)

    # (n_subtypes, n_total)
    full_labels = labels.T

    # Figure setup
    gene_track_h = 2.4

    fig = plt.figure(
        figsize=(14, 2 + 2 * gene_track_h + 0.3 * n_subtypes)
    )

    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1, gene_track_h, n_subtypes, gene_track_h],
        width_ratios=[40, 1],
        hspace=0.10,
        wspace=0.03,
    )

    ax_mask = fig.add_subplot(gs[0, 0])
    ax_gene = fig.add_subplot(gs[1, 0], sharex=ax_mask)
    ax_lab = fig.add_subplot(gs[2, 0], sharex=ax_mask)
    ax_st = fig.add_subplot(gs[3, 0], sharex=ax_mask)
    ax_cb = fig.add_subplot(gs[2, 1])

    frame_lanes = {
        3: (0.67, 1.0),
        2: (0.33, 0.66),
        1: (0.0, 0.33),
    }

    # Row 0: loss mask
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

    # Row 1: Gene track
    ax_gene.set_xlim(0, n_total)
    ax_gene.set_ylim(0, 1)
    ax_gene.axis("off")

    gene_bars = defaultdict(list)

    for gene, (start_hxb2, end_hxb2, frame) in GENES_RAW.items():
        start_ata = hxb2_to_ata[start_hxb2]
        end_ata = hxb2_to_ata[end_hxb2]

        y0, y1 = frame_lanes[frame]
        color = GENE_COLORS.get(gene, "grey")
        width = max(end_ata - start_ata, 1)

        gene_bars[
            (color, y0 + 0.02, (y1 - y0) - 0.04)
        ].append((start_ata, width))

        if width > n_total * 0.025:
            ax_gene.text(
                start_ata + width / 2,
                (y0 + y1) / 2,
                gene,
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    fc="white",
                    ec="none",
                    alpha=0.6,
                ),
            )

    for (color, y_bottom, height), xranges in gene_bars.items():
        ax_gene.broken_barh(
            xranges,
            (y_bottom, height),
            facecolors=color,
            alpha=0.40,
            linewidths=0,
        )

    for frame, (y0, y1) in frame_lanes.items():
        ax_gene.text(
            -n_total * 0.005,
            (y0 + y1) / 2,
            f"F{frame}",
            ha="right",
            va="center",
            fontsize=7,
            color="grey",
        )

    # Row 2: Subtype probability heatmap
    im = ax_lab.imshow(
        full_labels,
        aspect="auto",
        cmap="viridis",
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

    subtype_arr = np.full(
        n_total,
        fill_value="",
        dtype=object,
    )

    for r_start, r_end, subtype in regions_aligned:
        if subtype in JPHMM_TO_SBTR_LTR.keys():
            subtype = JPHMM_TO_SBTR_LTR[subtype]

        s = max(0, int(r_start))
        e = min(n_total, int(r_end))

        if s < e:
            subtype_arr[s:e] = str(subtype)

    frame_to_ranges = defaultdict(list)

    for gene, (start_hxb2, end_hxb2, frame) in GENES_RAW.items():
        s_ata = int(hxb2_to_ata[start_hxb2])
        e_ata = int(hxb2_to_ata[end_hxb2])
        frame_to_ranges[frame].append((s_ata, e_ata))

    pad = 0.015
    st_bars = defaultdict(list)

    for frame, gene_ranges in frame_to_ranges.items():
        y0, y1 = frame_lanes[frame]
        rect_h = (y1 - y0) - 2 * pad

        for g_s, g_e in gene_ranges:
            g_s = max(0, g_s)
            g_e = min(n_total, g_e)

            if g_s >= g_e:
                continue

            pos = g_s

            for val, grp in groupby(subtype_arr[g_s:g_e]):
                run_len = sum(1 for _ in grp)

                if val:
                    st_color = ST_COLORS.get(
                        val,
                        "#CCCCCC",
                    )

                    st_bars[
                        (st_color, y0 + pad, rect_h)
                    ].append((pos, run_len))

                pos += run_len

        ax_st.text(
            -n_total * 0.005,
            (y0 + y1) / 2,
            f"F{frame}",
            ha="right",
            va="center",
            fontsize=7,
            color="grey",
        )

    for (st_color, y_bottom, height), xranges in st_bars.items():
        ax_st.broken_barh(
            xranges,
            (y_bottom, height),
            facecolors=st_color,
            alpha=0.88,
            linewidths=0,
        )

    present_subtypes = sorted(
        st for st in set(subtype_arr)
        if st != ""
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
            title="Subtype(s)",
            title_fontsize=8,
            fontsize=7,
            ncol=min(len(legend_handles), 6),
            loc="upper left",
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

    fig.savefig(
        path,
        dpi=100,
    )

    plt.close(fig)


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

    METRICS_DIR = config.MODEL_CONFIG["metrics_dir"]
    VERSION = config.VERSION

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
    if save_path_loss.endswith('.html'):
        fig_loss.write_html(save_path_loss)
    elif save_path_loss.endswith('.png'):
        fig_loss.write_image(save_path_loss, scale=2)
    else:
        fig_loss.write_html(save_path_loss + '.html')
        fig_loss.write_image(save_path_loss + '.png', scale=2)
        print(f"Loss visualization saved to: {save_path_loss}", flush=True)


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
    if save_path_evol.endswith('.html'):
        fig_perf.write_html(save_path_evol)
    elif save_path_evol.endswith('.png'):
        fig_perf.write_image(save_path_evol, scale=2)
    else:
        fig_perf.write_html(save_path_evol + '.html')
        fig_perf.write_image(save_path_evol + '.png', scale=2)
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

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        elif save_path.endswith(".png"):
            fig.write_image(save_path, scale=2)
        else:
            fig.write_html(save_path + ".html")
            fig.write_image(save_path + ".png", scale=2)
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
    print(f"\nPlot saved at {save_path} (.html/.png/.pdf)")

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


if __name__ == "__main__":
    breakpoints_path = f"{workspace_path}/data/output/lanl_crf_breakpoints.csv"
    df_bp = pd.read_csv(breakpoints_path)

    df_bp.rename(columns={
        'position' : 'pos'
    }, inplace=True)
    df_bp.sort_values(by=['crf'],
                    key=lambda s: s.str.extract(r'CRF(\d+)', expand=False).astype(int),
                    inplace=True,
                    ascending=False)
    
    visualize_breakpoints(df_bp,
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

    fig = plot_reference_distribution_with_year(subtype_data,
                                      save_path=save_path_ref_dist)

    # rate array for diversity
    ata_to_hxb2, hxb2_to_ata = config.build_hxb2_ata_maps(hxb2_ata_seq)
    subtypes_with_data = ['A', 'B', 'C', 'D', 'AE', 'F', 'G']
    names = subtypes_with_data + ['avg']
    diversity_arrays = {}
    for name in names:
        rate_array_path = f"{workspace_path}/data/input/diversity/site_rates_{name}.npy"
        diversity_array = np.load(rate_array_path)
        diversity_arrays[name] = diversity_array

    visualize_diversity(
        diversity_arrays,
        hxb2_to_ata,
        window_size=100, 
        save_path=f"{workspace_path}/figs/empirical_diversity.html"
    )
    
    save_path_loss = f"{workspace_path}/figs/loss_evolution.html"
    save_path_evol = f"{workspace_path}/figs/metrics_evolution.html"
    visualize_metrics(save_path_loss=save_path_loss,
                      save_path_evol=save_path_evol)

    processing_times = f"{workspace_path}/data/processing_times.csv"
    save_path_time = f"{workspace_path}/figs/processing_time_per_10k.html"
    plot_time_per_10k(processing_times, save_path_time)
