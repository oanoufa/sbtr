import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotly.subplots import make_subplots
from Bio import SeqIO
import pandas as pd
import sys
import os
from collections import defaultdict
import plotly.io as pio
pio.defaults.default_format = "png"
print(pio.kaleido.scope, flush=True)
import config

from utils import build_hxb2_ata_maps


workspace_path = config.WORKSPACE_PATH

color_scheme = ['#072C4B', '#F28089', '#71cddd']



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

    genes_raw_data = {
        # LTR
        "5'LTR":   (1, 790, 1),
        "3'LTR":   (9417, 9719, 2),

        # Gag Subunits
        "p17":     (790, 1186, 1),
        "p24":     (1186, 1879, 1),
        "p2":      (1879, 1921, 1),
        "p7":      (1921, 2086, 1),
        "p1":      (2086, 2134, 1),
        "p6":      (2134, 2292, 1),
        
        # Pol Subunits (Frame 3)
        "prot":    (2085, 2550, 3),
        "p51_RT":  (2550, 3870, 3),
        "p15":     (3870, 4230, 3),
        "p31_int": (4230, 5096, 3),
        
        # Env Subunits (Frame 3)
        "gp120":   (6225, 7758, 3),
        "gp41":    (7758, 8795, 3),
        
        # Accessory/Regulatory Genes
        "vif":     (5041, 5619, 1),
        "vpr":     (5559, 5850, 3),
        "vpu":     (6062, 6310, 2),
        "nef":     (8797, 9417, 1),
        
        # Split Spliced Genes
        "tat1":    (5831, 6045, 2),
        "tat2":    (8379, 8469, 1),
        "rev1":    (5970, 6045, 3),
        "rev2":    (8379, 8653, 2),
    }

    # The restored color palette
    line_colors = {
        # Grey for LTRs
        "5'LTR": "#7f7f7f", "3'LTR": "#7f7f7f",
        "p17": "#1f77b4", "p24": "#ff7f0e", "p2": "#2ca02c", "p7": "#d62728", "p1": "#9467bd", "p6": "#8c564b",
        "prot": "#e377c2", "p51_RT": "#7f7f7f", "p15": "#bcbd22", "p31_int": "#17becf",
        "gp120": "#aec7e8", "gp41": "#ffbb78",
        "vif": "#98df8a", "vpr": "#ff9896", "vpu": "#c5b0d5", "nef": "#c49c94",
        "tat1": "#f7b6d2", "tat2": "#f7b6d2", "rev1": "#dbdb8d", "rev2": "#dbdb8d"
    }

    for i, (gene, (start, end, frame)) in enumerate(genes_raw_data.items()):
        y0, y1 = frame_lanes[frame]
        color = line_colors.get(gene, "grey")
        
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
        print(f"Mutation rate visualization saved to: {save_path}", flush=True)


def visualize_mutation_rates(
    rate_arrays: dict,  # name -> rate_array
    ata_to_hxb2,
    window_size=100,
    save_path=f"{workspace_path}/figs/mutation_rate_profile.html"):
    """
    Visualizes empirical mutation rate arrays with HIV genes in the background.
    rate_arrays: dict of {name: rate_array}, all must have the same length.
    """
    # Use first array for reference length/mean
    first_array = next(iter(rate_arrays.values()))
    ata_len = len(first_array)
    x_positions = np.arange(ata_len)
    y_max = max(arr.max() for arr in rate_arrays.values()) * 1.1

    kernel = np.ones(window_size) / window_size

    # Build inverse map: HXB2 -> ATA
    hxb2_to_ata = {}
    for ata_idx, hxb2_idx in enumerate(ata_to_hxb2):
        if hxb2_idx > 0 and hxb2_idx not in hxb2_to_ata:
            hxb2_to_ata[hxb2_idx] = ata_idx

    fig = make_subplots(
        rows=1, cols=1,
        row_heights=[1.0],
        subplot_titles=("<b>Empirical mutation rate across the genome</b>",)
    )

    # =========================================================================
    # GENE MAP BACKGROUND (unchanged)
    # =========================================================================
    genes_raw_data = {
        "5'LTR":   (1, 790, 1),    "3'LTR":   (9417, 9719, 2),
        "p17":     (790, 1186, 1), "p24":     (1186, 1879, 1),
        "p2":      (1879, 1921, 1),"p7":      (1921, 2086, 1),
        "p1":      (2086, 2134, 1),"p6":      (2134, 2292, 1),
        "prot":    (2085, 2550, 3),"p51_RT":  (2550, 3870, 3),
        "p15":     (3870, 4230, 3),"p31_int": (4230, 5096, 3),
        "gp120":   (6225, 7758, 3),"gp41":    (7758, 8795, 3),
        "vif":     (5041, 5619, 1),"vpr":     (5559, 5850, 3),
        "vpu":     (6062, 6310, 2),"nef":     (8797, 9417, 1),
        "tat1":    (5831, 6045, 2),"tat2":    (8379, 8469, 1),
        "rev1":    (5970, 6045, 3),"rev2":    (8379, 8653, 2),
    }

    line_colors = {
        "5'LTR": "#7f7f7f", "3'LTR": "#7f7f7f",
        "p17": "#1f77b4", "p24": "#ff7f0e", "p2": "#2ca02c", "p7": "#d62728", "p1": "#9467bd", "p6": "#8c564b",
        "prot": "#e377c2", "p51_RT": "#7f7f7f", "p15": "#bcbd22", "p31_int": "#17becf",
        "gp120": "#aec7e8", "gp41": "#ffbb78",
        "vif": "#98df8a", "vpr": "#ff9896", "vpu": "#c5b0d5", "nef": "#c49c94",
        "tat1": "#f7b6d2", "tat2": "#f7b6d2", "rev1": "#dbdb8d", "rev2": "#dbdb8d"
    }

    frame_lanes = {
        3: (0, y_max * 0.32),
        2: (y_max * 0.33, y_max * 0.64),
        1: (y_max * 0.65, y_max * 0.95)
    }

    for i, (gene, (start_hxb2, end_hxb2, frame)) in enumerate(genes_raw_data.items()):
        start_ata = hxb2_to_ata.get(start_hxb2, 0)
        end_ata   = hxb2_to_ata.get(end_hxb2, ata_len - 1)
        y0, y1    = frame_lanes[frame]
        color     = line_colors.get(gene, "grey")

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
            x=-0.02, y=(y0 + y1) / 2,
            xref="paper", yref="y",
            text=f"<b>F{frame}</b>", showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="right",
        )

    # =========================================================================
    # ONE RAW + SMOOTHED TRACE PAIR PER RATE ARRAY
    # =========================================================================
    # Plotly default color cycle
    trace_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for i, (name, rate_array) in enumerate(rate_arrays.items()):
        color          = trace_colors[i % len(trace_colors)]
        smoothed_rates = np.convolve(rate_array, kernel, mode='same')
        mean_rate      = np.mean(rate_array)

        fig.add_trace(go.Scatter(
            x=x_positions, y=rate_array,
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
    fig.update_yaxes(title_text="Mutation Rate", range=[0, y_max], row=1, col=1)

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        elif save_path.endswith('.png'):
            fig.write_image(save_path, scale=2)
        else:
            fig.write_html(save_path + '.html')
            fig.write_image(save_path + '.png', scale=2)
        print(f"Mutation rate visualization saved to: {save_path}", flush=True)


def visualize_sample(
    sample: dict,
    pure_st_to_id_dict: dict,
    idx: int = 0,
    path: str = None,
) -> None:
    attention_mask = sample["attention_mask"].numpy()
    loss_mask      = sample["loss_mask"].numpy()
    labels         = sample["labels"].numpy()
    if hasattr(attention_mask, 'ndim') and attention_mask.ndim > 1:
        attention_mask = attention_mask[0]
        loss_mask      = loss_mask[0]
        labels         = labels[0]

    real_mask      = attention_mask.astype(bool)
    real_loss_mask = loss_mask.astype(bool)
    mask_used      = real_loss_mask
    n_real         = mask_used.sum()
    n_total        = len(mask_used)

    # ── Build full-length label array (NaN where masked out) ──────────────
    id_to_st      = {v: k for k, v in pure_st_to_id_dict.items()}
    subtype_names = [id_to_st[i] for i in range(len(pure_st_to_id_dict))]
    n_subtypes    = len(subtype_names)

    # full_labels = np.full((n_subtypes, n_total), np.nan)          # NaN = masked
    # full_labels[:, mask_used] = labels[mask_used].T               # fill real positions
    full_labels = labels.T

    # ── Layout: 2 rows × 2 cols, right col is narrow colorbar ────────────
    fig = plt.figure(figsize=(14, 4 + 0.3 * n_subtypes))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[1, n_subtypes],
        width_ratios=[40, 1],          # main panels | colorbar
        hspace=0.15,
        wspace=0.03,
    )

    ax_mask = fig.add_subplot(gs[0, 0])
    ax_lab  = fig.add_subplot(gs[1, 0], sharex=ax_mask)   # ← shared x-axis
    ax_cb   = fig.add_subplot(gs[:, 1])                    # colorbar spans both rows

    # ── Top panel: attention/loss mask ────────────────────────────────────
    ax_mask.imshow(
        mask_used[np.newaxis, :], aspect="auto",
        cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
    )
    ax_mask.set_yticks([0])
    ax_mask.set_yticklabels(["loss\nmask"], fontsize=8)
    ax_mask.xaxis.set_major_locator(ticker.MultipleLocator(max(1, n_total // 10)))
    ax_mask.set_title(
        f"Sample {idx}  —  {n_real} real tokens / {n_total} total  "
        f"({n_total - n_real} padding)",
        fontsize=10,
    )
    plt.setp(ax_mask.get_xticklabels(), visible=False)     # hide redundant x labels

    # ── Bottom panel: per-token label heatmap (full length, NaN = grey) ──
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#cccccc")                          # NaN → light grey

    im = ax_lab.imshow(
        full_labels, aspect="auto",
        cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
    )
    ax_lab.set_yticks(range(n_subtypes))
    ax_lab.set_yticklabels(subtype_names, fontsize=7)
    ax_lab.set_xlabel("Token position", fontsize=9)
    ax_lab.xaxis.set_major_locator(ticker.MultipleLocator(max(1, n_total // 10)))

    # ── Colorbar in its own dedicated axes ────────────────────────────────
    fig.colorbar(im, cax=ax_cb, label="label (0/1)")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}", flush=True)


def visualize_metrics(save_path_loss,
                      save_path_evol):
    # Generate figures showing the evolution of the scores during training

    METRICS_DIR = config.MODEL_CONFIG["metrics_dir"]
    VERSION = config.VERSION

    train_metrics_df = pd.read_csv(os.path.join(METRICS_DIR, f"train_metrics_v{VERSION}.tsv"), sep='\t')
    val_metrics_df = pd.read_csv(os.path.join(METRICS_DIR, f"val_metrics_v{VERSION}.tsv"), sep='\t')
    test_metrics_df = pd.read_csv(os.path.join(METRICS_DIR, f"test_metrics_v{VERSION}.tsv"), sep='\t')

    # f1/micro	precision/micro	recall/micro	loss	step
    # 0.07481227070093155	0.05566808208823204	0.11402550339698792	5.273273804485798	2000
    # --- 1. Data Preparation ---
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

    # --- 2. Visualize Loss Evolution ---
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

    # Add Test Loss as a specific marker
    fig_loss.add_trace(go.Scatter(
        x=[final_step], 
        y=[test_metrics_df['loss'].iloc[0]],
        mode='markers', 
        name='Test Loss',
        marker=dict(size=12, symbol='star', color='gold', line=dict(width=2, color='black')),
        hovertemplate="Test Loss: %{y}<extra></extra>"
    ))

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


    # --- 3. Visualize Performance Metrics (F1, Precision, Recall) ---
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

    # Add Test performance metrics as diamond markers
    for metric in performance_metrics:
        fig_perf.add_trace(go.Scatter(
            x=[final_step], 
            y=[test_metrics_df[metric].iloc[0]],
            mode='markers', 
            name=f"Test {metric}",
            marker=dict(size=10, symbol='diamond'),
            showlegend=True
        ))

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
    fasta_path = (f"{workspace_path}/data/input/HIV1_PURE_REF.fasta")
    hxb2_ata_seq = ""
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            st_to_seq_dict[rec.id.split(".")[1]].append(str(rec.seq))

    # ---- rate array for mutation ----------------------------------------
    ata_to_hxb2 = build_hxb2_ata_maps(hxb2_ata_seq)
    names = ['A', 'B', 'C', 'avg']
    rate_arrays = {}
    for name in names:
        rate_array_path = f"{workspace_path}/data/input/mutation_rates/empirical_mutation_rates_{name}.npy"
        rate_array = np.load(rate_array_path)
        rate_arrays[name] = rate_array

    visualize_mutation_rates(
        rate_arrays,
        ata_to_hxb2,
        window_size=100, 
        save_path=f"{workspace_path}/figs/empirical_mutation_rates.html"
    )
    
    save_path_loss = f"{workspace_path}/figs/loss_evolution.html"
    save_path_evol = f"{workspace_path}/figs/metrics_evolution.html"
    visualize_metrics(save_path_loss=save_path_loss,
                      save_path_evol=save_path_evol)