import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Bio import SeqIO
import pandas as pd



workspace_path = "/workspaces/mpath/oanoufa"

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
        else:
            fig.write_image(save_path, scale=2)
        print(f"Mutation rate visualization saved to: {save_path}")


def visualize_mutation_rates(
    rate_array,
    ata_to_hxb2,
    window_size=100,
    save_path=f"{workspace_path}/figs/mutation_rate_profile.html"):
    """
    Visualizes the empirical mutation rate array with HIV genes in the background.
    """
    ata_len = len(rate_array)
    mean_rate = np.mean(rate_array)
    x_positions = np.arange(ata_len)
    
    # Smooth data
    kernel = np.ones(window_size) / window_size
    smoothed_rates = np.convolve(rate_array, kernel, mode='same')
    y_max = max(rate_array) * 1.1

    # Build inverse map: HXB2 -> ATA
    hxb2_to_ata = {}
    for ata_idx, hxb2_idx in enumerate(ata_to_hxb2):
        if hxb2_idx > 0 and hxb2_idx not in hxb2_to_ata:
            hxb2_to_ata[hxb2_idx] = ata_idx

    # Create subplots
    fig = make_subplots(
        rows=1, cols=1, 
        shared_xaxes=False,
        row_heights=[1.0,
                    #  0.3,
                     ],
        vertical_spacing=0.12,
        subplot_titles=(
            "<b>Empirical Mutation Rate across the Genome</b>", 
            # "<b>Distribution of Mutation Rates</b>",
        )
    )

    # =========================================================================
    # ADD BACKGROUND GENE MAP (TOP PANEL)
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

    # Map the 3 frames to physical Y-axis heights based on the data
    frame_lanes = {
        3: (0, y_max * 0.32),
        2: (y_max * 0.33, y_max * 0.64),
        1: (y_max * 0.65, y_max * 0.95)
    }

    for i, (gene, (start_hxb2, end_hxb2, frame)) in enumerate(genes_raw_data.items()):
        # Convert HXB2 to ATA coordinates
        start_ata = hxb2_to_ata.get(start_hxb2, 0)
        end_ata = hxb2_to_ata.get(end_hxb2, ata_len - 1)
        
        y0, y1 = frame_lanes[frame]
        color = line_colors.get(gene, "grey")
        
        # Add the colored lane
        fig.add_shape(
            type="rect",
            x0=start_ata, x1=end_ata,
            y0=y0, y1=y1,
            fillcolor=color,
            opacity=0.25,  # Kept light so the data lines pop
            layer="below",
            line_width=0,
            row=1, col=1
        )
        
        # Stagger annotations so they don't overlap as much
        y_pos = (y0 + y1) / 2
        y_offset = (y_max * 0.04) if (i % 2 == 0) else -(y_max * 0.04)
        
        fig.add_annotation(
            x=(start_ata + end_ata) / 2,
            y=y_pos + y_offset,
            text=f"<b>{gene}</b>",
            showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=color,
            borderwidth=1,
            row=1, col=1
        )

    # Frame labels on the far left
    for frame, (y0, y1) in frame_lanes.items():
        fig.add_annotation(
            x=0, y=(y0 + y1) / 2,
            text=f"<b>F{frame}</b>", showarrow=False, 
            font=dict(size=12, color="black"),
            xanchor="right",
            row=1, col=1
        )

    # =========================================================================
    # ADD DATA TRACES
    # =========================================================================
    
    # 1. Raw Data (Bottom Layer)
    # fig.add_trace(
    #     go.Scatter(
    #         x=x_positions, y=rate_array, 
    #         mode='lines', 
    #         line=dict(color='gray', width=1), 
    #         opacity=0.6, name='Raw Rate'
    #     ), row=1, col=1
    # )

    # 2. Smoothed Data (Middle Layer)
    fig.add_trace(
        go.Scatter(
            x=x_positions, y=smoothed_rates, 
            mode='lines', 
            line=dict(color='black', width=2), # Changed to black for contrast against colors
            name=f'{window_size}-bp Moving Average'
        ), row=1, col=1
    )

    # 3. Mean Line
    fig.add_hline(
        y=mean_rate, line_dash="dash", line_color="red", opacity=1.0,
        annotation_text=f"Mean: {mean_rate:.3f}", annotation_position="top right",
        layer="above", row=1, col=1
    )

    # # --- BOTTOM PANEL: Histogram ---
    # fig.add_trace(
    #     go.Histogram(
    #         x=rate_array, nbinsx=50, marker_color='#072C4B', opacity=0.8, name='Count'
    #     ), row=2, col=1
    # )
    # fig.add_vline(
    #     x=mean_rate, line_dash="dash", line_color="red", opacity=1.0,
    #     annotation_text=f"Mean", annotation_position="top right", row=2, col=1
    # )

    # --- LAYOUT STYLING ---
    fig.update_layout(
        template="plotly_white",
        height=500, width=1100,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="ATA Alignment Position (bp)", range=[0, ata_len], row=1, col=1)
    fig.update_yaxes(title_text="Mutation Rate", range=[0, y_max], row=1, col=1)
    # fig.update_xaxes(title_text="Mutation Rate (Probability)", row=2, col=1)
    # fig.update_yaxes(title_text="Count of Positions", row=2, col=1)

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)
        print(f"Mutation rate visualization saved to: {save_path}")

def build_hxb2_ata_maps(hxb2_ata_seq: str):
    seq       = np.frombuffer(hxb2_ata_seq.encode(), dtype=np.uint8)
    is_base   = seq != ord("-")
    ata_pos   = np.where(is_base)[0]
    ata_len   = len(hxb2_ata_seq)

    ata_to_hxb2            = np.zeros(ata_len, dtype=np.int32)
    ata_to_hxb2[ata_pos]   = np.arange(1, ata_pos.size + 1)
    for i in range(1, ata_len):
        if ata_to_hxb2[i] == 0:
            ata_to_hxb2[i] = ata_to_hxb2[i - 1]
    return ata_to_hxb2


if __name__ == "__main__":
    breakpoints_path = f"{workspace_path}/data/LANL/crf/lanl_crf_breakpoints.csv"
    segments_path = f"{workspace_path}/data/LANL/crf/lanl_crf_segments.csv"
    df_bp = pd.read_csv(breakpoints_path)
    df_seg = pd.read_csv(segments_path)

    df_bp.rename(columns={
        'position' : 'pos'
    }, inplace=True)
    df_bp.sort_values(by=['crf'],
                    key=lambda s: s.str.extract(r'CRF(\d+)', expand=False).astype(int),
                    inplace=True,
                    ascending=False)
    
    visualize_breakpoints(df_bp)
    
    st_to_seq_dict = defaultdict(list)
    fasta_path = (f"{workspace_path}/data/SBTR_DATA/LANL_SBTR_ALIGNMENT.fasta")
    hxb2_ata_seq = ""
    for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq)
        else:
            st_to_seq_dict[rec.id.split(".")[1]].append(str(rec.seq))

    # ---- rate array for mutation ----------------------------------------
    ata_to_hxb2 = build_hxb2_ata_maps(hxb2_ata_seq)
    rate_array_path = f"{workspace_path}/data/SBTR_DATA/empirical_mutation_rates.npy"
    rate_array = np.load(rate_array_path)

    visualize_mutation_rates(
        rate_array,
        ata_to_hxb2,
        window_size=50, 
        save_path=f"{workspace_path}/figs/empirical_mutation_rates.png"
    )