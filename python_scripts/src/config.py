"""Store paths and settings shared by the sbtr scripts."""

import torch
import numpy as np
from typing import Tuple

WORKSPACE_PATH = "/pasteur/helix/projects/mPath/oanoufa/sbtr"
PURE_REF_PATH = f"{WORKSPACE_PATH}/data/output/HIV1_PURE_REF.fasta"
CRF_REF_PATH = f"{WORKSPACE_PATH}/data/output/HIV1_CRF_REF.fasta"
COMBINED_REF_PATH = f"{WORKSPACE_PATH}/data/HIV1_COMBINED_REF.fasta"
TOKEN_PATH = f"{WORKSPACE_PATH}/hftoken.txt"
CRF_SEGMENTS_BD_ATA=f"{WORKSPACE_PATH}/data/lanl_crf_segments_brokedown_aln.csv"
MAPPING_PATH = f"{WORKSPACE_PATH}/data/hxb2_ata_mapping.csv"

# GLOBAL PARAMETERS
ATA_LEN = 10240 # max length of sequences in the dataset is (11954) (16980) (11561) NOW 10240 and multiple of 128 (10240)
HXB2_LEN = 9719
HXB2_START = 790
HXB2_END = 9417
HXB2_USED_LEN = HXB2_END - HXB2_START + 1 # [START, END]

# SEQUENCE GENERATION PARAMETERS
N_SEQ = 15000
RP = 0.95
TEST_SET_SIZE = 1000
MAX_YEAR = 2027
MIN_SEG_LEN = 50 # Min length of segments of a subtype in a recombinant sequence (in ATA positions)
MAX_SUBTYPES = 7
MAX_BREAKPOINTS = 10
PARTIAL_FRAC = 0.50 # Fraction of sequences that are partial (i.e., not full-length)
MIN_FRAG_LEN = 300 # Min length of a partial sequence (prot is around 300bp)
DIV_WINDOW_SIZE = 200
MIN_DIV = 15
MAX_RETRIES = 50
BP_PRIOR_LAMBDA = 1.0

# CRF REF BANK PARAMETERS
PCT_PER_CRF_BANK = 0.30 # Adaptive bank size depending on the number of sequences of the CRF
MIN_PER_CRF_BANK = 5 # Min bank size for each CRF
N_TEST = 5 # Test size for each CRF (including one gag and one pol sequence)

# DECODER PARAMETERS
SLIDING_WINDOW_SIZE = 40 # Twice the size of the smallest segments possible (window centered on position)
TOP_K = 30 # Number of top CRF reference sequences to consider in the margin
PROB_ZERO_THRESHOLD = 0.5 # Minimum output prob for a label to be considered a valid prediction (otherwise, label is set to 0)
CRF_MATCH_MARGIN = 0.02 # If several CRFs are within this value, take into account the ambiguity
CRF_ASSIGN_THR = 0.5 # Minimum match to assign a CRF
PARTIAL_THR = 7000 # Threshold to consider a sequence full (>=7000nt) or partial (<7000nt)
GATE_MIN_SEG_LEN = 200   # min active positions for a CRF segment to be evaluated by the gate
GATE_MIN_SUPPORT = 0.5   # min fraction of those positions compatible with the segment subtype

# INPUT CHECKS
MAX_SEQ_LEN        = 11000
MIN_INFORMATIVE_BP = 100    # non-N bases after dealigning
MAX_N_FRACTION     = 0.2    # warning only

# MODEL PARAMETERS
VERSION = "0.5"
PAD_LEN = 128
SEQ_LEN_AFTER_PAD = ((ATA_LEN + PAD_LEN - 1) // PAD_LEN) * PAD_LEN

MODEL_CONFIG = {
    # Model
    "model_name": "oanoufa/sbtr_ntv3_650M", # oanoufa/sbtr_ntv3_650M oanoufa/sbtr_custom
    "backbone": "InstaDeepAI/NTv3_650M_pre", # zhihan1996/DNABERT-2-117M InstaDeepAI/NTv3_650M_pre InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    "tokenizer": "InstaDeepAI/NTv3_650M_pre", # /pasteur/helix/projects/mPath/oanoufa/sbtr/data/model/tokenizer/custom_hiv_tokenizer
    "checkpoint_name": f"sbtr_v{VERSION}.pt",
    "load_checkpoint": True, # Whether to load from checkpoint to resume training or start fresh training
    "model_version": VERSION,

    # Data
    "labels_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}_{PARTIAL_FRAC}/labels_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy",
    "sequences_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}_{PARTIAL_FRAC}/sequences_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy",
    "loss_masks_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}_{PARTIAL_FRAC}/loss_masks_{N_SEQ}_{RP}_{PARTIAL_FRAC}.npy",
    "metadata_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}_{PARTIAL_FRAC}/metadata_{N_SEQ}_{RP}_{PARTIAL_FRAC}.tsv",
    "data_cache_dir": f"{WORKSPACE_PATH}/data/model",
    "checkpoint_dir": f"{WORKSPACE_PATH}/data/model/checkpoints/v{VERSION}",
    "metrics_dir": f"{WORKSPACE_PATH}/data/model/metrics",

    # Training
    "batch_size": 8,
    "num_steps_training": 30000,
    # Only batch_size * num_steps_training samples will be used for training (randomly sampled from the training split)
    "log_every_n_steps": 10000,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_proportion": 0.05,  # 5% of training steps for warmup
    "grad_clip_norm": 1.0,
    "backbone_learning_rate_multiplier": 0.1, # backbone learning rate = this * main learning rate
    "tv_weight": 0.03,
    "smooth_kernel": 5,
    "embed_layer": -1, # Which layer of the backbone to use for embeddings

    # Validation
    "validate_every_n_steps": 30000,
    "max_val_batches": 500,

    # Inference
    "inference_batch_size": 8, # 1 is faster if on cpu

    # General
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}

torch.manual_seed(MODEL_CONFIG["seed"])
np.random.seed(MODEL_CONFIG["seed"])

# Rates in units of 1e-3 subst/site/year, from Nasir et al. 2021 Table 2
# Weighted mean: (pol_rate * pol_len + env_rate * env_len) / (pol_len + env_len)
# pol window of 1056 nt, env window of 564 nt

_POL_LEN = 1056
_ENV_LEN = 564
_W = _POL_LEN + _ENV_LEN

CLOCK_RATES = {
    st: (pol * _POL_LEN + env * _ENV_LEN) / _W * 1e-3
    for st, pol, env in [
        ('A1',     1.25, 3.41),
        ('A6',     2.18, 3.94),
        ('B',      1.08, 2.33),
        ('C',      1.22, 2.74),
        ('D',      0.98, 2.93),
        ('F1',     2.05, 1.34),
        ('G',      1.78, 4.72),
        # ('01_AE',  1.48, 2.99),
        # ('02_AG',  1.29, 4.48),
        ('E',      1.48, 2.99),
    ]
}
CLOCK_RATES['avg'] = sum(CLOCK_RATES.values()) / len(CLOCK_RATES)

ST_TO_ID_DICT = {
    'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 'A6': 5, 'A7': 6, 'A8': 7,
    'B': 8, 'C': 9, 'D': 10, 'E': 11, 'F1': 12, 'F2': 13, 'G': 14,
    'H': 15, 'J': 16, 'K': 17, 'L': 18, 'N': 19, 'O': 20, 'P': 21,
    }

ST_COLORS = {
    # A family - blues
    'A':   '#378ADD',
    'A1':  '#185FA5',
    'A2':  '#85B7EB',
    'A3':  '#0C447C',
    'A4':  '#B5D4F4',
    'A5U':  '#B5D47C',
    'A5':  '#B5D47C',
    'A6':  '#042C53',
    'A7':  '#5B9FD4',
    'A8':  '#2A76C4',
    # B - coral
    'B':   '#D85A30',
    # C - teal
    'C':   '#1D9E75',
    # D - amber
    'D':   '#BA7517',
    # AE - coral/pink blend
    'AE':  '#F0997B',
    'E':  '#F0997B',
    '01_AE': '#F0997B',
    # F family - purple
    'F':   '#7F77DD',
    'F1':  '#534AB7',
    'F2':  '#AFA9EC',
    # G - green
    'G':   '#639922',
    # H - red
    'H':   '#E24B4A',
    # J - amber/warm
    'J':   '#EF9F27',
    # K - teal (lighter)
    'K':   '#5DCAA5',
    # L - pink
    'L':   '#D4537E',
    # N - golden yellow
    'N':   '#E2B93B',
    # O - indigo / violet
    'O':   '#4A2E80',
    # P - mint / sage
    'P':   '#2EA885',
    # U - neutral gray
    'U':   '#788496',
    # CPZ - olive/brown (SIVcpz, chimpanzee origin)
    'CPZ': '#8C6E3A',
    # GOR - muted brown (SIVgor, gorilla origin)
    'GOR': '#6E5039',
    # MAC - olive-tan/khaki (SIVmac, rhesus macaque origin)
    'MAC': '#9E824C',
    # SMM - warm clay/taupe-brown (SIVsmm, sooty mangabey origin)
    'SMM': '#82563B',
    # AG - blue/green recombinant (A + G blend)
    '02_AG': '#072C4B',
    # BC - coral/teal recombinant (B + C blend)
    '07_BC': '#7A7C52',
    # LTR sequence features & insertions - white / off-white
    "5'LTR":         '#E8E8E8',  # Pure white
    "5'-Insertion":  '#E8E8E8',  # Pure white
    "3'LTR":         '#E8E8E8',  # Light off-white / light gray
    "3'-Insertion":  '#E8E8E8',  # Light off-white / light gray
    "Other CRFs / URF": '#A6A6A6',
}

# GENE MAP BACKGROUND
GENES_RAW = {
    "5'LTR":   (1, 634, 1),    "3'LTR":   (9086, 9719, 2),
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

GENE_COLORS = {
    # LTRs
    "5'LTR": "#969696", "3'LTR": "#969696",
    # gag: blues
    "p17": "#4292c6", "p24": "#9ecae1", "p2": "#2171b5",
    "p7": "#c6dbef", "p1": "#6baed6", "p6": "#084594",
    # pol: greens
    "prot": "#a1dab4", "p51_RT": "#41ab5d", "p15": "#c7e9c0", "p31_int": "#238b45",
    # env: oranges
    "gp120": "#f16913", "gp41": "#fdae6b",
    # accessory: purples
    "vif": "#9e9ac8", "vpr": "#cbc9e2", "vpu": "#807dba", "nef": "#6a51a3",
    # regulatory: pinks (both exons share a colour)
    "tat1": "#dd3497", "tat2": "#dd3497", "rev1": "#fa9fb5", "rev2": "#fa9fb5",
}

COLOR_SCHEME = ['#072C4B', '#F28089','#71cddd', "a07a99", "#E69F00"]

# 01_AE HANDLING
# end of pol (HXB2): CRF01_AE is A1 up to here
AE_A1_END_HXB2 = (GENES_RAW["p31_int"][1] + GENES_RAW["vif"][0]) // 2
# half-width (ATA positions) of the loss-masked zone at the A1->AE switch; 0 disables. we add this because of the overlap between pol and vif (vif starts at 5041, pol ends at 5096)
AE_SPLIT_MASK = 30

# DRM and LTR masks

START_5LTR = [i for i in range(GENES_RAW["5'LTR"][0], GENES_RAW["5'LTR"][1]+1)]
NEF_3LTR = [i for i in range(GENES_RAW["nef"][1], GENES_RAW["3'LTR"][1]+1)]

HXB2_POL_FRAMES = {
    'Protease': 2253,      # PR starts at nucleotide 2253
    'RT': 2550,            # RT starts at nucleotide 2550
    'Integrase': 4230,     # IN starts at nucleotide 4230
}
 
# Stanford HIV DRM positions (amino acid, 1-based)
DRM_POSITIONS = {
    'RT_NRTI': [41, 65, 67, 69, 70, 74, 115, 151, 184, 210, 215, 219],
    'RT_NNRTI': [100, 101, 103, 106, 138, 181, 188, 190, 230],
    'Protease': [30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90],
    'Integrase': [66, 92, 118, 138, 140, 143, 147, 148, 155, 263]
}

# HXB2 positions of Stanford HIV DRM positions (nucleotide, 1-based)
DRM_HXB2_POSITIONS = [HXB2_POL_FRAMES['RT'] + (pos - 1) * 3 for pos in DRM_POSITIONS['RT_NRTI']] + [HXB2_POL_FRAMES['RT'] + (pos - 1) * 3 for pos in DRM_POSITIONS['RT_NNRTI']] + [HXB2_POL_FRAMES['Protease'] + (pos - 1) * 3 for pos in DRM_POSITIONS['Protease']] + [HXB2_POL_FRAMES['Integrase'] + (pos - 1) * 3 for pos in DRM_POSITIONS['Integrase']]

EXTENDED_DRM_HXB2_POSITIONS = DRM_HXB2_POSITIONS.copy()
EXTENDED_DRM_HXB2_POSITIONS.extend([pos + 1 for pos in DRM_HXB2_POSITIONS])
EXTENDED_DRM_HXB2_POSITIONS.extend([pos + 2 for pos in DRM_HXB2_POSITIONS])

MASKED_POSITIONS_HXB2 = START_5LTR + EXTENDED_DRM_HXB2_POSITIONS + NEF_3LTR


def build_hxb2_ata_maps(hxb2_ata_seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    ata_to_hxb2[col] = full-genome HXB2 position (1-based) of column ``col``; the
        first HXB2 base of the cropped alignment is HXB2_START. Gap columns carry
        the preceding position forward; columns before the first base carry 0.
    hxb2_to_ata : shape (HXB2_LEN + 1,), indexed by full-genome HXB2 position.
        Positions outside [HXB2_START, HXB2_START + n_bases - 1] were cropped and
        are clamped to the first / last column. Index 0 is unused.
    """
    seq     = np.frombuffer(hxb2_ata_seq.encode(), dtype=np.uint8)
    ata_pos = np.where(seq != ord("-"))[0]           # aln cols where HXB2 has a base
    first   = HXB2_START
    last    = first + ata_pos.size - 1
    if last > HXB2_LEN:
        raise ValueError(f"HXB2 row has {ata_pos.size} bases from {first}: runs past {HXB2_LEN}")

    ata_to_hxb2 = np.zeros(len(hxb2_ata_seq), dtype=np.int32)
    ata_to_hxb2[ata_pos] = np.arange(first, last + 1)
    ata_to_hxb2 = np.maximum.accumulate(ata_to_hxb2)      # forward-fill gap columns

    hxb2_to_ata = np.zeros(HXB2_LEN + 1, dtype=np.int32)
    hxb2_to_ata[first:last + 1] = ata_pos
    hxb2_to_ata[1:first]        = ata_pos[0]              # cropped 5' region
    hxb2_to_ata[last + 1:]      = ata_pos[-1]             # cropped 3' region
    return ata_to_hxb2, hxb2_to_ata