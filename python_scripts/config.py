import torch
import os
import numpy as np

WORKSPACE_PATH = "/pasteur/helix/projects/mPath/oanoufa/sbtr"
PURE_REF_PATH = f"{WORKSPACE_PATH}/data/input/HIV1_PURE_REF.fasta"
CRF_REF_PATH = f"{WORKSPACE_PATH}/data/output/HIV1_CRF_REF.fasta"
COMBINED_REF_PATH = f"{WORKSPACE_PATH}/data/output/HIV1_COMBINED_REF.fasta"

# SEQUENCE GENERATION PARAMETERS
N_SEQ = 600000
RP = 0.9
MAX_YEAR = 2030
SEQ_LEN = 11561 # max length of sequences in the dataset is (11954) (NOW 16980) NOW 11561 and multiple of 128 (11648)
PAD_LEN =  128
SEQ_LEN_AFTER_PAD = ((SEQ_LEN // PAD_LEN) + 1) * PAD_LEN

# NUCLEOTIDE TRANSFORMER PARAMETERS
VERSION = "8"

TOKEN_PATH = '/pasteur/appa/homes/oanoufa/ibenstoken.txt'

MODEL_CONFIG = {
    # Model
    "model_name": "InstaDeepAI/NTv3_650M_pre",
    "checkpoint_name": f"model_v{VERSION}.pt",
    "load_checkpoint": False, # Whether to load from checkpoint to resume training or start fresh training
    "model_version": VERSION,

    # Data
    "labels_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}/labels_{N_SEQ}_{RP}.npy",
    "sequences_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}/sequences_{N_SEQ}_{RP}.npy",
    "loss_masks_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}/loss_masks_{N_SEQ}_{RP}.npy",
    "metadata_path": f"{WORKSPACE_PATH}/data/output/seq_gen/{N_SEQ}_{RP}/metadata_{N_SEQ}_{RP}.tsv",
    "data_cache_dir": f"{WORKSPACE_PATH}/data/model",
    "checkpoint_dir": f"{WORKSPACE_PATH}/data/model/checkpoints",
    "metrics_dir": f"{WORKSPACE_PATH}/data/model/metrics",

    # Training
    "batch_size": 8,
    "num_steps_training": 40000,
    # Only batch_size * num_steps_training samples will be used for training (randomly sampled from the training split)
    "log_every_n_steps": 0.01,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_proportion": 0.05,  # 5% of training steps for warmup
    "grad_clip_norm": 1.0,
    "backbone_learning_rate_multiplier": 0.1, # backbone learning rate = this * main learning rate

    # Validation
    "validate_every_n_steps": 0.1,
    "max_val_batches": 500,

    # General
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}

os.makedirs(MODEL_CONFIG["data_cache_dir"], exist_ok=True)
os.makedirs(MODEL_CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(MODEL_CONFIG["metrics_dir"], exist_ok=True)
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
    # A family — blues
    'A':   '#378ADD',
    'A1':  '#185FA5',
    'A2':  '#85B7EB',
    'A3':  '#0C447C',
    'A4':  '#B5D4F4',
    'A5':  '#B5D47C',
    'A6':  '#042C53',
    'A7':  '#5B9FD4',
    'A8':  '#2A76C4',
    # B — coral
    'B':   '#D85A30',
    # C — teal
    'C':   '#1D9E75',
    # D — amber
    'D':   '#BA7517',
    # E — coral/pink blend
    'E':   '#F0997B',
    # F family — purple
    'F':   '#7F77DD',
    'F1':  '#534AB7',
    'F2':  '#AFA9EC',
    # G — green
    'G':   '#639922',
    # H — red
    'H':   '#E24B4A',
    # J — amber/warm
    'J':   '#EF9F27',
    # K — teal (lighter)
    'K':   '#5DCAA5',
    # L — pink
    'L':   '#D4537E',
    # N — light gray
    'N':   '#CCCCCC',
    # O — gray
    'O':   '#888888',
    # P — dark gray
    'P':   '#555555',
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
    "5'LTR": "#7f7f7f", "3'LTR": "#7f7f7f",
    "p17": "#1f77b4", "p24": "#ff7f0e", "p2": "#2ca02c", "p7": "#d62728", "p1": "#9467bd", "p6": "#8c564b",
    "prot": "#e377c2", "p51_RT": "#7f7f7f", "p15": "#bcbd22", "p31_int": "#17becf",
    "gp120": "#aec7e8", "gp41": "#ffbb78",
    "vif": "#98df8a", "vpr": "#ff9896", "vpu": "#c5b0d5", "nef": "#c49c94",
    "tat1": "#f7b6d2", "tat2": "#f7b6d2", "rev1": "#dbdb8d", "rev2": "#dbdb8d"
}

COLOR_SCHEME = ['#072C4B', '#F28089', '#71cddd']


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