import torch
import os
import numpy as np

WORKSPACE_PATH = "/pasteur/helix/projects/mPath/oanoufa/sbtr"
PURE_REF_PATH = f"{WORKSPACE_PATH}/data/input/HIV1_PURE_REF.fasta"
N_SEQ = 300000
RP = 0.9
VERSION = "5"

TOKEN_PATH = '/pasteur/appa/homes/oanoufa/ibenstoken.txt'

MODEL_CONFIG = {
    # Model
    "model_name": "InstaDeepAI/NTv3_650M_pre",
    "checkpoint_name": f"model_v{VERSION}.pt",
    "load_checkpoint": False, # Whether to load from checkpoint to resume training or start fresh training
    "model_version": VERSION,

    # Data
    "labels_path": f"{WORKSPACE_PATH}/data/output/{N_SEQ}_{RP}/labels_{N_SEQ}_{RP}.npy",
    "sequences_path": f"{WORKSPACE_PATH}/data/output/{N_SEQ}_{RP}/sequences_{N_SEQ}_{RP}.npy",
    "loss_masks_path": f"{WORKSPACE_PATH}/data/output/{N_SEQ}_{RP}/loss_masks_{N_SEQ}_{RP}.npy",
    "metadata_path": f"{WORKSPACE_PATH}/data/output/{N_SEQ}_{RP}/metadata_{N_SEQ}_{RP}.tsv",
    "data_cache_dir": f"{WORKSPACE_PATH}/data/model",
    "checkpoint_dir": f"{WORKSPACE_PATH}/data/model/checkpoints",
    "metrics_dir": f"{WORKSPACE_PATH}/data/model/metrics",
    "sequence_length": 12032, # max length of sequences in the dataset is 11954 and multiple of 128
    "pad_multiple_of": 128,

    # Training
    "batch_size": 8,
    "num_steps_training": 20000,
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


ST_LIST = {
    '100_01C', '110_BC', '78_cpx', 'N', '22_01A1', 'B', '146_BC', '12_BF', '39_BF1', '68_01B',
    '112_01B', '101_01B', '113_0107', '121_0107', '93_cpx', '133_A6B', 'K', '80_0107', '88_BC',
    '106_cpx', '03_A6B', '45_cpx', '02_AG', '38_BF1', '128_07B', '157_A6C', 'D', '124_cpx', '155_0755',
    'GOR', '16_A2D', '108_BC', 'A6', '94_cpx', '122_BF1', '109_0107', 'CPZ', '69_01B', '53_01B', '130_A1B',
    '79_0107', '58_01B', '120_0107', 'G', '97_01B', '42_BF1', '09_cpx', '34_01B', '04_cpx', '54_01B', '96_cpx',
    '62_BC', '114_0155', '131_A1B', '23_BG', 'J', '74_01B', '26_A5U', '77_cpx', '72_BF1', '140_0107', '67_01B',
    'A3', '75_BF1', '138_cpx', '103_01B', '70_BF1', '153_55B', 'A2', '116_0108', '134_0107', '36_cpx', '47_BF1',
    '14_BG', 'A8', '17_BF1', '31_BC', '87_cpx', '11_cpx', '37_cpx', 'F1', '25_cpx', '91_cpx', '86_BC', 'A1',
    '35_A1D', '08_BC', '40_BF1', '141_BF1', '29_BF1', '84_A1D', '95_02B', '82_cpx', 'O', '105_0108', '52_01B',
    '41_CD', '99_BF1', 'H', 'F2', '33_01B', 'A7', '104_0107', '111_01C', '61_BC', '10_CD', '126_0755', '83_cpx',
    '66_BF1', '76_01B', '73_BG', '102_0107', '28_BF1', '71_BF1', '154_0755', '24_BG', '50_A1D', '89_BF1', '117_0107',
    '63_02A6', '56_cpx', '18_cpx', '20_BG', '07_BC', '13_cpx', '65_cpx', 'L', '06_cpx', '137_0107', '01_AE', '55_01B',
    'C', '44_BF1', '49_cpx', '19_cpx', '90_BF1', '115_01C', '152_DG', '92_C2U', '118_BC', '132_94B', '151_0107',
    '32_06A6', '48_01B', '51_01B', '98_06B', '125_0107', '123_0107', '60_BC', '159_01103', '107_01B', '59_01B',
    '27_cpx', 'P', '57_BC', '05_DF', '143_cpx', '119_0107', '81_cpx', '21_A2D', '43_02G', 'A4', '15_01B', '156_0755',
    '64_BC', '85_BC', '46_BF1',
    }

ST_TO_ID_DICT = {
    'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A6': 4, 'A7': 5, 'A8': 6,
    'B': 7, 'C': 8, 'D': 9, 'F1': 10, 'F2': 11, 'G': 12, 'H': 13,
    'J': 14, 'K': 15, 'L': 16, 'E': 17,
    'O': 18, 'N': 19, 'P': 20,
    }

ST_COLORS = {
    # A family — blues
    'A':   '#378ADD',
    'A1':  '#185FA5',
    'A2':  '#85B7EB',
    'A3':  '#0C447C',
    'A4':  '#B5D4F4',
    'A6':  '#042C53',
    'A7':  '#5B9FD4',
    'A8':  '#2A76C4',
    # B — coral
    'B':   '#D85A30',
    # C — teal
    'C':   '#1D9E75',
    # D — amber
    'D':   '#BA7517',
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
    # E — coral/pink blend
    'E':   '#F0997B',
    # O — gray
    'O':   '#888888',
    # N — light gray
    'N':   '#CCCCCC',
    # P — dark gray
    'P':   '#555555',
}
