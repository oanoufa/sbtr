import torch
import os
import numpy as np

n_seq = 200000
rp = 0.8
workspace_path = "/workspaces/mpath/oanoufa"
model_version = "4"

config = {
    # Model
    "model_name": "InstaDeepAI/NTv3_650M_pre",
    "checkpoint_name": f"model_v{model_version}.pt",
    "load_checkpoint": False, # Whether to load from checkpoint to resume training or start fresh training
    "model_version": model_version,

    # Data
    "labels_path": f"{workspace_path}/data/GEN/labels_{n_seq}_{rp}.npy",
    "sequences_path": f"{workspace_path}/data/GEN/sequences_{n_seq}_{rp}.npy",
    "loss_masks_path": f"{workspace_path}/data/GEN/loss_masks_{n_seq}_{rp}.npy",
    "metadata_path": f"{workspace_path}/data/GEN/metadata_{n_seq}_{rp}.tsv",
    "data_cache_dir": f"{workspace_path}/data/NT_V3",
    "sequence_length": 10496, # max length of sequences in the dataset and multiple of 128
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

os.makedirs(config["data_cache_dir"], exist_ok=True)
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])