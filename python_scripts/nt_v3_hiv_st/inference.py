import gzip
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from Bio import SeqIO
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from typing import Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from huggingface_hub import login
token_path = '/workspaces/mpath/oanoufa/data/ibenstoken.txt'
with open(token_path, 'r') as f:
    token = f.read().strip()
login(token=token)


constants_path = "/users/mpath/oanoufa/HIV_PROJECT/scripts/utils/constants.py"
sys.path.append(os.path.dirname(constants_path))
import constants
import model_config
pure_st_to_id_dict = constants.ST_TO_ID_DICT
num_subtypes = len(pure_st_to_id_dict)
config = model_config.config
max_length = config["sequence_length"]
pad_multiple_of = config["pad_multiple_of"]
from hiv_dataset_class import HIVSequenceDataset, parse_labels_file
from hiv_nt_training_class import HFModelForHIVSubtyping, train_step, validation_step
from hiv_nt_metrics_class import HIVSubtypingMetrics


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    # Set device
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    # Create model
    model = HFModelForHIVSubtyping(
        model_name=config["model_name"],
        num_subtypes=num_subtypes
    )
    model = model.to(device)
    print(f"\nLoading best model from checkpoint for inference...")
    checkpoint = torch.load(os.path.join(config["data_cache_dir"], "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inference_sequences_path = "/users/mpath/oanoufa/HIV_PROJECT/data/GEN/gen_sequences.fasta"
    sequences = []
    for record in SeqIO.parse(inference_sequences_path, "fasta"):
        sequences.append(str(record.seq))

    print(f"\nRunning inference...")
