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
from hiv_nt_metrics_class import HIVSubtypingMetrics
from typing import Dict

constants_path = "/users/mpath/oanoufa/HIV_PROJECT/scripts/utils/constants.py"
sys.path.append(os.path.dirname(constants_path))
import constants
import model_config
pure_st_to_id_dict = constants.ST_TO_ID_DICT
num_subtypes = len(pure_st_to_id_dict)
config = model_config.config
max_length = config["sequence_length"]
device = torch.device(config["device"])


class HIVClassificationHead(nn.Module):
    """Proper multiclass head for HIV subtyping."""
    def __init__(self, embed_dim: int, num_subtypes: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_subtypes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.head(x)  # [batch, seq_len, num_subtypes]


class HFModelForHIVSubtyping(nn.Module):
    """NT-V3 backbone + HIV subtype classification head"""

    def __init__(
        self,
        model_name: str,
        num_subtypes: int,  # len(unique_subtypes) from your data
    ):
        super().__init__()

        # Load config and model (same backbone)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.backbone = backbone

        # Subtype classification head
        self.subtype_head = HIVClassificationHead(
            self.config.embed_dim,
            num_subtypes)

        self.model_name = model_name

    def forward(self, tokens: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Forward through backbone
        outputs = self.backbone(input_ids=tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]  # Last layer embeddings

        # Predict HIV subtypes [batch, seq_len, num_subtypes]
        subtype_logits = self.subtype_head(embedding)

        return {"subtype_logits": subtype_logits}


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: Dict[str, torch.Tensor],
    train_metrics: HIVSubtypingMetrics,
):
    tokens = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    # Extract the combined ambiguity + attention mask
    loss_mask = batch["loss_mask"].to(device).unsqueeze(-1)

    outputs = model(tokens=tokens, attention_mask=attention_mask)
    logits = outputs["subtype_logits"]

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    # Multiply unreduced loss by loss_mask. Ignored ambiguous zones evaluate to 0.
    loss_unreduced = loss_fn(logits, labels) * loss_mask         
    loss = loss_unreduced.sum() / loss_mask.sum().clamp(min=1)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        train_metrics.update(
            preds=(torch.sigmoid(logits) > 0.5).float(),
            targets=labels,
            loss_mask=loss_mask,
            loss=loss.item(),
            input_ids=tokens,
            loss_unreduced=loss_unreduced.detach(),
        )

    return loss.item()

def validation_step(
    model: nn.Module,
    batch: dict,
    metrics: HIVSubtypingMetrics,
):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    tokens = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device).unsqueeze(-1)

    with torch.no_grad():
        outputs = model(tokens=tokens, attention_mask=attention_mask)
        logits = outputs["subtype_logits"]

        loss = loss_fn(logits, labels) * loss_mask
        loss = loss.sum() / loss_mask.sum().clamp(min=1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        metrics.update(
            preds=preds,
            targets=labels,
            loss_mask=loss_mask,
            loss=loss.item(),
        )

    return loss.item()
