import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM
from tqdm import tqdm
import math
import re
from .metrics_class import HIVSubtypingMetrics
from typing import Dict
from . import config

workspace_path = config.WORKSPACE_PATH
pure_st_to_id_dict = config.ST_TO_ID_DICT
num_subtypes = len(pure_st_to_id_dict)
MODEL_CONFIG = config.MODEL_CONFIG
max_length = config.SEQ_LEN_AFTER_PAD
device = torch.device(MODEL_CONFIG["device"])
tv_weight = MODEL_CONFIG['tv_weight']


class HIVClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_subtypes: int, smooth_kernel: int = 5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim) # normalize embeddings before classification
        self.head = nn.Linear(embed_dim, num_subtypes) # linear layer to map embeddings to subtype logits
        # depthwise conv: each subtype channel smoothed independently
        self.smooth = nn.Conv1d(
            num_subtypes, num_subtypes, kernel_size=smooth_kernel,
            padding=smooth_kernel // 2, groups=num_subtypes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        logits = self.head(x)              # [batch, seq_len, num_subtypes]
        logits = logits.transpose(1, 2)    # [batch, num_subtypes, seq_len]
        logits = self.smooth(logits)
        return logits.transpose(1, 2)      # [batch, seq_len, num_subtypes]


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

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        backbone_inputs = {"input_ids": tokens}
        if attention_mask is not None:
            backbone_inputs["attention_mask"] = attention_mask
        outputs = self.backbone(**backbone_inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]
        subtype_logits = self.subtype_head(embedding)
        return {"subtype_logits": subtype_logits}

def tv_penalty(logits: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    diff = (probs[:, 1:, :] - probs[:, :-1, :]).abs()
    pair_mask = loss_mask[:, 1:, :] * loss_mask[:, :-1, :]  # skip pairs touching padding/ambiguous zones
    return (diff * pair_mask).sum() / pair_mask.sum().clamp(min=1)

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
    bce = loss_unreduced.sum() / loss_mask.sum().clamp(min=1)
    tv = tv_penalty(logits, loss_mask)
    loss = bce + tv_weight * tv

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        train_metrics.update(
            preds=torch.sigmoid(logits), # previously (torch.sigmoid(logits) > 0.5).float()
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

        loss_unreduced = loss_fn(logits, labels) * loss_mask
        bce = loss_unreduced.sum() / loss_mask.sum().clamp(min=1)
        tv = tv_penalty(logits, loss_mask)
        loss = bce + tv_weight * tv

        preds = torch.sigmoid(logits)
        # preds = (preds > 0.5).float()

        metrics.update(
            preds=preds,
            targets=labels,
            loss_mask=loss_mask,
            loss=loss.item(),
        )

    return loss.item()




"""
Small, from-scratch bidirectional full-attention transformer with
single-nucleotide tokenization. No conv up/downsampling anywhere:
one token in -> one token out, at every layer.

Used to test whether NT-v3's conv-based sequence compression is
degrading breakpoint localization, independent of its parameter
count. Drop-in replacement for `backbone` in HFModelForHIVSubtyping
(model_class.py) -- exposes the same `output_hidden_states=True`
/ `hidden_states[-1]` shape [batch, seq_len, embed_dim] that
HIVClassificationHead expects.
"""

# --- tokenizer --------------------------------------------------------
# Fractional one-hot over (A, C, G, T), encoding IUPAC ambiguity as an
# even split across the represented bases (e.g. W -> A/T -> .5,0,0,.5).
# Pad position is the zero vector -- distinguishable from any real base.
IUPAC_TO_PROBS = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "C": (0.0, 1.0, 0.0, 0.0),
    "G": (0.0, 0.0, 1.0, 0.0),
    "T": (0.0, 0.0, 0.0, 1.0),
    "R": (0.5, 0.0, 0.5, 0.0),  # A/G
    "Y": (0.0, 0.5, 0.0, 0.5),  # C/T
    "S": (0.0, 0.5, 0.5, 0.0),  # G/C
    "W": (0.5, 0.0, 0.0, 0.5),  # A/T
    "K": (0.0, 0.0, 0.5, 0.5),  # G/T
    "M": (0.5, 0.5, 0.0, 0.0),  # A/C
    "B": (0.0, 1 / 3, 1 / 3, 1 / 3),  # C/G/T
    "D": (1 / 3, 0.0, 1 / 3, 1 / 3),  # A/G/T
    "H": (1 / 3, 1 / 3, 0.0, 1 / 3),  # A/C/T
    "V": (1 / 3, 1 / 3, 1 / 3, 0.0),  # A/C/G
    "N": (0.25, 0.25, 0.25, 0.25),
    "-": (0.0, 0.0, 0.0, 0.0)
}
PAD_VEC = (0.0, 0.0, 0.0, 0.0)
NUM_BASES = 4  # A, C, G, T


def encode_sequence(seq: str, max_length: int = None) -> torch.Tensor:
    """Returns a [seq_len, 4] float tensor, one fractional-base row per position."""
    vecs = [IUPAC_TO_PROBS.get(b.upper(), IUPAC_TO_PROBS["N"]) for b in seq]
    if max_length is not None:
        vecs = vecs[:max_length] + [PAD_VEC] * (max_length - len(vecs))
    return torch.tensor(vecs, dtype=torch.float)


# --- positional encoding ----------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) so it costs no extra params and extrapolates
    a bit beyond max_length if a longer fragment ever shows up."""

    def __init__(self, embed_dim: int, max_length: int):
        super().__init__()
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --- backbone -----------------------------------------------------------
class FullAttentionNTBackbone(nn.Module):
    """Plain bidirectional transformer encoder, single-nt resolution
    preserved at every layer. Uses PyTorch's built-in SDPA kernel
    (Flash Attention when available) via nn.TransformerEncoderLayer.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_length: int = 13000,  # headroom above your ~12kb sequences
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Linear(NUM_BASES, embed_dim)
        self.pos_embed = SinusoidalPositionalEncoding(embed_dim, max_length)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable at this depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs
    ):
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)
        x = self.embed_dropout(x)

        # nn.TransformerEncoder wants True = "ignore this position"
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)

        # mimic HF's output_hidden_states=True interface so the rest
        # of model_class.py (subtype_head, hidden_states[-1]) needs
        # zero changes.
        class _Output:
            pass

        out = _Output()
        out.hidden_states = [x]  # only the final layer; extend if you
        # want to probe intermediate layers too
        return out


# --- full model (mirrors HFModelForHIVSubtyping) -----------------------
class FullAttentionModelForHIVSubtyping(nn.Module):
    """Same interface as HFModelForHIVSubtyping in model_class.py --
    swap this in for the NT-v3 backbone with no other pipeline changes.
    """

    def __init__(self, num_subtypes: int, **backbone_kwargs):
        super().__init__()
        from model_class import HIVClassificationHead  # reuse existing head

        self.backbone = FullAttentionNTBackbone(**backbone_kwargs)
        self.subtype_head = HIVClassificationHead(self.backbone.embed_dim, num_subtypes)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        outputs = self.backbone(input_ids=tokens, attention_mask=attention_mask)
        embedding = outputs.hidden_states[-1]
        subtype_logits = self.subtype_head(embedding)
        return {"subtype_logits": subtype_logits}


if __name__ == "__main__":
    # rough param count sanity check
    model = FullAttentionModelForHIVSubtyping(num_subtypes=20)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.1f}M")

    batch, seq_len = 2, 12000
    tokens = torch.randint(2, 6, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    out = model(tokens, mask)
    print(out["subtype_logits"].shape)  # [2, 12000, 20]