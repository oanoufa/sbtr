"""Define the sbtr model configuration and classification layers."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM, AutoConfig
from typing import Dict

class HIVSubtypingConfig(PretrainedConfig):
    model_type = "hiv_subtyping"

    def __init__(
        self,
        backbone_name: str = "InstaDeepAI/NTv3_pretrained",
        backbone_revision: str = None,   # commit SHA of the backbone repo; None = latest
        num_subtypes: int = 22,
        smooth_kernel: int = 5,
        embed_layer: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.backbone_revision = backbone_revision
        self.num_subtypes = num_subtypes
        self.smooth_kernel = smooth_kernel
        self.embed_layer = embed_layer

class DilatedSmoother(nn.Module):
    def __init__(self, num_subtypes: int, kernel_size: int = 5, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(
                num_subtypes, num_subtypes, kernel_size=kernel_size,
                padding=(kernel_size - 1) * d // 2, dilation=d,
                groups=num_subtypes
            )
            for d in dilations
        ])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_subtypes, seq_len]
        for conv in self.layers:
            x = x + self.act(conv(x))   # residual so raw logits aren't fully overwritten
        return x

class HIVClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_subtypes: int, hidden_dim: int = None,
                 smooth_kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subtypes),
        )
        self.smooth = DilatedSmoother(
            num_subtypes, kernel_size=5, dilations=(1, 2, 4, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        logits = self.mlp(x)
        logits = logits.transpose(1, 2)
        logits = self.smooth(logits)
        return logits.transpose(1, 2)

class HFModelForHIVSubtyping(PreTrainedModel):
    """Backbone + HIV subtype classification head"""
    config_class = HIVSubtypingConfig

    # Standard Hugging Face attribute for tied weights (e.g. LM head embeddings)
    _tied_weights_keys = []

    def __init__(self, config: HIVSubtypingConfig):
        super().__init__(config)
        self.config = config

        rev = config.backbone_revision
        backbone_config = AutoConfig.from_pretrained(
            config.backbone_name,
            trust_remote_code=True,
            revision=rev,
            code_revision=rev,
        )
        if not hasattr(backbone_config, "is_decoder"):
            backbone_config.is_decoder = False
        self.backbone = AutoModelForMaskedLM.from_config(
            backbone_config,
            trust_remote_code=True,
            code_revision=rev,
        )
        self.backbone.model.lm_head.requires_grad_(False)
        embed_dim = self.backbone.model.lm_head.head.in_features   # = conv_init_embed_dim

        self.subtype_head = HIVClassificationHead(
            embed_dim=embed_dim,
            num_subtypes=config.num_subtypes,
            smooth_kernel=config.smooth_kernel,
        )

        self.post_init()

    @classmethod
    def from_pretrained_backbone(cls, config: HIVSubtypingConfig):
        """Helper method used ONLY when starting training from scratch."""
        model = cls(config)
        model.backbone = AutoModelForMaskedLM.from_pretrained(
            config.backbone_name,
            trust_remote_code=True,
        )
        model.backbone.model.lm_head.requires_grad_(False)
        return model

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        backbone_inputs = {"input_ids": tokens}
        if attention_mask is not None:
            backbone_inputs["attention_mask"] = attention_mask
        outputs = self.backbone(**backbone_inputs, skip_lm_head=True)
        embedding = outputs.embedding

        subtype_logits = self.subtype_head(embedding)
        return {"subtype_logits": subtype_logits}
