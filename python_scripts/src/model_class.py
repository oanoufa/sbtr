import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM, AutoConfig
from typing import Dict

class HIVSubtypingConfig(PretrainedConfig):
    model_type = "hiv_subtyping"

    def __init__(
        self,
        backbone_name: str = "InstaDeepAI/NTv3_650M_pre",
        num_subtypes: int = 22,
        smooth_kernel: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.num_subtypes = num_subtypes
        self.smooth_kernel = smooth_kernel

class HIVClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_subtypes: int, smooth_kernel: int = 5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_subtypes)
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

class HFModelForHIVSubtyping(PreTrainedModel):
    """NT-V3 backbone + HIV subtype classification head"""
    config_class = HIVSubtypingConfig

    # 1. Standard Hugging Face attribute for tied weights (e.g. LM head embeddings)
    _tied_weights_keys = []

    def __init__(self, config: HIVSubtypingConfig):
        super().__init__(config)
        self.config = config

        backbone_config = AutoConfig.from_pretrained(
            config.backbone_name, 
            trust_remote_code=True
        )
        self.backbone = AutoModelForMaskedLM.from_config(
            backbone_config, 
            trust_remote_code=True
        )

        embed_dim = getattr(
            backbone_config, "embed_dim", getattr(backbone_config, "hidden_size", 1280)
        )

        self.subtype_head = HIVClassificationHead(
            embed_dim=embed_dim,
            num_subtypes=config.num_subtypes,
            smooth_kernel=config.smooth_kernel
        )

        # Initialize weights and property aliases for Hugging Face compatibility
        self.post_init()

    @classmethod
    def from_pretrained_backbone(cls, config: HIVSubtypingConfig):
        """Helper method used ONLY when starting training from scratch."""
        model = cls(config)
        # Load raw pretrained weights into the backbone for training
        model.backbone = AutoModelForMaskedLM.from_pretrained(
            config.backbone_name, 
            trust_remote_code=True
        )
        return model

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        backbone_inputs = {"input_ids": tokens}
        if attention_mask is not None:
            backbone_inputs["attention_mask"] = attention_mask
        outputs = self.backbone(**backbone_inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1]
        subtype_logits = self.subtype_head(embedding)
        return {"subtype_logits": subtype_logits}