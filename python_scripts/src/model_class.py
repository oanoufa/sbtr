"""Define the sbtr model configuration and classification layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM, AutoConfig
from typing import Dict

class HIVSubtypingConfig(PretrainedConfig):
    model_type = "hiv_subtyping"

    def __init__(
        self,
        backbone_name: str = "InstaDeepAI/NTv3_650M_pre",
        num_subtypes: int = 22,
        smooth_kernel: int = 5,
        embed_layer: int = -1,
        # only used when backbone_name == "custom"
        custom_vocab_size: int = 8,
        custom_embed_dim: int = 256,
        custom_num_layers: int = 9,
        custom_num_heads: int = 8,
        custom_ffn_dim: int = 1024,
        custom_max_length: int = 11648,
        custom_dropout: float = 0.1,
        custom_pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.num_subtypes = num_subtypes
        self.smooth_kernel = smooth_kernel
        self.embed_layer = embed_layer
        self.custom_vocab_size = custom_vocab_size
        self.custom_embed_dim = custom_embed_dim
        self.custom_num_layers = custom_num_layers
        self.custom_num_heads = custom_num_heads
        self.custom_ffn_dim = custom_ffn_dim
        self.custom_max_length = custom_max_length
        self.custom_dropout = custom_dropout
        self.custom_pad_token_id = custom_pad_token_id

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
    """Backbone + HIV subtype classification head"""
    config_class = HIVSubtypingConfig

    # Standard Hugging Face attribute for tied weights (e.g. LM head embeddings)
    _tied_weights_keys = []

    def __init__(self, config: HIVSubtypingConfig):
        super().__init__(config)
        self.config = config
        self.is_custom_backbone = (config.backbone_name == "custom")

        if self.is_custom_backbone:
            backbone_config = CustomBackboneConfig(
                vocab_size=config.custom_vocab_size,
                embed_dim=config.custom_embed_dim,
                num_layers=config.custom_num_layers,
                num_heads=config.custom_num_heads,
                ffn_dim=config.custom_ffn_dim,
                max_length=config.custom_max_length,
                dropout=config.custom_dropout,
                pad_token_id=config.custom_pad_token_id,
            )
            self.backbone = CustomBackbone(backbone_config)
            embed_dim = config.custom_embed_dim
        else:
            backbone_config = AutoConfig.from_pretrained(
                config.backbone_name,
                trust_remote_code=True,
            )
            if not hasattr(backbone_config, "is_decoder"):
                backbone_config.is_decoder = False
            self.backbone = AutoModelForMaskedLM.from_config(
                backbone_config,
                trust_remote_code=True,
            )
            embed_dim = getattr(
                backbone_config, "embed_dim", getattr(backbone_config, "hidden_size", 1280)
            )

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
        if config.backbone_name == "custom":
            return model  # custom backbone has no pretrained weights to load

        backbone = AutoModelForMaskedLM.from_pretrained(
                config.backbone_name,
                trust_remote_code=True,
            )
        model.backbone = backbone
        return model

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        if self.is_custom_backbone:
            embedding = self.backbone(input_ids=tokens, attention_mask=attention_mask)
        else:
            backbone_inputs = {"input_ids": tokens}
            if attention_mask is not None:
                backbone_inputs["attention_mask"] = attention_mask
            outputs = self.backbone(**backbone_inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[self.config.embed_layer]

        subtype_logits = self.subtype_head(embedding)
        return {"subtype_logits": subtype_logits}



class CustomBackboneConfig:
    def __init__(self, vocab_size=8, embed_dim=768, num_layers=12,
                 num_heads=12, ffn_dim=3072, max_length=11648, dropout=0.1,
                 pad_token_id=0):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_length = max_length
        self.dropout = dropout
        self.pad_token_id = pad_token_id


def build_rotary_cache(seq_len, head_dim, device, base=10000.0):
    # Standard RoPE: Split head_dim into two halves [0..d/2] and [d/2..d]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [seq_len, head_dim / 2]
    emb = torch.cat((freqs, freqs), dim=-1)               # [seq_len, head_dim]
    
    # Reshape to [1, 1, seq_len, head_dim] for explicit 4D broadcasting with [B, H, T, D]
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def rotate_half(x):
    # Split the last dimension into two equal halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    # x: [B, num_heads, T, head_dim]
    # cos, sin: [1, 1, T, head_dim]
    return (x * cos) + (rotate_half(x) * sin)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, num_heads, T, head_dim]

        # Apply fixed RoPE
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # Mask format for PyTorch SDPA: [B, 1, 1, T] or [B, 1, T, T]
            attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

        # F.scaled_dot_product_attention automatically invokes FlashAttention-2 
        # when inputs are CUDA tensors with half precision (FP16/BF16)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, config: CustomBackboneConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = HybridSelfAttention(config.embed_dim, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embed_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CustomBackbone(nn.Module):
    def __init__(self, config: CustomBackboneConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim,
                                         padding_idx=config.pad_token_id)
        self.head_dim = config.embed_dim // config.num_heads
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        B, T = input_ids.shape
        x = self.token_embed(input_ids)

        # Build RoPE cache once per forward pass
        cos, sin = build_rotary_cache(T, self.head_dim, input_ids.device)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask)

        return self.final_norm(x)



class HybridSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, window_size=512, global_stride=256):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size  # Total window size (e.g., 256 left + 256 right)
        self.global_stride = global_stride # Spacing for global anchor tokens

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def _create_hybrid_mask(self, T, device, key_padding_mask=None):
        # 1. Start with full restriction (-inf)
        mask = torch.full((T, T), float("-inf"), device=device)

        # 2. Local Banded Window: positions within window_size/2 attend to each other
        half_w = self.window_size // 2
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        local_mask = (col_idx >= row_idx - half_w) & (col_idx <= row_idx + half_w)
        mask[local_mask] = 0.0

        # 3. Global Anchor Tokens: Every `global_stride` token can see ALL, and ALL can see it
        # Token 0 ([CLS]) + evenly spaced tokens across the 11.6k genome
        global_indices = torch.arange(0, T, self.global_stride, device=device)
        
        mask[global_indices, :] = 0.0  # Global tokens attend to everyone
        mask[:, global_indices] = 0.0  # Everyone attends to global tokens

        # Broadcast mask to shape [1, 1, T, T] for batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 4. Merge Key Padding Mask if present
        if key_padding_mask is not None:
            # key_padding_mask is [B, T], True where padded
            pad_mask = key_padding_mask[:, None, None, :]  # [B, 1, 1, T]
            mask = mask.masked_fill(pad_mask, float("-inf"))

        return mask

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, num_heads, T, head_dim]

        # Apply RoPE
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Generate combined local-window + global-anchor mask
        attn_mask = self._create_hybrid_mask(T, x.device, key_padding_mask)

        # SDPA handles custom attention masks cleanly
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)