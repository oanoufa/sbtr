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

constants_path = "/users/mpath/oanoufa/HIV_PROJECT/scripts/utils/constants.py"
sys.path.append(os.path.dirname(constants_path))
import constants
import model_config

pure_st_to_id_dict = constants.ST_TO_ID_DICT
tokenizer  = AutoTokenizer.from_pretrained(model_config.config["model_name"], trust_remote_code=True)
max_length = model_config.config["sequence_length"]
pad_multiple_of = model_config.config["pad_multiple_of"]


def open_memmaps(sequences_path: str, labels_path: str, masks_path: str):
    """
    Open pre-existing memmap files in read-only mode, including the biological loss mask.
    """
    seq_mm = np.load(sequences_path, mmap_mode='r')
    lbl_mm = np.load(labels_path,    mmap_mode='r')
    mask_mm = np.load(masks_path,    mmap_mode='r')
    return seq_mm, lbl_mm, mask_mm

class HIVSequenceDataset(Dataset):
    """
    PyTorch Dataset for HIV per-token subtype labeling backed by memmap files.

    Neither the sequence array nor the label array is loaded into RAM at init.
    Each __getitem__ call pages in exactly two rows (one sequence, one label
    matrix) from disk via the OS page cache — making this suitable for
    datasets that don't fit in memory.

    NTv3 tokenizer notes
    --------------------
    - Single-base tokenization: 1 token == 1 nucleotide.
    - Sequences must be padded to a multiple of `pad_multiple_of` (128 for the
      7-downsample variant, 32 for the 5-downsample variant).
    - `add_special_tokens=False` — NTv3 is used without CLS/EOS.
    """
    def __init__(
        self,
        seq_mm: np.memmap,              # (n_seq, ata_len)  uint8
        lbl_mm: np.memmap,              # (n_seq, ata_len, n_subtypes)  bool (packed)
        mask_mm: np.memmap,             # (n_seq, ata_len) bool (ambiguity mask)
        metadata: pd.DataFrame,
        tokenizer: AutoTokenizer,
        n_subtypes: int,
        max_length: int,
        pad_multiple_of: int,
        split: str = "train",
    ):
        super().__init__()
        self.seq_mm          = seq_mm
        self.lbl_mm          = lbl_mm
        self.mask_mm         = mask_mm
        self.tokenizer       = tokenizer
        self.n_subtypes      = n_subtypes
        self.max_length      = max_length
        self.pad_multiple_of = pad_multiple_of
        self.pad_token_id    = tokenizer.pad_token_id
        self.n_token_id      = tokenizer.encode("N", add_special_tokens=False)[0]
        meta = metadata.reset_index(drop=True)
        self.indices = meta.index[meta["split"] == split].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row = self.indices[idx]

        seq_uint8 = self.seq_mm[row]                          
        packed    = self.lbl_mm[row]                          
        bio_mask  = self.mask_mm[row]
        
        per_site = np.unpackbits(packed, axis=-1, count=self.n_subtypes)

        seq_str = seq_uint8.tobytes().decode('ascii')
        seq_str  = seq_str[:self.max_length]
        per_site = per_site[:self.max_length]
        bio_mask = bio_mask[:self.max_length]

        enc = self.tokenizer(
            seq_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_multiple_of,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]                       

        # Standard transformer attention mask
        attention_mask = (input_ids != self.pad_token_id).long()

        # Combine attention_mask with the biological ambiguity mask
        bio_mask_tensor = torch.zeros(self.max_length, dtype=torch.long)
        bio_mask_tensor[:len(seq_str)] = torch.from_numpy(bio_mask.astype(np.int64))
        
        # Mask N tokens as well
        valid_nucleotide_mask = (input_ids != self.n_token_id).long()

        # Loss mask is 1 ONLY if it's a real token AND not a biologically ambiguous boundary
        loss_mask = attention_mask * bio_mask_tensor * valid_nucleotide_mask

        # Build token-level label tensor
        seq_len      = len(seq_str)
        token_labels = torch.zeros(self.max_length, self.n_subtypes, dtype=torch.float32)
        token_labels[:seq_len] = torch.from_numpy(per_site.astype(np.float32))

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "loss_mask":      loss_mask,       # Used for cross-entropy masking
            "labels":         token_labels,    
        }


def visualize_sample(
    sample: dict,
    pure_st_to_id_dict: dict,
    idx: int = 0,
    path: str = None,
) -> None:
    """
    Visualize one dataset sample:
      - Top panel   : attention mask (1 = real token, 0 = padding)
      - Bottom panel: per-token label heatmap (real tokens only)
    """
    attention_mask = sample["attention_mask"].numpy()
    labels         = sample["labels"].numpy()
    if hasattr(attention_mask, 'ndim') and attention_mask.ndim > 1:
            attention_mask = attention_mask[0]
            labels = labels[0]
    real_mask   = attention_mask.astype(bool)
    n_real      = real_mask.sum()
    real_labels = labels[real_mask].T                         # (n_subtypes, n_real)

    id_to_st      = {v: k for k, v in pure_st_to_id_dict.items()}
    subtype_names = [id_to_st[i] for i in range(len(pure_st_to_id_dict))]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 4 + 0.3 * len(subtype_names)),
        gridspec_kw={"height_ratios": [1, len(subtype_names)]},
    )

    ax_mask = axes[0]
    ax_mask.imshow(
        attention_mask[np.newaxis, :], aspect="auto",
        cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
    )
    ax_mask.set_yticks([0])
    ax_mask.set_yticklabels(["attention\nmask"], fontsize=8)
    ax_mask.xaxis.set_major_locator(ticker.MultipleLocator(max(1, len(attention_mask) // 10)))
    ax_mask.set_title(
        f"Sample {idx}  —  {n_real} real tokens / {len(attention_mask)} total  "
        f"({len(attention_mask) - n_real} padding)",
        fontsize=10,
    )
    if n_real < len(attention_mask):
        ax_mask.axvline(n_real - 0.5, color="red", linewidth=1.2, linestyle="--")

    ax_lab = axes[1]
    im = ax_lab.imshow(
        real_labels, aspect="auto",
        cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
    )
    ax_lab.set_yticks(range(len(subtype_names)))
    ax_lab.set_yticklabels(subtype_names, fontsize=7)
    ax_lab.set_xlabel("Token position (real tokens only)", fontsize=9)
    ax_lab.xaxis.set_major_locator(ticker.MultipleLocator(max(1, n_real // 10)))
    plt.colorbar(im, ax=ax_lab, fraction=0.02, pad=0.01, label="label (0/1)")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}")


if __name__ == "__main__":
    workspace_path = "/users/mpath/oanoufa/HIV_PROJECT"

    sequences_path = f"{workspace_path}/data/GEN/sequences.npy"
    labels_path    = f"{workspace_path}/data/GEN/labels.npy"
    metadata_path  = f"{workspace_path}/data/GEN/sequence_metadata.tsv"

    seq_mm, lbl_mm = open_memmaps(sequences_path, labels_path)
    metadata = pd.read_csv(metadata_path, sep='\t')

    assert seq_mm.shape[0] == len(metadata), "Mismatch between sequences and metadata counts"
    print(f"Sequences memmap shape : {seq_mm.shape}")
    print(f"Labels memmap shape    : {lbl_mm.shape}")

    dataset = HIVSequenceDataset(
        seq_mm          = seq_mm,
        lbl_mm          = lbl_mm,
        metadata        = metadata,
        tokenizer       = tokenizer,
        n_subtypes      = len(pure_st_to_id_dict),
        max_length      = max_length,
        pad_multiple_of = pad_multiple_of,
        split           = "train",
    )

    sample_idx = np.random.randint(len(dataset))
    sample     = dataset[sample_idx]
    print("Input IDs shape:     ", sample["input_ids"].shape)
    print("Attention mask shape:", sample["attention_mask"].shape)
    print("Labels shape:        ", sample["labels"].shape)
    print(f"Real tokens:          {sample['attention_mask'].sum().item()} / {max_length}")
    print(f"Active label slots:   {(sample['labels'].sum(dim=-1) > 0).sum().item()} tokens with >= 1 label")

    id_to_st = {v: k for k, v in pure_st_to_id_dict.items()}
    with open(f"{workspace_path}/figs/sample_vis/sample_labels.txt", "w") as f:
        for sample_idx in range(10):
            sample      = dataset[sample_idx]
            sample_name = metadata.loc[metadata["split"] == "train"].iloc[sample_idx]["sequence_name"]
            print(f"\nVisualizing sample {sample_idx}: {sample_name}")
            out_path = f"{workspace_path}/figs/sample_vis/sample_{sample_idx}_{sample_name}.png"
            visualize_sample(sample, pure_st_to_id_dict,
                             idx=str(sample_idx) + '_' + sample_name, path=out_path)
            f.write("start\tend\tsubtype\n")
            labels_np     = sample["labels"].numpy()
            current_label = None
            current_start = None
            for pos in range(labels_np.shape[0]):
                label_vec = labels_np[pos]
                if label_vec.sum() == 0:
                    label_str = None
                else:
                    active = [id_to_st[i] for i in range(len(pure_st_to_id_dict)) if label_vec[i] > 0]
                    label_str = "/".join(active)
                if label_str != current_label:
                    if current_label is not None:
                        f.write(f"{current_start}\t{pos}\t{current_label}\n")
                    current_label = label_str
                    current_start = pos
            if current_label is not None:
                f.write(f"{current_start}\t{labels_np.shape[0]}\t{current_label}\n")