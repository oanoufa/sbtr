import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from Bio import SeqIO
import os
import sys

from . import config

workspace_path = config.WORKSPACE_PATH
pure_st_to_id_dict = config.ST_TO_ID_DICT
tokenizer  = AutoTokenizer.from_pretrained("oanoufa/sbtr_ntv3_650M", trust_remote_code=True, revision="main")
max_length = config.SEQ_LEN_AFTER_PAD
pad_multiple_of = config.PAD_LEN



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
        hxb2_to_ata: dict,
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
        self.hxb2_to_ata     = hxb2_to_ata
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

        # Mask pad tokens
        attention_mask = (input_ids != self.pad_token_id).long()

        # Mask biological ambiguity
        bio_mask_tensor = torch.zeros(self.max_length, dtype=torch.long)
        bio_mask_tensor[:len(seq_str)] = torch.from_numpy(bio_mask.astype(np.int64))
        
        # Mask 5', 3' and DRMs
        drm_mask = torch.ones(self.max_length, dtype=torch.long)
        for hxb2_pos in config.MASKED_POSITIONS_HXB2:
            ata_pos = self.hxb2_to_ata[hxb2_pos]
            drm_mask[ata_pos] = 0


        # Mask N tokens
        valid_nucleotide_mask = (input_ids != self.n_token_id).long()

        # Loss mask is 1 where the token is valid, 0 elsewhere
        loss_mask = attention_mask * bio_mask_tensor * valid_nucleotide_mask * drm_mask

        # Build token-level label tensor
        seq_len      = len(seq_str)
        token_labels = torch.zeros(self.max_length, self.n_subtypes, dtype=torch.float32)
        token_labels[:seq_len] = torch.from_numpy(per_site.astype(np.float32))

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "loss_mask":      loss_mask,
            "labels":         token_labels,
        }
