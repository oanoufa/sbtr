# inference.py
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
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split   # fixed typo
from typing import Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from pathlib import Path


from huggingface_hub import login
import config
TOKEN_PATH = config.TOKEN_PATH
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
login(token=token)

WORKSPACE_PATH     = config.WORKSPACE_PATH
ST_TO_ID_DICT      = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
MAX_LENGTH         = MODEL_CONFIG["sequence_length"]
PAD_MULTIPLE_OF    = MODEL_CONFIG["pad_multiple_of"]
PURE_REF_PATH      = config.PURE_REF_PATH

from hiv_dataset_class import HIVSequenceDataset
from hiv_nt_training_class import HFModelForHIVSubtyping, train_step, validation_step
from hiv_nt_metrics_class import HIVSubtypingMetrics
from utils import build_hxb2_ata_maps
from figs import visualize_sample

import argparse

parser = argparse.ArgumentParser(
    description='Infer HIV-1 subtype per position for sequences aligned to the reference alignment.'
)
parser.add_argument('--sequences', type=str, required=True,
                    help='FASTA file of sequences aligned to the HIV1 subtype reference alignment.')
parser.add_argument('--metadata', type=str, default=None,
                    help="Metadata TSV; if omitted one is generated with split='inference' for every sample.")
parser.add_argument("--tag", type=str, default="inference",
                    help="Text appended to the end of all generated file names.")
parser.add_argument("--out_dir", type=str, default=".",
                    help="Output directory.")
args = parser.parse_args()

sequences = Path(args.sequences)
tag       = args.tag
out_dir   = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    n_packed = int(np.ceil(NUM_SUBTYPES / 8))

    # ---- 1. Load HXB2 reference -------------------------------------------
    print(f"Loading HXB2 reference from: {PURE_REF_PATH}")
    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(PURE_REF_PATH, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: pure_ref FASTA is empty.")

    ata_len     = len(hxb2_ata_seq)
    hxb2_to_ata = build_hxb2_ata_maps(hxb2_ata_seq)
    max_hxb2    = len(hxb2_to_ata) - 1
    print(f"  ATA length     : {ata_len}")
    print(f"  HXB2 positions : 1 – {max_hxb2}")

    # ---- 2. Load query sequences ------------------------------------------
    print(f"\nLoading query sequences from: {sequences}")
    records = list(SeqIO.parse(str(sequences), "fasta"))
    N = len(records)
    if N == 0:
        sys.exit("ERROR: query FASTA is empty.")
    print(f"  Sequences found: {N}")

    seq_lens = [len(r.seq) for r in records]
    if len(set(seq_lens)) != 1:
        print("  [warn] Not all sequences have the same length — may not be aligned.",
              file=sys.stderr)
    if seq_lens[0] != ata_len:
        sys.exit(
            f"ERROR: query sequences have length {seq_lens[0]} "
            f"but ATA alignment has length {ata_len}."
        )

    # ---- 4. Build / load metadata -----------------------------------------
    # If the user supplied a metadata file, load it.
    # Otherwise generate one so that every sequence lands in the
    # 'inference' split and the rest of the pipeline is unchanged.
    if args.metadata is not None:
        metadata = pd.read_csv(args.metadata, sep="\t")
        print(f"\nLoaded metadata from: {args.metadata}  ({len(metadata)} rows)")
    else:
        seq_names = [rec.id.split()[0] for rec in records]
        metadata  = pd.DataFrame({
            "sequence_name": seq_names,
            "split":         "inference",
        })
        generated_meta_path = out_dir / f"metadata_{tag}.tsv"
        metadata.to_csv(generated_meta_path, sep="\t", index=False)
        print(f"\nNo metadata provided — generated: {generated_meta_path}")

    # ---- 5. Allocate memmaps ----------------------------------------------
    out_seqs  = out_dir / f"sequences_{tag}.npy"
    out_lbls  = out_dir / f"labels_{tag}.npy"
    out_masks = out_dir / f"loss_masks_{tag}.npy"

    seq_mm  = np.lib.format.open_memmap(str(out_seqs),  mode="w+", dtype=np.uint8,
                                         shape=(N, ata_len))
    lbl_mm  = np.lib.format.open_memmap(str(out_lbls),  mode="w+", dtype=np.uint8,
                                         shape=(N, ata_len, n_packed))
    mask_mm = np.lib.format.open_memmap(str(out_masks), mode="w+", dtype=bool,
                                         shape=(N, ata_len))

    print(f"\nAllocated memmaps:")
    print(f"  sequences  : {out_seqs}   shape={seq_mm.shape}")
    print(f"  labels     : {out_lbls}  shape={lbl_mm.shape}")
    print(f"  loss_masks : {out_masks} shape={mask_mm.shape}")

    # ---- 6. Fill memmaps --------------------------------------------------
    print("\nProcessing sequences …")
    zero_lbl_packed = np.zeros((ata_len, n_packed), dtype=np.uint8)
    zero_mask       = np.ones(ata_len,             dtype=bool)

    for i, rec in enumerate(records):
        raw = str(rec.seq).upper()
        arr = np.frombuffer(raw.encode(), dtype=np.uint8).copy()
        arr[arr == ord("-")] = ord("N")
        seq_mm[i] = arr
        lbl_mm[i]  = zero_lbl_packed
        mask_mm[i] = zero_mask

        if (i + 1) % max(1, N // 10) == 0 or i == N - 1:
            print(f"  [{i+1}/{N}]")

    seq_mm.flush()
    lbl_mm.flush()
    mask_mm.flush()

    # ---- 7. Model + tokenizer --------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"], trust_remote_code=True
    )
    device = torch.device(MODEL_CONFIG["device"])
    print(f"\nUsing device: {device}")

    model = HFModelForHIVSubtyping(
        model_name=MODEL_CONFIG["model_name"], num_subtypes=NUM_SUBTYPES
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 8. Dataset + loader ---------------------------------------------
    inference_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="inference",
    )
    inference_loader = DataLoader(
        inference_dataset, batch_size=1,
        shuffle=False, num_workers=MODEL_CONFIG["num_workers"],
    )
    print(f"Inference samples: {len(inference_dataset)}")

    # ---- 9. Metrics (only when labels are available) ---------------------
    id_to_st   = {v: k for k, v in ST_TO_ID_DICT.items()}
    n_token_id = tokenizer.encode("N", add_special_tokens=False)[0]

    # ---- 10. Load checkpoint ---------------------------------------------
    print(f"\nLoading checkpoint …")
    checkpoint = torch.load(
        os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # ---- 11. Inference loop ----------------------------------------------
    sample_vis_dir = Path(WORKSPACE_PATH) / "figs" / f"sample_vis_{tag}"
    sample_vis_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate a memory-mapped file for all predictions so we never
    # have to hold the full (N, ata_len, NUM_SUBTYPES) float array in RAM.
    out_preds_path = out_dir / f"predictions_{tag}.npy"
    pred_mm = np.lib.format.open_memmap(
        str(out_preds_path), mode="w+", dtype=np.float32,
        shape=(N, MODEL_CONFIG["sequence_length"], NUM_SUBTYPES),
    )
    print(f"Prediction memmap: {out_preds_path}  shape={pred_mm.shape}")

    # We'll also collect per-sample metadata rows for a summary TSV.
    inference_rows = (
        metadata[metadata["split"] == "inference"]
        .reset_index(drop=True)
    )

    print("\nRunning inference …")
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(inference_loader),
            total=len(inference_loader),
            mininterval=30,
            desc="Inference",
        ):
            sample_name = inference_rows.iloc[i]["sequence_name"]

            # Forward pass
            logits     = model(batch["input_ids"].to(device))["subtype_logits"]
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (ata_len, NUM_SUBTYPES)

            # ---- Save predictions to memmap (one row per sample) ----
            pred_mm[i] = pred_probs.astype(np.float32)

            # ---- Per-sample visualisations -----------------
            sample_pred = {
                "input_ids":      batch["input_ids"][0].cpu().detach(),
                "loss_mask":      batch["loss_mask"][0].cpu().detach(),
                "attention_mask": batch["attention_mask"][0].cpu().detach(),
                "labels":         torch.from_numpy(pred_probs),
            }
            out_path = str(sample_vis_dir / f"inference_sample_{i}_{sample_name}.png")
            visualize_sample(sample=sample_pred,
                                pure_st_to_id_dict=ST_TO_ID_DICT,
                                idx=f"inference_{i}_{sample_name}",
                                path=out_path)

    pred_mm.flush()
    print(f"\nPredictions saved  → {out_preds_path}")