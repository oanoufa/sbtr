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
from sklearn.model_selection import train_test_split
from typing import Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


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
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
PAD_MULTIPLE_OF    = config.PAD_LEN
PURE_REF_PATH      = config.PURE_REF_PATH
CRF_REF_PATH       = config.CRF_REF_PATH
VERSION = config.VERSION

from dataset_class import HIVSequenceDataset
from model_class import HFModelForHIVSubtyping, train_step, validation_step
from metrics_class import HIVSubtypingMetrics
from utils import build_hxb2_ata_maps
from figs import visualize_sample
from hmm_decoder_class import HIVDecoder
from crf_decoder_class import CRFReferenceDecoder

import argparse

parser = argparse.ArgumentParser(
    description='Infer HIV-1 subtype per position for sequences aligned to the reference alignment.'
)
parser.add_argument('--sequences_ata', type=str, required=True,
                    help='FASTA file of sequences aligned to the HIV1 subtype reference alignment.')
parser.add_argument('--metadata', type=str, default=None,
                    help="Metadata TSV; if omitted one is generated with split='inference' for every sample.")
parser.add_argument("--tag", type=str, default="inference",
                    help="Text appended to the end of all generated file names.")
parser.add_argument("--out_dir", type=str, default=".",
                    help="Output directory.")
args = parser.parse_args()

sequences_ata = Path(args.sequences_ata)
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
    ata_to_hxb2, hxb2_to_ata = build_hxb2_ata_maps(hxb2_ata_seq)
    print(f"  ATA length, HXB2 length     : {ata_len, int(max(ata_to_hxb2))}")

    # ---- 2. Load query sequences ------------------------------------------
    print(f"\nLoading query sequences from: {sequences_ata}")
    records = list(SeqIO.parse(str(sequences_ata), "fasta"))
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
    zero_lbl_packed = np.zeros((ata_len, n_packed), dtype=np.uint8)
    zero_mask = np.ones(ata_len,             dtype=bool)
    gap_masks = {}

    for i, rec in enumerate(records):
        raw = str(rec.seq).upper()
        
        is_real = np.array([c != '-' for c in raw], dtype=bool)
        gap_masks[rec.id.split()[0]] = is_real

        arr = np.frombuffer(raw.encode(), dtype=np.uint8).copy()
        arr[arr == ord("-")] = ord("N")
        seq_mm[i] = arr
        lbl_mm[i]  = zero_lbl_packed
        mask_mm[i] = zero_mask

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

    # ---- 9. Load checkpoint ---------------------------------------------
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

    out_preds_path = out_dir / f"predictions_{tag}.npy"
    pred_mm = np.lib.format.open_memmap(
        str(out_preds_path), mode="w+", dtype=np.float32,
        shape=(N, MAX_LENGTH, NUM_SUBTYPES),
    )
    print(f"Prediction memmap: {out_preds_path}  shape={pred_mm.shape}")

    inference_rows = (
        metadata[metadata["split"] == "inference"]
        .reset_index(drop=True)
    )

    ID_TO_ST_DICT = {v: k for k, v in ST_TO_ID_DICT.items()}
    hmm_decoder = HIVDecoder(
        id_to_subtype    = ID_TO_ST_DICT,
        crf_labels_path  = f'{WORKSPACE_PATH}/data/output/lanl_crf_label_seqs.npz',
        epsilon          = 1e-6,
        purity_threshold = 0.95,
    )

    crf_decoder = CRFReferenceDecoder(
        bank_path=f'{WORKSPACE_PATH}/data/model/reference_bank/crf_reference_bank_v{VERSION}.npz',
        top_k=5,
    )

    sample_regions_dir = out_dir / f"sample_regions_{tag}"
    sample_regions_dir.mkdir(parents=True, exist_ok=True)

    result_csv_path = out_dir / f"inference_results_{tag}.csv"
    with open(result_csv_path, 'w') as f:
        f.write(
            "sample_name,classification,composition,dominant_subtype,dominant_fraction,"
            "hmm_best_crf,hmm_best_crf_score,hmm_top5,"
            "ref_best_crf,ref_best_distance,ref_top5"
            "\n"
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
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (MAX_LENGTH, NUM_SUBTYPES)
            pred_mm[i] = pred_probs.astype(np.float32)
 
            # Trim to ATA length and decode
            preds    = pred_probs[:ata_len]
            is_real  = gap_masks[sample_name]
 
            hmm_result = hmm_decoder.decode(
                probs       = preds,
                gap_mask    = is_real,
                sample_name = sample_name,
                ata_to_hxb2 = ata_to_hxb2,
            )
            # ---- Write region TSVs ----------------------------------------
            dealigned_regions_path = sample_regions_dir / f"dealigned_regions_{sample_name}.tsv"
            aligned_regions_path   = sample_regions_dir / f"aligned_regions_{sample_name}.tsv"
            hxb2_aligned_path      = sample_regions_dir / f"hxb2_aligned_{sample_name}.tsv"
 
            with open(dealigned_regions_path, 'w') as f_d, \
                 open(aligned_regions_path,   'w') as f_a, \
                 open(hxb2_aligned_path,      'w') as f_h:
 
                for start, end, st in hmm_result.regions_dealigned:
                    f_d.write(f"{start}\t{end}\t{st}\n")
 
                for start, end, st in hmm_result.regions_aligned:
                    f_a.write(f"{start}\t{end}\t{st}\n")
                    hxb2_start = ata_to_hxb2[start]
                    hxb2_end   = ata_to_hxb2[end - 1]
                    f_h.write(f"{hxb2_start}\t{hxb2_end}\t{st}\n")
 
            # ---- Visualisation -------------------------------------------
            sample_pred = {
                "input_ids":      batch["input_ids"][0].cpu().detach(),
                "loss_mask":      batch["loss_mask"][0].cpu().detach(),
                "attention_mask": batch["attention_mask"][0].cpu().detach(),
                "labels":         torch.from_numpy(pred_probs),
            }
            out_path = str(sample_vis_dir / f"inference_sample_{i}_{sample_name}.png")
            visualize_sample(sample=sample_pred,
                             hxb2_to_ata=hxb2_to_ata,
                             pure_st_to_id_dict=ST_TO_ID_DICT,
                             idx=f"inference_{i}_{sample_name}",
                             path=out_path)

            preds_normalized = preds / (preds.sum(axis=-1, keepdims=True) + 1e-9)
            # Subtype/CRF/Novel recombinant prediction
            crf_result = crf_decoder.query(
                probs=preds_normalized,
                query_mask=is_real,
            )

            # ---- Summary CSV row -----------------------------------------
            top5_hmm = "  ".join(
                f"{crf}:{score:.4f}"
                for crf, score in list(hmm_result.top_crf_matches.items())[:5]
            )

            top5_ref = "  ".join(
                f"{r['crf_type']}:{r['max_score']:.4f}"
                for r in crf_result["top_crf_types"][:5]
            )

            best_ref_crf      = crf_result["top_sequences"][0]["name"]
            best_ref_distance = crf_result["top_crf_types"][0]["max_score"]

            with open(result_csv_path, 'a') as f:
                f.write(
                    f"{sample_name},{hmm_result.classification},{hmm_result.composition_str},"
                    f"{hmm_result.dominant_subtype},{hmm_result.dominant_fraction:.4f},"
                    f"{hmm_result.best_crf},{hmm_result.best_crf_score:.4f},{top5_hmm},"
                    f"{best_ref_crf},{best_ref_distance:.4f},{top5_ref}"
                    f"\n"
                )
 
    pred_mm.flush()
    print(f"\nPredictions saved → {out_preds_path}")
    print(f"Results CSV      → {result_csv_path}")