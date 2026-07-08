# inference.py
import gzip
import numpy as np
import random
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
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter

from huggingface_hub import login
from mutator_class import SequenceMutator

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
ATA_LEN            = config.SEQ_LEN
PAD_MULTIPLE_OF    = config.PAD_LEN
PURE_REF_PATH      = config.PURE_REF_PATH
VERSION            = config.VERSION

from dataset_class import HIVSequenceDataset
from model_class import HFModelForHIVSubtyping
from utils import build_hxb2_ata_maps

import argparse

parser = argparse.ArgumentParser(
    description='Generate a bank of reference predictions for each CRF'
)
parser.add_argument('--crf_file_path', type=str, required=True,
                    help='FASTA file of CRF aligned to the HIV1 subtype reference alignment.')
args = parser.parse_args()

CRF_FILE_PATH = Path(args.crf_file_path)
out_dir = Path(WORKSPACE_PATH) / "data" / "model" / "reference_bank"
out_dir.mkdir(parents=True, exist_ok=True)

mutator = SequenceMutator(
        iqtree_dir=f"{WORKSPACE_PATH}/data/output/rates/",
        ata_len=ATA_LEN,
        seed=42,
        cache_dir=f"{WORKSPACE_PATH}/data/input/diversity/",
    )

def build_crf_reference_bank(
    crf_ref_path: str,
    max_per_crf: int = 5,
    max_per_test: int = 3,
    seed:         int = 42,
    mutator:      Optional[SequenceMutator] = None) -> tuple:
    """
    Build a CRF reference bank from a FASTA file.

    Steps
    1. Parse all sequences from *crf_ref_path*.
    2. Extract (CRF type, accession) from each sequence ID, handling two formats:
         • ``Ref.01_AE.CN.05.FJ051.DQ859178``    (with ``Ref.`` prefix)
         • ``01_AE.TH.2007.AA028a_wg7.JX447031``  (without prefix)
    3. Group by CRF type; deduplicate on accession (first occurrence kept).
    4. Retain at most *max_per_crf* sequences per CRF by random sampling.

    Parameters
    crf_ref_path : str
        Path to the CRF reference FASTA file.
    max_per_crf : int
        Maximum number of sequences to retain per CRF type (default 5).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    list[SeqRecord]
        Randomly sampled CRF reference sequences.
    """

    #  1. Load sequences                                                   #
    random.seed(seed)
    print(f"\nBuilding CRF reference bank from: {crf_ref_path}")
    all_records: list[SeqRecord] = list(SeqIO.parse(crf_ref_path, "fasta"))
    if not all_records:
        sys.exit("ERROR: CRF reference FASTA is empty.")
    print(f"  Loaded {len(all_records)} CRF reference sequences")

    #  2. Parse CRF type + accession                                       #
    _REF_PREFIX = re.compile(r"^Ref\.")

    def parse_id(record_id: str) -> tuple[str, str]:
        """
        Strip the optional ``Ref.`` prefix, then return
        (first field, last field) as (crf_type, accession).

        Examples
        --------
        ``'Ref.01_AE.CN.05.FJ051.DQ859178'``    → ``('01_AE', 'DQ859178')``
        ``'01_AE.TH.2007.AA028a_wg7.JX447031'`` → ``('01_AE', 'JX447031')``
        """
        clean = _REF_PREFIX.sub("", record_id)
        parts = clean.split(".")
        return parts[0], parts[-1]

    #  3. Group by CRF; deduplicate on accession                           #
    crf_groups: dict[str, dict[str, SeqRecord]] = defaultdict(dict)

    for rec in all_records:
        if "HXB2" in rec.id:
            continue
        crf_type, accession = parse_id(rec.id)
        crf_groups[crf_type].setdefault(accession, rec)  # first occurrence wins

    # Sort by numeric CRF prefix
    crf_groups = dict(sorted(crf_groups.items(), key=lambda x: int(x[0].split("_")[0])))

    print(f"  Found {len(crf_groups)} CRF type(s): {', '.join(crf_groups)}")
    for crf, acc_map in crf_groups.items():
        print(f"    {crf:<12s}: {len(acc_map):3d} unique sequence(s)")

    #  4. Random sampling – at most max_per_crf per CRF                   #
    bank:     list[SeqRecord] = []
    test_set: list[SeqRecord] = []

    for crf_type, acc_map in crf_groups.items():
        records = list(acc_map.values())

        # Augment the full pool to (max_per_crf + max_per_test) if needed
        target_total = max_per_crf + max_per_test
        if mutator is not None and len(records) < target_total:
            records = mutator.augment_to_target(
                records, target_count=target_total, subtype_key='avg'
            )

        chosen   = random.sample(records, min(max_per_crf, len(records)))
        leftover = [r for r in records if r.id not in {r2.id for r2 in chosen}]
        test     = random.sample(leftover, min(max_per_test, len(leftover)))

        bank.extend(chosen)
        test_set.extend(test)

        print(f"    {crf_type:<12s}: bank {len(chosen)}/{len(records)}, test {len(test)}/{len(leftover)} leftover")

    print(
        f"\n  CRF reference bank ready : {len(bank)} sequences "
        f"({len(crf_groups)} CRF type(s), ≤{max_per_crf} per type)"
    )
    print(
        f"  CRF test set ready       : {len(test_set)} sequences "
        f"({len(crf_groups)} CRF type(s), ≤{max_per_test} per type)"
    )
    return bank, test_set

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_packed = int(np.ceil(NUM_SUBTYPES / 8))

    # ---- Load HXB2 reference -------------------------------------------
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

    # ---- Model + tokenizer --------------------------------------------
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

    # ---- Load checkpoint ---------------------------------------------
    print(f"\nLoading checkpoint …")
    checkpoint = torch.load(
        os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # ---- CRF reference bank ------------------------------------------
    # Tokenize each CRF reference sequence directly (no HIVSequenceDataset wrapper needed).  We construct the attention mask ourselves from pad_token_id so we never rely on the tokenizer returning it.
    print(f"\nBuilding CRF reference bank from: {CRF_FILE_PATH}", flush=True)
    sequence_bank, test_set = build_crf_reference_bank(
            crf_ref_path=CRF_FILE_PATH,
            max_per_crf=5,
            max_per_test=3,
            mutator=mutator,
        )
    N = len(sequence_bank)
    # Filter records to keep 5 per CRF sampled according to their divergence
    # ---- Build metadata -----------------------------------------
    seq_names = [rec.id.split()[0] for rec in sequence_bank]
    metadata  = pd.DataFrame({
        "sequence_name": seq_names,
        "split":         "crf_bank",
    })
    generated_meta_path = out_dir / f"metadata_v{VERSION}.tsv"
    metadata.to_csv(generated_meta_path, sep="\t", index=False)
    print(f"\nNo metadata provided — generated: {generated_meta_path}")
    metadata_df = (
        metadata[metadata["split"] == "crf_bank"]
        .reset_index(drop=True)
    )

    # ---- Allocate memmaps ----------------------------------------------
    out_seqs  = out_dir / f"sequences_v{VERSION}.npy"
    out_lbls  = out_dir / f"labels_v{VERSION}.npy"
    out_masks = out_dir / f"loss_masks_v{VERSION}.npy"

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

    for i, rec in enumerate(sequence_bank):
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

    bank_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="crf_bank",
    )
    bank_loader = DataLoader(
        bank_dataset, batch_size=1,
        shuffle=False, num_workers=MODEL_CONFIG["num_workers"],
    )
    print(f"Bank samples: {len(bank_dataset)}")

    crf_names:    List[str]        = []
    crf_profiles: List[np.ndarray] = []   # each (MAX_LENGTH, NUM_SUBTYPES) float32

    crf_dict = {rec.id.split('.')[1] : 0 for rec in sequence_bank}
    # Store the number of times the crf was seen so that each CRF is only computed at most 3 times

    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(bank_loader),
            total=len(bank_loader),
            mininterval=30,
            desc="Generating CRF bank",
        ):

            sample_name = metadata_df.iloc[i]["sequence_name"]
            crf_id = sample_name.split('.')[1]
            if crf_id == 'B': # HXB2 ref
                continue
            if crf_dict.get(crf_id, 0) >= 10:
                continue
            crf_dict[crf_id] += 1

            # Forward pass
            logits     = model(batch["input_ids"].to(device))["subtype_logits"]
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (MAX_LENGTH, NUM_SUBTYPES)
            # Trim to ATA length and decode
            probs    = pred_probs[:ATA_LEN]
            probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-9)
            is_real  = gap_masks[sample_name]
            probs = probs * is_real[:, None]
            crf_names.append(sample_name)
            crf_profiles.append(probs.astype(np.float32))

    reference_bank  = np.stack(crf_profiles,    axis=0)  # (R, ATA_LEN, NUM_SUBTYPES)
    reference_names = np.array(crf_names)

    print(f"  Reference bank  : {reference_bank.shape}  "
          f"dtype={reference_bank.dtype}")

    # Save the reference bank to a compressed npz for later use in inference.
    out_path = out_dir / f"crf_reference_bank_v{VERSION}.npz"
    np.savez_compressed(
        out_path,
        reference_bank=reference_bank,
        reference_names=reference_names,
    )
    print(f"\nSaved CRF reference bank to: {out_path}")
    
    out_path_test = out_dir / f"crf_test_set_v{VERSION}.fasta"
    with open(out_path_test, "w") as output_f:
        writer = FastaWriter(output_f, wrap=100000)
        for record in test_set:
            record.description = ""
            seq = record.seq.upper()
            seq = seq.replace('-', "")
            seq = seq.strip('N')
            record.seq = seq
            record.id = str(record.id).replace("Ref.", "")
            writer.write_record(record)