"""Build a reference bank of HIV sequences for CRF comparisons."""

import gzip
import numpy as np
import io
import random
import torch
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import re
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter
from huggingface_hub import login, HfApi

from src.mutator_class import SequenceMutator
from src import config
from src.dataset_class import HIVSequenceDataset
from src.model_class import HFModelForHIVSubtyping

TOKEN_PATH = config.TOKEN_PATH
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
login(token=token)

WORKSPACE_PATH     = config.WORKSPACE_PATH
ST_TO_ID_DICT      = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
ATA_LEN            = config.ATA_LEN
PAD_MULTIPLE_OF    = config.PAD_LEN
PURE_REF_PATH      = config.PURE_REF_PATH
VERSION            = config.VERSION

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

GAG_HXB2 = (790, 2292)
POL_HXB2 = (2085, 5096)
PCT_PER_CRF_BANK = config.PCT_PER_CRF_BANK # Adaptive bank size depending on the number of sequences of the CRF
MIN_PER_CRF_BANK = config.MIN_PER_CRF_BANK # Min bank size for each CRF
N_TEST           = config.N_TEST # Min test size for each CRF (including one gag and one pol sequence)

def _crop_record(rec: SeqRecord, ata_start: int, ata_end: int, suffix: str) -> SeqRecord:
    cropped = rec[ata_start:ata_end]
    cropped.id = f"{rec.id}_{suffix}"
    cropped.description = ""
    return cropped

def build_crf_reference_bank(
    crf_ref_path: str,
    pct_per_crf:  float = 0.10,
    min_per_crf:  int = 3,
    n_test:       int = 5,
    seed:         int = 42,
    mutator:      Optional[SequenceMutator] = None,
    hxb2_to_ata:  Optional[np.ndarray] = None) -> tuple:
    """
    Build a CRF reference bank from a FASTA file.

    Steps
    1. Parse all sequences from *crf_ref_path*.
    2. Extract (CRF type, accession) from each sequence ID, handling two formats:
         - ``Ref.01_AE.CN.05.FJ051.DQ859178``    (with ``Ref.`` prefix)
         - ``01_AE.TH.2007.AA028a_wg7.JX447031``  (without prefix)
    3. Group by CRF type; deduplicate on accession (first occurrence kept).
    4. Retain round(pct_per_crf * n_unique) sequences per CRF (floored at
       min_per_crf), computed on the unaugmented unique count, by random
       sampling.

    Parameters
    crf_ref_path : str
        Path to the CRF reference FASTA file.
    pct_per_crf : float
        Fraction of each CRF's unique (pre-augmentation) sequence count to
        retain in the bank (default 0.10).
    min_per_crf : int
        Minimum number of sequences to retain per CRF type, regardless of
        percentage (default 3).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    list[SeqRecord]
        Randomly sampled CRF reference sequences.
    """

    #  1. Load sequences
    random.seed(seed)
    print(f"\nBuilding CRF reference bank from: {crf_ref_path}")
    all_records: list[SeqRecord] = list(SeqIO.parse(crf_ref_path, "fasta"))
    if not all_records:
        sys.exit("ERROR: CRF reference FASTA is empty.")
    print(f"  Loaded {len(all_records)} CRF reference sequences")

    #  2. Parse CRF type + accession
    _REF_PREFIX = re.compile(r"^Ref\.")

    def parse_id(record_id: str) -> tuple[str, str]:
        """
        Strip the optional ``Ref.`` prefix, then return
        (first field, last field) as (crf_type, accession).

        Examples
        --------
        ``'Ref.01_AE.CN.05.FJ051.DQ859178'``    -> ``('01_AE', 'DQ859178')``
        ``'01_AE.TH.2007.AA028a_wg7.JX447031'`` -> ``('01_AE', 'JX447031')``
        """
        clean = _REF_PREFIX.sub("", record_id)
        parts = clean.split(".")
        return clean, parts[0], parts[-1]

    #  3. Group by CRF; clean id, deduplicate on accession
    crf_groups: dict[str, dict[str, SeqRecord]] = defaultdict(dict)

    for rec in all_records:
        if "HXB2" in rec.id:
            continue
        clean_id, crf_type, accession = parse_id(rec.id)
        rec.id = clean_id
        crf_groups[crf_type].setdefault(accession, rec)  # first occurrence wins

    # Sort by numeric CRF prefix
    crf_groups = dict(sorted(crf_groups.items(), key=lambda x: int(x[0].split("_")[0])))

    print(f"  Found {len(crf_groups)} CRF type(s): {', '.join(crf_groups)}")
    for crf, acc_map in crf_groups.items():
        print(f"    {crf:<12s}: {len(acc_map):3d} unique sequence(s)")

    #  4. Random sampling - dynamic count per CRF (pct_per_crf, min_per_crf)
    bank:     list[SeqRecord] = []
    test_set: list[SeqRecord] = []

    if hxb2_to_ata is not None:
        gag_ata_start, gag_ata_end = int(hxb2_to_ata[GAG_HXB2[0]]), int(hxb2_to_ata[GAG_HXB2[1]])
        pol_ata_start, pol_ata_end = int(hxb2_to_ata[POL_HXB2[0]]), int(hxb2_to_ata[POL_HXB2[1]])

    for crf_type, acc_map in crf_groups.items():
        records = list(acc_map.values())

        # Dynamic bank size: pct_per_crf of the unaugmented unique count,
        # floored at min_per_crf.
        n_bank_target = max(round(pct_per_crf * len(records)), min_per_crf)

        # Augment the full pool to (n_bank_target + n_test) if needed
        target_total = n_bank_target + n_test
        if mutator is not None and len(records) < target_total:
            records = mutator.augment_to_target(
                records, target_count=target_total, subtype_key='avg'
            )

        chosen   = random.sample(records, min(n_bank_target, len(records)))
        leftover = [r for r in records if r.id not in {r2.id for r2 in chosen}]
        test     = random.sample(leftover, min(n_test, len(leftover)))

        bank.extend(chosen)

        test_full = test[:3]
        test_set.extend(test_full)

        if hxb2_to_ata is not None:
            if len(test) >= 4:
                test_set.append(_crop_record(test[3], gag_ata_start, gag_ata_end, "gag"))
            if len(test) >= 5:
                test_set.append(_crop_record(test[4], pol_ata_start, pol_ata_end, "pol"))
        else:
            test_set.extend(test[3:])

        print(f"    {crf_type:<12s}: bank {len(chosen)}/{len(records)}, test {len(test)}/{len(leftover)} leftover")

    print(
        f"\n  CRF reference bank ready : {len(bank)} sequences "
        f"({len(crf_groups)} CRF type(s), {pct_per_crf:.0%} per type, min {min_per_crf})"
    )
    print(
        f"  CRF test set ready       : {len(test_set)} sequences "
        f"({len(crf_groups)} CRF type(s), ≤{n_test} per type)"
    )
    return bank, test_set

if __name__ == "__main__":
    n_packed = int(np.ceil(NUM_SUBTYPES / 8))

    # Load HXB2 reference
    print(f"Loading HXB2 reference from: {PURE_REF_PATH}")
    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(PURE_REF_PATH, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: pure_ref FASTA is empty.")

    ata_to_hxb2, hxb2_to_ata = config.build_hxb2_ata_maps(hxb2_ata_seq)
    print(f"  ATA length, HXB2 length     : {ATA_LEN, int(max(ata_to_hxb2))}")

    # Model + tokenizer
    model_used = "oanoufa/sbtr_ntv3_650M"
    tokenizer = AutoTokenizer.from_pretrained(model_used, trust_remote_code=True)
    model = HFModelForHIVSubtyping.from_pretrained(model_used)
    device = torch.device(MODEL_CONFIG["device"])
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # CRF reference bank
    # Tokenize each CRF reference sequence directly (no HIVSequenceDataset wrapper needed).  We construct the attention mask ourselves from pad_token_id so we never rely on the tokenizer returning it.
    print(f"\nBuilding CRF reference bank from: {CRF_FILE_PATH}", flush=True)
    sequence_bank, test_set = build_crf_reference_bank(
            crf_ref_path=CRF_FILE_PATH,
            pct_per_crf=PCT_PER_CRF_BANK,
            min_per_crf=MIN_PER_CRF_BANK,
            n_test=N_TEST,
            mutator=mutator,
            hxb2_to_ata=hxb2_to_ata,
        )
    N = len(sequence_bank)

    seq_names = [rec.id for rec in sequence_bank]
    metadata  = pd.DataFrame({
        "sequence_name": seq_names,
        "split":         "crf_bank",
    })
    generated_meta_path = out_dir / f"metadata.tsv"
    metadata_df = (
        metadata[metadata["split"] == "crf_bank"]
        .reset_index(drop=True)
    )
    metadata.to_csv(generated_meta_path, sep="\t", index=False)
    print(f"\nGenerated: {generated_meta_path}")

    # Allocate memmaps
    out_seqs  = out_dir / f"sequences_v{VERSION}.npy"
    out_lbls  = out_dir / f"labels_v{VERSION}.npy"
    out_masks = out_dir / f"loss_masks_v{VERSION}.npy"

    seq_mm  = np.lib.format.open_memmap(str(out_seqs),  mode="w+", dtype=np.uint8,
                                         shape=(N, ATA_LEN))
    lbl_mm  = np.lib.format.open_memmap(str(out_lbls),  mode="w+", dtype=np.uint8,
                                         shape=(N, ATA_LEN, n_packed))
    mask_mm = np.lib.format.open_memmap(str(out_masks), mode="w+", dtype=bool,
                                         shape=(N, ATA_LEN))

    print(f"\nAllocated memmaps:")
    print(f"  sequences  : {out_seqs}   shape={seq_mm.shape}")
    print(f"  labels     : {out_lbls}  shape={lbl_mm.shape}")
    print(f"  loss_masks : {out_masks} shape={mask_mm.shape}")

    # Fill memmaps
    zero_lbl_packed = np.zeros((ATA_LEN, n_packed), dtype=np.uint8)
    zero_mask = np.ones(ATA_LEN,             dtype=bool)
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

    # Store the number of times the crf was seen so that each CRF is only computed at most X times

    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(bank_loader),
            total=len(bank_loader),
            mininterval=30,
            desc="Generating CRF bank",
        ):

            sample_name = metadata_df.iloc[i]["sequence_name"]
            # Forward pass
            logits     = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))["subtype_logits"]
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (MAX_LENGTH, NUM_SUBTYPES)
            # Trim to ATA length and decode
            probs    = pred_probs[:ATA_LEN]
            probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-9)
            is_real  = gap_masks[sample_name]
            probs = probs * is_real[:, None]
            crf_names.append(sample_name)
            crf_profiles.append(probs.astype(np.float32))

    reference_bank = np.stack(crf_profiles, axis=0).astype(np.float16)  # (R, ATA_LEN, NUM_SUBTYPES)
    reference_names = np.array(crf_names)
    print(f"  Reference bank  : {reference_bank.shape}  "
          f"dtype={reference_bank.dtype}")

    # Save the reference bank to a compressed file for later use in inference.
    out_path = out_dir / "crf_reference_bank.npz"
    if out_path.exists():
        os.remove(out_path)

    np.savez_compressed(out_path, reference_bank=reference_bank, reference_names=reference_names)

    print(f"\nSaved CRF reference bank to: {out_path} with shape {reference_bank.shape}")

    test_dir = Path(WORKSPACE_PATH) / "data" / "output" / "test" /f"crf_v{VERSION}"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_path_test = test_dir / f"crf_test_set_v{VERSION}.fasta"
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
    print(f"\nSaved test set to: {out_path_test}")


    os.remove(out_seqs)
    os.remove(out_lbls)
    os.remove(out_masks)

    print(f"Uploading crf_ref_bank to HF")
    api = HfApi()

    # Upload a single file
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo="crf_reference_bank.npz",
        repo_id="oanoufa/sbtr_necessary_data",
        repo_type="dataset",
    )

    HIV1_COMBINED_REF = config.COMBINED_REF_PATH
    with open(HIV1_COMBINED_REF, "rb") as f_in:
        compressed_buffer = io.BytesIO(gzip.compress(f_in.read()))

    api.upload_file(
        path_or_fileobj=compressed_buffer,
        path_in_repo="HIV1_COMBINED_REF.fasta.gz",  # Use .gz extension
        repo_id="oanoufa/sbtr_necessary_data",
        repo_type="dataset",
    )