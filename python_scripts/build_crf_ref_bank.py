# build_crf_ref_bank.py
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
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter
from huggingface_hub import login

from src.mutator_class import SequenceMutator
from src import config
from src.dataset_class import HIVSequenceDataset
from src.model_class import HFModelForHIVSubtyping
from src.crf_decoder_class import CRFReferenceDecoder

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
SEED               = MODEL_CONFIG["seed"]

import argparse

parser = argparse.ArgumentParser(
    description='Generate a bank of reference predictions for each CRF'
)
parser.add_argument('--crf_file_path', type=str, required=True,
                    help='FASTA file of CRF aligned to the HIV1 subtype reference alignment.')
args = parser.parse_args()

CRF_FILE_PATH = Path(args.crf_file_path)
out_dir = Path(WORKSPACE_PATH) / "data" / "reference_bank"
out_dir.mkdir(parents=True, exist_ok=True)

GAG_HXB2 = (790, 2292)
POL_HXB2 = (2085, 5096)
PCT_PER_CRF_BANK = config.PCT_PER_CRF_BANK # Adaptive bank size depending on the number of sequences of the CRF
MIN_PER_CRF_BANK = config.MIN_PER_CRF_BANK # Min bank size for each CRF
N_TEST           = config.N_TEST # Min test size for each CRF (including one gag and one pol sequence)


def _crop_record(rec: SeqRecord, ata_start: int, ata_end: int, suffix: str) -> SeqRecord:
    cropped = rec[ata_start:ata_end]
    cropped.id = f"{rec.id}_{suffix}"
    cropped.description = ""
    return cropped


def _pairwise_hamming_distance(
    num_paths: np.ndarray,             # (n, L) int8
    uninformative_codes: Tuple[int, ...],
) -> np.ndarray:
    """
    All-pairs distance matrix over num_path arrays.

    distance(i, j) = 1 - (agreement fraction), computed only over positions
    informative in *both* sequences (i.e. not U / 5'LTR / 3'LTR in either),
    mirroring CRFReferenceDecoder._agreement_scores. Pairs with no
    comparable positions get a fallback distance of 1.0 (treated as
    maximally divergent) so they don't artificially collapse the
    farthest-point search.
    """
    n, L = num_paths.shape
    informative = ~np.isin(num_paths, uninformative_codes)  # (n, L) bool

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        comparable = informative[i][None, :] & informative[i + 1:]                 # (n-i-1, L)
        agree      = (num_paths[i + 1:] == num_paths[i][None, :]) & comparable      # (n-i-1, L)
        n_comp     = comparable.sum(axis=1)
        n_agree    = agree.sum(axis=1)

        d = np.ones(n - i - 1, dtype=np.float32)  # fallback = maximally divergent
        has_overlap = n_comp > 0
        d[has_overlap] = 1.0 - (n_agree[has_overlap] / n_comp[has_overlap])

        dist[i, i + 1:] = d
        dist[i + 1:, i] = d

    return dist


def _farthest_point_sampling(
    dist: np.ndarray,
    k: int,
    seed: int = SEED,
) -> List[int]:
    """
    Greedy max-min (farthest-point) sampling over a precomputed distance
    matrix: iteratively pick the point maximizing its minimum distance to
    the already-chosen set, so the selection spans maximum divergence.
    """
    n = dist.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))

    rng = random.Random(seed)
    selected = [rng.randrange(n)]
    min_dist = dist[selected[0]].copy()
    min_dist[selected[0]] = -np.inf

    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist[nxt])
        min_dist[selected] = -np.inf

    return selected


def _compute_num_paths_for_pool(
    records: List[SeqRecord],
    model,
    tokenizer,
    device,
    decoder: CRFReferenceDecoder,
    hxb2_to_ata: np.ndarray,
    ata_len: int,
    n_packed: int,
    num_subtypes: int,
    max_length: int,
    pad_multiple_of: int,
    num_workers: int,
    tmp_dir: Path,
) -> np.ndarray:
    """
    Run the subtyping model over `records` and decode each prediction into a
    per-position label path (num_path) via `decoder.query(...)` (no bank
    loaded on `decoder`, so it returns the raw num_path array).

    Returns
    -------
    np.ndarray, shape (len(records), ata_len), dtype int8 — index-aligned
    with `records`.
    """
    N = len(records)
    tmp_seqs  = tmp_dir / "_tmp_pool_sequences.npy"
    tmp_lbls  = tmp_dir / "_tmp_pool_labels.npy"
    tmp_masks = tmp_dir / "_tmp_pool_masks.npy"

    seq_mm  = np.lib.format.open_memmap(str(tmp_seqs),  mode="w+", dtype=np.uint8,
                                         shape=(N, ata_len))
    lbl_mm  = np.lib.format.open_memmap(str(tmp_lbls),  mode="w+", dtype=np.uint8,
                                         shape=(N, ata_len, n_packed))
    mask_mm = np.lib.format.open_memmap(str(tmp_masks), mode="w+", dtype=bool,
                                         shape=(N, ata_len))

    zero_lbl_packed = np.zeros((ata_len, n_packed), dtype=np.uint8)
    zero_mask = np.ones(ata_len, dtype=bool)

    is_real_list: List[np.ndarray] = []
    names: List[str] = []
    for i, rec in enumerate(records):
        raw = str(rec.seq).upper()
        is_real = np.array([c != '-' for c in raw], dtype=bool)
        is_real_list.append(is_real)
        names.append(rec.id)

        arr = np.frombuffer(raw.encode(), dtype=np.uint8).copy()
        arr[arr == ord("-")] = ord("N")
        seq_mm[i]  = arr
        lbl_mm[i]  = zero_lbl_packed
        mask_mm[i] = zero_mask

    seq_mm.flush()
    lbl_mm.flush()
    mask_mm.flush()

    metadata = pd.DataFrame({"sequence_name": names, "split": "crf_pool"})
    pool_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=num_subtypes, hxb2_to_ata=hxb2_to_ata,
        max_length=max_length, pad_multiple_of=pad_multiple_of, split="crf_pool",
    )
    pool_loader = DataLoader(
        pool_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
    )

    num_paths = np.zeros((N, ata_len), dtype=np.int8)
    with torch.no_grad():
        for i, batch in tqdm(
            enumerate(pool_loader), total=len(pool_loader),
            mininterval=30, desc="Scoring CRF pool for diversity selection",
        ):
            logits = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )["subtype_logits"]
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            probs = pred_probs[:ata_len]
            probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-9)

            num_path = decoder.query(
                sample_name=names[i],
                probs=probs,
                compactmapout_entry=[],
                hxb2_to_ata=hxb2_to_ata,
                query_mask=is_real_list[i],
            )
            num_paths[i] = num_path

    del seq_mm, lbl_mm, mask_mm
    os.remove(tmp_seqs)
    os.remove(tmp_lbls)
    os.remove(tmp_masks)

    return num_paths


def build_crf_reference_bank(
    crf_ref_path:    str,
    model,
    tokenizer,
    device,
    hxb2_to_ata:     np.ndarray,
    ata_len:         int,
    num_subtypes:    int,
    max_length:      int,
    pad_multiple_of: int,
    num_workers:     int,
    tmp_dir:         Path,
    pct_per_crf:     float = 0.10,
    min_per_crf:     int = 1,
    n_test:          int = 5,
    seed:            int = SEED,
    mutator:         Optional[SequenceMutator] = None,
) -> tuple:
    """
    Build a CRF reference bank from a FASTA file.

    Steps
    1. Parse all sequences from *crf_ref_path*.
    2. Extract (CRF type, accession) from each sequence ID, handling two formats:
         - ``Ref.01_AE.CN.05.FJ051.DQ859178``    (with ``Ref.`` prefix)
         - ``01_AE.TH.2007.AA028a_wg7.JX447031``  (without prefix)
    3. Group by CRF type; deduplicate on accession (first occurrence kept).
    4. Augment each CRF's pool (if needed) so it has >= n_bank_target + n_test
       sequences. n_bank_target = round(pct_per_crf * n_unique), floored at
       min_per_crf, computed on the unaugmented unique count.
    5. Score every pool sequence with the subtyping model and decode it
       (no bank attached) into a per-position label path (num_path), using
       CRFReferenceDecoder so bank entries are generated with exactly the
       same logic used at inference time.
    6. For each CRF, greedily select n_bank_target bank sequences via
       farthest-point sampling over num_path Hamming distance (ignoring
       U/LTR positions), so the bank spans maximum observed divergence.
       Remaining ("leftover") sequences are randomly split into a test set,
       unchanged from the previous behaviour.

    Returns
    -------
    reference_bank  : np.ndarray (R, ata_len) int8 — num_path per bank entry.
    reference_names : np.ndarray (R,) str
    test_set        : list[SeqRecord]
    """

    #  1. Load sequences
    random.seed(SEED)
    print(f"\nBuilding CRF reference bank from: {crf_ref_path}")
    all_records: list[SeqRecord] = list(SeqIO.parse(crf_ref_path, "fasta"))
    if not all_records:
        sys.exit("ERROR: CRF reference FASTA is empty.")
    print(f"  Loaded {len(all_records)} CRF reference sequences")

    #  2. Parse CRF type + accession
    _REF_PREFIX = re.compile(r"^Ref\.")

    def parse_id(record_id: str) -> tuple[str, str, str]:
        """
        Strip the optional ``Ref.`` prefix, then return
        (clean_id, first field, last field) as (clean, crf_type, accession).
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

    #  4. Build (possibly augmented) pools + dynamic bank-size targets
    crf_pool_records:     Dict[str, List[SeqRecord]] = {}
    n_bank_target_by_crf: Dict[str, int] = {}

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

        crf_pool_records[crf_type]     = records
        n_bank_target_by_crf[crf_type] = n_bank_target

    #  5. Flatten pools (index-aligned, not id-keyed, to avoid any risk of
    #     accession collisions across CRF groups) and score with the model.
    all_pool_records: List[SeqRecord] = []
    crf_index_ranges: Dict[str, Tuple[int, int]] = {}
    for crf_type, records in crf_pool_records.items():
        start = len(all_pool_records)
        all_pool_records.extend(records)
        crf_index_ranges[crf_type] = (start, len(all_pool_records))

    n_packed = int(np.ceil(num_subtypes / 8))
    decoder  = CRFReferenceDecoder(bank_path=None)
    uninformative_codes = (decoder.code_u, decoder.code_5ltr, decoder.code_3ltr)

    print(
        f"\n  Scoring {len(all_pool_records)} pool sequence(s) across "
        f"{len(crf_pool_records)} CRF type(s) for diversity-based selection..."
    )
    pool_num_paths = _compute_num_paths_for_pool(
        all_pool_records, model=model, tokenizer=tokenizer, device=device,
        decoder=decoder, hxb2_to_ata=hxb2_to_ata, ata_len=ata_len,
        n_packed=n_packed, num_subtypes=num_subtypes, max_length=max_length,
        pad_multiple_of=pad_multiple_of, num_workers=num_workers, tmp_dir=tmp_dir,
    )

    #  6. Diversity-based bank selection (farthest-point sampling) + random
    #     test-set selection from the leftover pool (unchanged behaviour).
    bank_names:     List[str] = []
    bank_num_paths: List[np.ndarray] = []
    test_set:       List[SeqRecord]  = []

    gag_ata_start, gag_ata_end = int(hxb2_to_ata[GAG_HXB2[0]]), int(hxb2_to_ata[GAG_HXB2[1]])
    pol_ata_start, pol_ata_end = int(hxb2_to_ata[POL_HXB2[0]]), int(hxb2_to_ata[POL_HXB2[1]])

    for crf_type, records in crf_pool_records.items():
        start, end    = crf_index_ranges[crf_type]
        num_paths     = pool_num_paths[start:end]                  # (n_pool, L)
        n_bank_target = n_bank_target_by_crf[crf_type]
        n_bank        = min(n_bank_target, len(records))

        if n_bank > 0:
            dist = _pairwise_hamming_distance(num_paths, uninformative_codes)
            chosen_idx = _farthest_point_sampling(dist, n_bank, seed=seed)
        else:
            chosen_idx = []

        chosen_ids = {records[i].id for i in chosen_idx}
        for i in chosen_idx:
            bank_names.append(records[i].id)
            bank_num_paths.append(num_paths[i])

        leftover = [r for r in records if r.id not in chosen_ids]
        test     = random.sample(leftover, min(n_test, len(leftover)))

        test_full = test[:3]
        test_set.extend(test_full)

        if len(test) >= 4:
            test_set.append(_crop_record(test[3], gag_ata_start, gag_ata_end, "gag"))
        if len(test) >= 5:
            test_set.append(_crop_record(test[4], pol_ata_start, pol_ata_end, "pol"))

        print(
            f"    {crf_type:<12s}: bank {len(chosen_idx)}/{len(records)} "
            f"(diversity-selected), test {len(test)}/{len(leftover)} leftover"
        )

    reference_bank = (
        np.stack(bank_num_paths, axis=0).astype(np.int8)
        if bank_num_paths else np.zeros((0, ata_len), dtype=np.int8)
    )
    reference_names = np.array(bank_names)

    print(
        f"\n  CRF reference bank ready : {len(bank_names)} sequences "
        f"({len(crf_groups)} CRF type(s), {pct_per_crf:.0%} per type, min {min_per_crf}, "
        f"diversity-selected via farthest-point sampling)"
    )
    print(
        f"  CRF test set ready       : {len(test_set)} sequences "
        f"({len(crf_groups)} CRF type(s), ≤{n_test} per type)"
    )
    return reference_bank, reference_names, test_set


if __name__ == "__main__":

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

    mutator = SequenceMutator(
            iqtree_dir=f"{WORKSPACE_PATH}/data/output/rates/",
            ata_len=ATA_LEN,
            hxb2_to_ata=hxb2_to_ata,
            seed=42,
        )
    # Model + tokenizer
    model_used = "oanoufa/sbtr_ntv3_650M"
    tokenizer = AutoTokenizer.from_pretrained(model_used, trust_remote_code=True)
    model = HFModelForHIVSubtyping.from_pretrained(model_used)
    device = torch.device(MODEL_CONFIG["device"])
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # CRF reference bank (diversity-selected) + test set
    print(f"\nBuilding CRF reference bank from: {CRF_FILE_PATH}", flush=True)
    reference_bank, reference_names, test_set = build_crf_reference_bank(
        crf_ref_path=CRF_FILE_PATH,
        model=model,
        tokenizer=tokenizer,
        device=device,
        hxb2_to_ata=hxb2_to_ata,
        ata_len=ATA_LEN,
        num_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH,
        pad_multiple_of=PAD_MULTIPLE_OF,
        num_workers=MODEL_CONFIG["num_workers"],
        tmp_dir=out_dir,
        pct_per_crf=PCT_PER_CRF_BANK,
        min_per_crf=MIN_PER_CRF_BANK,
        n_test=N_TEST,
        mutator=mutator,
    )

    print(f"  Reference bank  : {reference_bank.shape}  dtype={reference_bank.dtype}")

    # Record keeping
    metadata = pd.DataFrame({"sequence_name": reference_names, "split": "crf_bank"})
    generated_meta_path = out_dir / "metadata.tsv"
    metadata.to_csv(generated_meta_path, sep="\t", index=False)
    print(f"\nGenerated: {generated_meta_path}")

    # Save the reference bank (num_path per sequence) to a compressed file
    # for later use in inference.
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
