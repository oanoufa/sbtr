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
import copy
import subprocess
import shutil

from python_scripts.figs import visualize_combined_light
from python_scripts.src.mutator_class import SequenceMutator
from python_scripts.src import config
from python_scripts.src.preprocessing_aux import clean_for_alignment, align_to_ata, decode_sample
from python_scripts.src.dataset_class import HIVSequenceDataset
from python_scripts.src.model_class import HFModelForHIVSubtyping
from python_scripts.src.decoder_class import CRFReferenceDecoder

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
PROB_ZERO_THRESHOLD = config.PROB_ZERO_THRESHOLD
BATCH_SIZE         = MODEL_CONFIG["batch_size"]

import argparse

parser = argparse.ArgumentParser(
    description='Generate a bank of reference predictions for each CRF'
)
parser.add_argument('--crf_file_path', type=str, required=True,
                    help='FASTA file of CRF aligned to the HIV1 subtype reference alignment.')
parser.add_argument('--bank_selection', choices=['diversity', 'reference'], default='reference',
                    help="'diversity': farthest-point sampling over the CRF pool. "
                         "'reference': keep the pool sequences whose accession is in --ref_fasta.")
parser.add_argument('--ref_fasta', type=str,
                    default='/pasteur/helix/projects/mPath/oanoufa/sbtr/data/input/HIV1_REF_2026_genome_DNA_seq.fasta.gz',
                    help="LANL reference FASTA (.fasta or .fasta.gz); only used with --bank_selection reference.")
parser.add_argument('--sbtr_path', type=str, default=str(Path(__file__).resolve().parent / 'sbtr.py'))
parser.add_argument('--mafft_bin', type=str, default='mafft')
parser.add_argument('--num_cpu', type=int, default=1)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

CRF_FILE_PATH = Path(args.crf_file_path)
out_dir = Path(WORKSPACE_PATH) / "data" / "reference_bank"
out_dir.mkdir(parents=True, exist_ok=True)

GAG_HXB2 = (790, 2292)
POL_HXB2 = (2085, 5096)
PCT_PER_CRF_BANK = config.PCT_PER_CRF_BANK # Adaptive bank size depending on the number of sequences of the CRF
MIN_PER_CRF_BANK = config.MIN_PER_CRF_BANK # Min bank size for each CRF
N_TEST           = config.N_TEST # Min test size for each CRF (including one gag and one pol sequence)


_ACC_RE = re.compile(r"^[A-Za-z]{1,2}_?\d{5,8}$")

def _accession(clean_id: str) -> str:
    """Accession of a LANL-style id (``01_AE.TH.90.CM240.U54771`` -> ``U54771``): last dot-separated
    field that looks like an accession, else the last field. Upper-cased."""
    parts = clean_id.split(".")
    for p in reversed(parts):
        if _ACC_RE.match(p):
            return p.upper()
    return parts[-1].upper()


def load_reference_accessions(ref_fasta_path) -> Dict[str, str]:
    """{accession: type} for every record of a reference FASTA (.fasta / .fasta.gz).
    type = first dot-separated field once the optional ``Ref.`` prefix is stripped (e.g. ``01_AE``, ``A1``)."""
    ref_path = Path(ref_fasta_path)
    opener = gzip.open if ref_path.suffix == ".gz" else open
    acc_to_type: Dict[str, str] = {}
    with opener(ref_path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            clean = re.sub(r"^Ref\.", "", rec.id)
            acc_to_type[_accession(clean)] = clean.split(".")[0]
    return acc_to_type

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

def _run_sbtr(records, tag, tmp_dir, sbtr_path, mafft_bin, num_cpu, batch_size, gpu):
    """Run `sbtr.py --no_decoder` on records. Returns (preds memmap, is_real, rows), rows[k] = row of records[k]."""
    run_dir = Path(tmp_dir) / f"_sbtr_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fasta = run_dir / "input.fasta"
    # index ids: LANL ids clash with the reference ids inside sbtr.py
    SeqIO.write([SeqRecord(r.seq, id=f"q{k}", description="") for k, r in enumerate(records)],
                str(fasta), "fasta")
    cmd = [sys.executable, str(sbtr_path), "--seq", str(fasta), "--out_dir", str(run_dir),
           "--tag", tag, "--mafft_bin", mafft_bin, "--num_cpu", str(num_cpu),
           "--batch_size", str(batch_size), "--no_decoder"]
    if gpu:
        cmd.append("--gpu")
    subprocess.run(cmd, check=True)

    with open(run_dir / f"names_{tag}.txt") as f:
        order = {l.strip(): j for j, l in enumerate(f)}
    missing = [k for k in range(len(records)) if f"q{k}" not in order]
    if missing:
        sys.exit(f"ERROR: {len(missing)} sequence(s) dropped by sbtr.py: {missing[:10]}")
    preds   = np.load(run_dir / f"predictions_{tag}.npy", mmap_mode="r")
    is_real = np.load(run_dir / f"is_real_{tag}.npy")
    return preds, is_real, [order[f"q{k}"] for k in range(len(records))]


def _compute_num_paths_for_pool(records, decoder, hxb2_to_ata, ata_len, sbtr_kw):
    preds, is_real, rows = _run_sbtr(records, **sbtr_kw)
    num_paths = np.zeros((len(records), ata_len), dtype=np.int8)
    for k, rec in enumerate(records):
        j = rows[k]
        num_paths[k] = decode_sample(decoder, rec.id, np.asarray(preds[j, :ata_len]),
                                     is_real[j], hxb2_to_ata)
    return num_paths, preds, rows


def _plot_bank_figures(bank_records, bank_num_paths, pool_preds, bank_rows,
                       decoder, hxb2_to_ata, ata_len, fig_dir):
    for i, rec in enumerate(bank_records):
        str_path = np.array([decoder.code_to_label[int(c)] for c in bank_num_paths[i]], dtype=object)
        regions_aligned = decoder._extract_breakpoints(str_path, skip_label="U")
        safe_id = re.sub(r"[^\w.\-]", "_", rec.id)
        visualize_combined_light(
            regions_aligned=regions_aligned, hxb2_to_ata=hxb2_to_ata,
            preds_slice=np.asarray(pool_preds[bank_rows[i], :ata_len]),
            pure_st_to_id_dict=ST_TO_ID_DICT, n_total=ata_len,
            path=str(fig_dir / f"{safe_id}.svg"),
            min_peak=PROB_ZERO_THRESHOLD, floor=0.01,
        )

def build_crf_reference_bank(
    crf_ref_path:    str,
    hxb2_to_ata:     np.ndarray,
    sbtr_kw:         dict,
    ata_len:         int,
    tmp_dir:         Path,
    pct_per_crf:     float = 0.10,
    min_per_crf:     int = 1,
    n_test:          int = 5,
    seed:            int = SEED,
    mutator:         Optional[SequenceMutator] = None,
    fig_dir:         Optional[Path] = None,
    bank_selection:  str = "diversity",
    ref_acc_to_type: Optional[Dict[str, str]] = None,
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
    ref_ids_by_crf:       Dict[str, set] = {}
    fallback_crfs:        List[str] = []
    n_bank_target_by_crf: Dict[str, int] = {}

    for crf_type, acc_map in crf_groups.items():
        records = list(acc_map.values())

        # Dynamic bank size: pct_per_crf of the unaugmented unique count, floored at min_per_crf.
        n_bank_target = max(round(pct_per_crf * len(records)), min_per_crf)

        if bank_selection == "reference":
            # originals only, never augmented sequences
            ref_ids = {r.id for r in records if _accession(r.id) in ref_acc_to_type}
            if ref_ids:
                ref_ids_by_crf[crf_type] = ref_ids
                n_bank_target = len(ref_ids)
            else:
                fallback_crfs.append(crf_type)    # no reference for this CRF: diversity selection instead

        # Augment the full pool to (n_bank_target + n_test) if needed
        target_total = n_bank_target + n_test
        if mutator is not None and len(records) < target_total:
            records = mutator.augment_to_target(
                records, target_count=target_total, subtype_key='avg'
            )

        crf_pool_records[crf_type]     = records
        n_bank_target_by_crf[crf_type] = n_bank_target

    #  5. Flatten pools (index-aligned, not id-keyed, to avoid any risk of accession collisions across CRF groups) and score with the model.
    all_pool_records: List[SeqRecord] = []
    crf_index_ranges: Dict[str, Tuple[int, int]] = {}
    for crf_type, records in crf_pool_records.items():
        start = len(all_pool_records)
        all_pool_records.extend(records)
        crf_index_ranges[crf_type] = (start, len(all_pool_records))

    decoder  = CRFReferenceDecoder(bank_path=None)
    uninformative_codes = (decoder.code_u, decoder.code_5ltr, decoder.code_3ltr)

    print(
        f"\n  Scoring {len(all_pool_records)} pool sequence(s) across "
        f"{len(crf_pool_records)} CRF type(s) for diversity-based selection..."
    )
    # diversity mode needs every pool sequence; reference mode only the selected ones (+ fallback CRFs)
    score_mask = np.ones(len(all_pool_records), dtype=bool)
    if bank_selection == "reference":
        score_mask[:] = False
        for crf_type, (s0, e0) in crf_index_ranges.items():
            if crf_type in fallback_crfs:
                score_mask[s0:e0] = True
            else:
                for k in range(s0, e0):
                    score_mask[k] = all_pool_records[k].id in ref_ids_by_crf[crf_type]
        matched_acc = {_accession(r.id) for ids in ref_ids_by_crf.values() for r in all_pool_records if r.id in ids}
        unmatched = sorted(a for a, t in ref_acc_to_type.items() if "_" in t and a not in matched_acc)
        print(f"  Reference selection: {sum(len(v) for v in ref_ids_by_crf.values())} bank sequences matched "
              f"over {len(ref_ids_by_crf)} CRF type(s); {len(unmatched)} CRF reference accession(s) of the "
              f"reference FASTA not found in the CRF pool" + (f": {unmatched[:20]}" if unmatched else ""))
        if fallback_crfs:
            print(f"  WARNING: no reference sequence found for {len(fallback_crfs)} CRF type(s), "
                  f"diversity selection used instead: {fallback_crfs}")
    score_idx = np.flatnonzero(score_mask)

    pool_num_paths = np.zeros((len(all_pool_records), ata_len), dtype=np.int8)
    scored = [all_pool_records[k] for k in score_idx]
    pool_num_paths[score_idx], pool_preds, rows = _compute_num_paths_for_pool(
        scored, decoder, hxb2_to_ata, ata_len, sbtr_kw)
    pool_row = {int(k): rows[p] for p, k in enumerate(score_idx)}

    #  6. Diversity-based bank selection (farthest-point sampling) + random test-set selection from the leftover pool (unchanged behaviour).
    bank_names:     List[str] = []
    bank_num_paths: List[np.ndarray] = []
    bank_records:   List[SeqRecord]  = []
    test_set:       List[SeqRecord]  = []
    bank_pool_idx:  List[int] = []

    gag_ata_start, gag_ata_end = int(hxb2_to_ata[GAG_HXB2[0]]), int(hxb2_to_ata[GAG_HXB2[1]])
    pol_ata_start, pol_ata_end = int(hxb2_to_ata[POL_HXB2[0]]), int(hxb2_to_ata[POL_HXB2[1]])

    for crf_type, records in crf_pool_records.items():
        start, end    = crf_index_ranges[crf_type]
        num_paths     = pool_num_paths[start:end]                  # (n_pool, L)
        n_bank_target = n_bank_target_by_crf[crf_type]
        n_bank        = min(n_bank_target, len(records))

        if bank_selection == "reference" and crf_type not in fallback_crfs:
            chosen_idx = [i for i, r in enumerate(records) if r.id in ref_ids_by_crf[crf_type]]
        elif n_bank > 0:
            dist = _pairwise_hamming_distance(num_paths, uninformative_codes)
            chosen_idx = _farthest_point_sampling(dist, n_bank, seed=seed)
        else:
            chosen_idx = []

        chosen_ids = {records[i].id for i in chosen_idx}
        for i in chosen_idx:
            bank_names.append(records[i].id)
            bank_num_paths.append(num_paths[i])
            bank_records.append(records[i])
            bank_pool_idx.append(start + i)
            assert all_pool_records[start + i].id == records[i].id
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
            f"({'reference' if crf_type in ref_ids_by_crf else 'diversity'}-selected), test {len(test)}/{len(leftover)} leftover"
        )

    reference_bank = (
        np.stack(bank_num_paths, axis=0).astype(np.int8)
        if bank_num_paths else np.zeros((0, ata_len), dtype=np.int8)
    )
    reference_names = np.array(bank_names)
    if fig_dir is not None and bank_records:
        _plot_bank_figures(bank_records, reference_bank, pool_preds,
                           [pool_row[g] for g in bank_pool_idx],
                           decoder, hxb2_to_ata, ata_len, Path(fig_dir))
    shutil.rmtree(Path(tmp_dir) / f"_sbtr_{sbtr_kw['tag']}", ignore_errors=True)

    print(
        f"\n  CRF reference bank ready : {len(bank_names)} sequences "
        f"({len(crf_groups)} CRF type(s), {pct_per_crf:.0%} per type, min {min_per_crf}, "
        f"diversity-selected via farthest-point sampling)"
    )
    print(
        f"  CRF test set ready       : {len(test_set)} sequences "
        f"({len(crf_groups)} CRF type(s), {n_test} per type)"
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
            seed=42,
        )

    # CRF reference bank (diversity-selected) + test set
    print(f"\nBuilding CRF reference bank from: {CRF_FILE_PATH}", flush=True)
    fig_dir = Path(out_dir) / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ref_acc_to_type = None
    if args.bank_selection == "reference":
        ref_acc_to_type = load_reference_accessions(args.ref_fasta)
        n_crf_refs = sum("_" in t for t in ref_acc_to_type.values())
        print(f"Loaded {len(ref_acc_to_type)} reference accessions from {args.ref_fasta} ({n_crf_refs} CRF-type)")

    sbtr_kw = dict(tag="pool", tmp_dir=out_dir, sbtr_path=args.sbtr_path,
                   mafft_bin=args.mafft_bin, num_cpu=args.num_cpu,
                   batch_size=BATCH_SIZE, gpu=args.gpu)
    reference_bank, reference_names, test_set = build_crf_reference_bank(
        crf_ref_path=CRF_FILE_PATH,
        hxb2_to_ata=hxb2_to_ata,
        sbtr_kw=sbtr_kw,
        ata_len=ATA_LEN,
        tmp_dir=out_dir,
        pct_per_crf=PCT_PER_CRF_BANK,
        min_per_crf=MIN_PER_CRF_BANK,
        n_test=N_TEST,
        mutator=mutator,
        fig_dir=fig_dir,
        bank_selection=args.bank_selection,
        ref_acc_to_type=ref_acc_to_type,
    )

    print(f"  Reference bank  : {reference_bank.shape}  dtype={reference_bank.dtype}")

    # Record keeping
    metadata = pd.DataFrame({"sequence_name": reference_names, "split": "crf_bank"})
    generated_meta_path = out_dir / "metadata.tsv"
    metadata.to_csv(generated_meta_path, sep="\t", index=False)
    print(f"\nGenerated: {generated_meta_path}")

    # Save the reference bank (num_path per sequence) to a compressed file for later use in inference.
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
