import numpy as np
import torch
import pandas as pd
import json
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
import os
import sys
import subprocess
import tempfile
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.dataset_class import HIVSequenceDataset
from src.model_class import HFModelForHIVSubtyping
from src.utils import build_hxb2_ata_maps
from src.figs import visualize_sample_probs
from src.crf_decoder_class import CRFReferenceDecoder

ST_TO_ID_DICT      = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
ATA_LEN            = config.ATA_LEN
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
PAD_MULTIPLE_OF    = config.PAD_LEN
VERSION            = config.VERSION

# Import reference FASTA file
_combined_ref_gz = hf_hub_download(
    repo_id="oanoufa/sbtr_necessary_data",
    filename="HIV1_COMBINED_REF.fasta.gz",
    repo_type="dataset",
)
COMBINED_REF_PATH = Path(_combined_ref_gz).with_suffix("")
if not COMBINED_REF_PATH.exists():
    with gzip.open(_combined_ref_gz, "rb") as f_in, open(COMBINED_REF_PATH, "wb") as f_out:
        f_out.write(f_in.read())

import argparse

parser = argparse.ArgumentParser(
    description='Infer HIV-1 subtype per position for sequences aligned to the reference alignment.'
)
parser.add_argument('--seq', type=str, required=True,
                    help='FASTA/txt file of query sequences, or an alignment thereof. '
                    'Will be dealigned and aligned to the HIV1 reference internally.')
parser.add_argument('--mafft_bin', type=str, default='mafft',
                    help='Path to the mafft executable (default: "mafft", assumed to be on PATH).')
parser.add_argument("--tag", type=str, default="inference",
                    help="Text appended to the end of all generated file names.")
parser.add_argument("--out_dir", type=str, default=".",
                    help="Output directory.")
parser.add_argument("--gpu",  action="store_true",
                    help="If true, try to use CUDA gpu.")
parser.add_argument("--num_cpu", type=int, default="1",
                    help="Number of CPUs to use for concurrent processing.")
parser.add_argument("--wto", type=str, default="",
                    help="What-to-output string to know what output the user needs. The letters can be concatenated in any order."
                    "The model will always output at least the results csv. "
                    "Options: "
                    "'f': output figures showing the prediction of the model for each sequence "
                    "'r': output regions csv with col (start, end, subtype) "
                    "'p': output the raw npy file of predictions "
                    "'a': output the attention masks")
args = parser.parse_args()

seq_path      = Path(args.seq)
mafft_bin     = args.mafft_bin
tag           = args.tag
out_dir       = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
num_workers   = args.num_cpu
wto           = args.wto
gpu           = args.gpu

if 'r' in wto:
    print("r in wto: saving subtype regions in a csv", flush=True)
if 'p' in wto:
    print("p in wto: saving prediction array in a npy file", flush=True)
if 'f' in wto:
    print("f in wto: saving figures in figs folder", flush=True)
if 'a' in wto:
    print("a in wto: saving attention masks in a npy file", flush=True)

num_workers = min(os.cpu_count(), num_workers)
print(f"Using {num_workers} CPUs", flush=True)

device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
print(f"Using device: {device}", flush=True)
device = torch.device(device) if isinstance(device, str) else device


def parse_compactmapout(compactmapout_path: Path) -> Dict[str, List[int]]:
    """
    Parse the MAFFT compactmapout file to get the mapping from aligned positions to original positions.
    Returns a dictionary where keys are sequence names and values are lists of original positions.
    
    The file is structured as follows:
    
    # Insertion in added sequence > Position in reference
    >r_K+G+C_2013
    2433c - 2444g > 7810v7811
    >r_A7+C+F2_2019
    8584a - 8590g > 10908v10909
    8605a - 8607a > 10966v10967
    8618g - 8626t > 10995v10996
    8756t - 8756t > 11220v11221

    The goal is to output a dictionary of sample_name:
    """
    mapping = {}
    with open(compactmapout_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                seq_name = line[1:].strip()
                mapping[seq_name] = []
            else:
                parts = line.strip().split('>')
                start_pos = parts[0].split('-')[0].strip()[:-1]  # Get the start position
                end_pos = parts[0].split('-')[1].strip()[:-1]    # Get the end position
                ata_pos = parts[1].strip().split('v')[0]  # Get the ATA position
                mapping[seq_name].append((int(ata_pos), int(start_pos), int(end_pos)))  # Store as a tuple of (ATA position, start, end)

    return mapping


def dealign_to_records(input_path: Path) -> List:
    """
    Strip gaps from an alignment (or pass already-flat sequences through),
    producing query records ready to be aligned to the reference with MAFFT.
    Mirrors the previous standalone alignment_to_seq.py script.
    """
    records = []
    for index, record in enumerate(SeqIO.parse(str(input_path), "fasta")):
        if index == 0 and "HXB2" in record.id:
            continue
        record.description = ""
        seq = str(record.seq).upper().replace('-', '').strip('N')
        record.seq = Seq(seq)
        record.id = record.id.replace("Ref.", "")
        records.append(record)
    return records


def load_reference_ids(reference_path: Path) -> Set[str]:
    """IDs present in the reference alignment, used to strip it back out post-MAFFT."""
    return {rec.id for rec in SeqIO.parse(str(reference_path), "fasta")}


def run_mafft_addfragments(
    query_records: List,
    reference_path: Path,
    mafft_bin: str,
    threads: int,
    tmp_dir: Path,
) -> Tuple[Path, Path]:
    """
    Align query_records onto reference_path with `mafft --addfragments`.
    Returns (aligned_fasta_path, compactmapout_path).
    """
    query_fasta = tmp_dir / "query.fasta"
    SeqIO.write(query_records, str(query_fasta), "fasta")

    aligned_fasta = tmp_dir / "aligned.fasta"
    map_file = Path(str(query_fasta) + ".map")  # mafft's --compactmapout naming convention

    cmd = [
        mafft_bin, "--auto", "--keeplength", "--compactmapout", "--quiet",
        "--thread", str(threads), "--addfragments", str(query_fasta), str(reference_path),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    with open(aligned_fasta, "w") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        sys.exit(f"ERROR: mafft failed (exit {result.returncode}):\n{result.stderr}")
    if not aligned_fasta.exists() or aligned_fasta.stat().st_size == 0:
        sys.exit(f"ERROR: mafft produced an empty alignment.\nstderr:\n{result.stderr}")
    if not map_file.exists():
        sys.exit(f"ERROR: expected mafft compactmapout file not found at {map_file}")

    return aligned_fasta, map_file


# Worker function to run in parallel on CPU worker processes
def process_single_sample_worker(
    idx: int,
    sample_name: str,
    preds_slice: np.ndarray,      # (L, C)
    ploss_slice: np.ndarray,
    compactmapout_entry: list,
    is_real: np.ndarray,          # (L,)
    crf_decoder: CRFReferenceDecoder,
    wto: str,
) -> tuple[str, list[str], dict]:
    """Runs CRF decoding on CPU thread/process and prepares data for figure rendering."""
    preds_normalized = preds_slice / (preds_slice.sum(axis=-1, keepdims=True) + 1e-9)

    # 1. CRF Decoder Query
    crf_result = crf_decoder.query(
        sample_name=sample_name,
        probs=preds_normalized,
        compactmapout_entry=compactmapout_entry,
        hxb2_to_ata=HXB2_TO_ATA,
        query_mask=is_real,
    )

    # 2. String formatting for main CSV
    top5_ref = "  ".join(
        f"{r['crf_type']}:{r['max_score']:.4f}"
        for r in crf_result["top_crf_types"][:5]
    )
    best_ref_crf = crf_result["top_sequences"][0]["name"]
    best_ref_distance = crf_result["top_crf_types"][0]["max_score"]

    result_line = (
        f"{sample_name},"
        f"{crf_result['composition_str']},"
        f"{crf_result['dominant_subtype']},{crf_result['dominant_fraction']:.4f},"
        f"{best_ref_crf},{best_ref_distance:.4f},{top5_ref},{crf_result['final_decision']}\n"
    )

    # 3. Region output formatting
    region_lines = []
    if 'r' in wto:
        for start, end, subtype in crf_result["regions_dealigned"]:
            region_lines.append(f"{sample_name},{start},{end},{subtype},{end - start + 1}\n")

    if 'f' in wto:
        out_path = str(sample_vis_dir / f"{sample_name}_preds.png")
        visualize_sample_probs(
            preds_slice=preds_slice,
            ploss_slice=ploss_slice,
            sample_idx=idx,
            sample_name=sample_name,
            regions_aligned=crf_result["regions_aligned"],
            pure_st_to_id_dict=ST_TO_ID_DICT,
            hxb2_to_ata=HXB2_TO_ATA,
            path=out_path,
        )

    return result_line, region_lines


def global_results(df, summary_json_path):

    def parse_decision(d):
        parts = d.split('.')
        kind = parts[0]                                   # pure | recombinant
        status = parts[3] if len(parts) > 3 else None      # like | assigned | unassigned
        crf = parts[4] if len(parts) > 4 else None         # e.g. 24_BG, or 51_01B+179_12B+...
        return kind, status, crf

    df[['kind', 'status', 'crf_raw']] = df['final_decision'].apply(
        lambda d: pd.Series(parse_decision(d))
    )
    # For assigned recombinants, crf_raw is a single CRF name; keep as-is.
    # For pure "like" calls, crf_raw is a "+"-joined list of nearest refs — not a CRF assignment.
    df['crf_assigned'] = df['crf_raw'].where(df['status'] == 'assigned')
    df['crf_like']     = df['crf_raw'].where(df['status'] == 'like')

    n = len(df)
    composition_counts = df['composition'].value_counts()
    dominant_counts = df['dominant_subtype'].value_counts()
    kind_counts = df['kind'].value_counts()
    status_counts = df.loc[df['kind'] == 'recombinant', 'status'].value_counts()
    crf_assigned_counts = df['crf_assigned'].value_counts()
    ref_best_counts = df['ref_best_crf'].value_counts()
    crf_assigned_size = df['crf_assigned'].shape[0]
    crf_like_size = df['crf_like'].shape[0]

    # Combined: assigned CRF (single value) + "like" pure calls (may list several nearest refs)
    like_mask = df['status'] == 'like'
    like_exploded = (
        df.loc[like_mask, 'crf_raw']
        .str.split('+')
        .explode()
    )
    combined_series = pd.concat([
        df.loc[df['status'] == 'assigned', 'crf_raw'],
        like_exploded,
    ])

    crf_combined_size = crf_assigned_size + crf_like_size
    crf_counts_combined = combined_series.value_counts()

    summary = {
        "n_sequences": n,
        "composition_prevalence": (composition_counts / n).round(4).to_dict(),
        "dominant_subtype_prevalence": (dominant_counts / n).round(4).to_dict(),
        "pure_vs_recombinant": (kind_counts / n).round(4).to_dict(),
        "recombinant_assigned_vs_unassigned": (
            (status_counts / status_counts.sum()).round(4).to_dict()
            if status_counts.sum() > 0 else {}
        ),
        "crf_prevalence_among_assigned": (
            (crf_assigned_counts / crf_assigned_size).round(4).to_dict()
            if crf_assigned_size > 0 else {}
        ),
        "crf_prevalence_among_like_and_assigned": (
            (crf_counts_combined / crf_combined_size).round(4).to_dict()
            if crf_combined_size > 0 else {}
        ),
        "dominant_fraction_stats": df['dominant_fraction'].describe()[
            ['mean', 'std', 'min', '50%', 'max']
        ].round(4).to_dict(),
        "low_confidence_fraction": round((df['dominant_fraction'] < 0.5).mean(), 4),
        "repeated_best_ref": {
            k: int(v) for k, v in ref_best_counts.items() if v > 1
        },
    }

    print("\n=== Global summary ===")
    print(f"N sequences:              {n}")
    print(f"Pure / recombinant:       {summary['pure_vs_recombinant']}")
    print(f"Top compositions:         {composition_counts.head(5).to_dict()}")
    print(f"Top dominant subtypes:    {dominant_counts.head(5).to_dict()}")
    print(f"CRF prevalence (assigned):{summary['crf_prevalence_among_assigned']}")
    print(f"Recombinant assigned/unassigned: {summary['recombinant_assigned_vs_unassigned']}")
    print(f"Low-confidence fraction (<0.5):  {summary['low_confidence_fraction']}")
    if summary['repeated_best_ref']:
        print(f"Repeated best_ref hits:   {summary['repeated_best_ref']}")

    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON      {summary_json_path}", flush=True)





if __name__ == "__main__":

    n_packed = int(np.ceil(NUM_SUBTYPES / 8))
    torch.set_float32_matmul_precision('high')

    # ---- 1. Load combined reference (HXB2 coords + mafft target + ID filter) ----
    print(f"Loading combined reference from: {COMBINED_REF_PATH}", flush=True)
    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(str(COMBINED_REF_PATH), "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: combined reference FASTA is empty.")

    ATA_TO_HXB2, HXB2_TO_ATA = build_hxb2_ata_maps(hxb2_ata_seq)
    print(f"  ATA length, HXB2 length     : {ATA_LEN, int(max(ATA_TO_HXB2))}", flush=True)

    reference_ids = load_reference_ids(COMBINED_REF_PATH)

    # ---- 2. Dealign input, align to reference with MAFFT, drop reference seqs ----
    print(f"\nDealigning input sequences from: {seq_path}", flush=True)
    query_records = dealign_to_records(seq_path)
    if len(query_records) == 0:
        sys.exit("ERROR: no query sequences found after dealigning input.")
    print(f"  Query sequences: {len(query_records)}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"sbtr_{tag}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        print(f"\nRunning MAFFT alignment to reference ({num_workers} threads)...", flush=True)
        aligned_fasta, map_file = run_mafft_addfragments(
            query_records=query_records,
            reference_path=COMBINED_REF_PATH,
            mafft_bin=mafft_bin,
            threads=num_workers,
            tmp_dir=tmp_dir,
        )

        mafft_mapping = parse_compactmapout(map_file)

        records_ali = [
            rec for rec in SeqIO.parse(str(aligned_fasta), "fasta")
            if rec.id not in reference_ids
        ]

    # Clean the potential info added at the beginning of the rec id (r_B+K+A3_2015 became 4ins:5192g-5197a,etc|r_B+K+A3_2015) but stay robust to the eventual presence of other |
    for rec in records_ali:
        rec.id = ''.join(rec.id.split('|')[1:]) if 'ins:' in rec.id else rec.id
    N = len(records_ali)
    if N == 0:
        sys.exit("ERROR: no query sequences remain after removing reference sequences.")
    print(f"  Sequences found: {N}", flush=True)

    seq_lens = [len(r.seq) for r in records_ali]
    if len(set(seq_lens)) != 1:
        print("  [warn] Not all sequences have the same length — may not be aligned.",
              flush=True)
    if seq_lens[0] != ATA_LEN:
        sys.exit(
            f"ERROR: query sequences have length {seq_lens[0]} "
            f"but ATA alignment has length {ATA_LEN}."
        )

    # ---- 4. Build / load metadata -----------------------------------------
    seq_names = [rec.id.split()[0] for rec in records_ali]
    metadata  = pd.DataFrame({
        "sequence_name": seq_names,
        "split":         "inference",
    })

    # ---- 5. Allocate memmaps ----------------------------------------------
    out_seqs         = out_dir / f"sequences_{tag}.npy"
    out_lbls         = out_dir / f"labels_{tag}.npy"
    out_masks        = out_dir / f"loss_masks_{tag}.npy"

    seq_mm  = np.lib.format.open_memmap(str(out_seqs),  mode="w+", dtype=np.uint8,
                                         shape=(N, ATA_LEN))
    lbl_mm  = np.lib.format.open_memmap(str(out_lbls),  mode="w+", dtype=np.uint8,
                                         shape=(N, ATA_LEN, n_packed))
    mask_mm = np.lib.format.open_memmap(str(out_masks), mode="w+", dtype=bool,
                                         shape=(N, ATA_LEN))


    print(f"\nAllocated memmaps:", flush=True)
    print(f"  sequences  : {out_seqs}   shape={seq_mm.shape}", flush=True)
    print(f"  labels     : {out_lbls}  shape={lbl_mm.shape}", flush=True)
    print(f"  loss_masks : {out_masks} shape={mask_mm.shape}", flush=True)

    # ---- 6. Fill memmaps --------------------------------------------------
    zero_lbl_packed = np.zeros((ATA_LEN, n_packed), dtype=np.uint8)
    zero_mask = np.ones(ATA_LEN,             dtype=bool)
    gap_masks = {}

    for i, rec in enumerate(records_ali):
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
    del records_ali

    # ---- 7. Model + tokenizer --------------------------------------------
    model_used = "oanoufa/sbtr_ntv3_650M"
    tokenizer = AutoTokenizer.from_pretrained(model_used, trust_remote_code=True)
    model = HFModelForHIVSubtyping.from_pretrained(model_used)

    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ---- 8. Dataset + loader ---------------------------------------------
    inference_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="inference",
    )
    inference_loader = DataLoader(
        inference_dataset, batch_size=MODEL_CONFIG["inference_batch_size"], # Batch size 1 is much faster
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"Inference samples: {len(inference_dataset)}", flush=True)

    # ---- 11. Inference loop ----------------------------------------------

    out_preds_path = out_dir / f"predictions_{tag}.npy"
    pred_mm = np.lib.format.open_memmap(
        str(out_preds_path), mode="w+", dtype=np.float32,
        shape=(N, MAX_LENGTH, NUM_SUBTYPES))
    print(f"Prediction memmap: {out_preds_path}  shape={pred_mm.shape}", flush=True)

    out_att_path = out_dir / f"attention_masks_{tag}.npy"
    att_mm = np.lib.format.open_memmap(
        str(out_att_path), mode="w+", dtype=bool,
        shape=(N, MAX_LENGTH))
    print(f"Attention masks memmap: {out_att_path} shape={att_mm.shape}", flush=True)

    out_post_loss_path = out_dir / f"post_loss_masks_{tag}.npy"
    ploss_mm = np.lib.format.open_memmap(
        str(out_post_loss_path), mode="w+", dtype=bool,
        shape=(N, MAX_LENGTH))
    print(f"Loss masks after dataset memmap: {out_post_loss_path} shape={ploss_mm.shape}", flush=True)

    inference_rows = (
        metadata[metadata["split"] == "inference"]
        .reset_index(drop=True)
    )

    bank_path = hf_hub_download(
        repo_id="oanoufa/sbtr_necessary_data",
        filename="crf_reference_bank.npz",
        repo_type="dataset",
    )

    crf_decoder = CRFReferenceDecoder(bank_path=bank_path)
    # ------------------------------------------------------------------
    # Phase 1: Pure GPU Model Forward Pass & Memmap Writing
    # ------------------------------------------------------------------
    print("\nRunning Model Inference (GPU Phase)...", flush=True)

    global_idx = 0
    with torch.no_grad(), torch.amp.autocast('cuda'): # Mixed precision speeds up inference
        for batch in tqdm(inference_loader, desc="Forward Pass", mininterval=30):
            B = batch["input_ids"].shape[0]

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            logits = model(tokens=input_ids, attention_mask=attention_mask)["subtype_logits"]
            pred_probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32) # (B, MAX_LENGTH, NUM_SUBTYPES)
            
            # Write directly to your memmap in batch slices (fast disk writes)
            pred_mm[global_idx : global_idx + B] = pred_probs
            att_mm[global_idx : global_idx + B] = batch["attention_mask"].cpu().numpy()
            ploss_mm[global_idx : global_idx + B] = batch["loss_mask"].cpu().numpy()

            global_idx += B

    pred_mm.flush()
    att_mm.flush()
    ploss_mm.flush()
    del inference_loader

    # ------------------------------------------------------------------
    # Phase 2: Multiprocessed CPU Post-Processing & Figure Generation
    # ------------------------------------------------------------------
    print(f"\nRunning Parallel CRF Decoding across available CPUs...", flush=True)
    sample_vis_dir = out_dir / "figs"
    if 'f' in wto: 
        sample_vis_dir.mkdir(parents=True, exist_ok=True)

    results_buffer, regions_buffer = [], []

    # Prepare task arguments
    def task_generator():
        for idx in range(N):
            sample_name = metadata.iloc[idx]["sequence_name"]
            yield (
                idx,
                sample_name,
                pred_mm[idx, :ATA_LEN],
                ploss_mm[idx, :ATA_LEN],
                mafft_mapping.get(sample_name, []),
                gap_masks[sample_name],
                crf_decoder,
                wto,
            )

    # Execute tasks concurrently across multiple CPU processes
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_sample_worker, *args): args[0] for args in task_generator()}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=N, desc="Parallel Decoding", mininterval=5):
            res_line, reg_lines = future.result()

            results_buffer.append(res_line)
            regions_buffer.extend(reg_lines)


    # Write results out in bulk
    result_csv_path = out_dir / f"results_{tag}.csv"
    with open(result_csv_path, 'w') as f:
        f.write(
            "sample_name,"
            "composition,"
            "dominant_subtype,dominant_fraction,"
            "ref_best_crf,ref_best_score,ref_top5,"
            "final_decision"
            "\n"
        )
        f.writelines(results_buffer)


    if 'r' in wto:
        regions_dealigned_csv_path = out_dir / f"regions_dealigned_{tag}.csv"
        with open(regions_dealigned_csv_path, 'w') as f:
            f.write("sample_name,start,end,subtype,length\n")
            f.writelines(regions_buffer)

    print(f"Results CSV      {result_csv_path}", flush=True)


    # Global result on the set of sequences given
    df = pd.read_csv(result_csv_path)
    summary_json_path = out_dir / f"summary_{tag}.json"
    global_results(df, summary_json_path)

    # Remove memmaps files
    os.remove(out_seqs)
    os.remove(out_lbls)
    os.remove(out_masks)
    if 'p' in wto:
        print(f"\nPredictions saved: {out_preds_path}", flush=True)
    else:
        os.remove(out_preds_path)

    if 'a' in wto:
        print(f"\Attention masks saved: {out_att_path}", flush=True)
    else:
        os.remove(out_att_path)

    os.remove(out_post_loss_path)