# inference.py
import numpy as np
import torch
import pandas as pd
from Bio import SeqIO
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login

from src import config
from src.dataset_class import HIVSequenceDataset
from src.model_class import HFModelForHIVSubtyping
from src.utils import build_hxb2_ata_maps
from src.figs import visualize_sample_probs
from src.crf_decoder_class import CRFReferenceDecoder


TOKEN_PATH = config.TOKEN_PATH
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
login(token=token)

WORKSPACE_PATH     = config.WORKSPACE_PATH
ST_TO_ID_DICT      = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
ATA_LEN            = config.ATA_LEN
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
PAD_MULTIPLE_OF    = config.PAD_LEN
PURE_REF_PATH      = config.PURE_REF_PATH
CRF_REF_PATH       = config.CRF_REF_PATH
VERSION            = config.VERSION

import argparse

parser = argparse.ArgumentParser(
    description='Infer HIV-1 subtype per position for sequences aligned to the reference alignment.'
)
parser.add_argument('--alignment', type=str, required=True,
                    help='FASTA file of sequences aligned to the HIV1 subtype reference alignment.')
parser.add_argument('--mafft_map', type=str, required=True,
                    help='MAFFT compactmapout output, used to map the aligned sequences back to the original sequences.')

parser.add_argument("--tag", type=str, default="inference",
                    help="Text appended to the end of all generated file names.")
parser.add_argument("--out_dir", type=str, default=".",
                    help="Output directory.")
parser.add_argument("--num_workers", type=int, default="1",
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

alignment     = Path(args.alignment)
mafft_map     = Path(args.mafft_map)
tag           = args.tag
out_dir       = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
num_workers   = args.num_workers
wto           = args.wto

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

if __name__ == "__main__":

    n_packed = int(np.ceil(NUM_SUBTYPES / 8))
    torch.set_float32_matmul_precision('high')

    # ---- 1. Load HXB2 reference -------------------------------------------
    print(f"Loading HXB2 reference from: {PURE_REF_PATH}", flush=True)
    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(PURE_REF_PATH, "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: pure_ref FASTA is empty.")

    ATA_TO_HXB2, HXB2_TO_ATA = build_hxb2_ata_maps(hxb2_ata_seq)
    print(f"  ATA length, HXB2 length     : {ATA_LEN, int(max(ATA_TO_HXB2))}", flush=True)

    # ---- 2. Load MAFFT compactmapout ---------------------------------------
    print(f"\nLoading MAFFT compactmapout from: {mafft_map}", flush=True)
    mafft_mapping = parse_compactmapout(mafft_map)

    # ---- 3. Load query sequences ------------------------------------------
    print(f"\nLoading query sequences from: {alignment}", flush=True)
    records_ali = list(SeqIO.parse(str(alignment), "fasta"))
    # Clean the potential info added at the beginning of the rec id (r_B+K+A3_2015 became 4ins:5192g-5197a,etc|r_B+K+A3_2015) but stay robust to the eventual presence of other |
    for rec in records_ali:
        rec.id = ''.join(rec.id.split('|')[1:]) if 'ins:' in rec.id else rec.id
    N = len(records_ali)
    if N == 0:
        sys.exit("ERROR: query FASTA is empty.")
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"], trust_remote_code=True
    )
    device = torch.device(MODEL_CONFIG["device"])
    print(f"\nUsing device: {device}", flush=True)

    model = HFModelForHIVSubtyping(
        model_name=MODEL_CONFIG["model_name"], num_subtypes=NUM_SUBTYPES
    )
    model = model.to(device)
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

    # ---- 9. Load checkpoint ---------------------------------------------
    print(f"\nLoading checkpoint …", flush=True)
    checkpoint = torch.load(
        os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

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

    crf_decoder = CRFReferenceDecoder(
        bank_path=f'{WORKSPACE_PATH}/data/model/reference_bank/crf_reference_bank.npz',
    )

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
