"""Run sbtr to predict HIV-1 subtypes across input sequences."""

import numpy as np
import torch
import pandas as pd
import json
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
import os
import sys
import tempfile
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Set, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re

from python_scripts.figs import visualize_sample_probs, visualize_combined_light
from python_scripts.src import config
from python_scripts.src.dataset_class import HIVSequenceDataset
from python_scripts.src.preprocessing_aux import run_mafft_addfragments, load_reference_ids, dealign_to_records, parse_compactmapout, decode_sample
from python_scripts.src.model_class import HFModelForHIVSubtyping
from python_scripts.src.decoder_class import CRFReferenceDecoder

ST_TO_ID_DICT      = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
ATA_LEN            = config.ATA_LEN
MAX_LENGTH         = config.SEQ_LEN_AFTER_PAD
PAD_MULTIPLE_OF    = config.PAD_LEN

import argparse

parser = argparse.ArgumentParser(
    description="sbtr: A novel HIV-1 fine-grain subtyping tool leveraging genomic language models.\n\n"
    "Infers HIV-1 subtype per position from a set of sequences by dealigning, aligning\n"
    "to the HIV-1 reference via MAFFT, running through the language model, and comparing\n"
    "against a bank of CRF reference sequences.\n\n"
    "Outputs generated:\n"
    "  • results_tag.csv (always)\n"
    "  • summary_tag.json (always)\n"
    "  • Optional outputs via --wto: regions CSV, prediction figures, raw predictions, or attention masks.",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument(
    "--seq",
    type=str,
    required=True,
    help=(
        "Path to query sequences or an alignment file. "
        "Supported formats: FASTA, FASTQ, PHYLIP, NEXUS. "
        "Will be dealigned and re-aligned internally."
    ),
)
parser.add_argument(
    "--mafft_bin",
    type=str,
    default="mafft",
    help='Path to the MAFFT executable (default: "mafft", assumes on PATH).',
)
parser.add_argument(
    "--tag",
    type=str,
    default="sbtr",
    help="Text appended to all output file names (default: 'sbtr').",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./sbtr_output",
    help="Output directory path (default: ./sbtr_output).",
)
parser.add_argument(
    "--gpu", action="store_true", help="Enable CUDA GPU acceleration if available."
)
parser.add_argument(
    "--num_cpu",
    type=int,
    default=1,
    help="Number of CPUs for concurrent processing (default: 1).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for forward pass (default: 1, increase for GPU).",
)
parser.add_argument(
    "--wto",
    type=str,
    default="",
    help="""Flags determining extra output files. Combine letters in any order:
  f: Output prediction figures per sequence (increases runtime)
  l: Output light svg prediction figures
  r: Output regions CSV with (start, end, subtype)
  p: Output raw .npy file of predictions
  a: Output attention masks""",
)
parser.add_argument(
    "--no_decoder",
    action="store_true",
    help="Stop after the model forward pass: save predictions, is_real masks and names; skip CRF decoding.",
)
parser.add_argument(
    "--save_ali",
    action="store_true",
    help="Save the MAFFT alignment fed to the model (query sequences only, reference removed) "
         "as <input name without extension>_ali.fasta in the output directory.",
)

args = parser.parse_args()

seq_path      = Path(args.seq)
mafft_bin     = args.mafft_bin
tag           = args.tag
out_dir       = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
num_workers   = args.num_cpu
batch_size    = args.batch_size
wto           = args.wto
gpu           = args.gpu
no_decoder    = args.no_decoder
save_ali      = args.save_ali

if 'r' in wto:
    print("r in wto: saving subtype regions in a csv", flush=True)
if 'p' in wto:
    print("p in wto: saving prediction array in a npy file", flush=True)
if 'f' in wto:
    print("f in wto: saving figures in figs folder", flush=True)
if 'l' in wto:
    print("l in wto: saving light svg figures in figs folder", flush=True)
if 'a' in wto:
    print("a in wto: saving attention masks in a npy file", flush=True)

num_workers = min(os.cpu_count(), num_workers)
if num_workers == 1:
    print(f"Using {num_workers} CPU", flush=True)
else:
    print(f"Using {num_workers} CPUs", flush=True)

device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} → {torch.cuda.get_device_name(i)}")
device = torch.device(device) if isinstance(device, str) else device

# Points to the directory where this python file is located
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_PATH = SCRIPT_DIR.parents[0]

BANK_PATH = Path(f"{WORKSPACE_PATH}/data/reference_bank/crf_reference_bank.npz")
COMBINED_REF_PATH = Path(f"{WORKSPACE_PATH}/data/HIV1_COMBINED_REF.fasta")
gz_path = COMBINED_REF_PATH.with_name(COMBINED_REF_PATH.name + ".gz")

if not COMBINED_REF_PATH.exists():
    with gzip.open(gz_path, "rb") as f_in, open(COMBINED_REF_PATH, "wb") as f_out:
        f_out.write(f_in.read())

MAX_SEQ_LEN        = config.MAX_SEQ_LEN
MIN_INFORMATIVE_BP = config.MIN_INFORMATIVE_BP
MAX_N_FRACTION     = config.MAX_N_FRACTION
PROB_ZERO_THRESHOLD = config.PROB_ZERO_THRESHOLD
VALID_NT           = set("ACGTRYKMSWBDHVN")  # IUPAC DNA
_BAD_ID_CHARS      = re.compile(r"[^A-Za-z0-9._+\-]")

def _preview(items: List[str], n: int = 10) -> str:
    return ", ".join(items[:n]) + (f", ... (+{len(items) - n} more)" if len(items) > n else "")

def sanitize_ids(records: List, reference_ids: Set[str], mapping_path: Path = None) -> None:
    """Replace problematic characters by '.', and make IDs unique (also w.r.t. reference IDs)."""
    taken = {i.lower() for i in reference_ids}
    mapping = []
    for n, rec in enumerate(records, start=1):
        orig = rec.id
        new = _BAD_ID_CHARS.sub(".", orig) or f"seq{n}"
        base, k = new, 1
        while new.lower() in taken:      # duplicate or collision with a reference ID
            k += 1
            new = f"{base}_{k}"
        taken.add(new.lower())
        if new != orig:
            mapping.append((orig, new))
            rec.id = new

    if mapping:
        print(f"  [warn] {len(mapping)} sequence ID(s) renamed (special characters, duplicates "
              f"or clash with reference IDs): {_preview([f'{o} -> {n}' for o, n in mapping], 5)}",
              flush=True)
        if mapping_path is not None:
            pd.DataFrame(mapping, columns=["original_id", "new_id"]).to_csv(mapping_path, index=False)
            print(f"  ID mapping saved: {mapping_path}", flush=True)

def check_inputs(
    records: List,
    reference_ids: Set[str],
    id_mapping_path: Path = None,
    skipped_path: Path = None,
    ) -> List:
    """Sanitize IDs, convert RNA to DNA, and drop invalid sequences
    (non-nucleotide characters, too long, too few informative bp).
    Returns the kept records; exits only if none remain."""
    sanitize_ids(records, reference_ids, id_mapping_path)

    kept, skipped, high_n = [], [], []
    n_rna = 0
    for rec in records:
        seq = str(rec.seq)
        if "U" in seq:
            seq = seq.replace("U", "T")
            rec.seq = Seq(seq)
            n_rna += 1

        n_count = seq.count("N")
        informative = len(seq) - n_count
        invalid = set(seq) - VALID_NT

        if invalid:
            skipped.append((rec.id, f"non-nucleotide characters: {''.join(sorted(invalid))}"))
        elif len(seq) > MAX_SEQ_LEN:
            skipped.append((rec.id, f"longer than {MAX_SEQ_LEN} bp after dealigning ({len(seq)} bp)"))
        elif informative < MIN_INFORMATIVE_BP:
            skipped.append((rec.id, f"fewer than {MIN_INFORMATIVE_BP} informative bp ({informative} bp)"))
        else:
            if n_count / len(seq) > MAX_N_FRACTION:
                high_n.append(f"{rec.id} ({n_count / len(seq):.0%} N)")
            kept.append(rec)

    if n_rna:
        print(f"  [info] RNA detected (U) in {n_rna} sequence(s): converting to DNA (U -> T).", flush=True)
    if high_n:
        print(f"  [warn] {len(high_n)} sequence(s) with > {MAX_N_FRACTION:.0%} N (kept): {_preview(high_n)}", flush=True)
    if skipped:
        print(f"  [warn] {len(skipped)} sequence(s) skipped: "
              f"{_preview([f'{i} ({r})' for i, r in skipped], 5)}", flush=True)
        if skipped_path is not None:
            pd.DataFrame(skipped, columns=["sequence_name", "reason"]).to_csv(skipped_path, index=False)
            print(f"  Skipped sequences saved: {skipped_path}", flush=True)

    if not kept:
        sys.exit("ERROR: no valid query sequences remain after input checks.")
    return kept


# Worker function to run in parallel on CPU worker processes
def process_single_sample_worker(
    sample_name: str,
    preds_slice: np.ndarray,      # (L, C)
    ploss_slice: np.ndarray,
    compactmapout_entry: list,
    is_real: np.ndarray,          # (L,)
    crf_decoder: CRFReferenceDecoder,
    wto: str,
    ) -> tuple[str, list[str], dict]:
    """Runs CRF decoding on CPU thread/process and prepares data for figure rendering."""

    # 1. CRF Decoder Query
    crf_result = decode_sample(crf_decoder, sample_name, preds_slice, is_real,
                               HXB2_TO_ATA, compactmapout_entry)

    # 2. String formatting for main CSV
    top5_ref = "  ".join(
        f"{r['crf_type']}:{r['max_score']:.4f}"
        for r in crf_result["top_crf_types"][:5]
    )
    best_ref_crf = crf_result["top_sequences"][0]["name"]
    best_ref_distance = crf_result["top_crf_types"][0]["max_score"]
    active_site_equivalents = [
        eq["crf_type"]
        for eq in crf_result.get("active_site_equivalents", [])
    ]
    active_site_equivalents_str = '+'.join(active_site_equivalents)

    result_line = (
        f"{sample_name},"
        f"{crf_result['composition_str']}, {crf_result['active_positions']},"
        f"{crf_result['dominant_subtype']},{crf_result['dominant_fraction']:.4f},"
        f"{best_ref_crf},{best_ref_distance:.4f},{top5_ref},{active_site_equivalents_str},"
        f"{crf_result['final_decision']}\n"
    )

    # 3. Region output formatting
    region_lines = []
    if 'r' in wto:
        for start, end, subtype in crf_result["regions_dealigned"]:
            region_lines.append(f"{sample_name},{start},{end},{subtype},{end - start + 1}\n")

    if 'l' in wto:
        out_path = str(sample_vis_dir / f"{sample_name}_light_preds.svg")
        visualize_combined_light(
            regions_aligned=crf_result["regions_aligned"],
            hxb2_to_ata=HXB2_TO_ATA,
            preds_slice=preds_slice,
            pure_st_to_id_dict=ST_TO_ID_DICT,
            n_total=ATA_LEN,
            path=out_path,
            min_peak=PROB_ZERO_THRESHOLD,
            floor=0.01
        )

    elif 'f' in wto:
        out_path = str(sample_vis_dir / f"{sample_name}_preds.png")
        visualize_sample_probs(
            preds_slice=preds_slice,
            ploss_slice=ploss_slice,
            sample_name=sample_name,
            regions_aligned=crf_result["regions_aligned"],
            pure_st_to_id_dict=ST_TO_ID_DICT,
            hxb2_to_ata=HXB2_TO_ATA,
            path=out_path,
        )


    return result_line, region_lines


def global_results(
    df, 
    summary_json_path,
    ) -> None:

    def parse_decision(d):
        parts = d.split('.')
        kind = parts[0]                                   # pure | recombinant
        full_or_partial = parts[2] if len(parts) > 2 else None  # full | partial
        status = parts[3] if len(parts) > 3 else None      # like | assigned | unassigned
        crf = parts[4] if len(parts) > 4 else None         # e.g. 24_BG, or 51_01B+179_12B+...
        return kind, full_or_partial, status, crf

    df[['kind', 'full_or_partial', 'status', 'crf_raw']] = df['final_decision'].apply(
        lambda d: pd.Series(parse_decision(d))
    )
    # For assigned recombinants, crf_raw is a single CRF name; keep as-is.
    # For pure "like" calls, crf_raw is a "+"-joined list of nearest refs, not a CRF assignment.
    df['crf_assigned'] = df['crf_raw'].where(df['status'] == 'assigned')
    df['crf_like']     = df['crf_raw'].where(df['status'] == 'like')

    n = len(df)
    composition_counts = df['composition'].value_counts()
    dominant_counts = df['dominant_subtype'].value_counts()
    kind_counts = df['kind'].value_counts()
    full_or_partial_counts = df['full_or_partial'].value_counts()
    status_counts = df.loc[df['kind'] == 'recombinant', 'status'].value_counts()
    crf_assigned_counts = df['crf_assigned'].value_counts()
    ref_best_counts = df['ref_best_crf'].value_counts()
    crf_assigned_size = df['crf_assigned'].count()
    crf_like_size = df['crf_like'].count()

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

    crf_combined_size = len(combined_series)
    crf_counts_combined = combined_series.value_counts()

    summary = {
        "n_sequences": n,
        "composition_prevalence": (composition_counts / n).round(4).to_dict(),
        "dominant_subtype_prevalence": (dominant_counts / n).round(4).to_dict(),
        "pure_vs_recombinant": (kind_counts / n).round(4).to_dict(),
        "full_vs_partial": (full_or_partial_counts / n).round(4).to_dict(),
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
    print(f"Full / partial:           {summary['full_vs_partial']}")
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

    # Load combined reference (HXB2 coords + mafft target + ID filter)
    print(f"Loading combined reference from: {COMBINED_REF_PATH}", flush=True)
    hxb2_ata_seq = None
    for i, rec in enumerate(SeqIO.parse(str(COMBINED_REF_PATH), "fasta")):
        if i == 0:
            hxb2_ata_seq = str(rec.seq).upper()
            print(f"  HXB2 record id : {rec.id}")
            break
    if hxb2_ata_seq is None:
        sys.exit("ERROR: combined reference FASTA is empty.")

    ATA_TO_HXB2, HXB2_TO_ATA = config.build_hxb2_ata_maps(hxb2_ata_seq)
    print(f"  ATA length, HXB2 length     : {ATA_LEN, int(max(ATA_TO_HXB2))}", flush=True)

    reference_ids = load_reference_ids(COMBINED_REF_PATH)

    # Dealign input, align to reference with MAFFT, drop reference seqs
    print(f"\nDealigning input sequences from: {seq_path}", flush=True)
    query_records = dealign_to_records(seq_path)
    if len(query_records) == 0:
        sys.exit("ERROR: no query sequences found after dealigning input.")
    check_inputs(query_records, reference_ids, out_dir / f"id_mapping_{tag}.csv")
    print(f"  Query sequences: {len(query_records)}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"sbtr_{tag}_", ignore_cleanup_errors=True) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        print(f"\nAligning input sequences to reference using MAFFT ({num_workers} threads)...", flush=True)
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

    # Delete
    if COMBINED_REF_PATH.exists():
        COMBINED_REF_PATH.unlink()
    # Clean the potential info added at the beginning of the rec id (r_B+K+A3_2015 became 4ins:5192g-5197a,etc|r_B+K+A3_2015) but stay robust to the eventual presence of other |
    for rec in records_ali:
        rec.id = ''.join(rec.id.split('|')[1:]) if 'ins:' in rec.id else rec.id
    N = len(records_ali)
    if N == 0:
        sys.exit("ERROR: no query sequences remain after removing reference sequences.")

    seq_lens = [len(r.seq) for r in records_ali]
    if len(set(seq_lens)) != 1:
        print("  [warn] Not all sequences have the same length — may not be aligned.",
              flush=True)
    if seq_lens[0] != ATA_LEN:
        sys.exit(
            f"ERROR: query sequences have length {seq_lens[0]} "
            f"but ATA alignment has length {ATA_LEN}."
        )

    seq_names = [rec.id.split()[0] for rec in records_ali]

    if save_ali:
        ali_path = out_dir / f"{seq_path.stem}_ali.fasta"
        for rec in records_ali:
            rec.description = ""
        SeqIO.write(records_ali, str(ali_path), "fasta")
        print(f"User entered --save_ali, alignment saved at: {ali_path}", flush=True)

    metadata  = pd.DataFrame({
        "sequence_name": seq_names,
        "split": tag,
    })

    # Allocate memmaps
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

    # Fill memmaps
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

    # load model + tokenizer
    model_used = MODEL_CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_used, trust_remote_code=True, revision="main")
    model = HFModelForHIVSubtyping.from_pretrained(model_used, trust_remote_code=True, revision="main")

    print(f"\nModel {model_used} with architecture:\n{model}")
    print(f"\nBackbone: {sum(p.numel() for p in model.backbone.parameters()):,} params")
    print(f"Head:     {sum(p.numel() for p in model.subtype_head.parameters()):,} params")

    model = model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Dataset + loader
    inference_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, hxb2_to_ata=HXB2_TO_ATA,
        split=tag,
    )
    inference_loader = DataLoader(
        inference_dataset, batch_size=batch_size, # Batch size 1 is much faster
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

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
        metadata[metadata["split"] == tag]
        .reset_index(drop=True)
    )


    crf_decoder = None if no_decoder else CRFReferenceDecoder(bank_path=BANK_PATH)
    # Model forward pass
    print("\nRunning model inference...", flush=True)

    global_idx = 0
    with torch.no_grad(), torch.amp.autocast('cuda'): # Mixed precision speeds up inference
        for batch in tqdm(inference_loader, desc="Forward pass...", mininterval=30):
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

    if no_decoder:
        np.save(out_dir / f"is_real_{tag}.npy", np.stack([gap_masks[n] for n in seq_names]))
        with open(out_dir / f"names_{tag}.txt", "w") as f:
            f.write("\n".join(seq_names) + "\n")
        for p in (out_seqs, out_lbls, out_masks, out_att_path):
            os.remove(p)
        print(f"--no_decoder: predictions saved in {out_preds_path}", flush=True)
        sys.exit(0)

    # CRF decoding and figures
    print(f"\nRunning CRF decoding across available CPUs...", flush=True)
    sample_vis_dir = out_dir / "figs"
    if 'f' in wto or 'l' in wto:
        sample_vis_dir.mkdir(parents=True, exist_ok=True)

    results_buffer, regions_buffer = [], []

    # Prepare task arguments
    def task_generator():
        for idx in range(N):
            sample_name = metadata.iloc[idx]["sequence_name"]
            yield (
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

    def sort_key(x):
        try:
            # Try to extract and convert the prefix to an integer
            val = int(x.split(',')[0].split('_')[0])
            return (0, val)  # Group 0: Numbered items come first, sorted by number
        except ValueError:
            return (1, x)    # Group 1: Unnumbered items go to the end, sorted alphabetically
    def region_sort_key(item):
        idx, x = item
        try:
            parts = x.split(',')
            region_name = parts[0].strip().lower()
            region_val = int(parts[1].strip())
            return (0, region_name, region_val, idx)  # Group 0: Sorted valid entries (with idx as tie-breaker)
        except (IndexError, ValueError):
            return (1, idx, 0)                        # Group 1: Malformed entries kept in their exact original order

    # Enumerate to keep track of original positions, sort, then extract the values back out
    regions_buffer = [x for idx, x in sorted(enumerate(regions_buffer), key=region_sort_key)]
    results_buffer = sorted(results_buffer, key=sort_key)

    # Write results out in bulk
    result_csv_path = out_dir / f"results_{tag}.csv"
    with open(result_csv_path, 'w') as f:
        f.write(
            "sample_name,"
            "composition,active_positions,"
            "dominant_subtype,dominant_fraction,"
            "ref_best_crf,ref_best_score,ref_top5,"
            "active_site_equivalents,"
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