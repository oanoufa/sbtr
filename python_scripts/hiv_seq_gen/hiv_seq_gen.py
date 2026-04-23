import random
import json
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from argparse import ArgumentParser
import os

workspace_path = "/workspaces/mpath/oanoufa"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = ArgumentParser(description="Generate synthetic HIV sequences with detailed labels.")
parser.add_argument("--n_seq",     type=int,   default=100,  help="Number of sequences to generate.")
parser.add_argument("--rp",        type=float, default=0.4,  help="Proportion of recombinant sequences.")
parser.add_argument("--seed",      type=int,   default=42,   help="Random seed for reproducibility.")
parser.add_argument("--n_workers", type=int,   default=None, help="Number of worker processes (default: CPU count).")
parser.add_argument("--json_bp_path", type=str, default=f"{workspace_path}/data/LANL/crf/synthetic_blueprints_500000_no_acz.json",
                    help="Path to JSON file containing recombinant blueprint definitions.")
parser.add_argument("--window_size", type=int, default=50, help="Window size for breakpoint optimization.")
parser.add_argument("--min_div_threshold", type=int, default=20,
                    help="Minimum divergence threshold for accepting a breakpoint during optimization.")
parser.add_argument("--max_attempts", type=int, default=50,
                    help="Maximum attempts to find a valid recombinant blueprint before giving up and trying a new one.")

args = parser.parse_args()
N_SEQ     = args.n_seq
RP        = args.rp
SEED      = args.seed
N_WORKERS = args.n_workers or mp.cpu_count()
JSON_BP_PATH = Path(args.json_bp_path)
WINDOW_SIZE = args.window_size
MIN_DIV_THRESHOLD = args.min_div_threshold
MAX_ATTEMPTS = args.max_attempts

out_dir    = f"{workspace_path}/data/GEN"
out_seqs   = f"{out_dir}/sequences_{N_SEQ}_{RP}.npy"   # uint8  (N_SEQ, ata_len)
out_labels = f"{out_dir}/labels_{N_SEQ}_{RP}.npy"      # uint8  (N_SEQ, ata_len, n_packed)
out_names  = f"{out_dir}/sequence_metadata_{N_SEQ}_{RP}.tsv"

os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Subtype divergence-aware sampling
# ---------------------------------------------------------------------------

SUBTYPE_GROUPS: dict = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A6": 0, "A7": 0, "A8": 0,
    "F1": 1, "F2": 1,
    "B": 2, "C": 2, "D": 2, "G": 2, "H": 2, "J": 2, "K": 2, "L": 2,
    "E": 2,
}

DIVERGENCE_WEIGHTS: list = [
    #  g0    g1    g2
    [0.05, 1.00, 1.00],   # g0 (A sub-subtypes)
    [1.00, 0.05, 1.00],   # g1 (F sub-subtypes)
    [1.00, 1.00, 1.00],   # g2 (main subtypes)
]

# ---------------------------------------------------------------------------
# Position-conversion helpers
# ---------------------------------------------------------------------------

def build_hxb2_ata_maps(hxb2_ata_seq: str):
    seq           = np.frombuffer(hxb2_ata_seq.encode(), dtype=np.uint8)
    is_base       = seq != ord('-')
    ata_positions = np.where(is_base)[0]
    n_hxb2        = ata_positions.size

    hxb2_to_ata     = np.empty(n_hxb2 + 1, dtype=np.int32)
    hxb2_to_ata[1:] = ata_positions

    ata_len                    = len(hxb2_ata_seq)
    ata_to_hxb2                = np.zeros(ata_len, dtype=np.int32)
    ata_to_hxb2[ata_positions] = np.arange(1, n_hxb2 + 1)
    for i in range(1, ata_len):
        if ata_to_hxb2[i] == 0:
            ata_to_hxb2[i] = ata_to_hxb2[i - 1]

    return hxb2_to_ata, ata_to_hxb2

# ---------------------------------------------------------------------------
# Breakpoint validation functions
# ---------------------------------------------------------------------------

def calculate_hamming_distance(seq1: np.ndarray, seq2: np.ndarray) -> int:
    """
    Calculate Hamming distance between two sequences in a window.
    Only counts positions where both sequences have valid bases (A, T, C, G).
    """
    ATCG = np.array([ord('A'), ord('T'), ord('C'), ord('G')], dtype=np.uint8)
    
    # Create masks for valid bases
    valid1 = np.isin(seq1, ATCG)
    valid2 = np.isin(seq2, ATCG)
    both_valid = valid1 & valid2
    
    if not both_valid.any():
        return 0  # No valid positions to compare
    
    # Count differences where both sequences have valid bases
    differences = (seq1 != seq2) & both_valid
    return int(np.sum(differences))

def validate_breakpoint_divergence(bp_pos_ata: int, subtype1: str, subtype2: str, 
                                 st_to_arr_dict: dict, window_size: int, 
                                 min_div_threshold: int, py_rng) -> tuple:
    """
    Validate that two subtypes are sufficiently different at a breakpoint location.
    
    Args:
        bp_pos_ata: Breakpoint position in ATA coordinates
        subtype1: First subtype (before breakpoint)  
        subtype2: Second subtype (after breakpoint)
        st_to_arr_dict: Dictionary mapping subtypes to their sequence arrays
        window_size: Window size around breakpoint to check
        min_div_threshold: Minimum number of differences required
        py_rng: Random number generator
        
    Returns:
        tuple: (is_valid: bool, selected_seq1: np.ndarray, selected_seq2: np.ndarray, divergence: int)
    """
    if subtype1 == subtype2:
        return False, None, None, 0
    
    # Get available sequences for each subtype
    seqs1 = st_to_arr_dict.get(subtype1, [])
    seqs2 = st_to_arr_dict.get(subtype2, [])
    
    if not seqs1 or not seqs2:
        return False, None, None, 0
    
    # Define window bounds
    ata_len = len(seqs1[0])  # Assuming all sequences have same length
    window_start = max(0, bp_pos_ata - window_size)
    window_end = min(ata_len, bp_pos_ata + window_size + 1)
    
    best_divergence = 0
    best_seq1 = None
    best_seq2 = None
    
    # Try multiple combinations to find the most divergent pair
    max_combinations = min(10, len(seqs1) * len(seqs2))  # Limit combinations for performance
    combinations_tried = 0
    
    # Shuffle sequences to get random sampling
    shuffled_seqs1 = py_rng.sample(seqs1, min(5, len(seqs1)))
    shuffled_seqs2 = py_rng.sample(seqs2, min(5, len(seqs2)))
    
    for seq1 in shuffled_seqs1:
        for seq2 in shuffled_seqs2:
            if combinations_tried >= max_combinations:
                break
                
            # Extract window regions
            window1 = seq1[window_start:window_end]
            window2 = seq2[window_start:window_end]
            
            # Calculate divergence in this window
            divergence = calculate_hamming_distance(window1, window2)
            
            # Update best pair if this is more divergent
            if divergence > best_divergence:
                best_divergence = divergence
                best_seq1 = seq1
                best_seq2 = seq2
            combinations_tried += 1

    is_valid = best_divergence >= min_div_threshold
    return is_valid, best_seq1, best_seq2, best_divergence

def optimize_blueprint_breakpoints(blueprint: list, bpid_to_st: dict, 
                                 st_to_arr_dict: dict, hxb2_to_ata: np.ndarray,
                                 window_size: int, min_div_threshold: int, 
                                 py_rng) -> dict:
    """
    Optimize breakpoints to ensure sufficient divergence between transitioning subtypes.
    
    Returns:
        dict: Mapping of segment indices to optimized breakpoint info
              {segment_idx: {"pos": ata_pos, "p1": seq_array1, "p2": seq_array2, "divergence": int}}
    """
    optimized_bps = {}

    for i in range(len(blueprint) - 1):
        current_seg = blueprint[i]
        next_seg    = blueprint[i + 1]

        current_subtypes = [bpid_to_st[bpid] for bpid in current_seg[2]]
        next_subtypes    = [bpid_to_st[bpid] for bpid in next_seg[2]]

        if set(current_subtypes) != set(next_subtypes):
            bp_hxb2 = next_seg[0]
            bp_ata  = int(hxb2_to_ata[bp_hxb2])

            all_subtypes = set(current_subtypes) | set(next_subtypes)
            subtype1 = list(all_subtypes)[0]
            subtype2 = list(all_subtypes)[1] if len(all_subtypes) > 1 else subtype1

            is_valid, seq1, seq2, divergence = validate_breakpoint_divergence(
                bp_ata, subtype1, subtype2, st_to_arr_dict,
                window_size, min_div_threshold, py_rng
            )

            if is_valid:
                # Store ONLY under index i  (= "transition after segment i")
                optimized_bps[i] = {
                    "pos": bp_ata,
                    "parents": {subtype1: seq1, subtype2: seq2},
                    "divergence": divergence,
                }
            else:
                return None

    return optimized_bps

# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def build_rate_array(ata_to_hxb2: np.ndarray, within_st_div_rate: dict) -> np.ndarray:
    rates = np.full(len(ata_to_hxb2), 0.03, dtype=np.float32)
    for (start, end), rate in within_st_div_rate.items():
        mask = (ata_to_hxb2 >= start) & (ata_to_hxb2 <= end)
        rates[mask] = rate
    return rates


def mutate_sequence_vec(seg_arr: np.ndarray, seg_start_ata: int,
                        rate_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ATCG        = np.array([ord('A'), ord('T'), ord('C'), ord('G')], dtype=np.uint8)
    is_base     = np.isin(seg_arr, ATCG)
    n           = len(seg_arr)
    local_rates = rate_array[seg_start_ata: seg_start_ata + n]
    rand_vals   = rng.random(n).astype(np.float32)
    mutate_mask = is_base & (rand_vals < local_rates)

    if mutate_mask.any():
        idx          = np.where(mutate_mask)[0]
        base_to_idx  = np.zeros(256, dtype=np.int8)
        for i, b in enumerate(ATCG):
            base_to_idx[b] = i
        current_idx  = base_to_idx[seg_arr[idx]]
        offsets      = rng.integers(1, 4, size=len(idx), dtype=np.int8)
        new_idx      = (current_idx + offsets) % 4
        seg_arr[idx] = ATCG[new_idx]

    return seg_arr

def sample_divergent_subtype(already_assigned: set, candidates: list, py_rng) -> str:
    if not already_assigned or not candidates:
        return py_rng.choice(candidates)

    weights = []
    for st in candidates:
        g_st = SUBTYPE_GROUPS.get(st, 2)
        w    = max(DIVERGENCE_WEIGHTS[g_st][SUBTYPE_GROUPS.get(a, 2)] for a in already_assigned)
        weights.append(w)

    total = sum(weights)
    r     = py_rng.random() * total
    cumul = 0.0
    for st, w in zip(candidates, weights):
        cumul += w
        if r <= cumul:
            return st
    return candidates[-1]

# ---------------------------------------------------------------------------
# Worker — writes directly into its disjoint row slice of the final memmaps.
# No temporary files, no merge step.
# ---------------------------------------------------------------------------

def _worker(args: dict) -> list:
    """
    Each worker:
      1. Opens the final memmaps with mmap_mode='r+' (no data copied into RAM).
      2. Writes only into rows [row_start, row_end) — disjoint from all other
         workers, so no locking is needed.
      3. Flushes and returns the name list for its chunk.

    This eliminates the merge step entirely: when all workers finish, the
    final memmaps are already complete.
    """
    row_start          = args["row_start"]
    row_end            = args["row_end"]          # exclusive
    chunk_size         = row_end - row_start
    worker_id          = args["worker_id"]
    worker_seed        = args["worker_seed"]
    pure_st_to_id_dict = args["pure_st_to_id_dict"]
    n_subtypes         = len(pure_st_to_id_dict)
    ata_len            = args["ata_len"]
    n_packed           = args["n_packed"]
    rate_array         = args["rate_array"]
    window_size        = args["window_size"]
    min_div_threshold  = args["min_div_threshold"]
    max_attempts       = args["max_attempts"]

    # Open final memmaps
    seq_mm = np.load(args["out_seqs"],   mmap_mode='r+')
    lbl_mm = np.load(args["out_labels"], mmap_mode='r+')

    # Pre-convert sequences to numpy arrays for fast Hamming distance math
    st_to_arr_dict = {
        st: [np.frombuffer(s.encode(), dtype=np.uint8) for s in seqs]
        for st, seqs in args["st_to_seq_dict"].items()
    }

    rng    = np.random.default_rng(worker_seed)
    py_rng = random.Random(worker_seed)
    names  = []

    for local_i in range(chunk_size):
        sequence_finalized = False
        attempts = 0
        is_recombinant = py_rng.random() <= RP

        while not sequence_finalized:
            if attempts >= max_attempts:
                is_recombinant = False
            attempts += 1
            remaining_st   = args["pure_st_list"].copy()
            seq_row        = np.empty(ata_len, dtype=np.uint8)
            lbl_row        = np.zeros((ata_len, n_subtypes), dtype=bool)

            if not is_recombinant:
                # Pure subtype sequence
                st = py_rng.choice(remaining_st)
                src_seqs = st_to_arr_dict[st]
                src_arr = py_rng.choice(src_seqs)
                seq_row[:] = mutate_sequence_vec(src_arr.copy(), 0, rate_array, rng)
                lbl_row[:, pure_st_to_id_dict[st]] = True
                name = f"pure_{st}"
                sequence_finalized = True
            else:
                # Recombinant sequence
                blueprint = py_rng.choice(args["blueprints"])
                
                # Assign subtypes to blueprint IDs with divergence bias
                bpid_to_st = {}
                already_assigned = set()
                
                # Collect all blueprint IDs first
                all_bpids = set()
                for seg_start_hxb2, seg_stop_hxb2, seg_st_bpid_list in blueprint:
                    for bpid in seg_st_bpid_list:
                        all_bpids.add(bpid)
                
                # Assign subtypes to blueprint IDs
                for bpid in sorted(all_bpids):
                    st = sample_divergent_subtype(already_assigned, remaining_st, py_rng)
                    bpid_to_st[bpid] = st
                    already_assigned.add(st)
                
                # Optimize breakpoints based on divergence
                optimized_bps = optimize_blueprint_breakpoints(
                    blueprint, bpid_to_st, st_to_arr_dict, args["hxb2_to_ata"],
                    window_size, min_div_threshold, py_rng
                )
                
                if optimized_bps is None:
                    # Failed to find valid breakpoints, try again
                    continue
                
                # Build recombinant sequence using optimized breakpoints
                current_pos = 0
                for i, (seg_start_hxb2, seg_stop_hxb2, seg_st_bpid_list) in enumerate(blueprint):

                    # ---- end position ----
                    # Transition AFTER this segment?  → end at the breakpoint
                    if i in optimized_bps:
                        seg_end_ata = optimized_bps[i]["pos"]
                    elif i == len(blueprint) - 1:
                        seg_end_ata = ata_len
                    else:
                        seg_end_ata = int(args["hxb2_to_ata"][seg_stop_hxb2])

                    # ---- source parent ----
                    seg_subtype = bpid_to_st[seg_st_bpid_list[0]]

                    if i in optimized_bps and seg_subtype in optimized_bps[i]["parents"]:
                        # use the parent validated at the downstream breakpoint
                        src_arr = optimized_bps[i]["parents"][seg_subtype]
                    elif (i - 1) in optimized_bps and seg_subtype in optimized_bps[i - 1]["parents"]:
                        # use the parent validated at the upstream breakpoint
                        src_arr = optimized_bps[i - 1]["parents"][seg_subtype]
                    else:
                        src_arr = py_rng.choice(st_to_arr_dict[seg_subtype])

                    # ---- write ----
                    if seg_end_ata - current_pos > 0:
                        chunk = src_arr[current_pos:seg_end_ata].copy()
                        seq_row[current_pos:seg_end_ata] = mutate_sequence_vec(
                            chunk, current_pos, rate_array, rng
                        )
                        for bpid in seg_st_bpid_list:
                            lbl_row[current_pos:seg_end_ata,
                                    pure_st_to_id_dict[bpid_to_st[bpid]]] = True

                    current_pos = seg_end_ata

                name = f"recombinant_{'_'.join(bpid_to_st.values())}"
                sequence_finalized = True

        # Final Cleanup & Write to Memmap
        if sequence_finalized:
            seq_row[seq_row == ord('-')] = ord('N')
            lbl_row[seq_row == ord('N')] = False
            seq_mm[row_start + local_i] = seq_row
            lbl_mm[row_start + local_i] = np.packbits(lbl_row, axis=-1)
            names.append(name)
        else:
            # Fallback for rare cases where MAX_ATTEMPTS is hit
            # This should never happen as we fallback to pure sequence generation
            names.append("failed_generation")

    seq_mm.flush()
    lbl_mm.flush()
    return names

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    mp.set_start_method("fork", force=True)

    # Load pure subtypes
    st_to_seq_dict: dict = defaultdict(list)
    pure_st_seq_path = (
        f"{workspace_path}/data/LANL/curated_alignments/LANL_SBTR_ALIGNMENT.fasta"
    )
    hxb2_ata_seq: str = ""
    for index, record in enumerate(SeqIO.parse(pure_st_seq_path, "fasta")):
        if index == 0:
            hxb2_ata_seq = str(record.seq)
        else:
            st = record.id.split(".")[1]
            st_to_seq_dict[st].append(str(record.seq))

    pure_st_list       = list(st_to_seq_dict.keys())
    pure_st_to_id_dict = {st: i for i, st in enumerate(pure_st_list)}
    n_subtypes         = len(pure_st_to_id_dict)
    n_packed           = int(np.ceil(n_subtypes / 8))
    ata_len            = len(hxb2_ata_seq)

    print(f"Pure subtypes    : {pure_st_list}")
    print(f"Mapping with ids : {pure_st_to_id_dict}")
    print(f"Alignment length : {ata_len}")
    print(f"Workers          : {N_WORKERS}")
    print(f"Output shape     : sequences ({N_SEQ}, {ata_len})  "
          f"labels ({N_SEQ}, {ata_len}, {n_packed}) [packed from {n_subtypes} subtypes]")
    print(f"Recombinant prop : {RP:.2%}")
    print(f"Random seed      : {SEED}")
    print(f"Breakpoint checking parameters: window_size={WINDOW_SIZE}, min_div_threshold={MIN_DIV_THRESHOLD}, max_attempts={MAX_ATTEMPTS}")
    print(f"Using {JSON_BP_PATH.name} for recombinant blueprint sampling")

    with open(JSON_BP_PATH) as f:
        blueprints = json.load(f)

    hxb2_to_ata, ata_to_hxb2 = build_hxb2_ata_maps(hxb2_ata_seq)

    within_st_div_rate = {
        (0,    2292): 0.05,
        (2293, 6225): 0.02,
        (6226, 8795): 0.13,
        (8796, 9719): 0.04,
    }
    rate_array = build_rate_array(ata_to_hxb2, within_st_div_rate)

    # Allocate final memmaps upfront, then close in main process.
    # Workers reopen with mmap_mode='r+' and write to disjoint row ranges.
    seq_mm = np.lib.format.open_memmap(out_seqs,   mode='w+', dtype=np.uint8, shape=(N_SEQ, ata_len))
    lbl_mm = np.lib.format.open_memmap(out_labels, mode='w+', dtype=np.uint8, shape=(N_SEQ, ata_len, n_packed))
    del seq_mm, lbl_mm   # close before workers open them

    # Split rows into contiguous chunks
    chunks      = np.array_split(np.arange(N_SEQ), min(N_WORKERS, N_SEQ))
    ss          = np.random.SeedSequence(SEED)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(chunks))]

    worker_args = [
        {
            "worker_id":          wid,
            "row_start":          int(chunk[0]),
            "row_end":            int(chunk[-1]) + 1,
            "worker_seed":        child_seeds[wid],
            "out_seqs":           out_seqs,
            "out_labels":         out_labels,
            "st_to_seq_dict":     dict(st_to_seq_dict),
            "pure_st_list":       pure_st_list,
            "pure_st_to_id_dict": pure_st_to_id_dict,
            "blueprints":         blueprints,
            "hxb2_to_ata":        hxb2_to_ata,
            "rate_array":         rate_array,
            "ata_len":            ata_len,
            "n_packed":           n_packed,
            "window_size":        WINDOW_SIZE,
            "min_div_threshold":  MIN_DIV_THRESHOLD,
            "max_attempts":       MAX_ATTEMPTS,
        }
        for wid, chunk in enumerate(chunks)
        if len(chunk) > 0
    ]

    print(f"\nLaunching {len(worker_args)} workers (no merge step — workers write directly into final memmaps)…")
    with mp.Pool(processes=len(worker_args)) as pool:
        results = list(tqdm(
            pool.imap(_worker, worker_args),
            total=len(worker_args),
            desc="Chunks completed",
        ))

    # Reassemble names in row order — O(n_seq) strings, trivially fast
    names = [name for chunk in results for name in chunk]

    print(f"\nRecombinant proportion: "
          f"{sum(1 for n in names if n.startswith('recombinant')) / len(names):.2%}")

    print(f"\nPercentage of failed generations (should be very low): "
          f"{sum(1 for n in names if n == 'failed_generation') / len(names):.2%}")
    st_appearance_count = defaultdict(int)
    for name in names:
        subtypes = '/'.join(name.split('_')[1:])
        for st in subtypes.split('/'):
            st_appearance_count[st] += 1
    print("\nSubtype appearance counts:")
    for st in pure_st_list:
        print(f"  {st}: {st_appearance_count[st]} ({st_appearance_count[st] / len(names):.2%})")

    split_list = np.random.default_rng(SEED).choice(
        ["train", "val", "test"], size=len(names), p=[0.9, 0.05, 0.05]
    )
    with open(out_names, 'w') as f:
        f.write("sequence_id\tsequence_name\tpure_or_recombinant\tsubtypes\tn_subtypes\tsplit\n")
        for i, name in enumerate(names, 1):
            subtypes = '/'.join(name.split('_')[1:])
            n_sub    = len(subtypes.split('/'))
            f.write(f"{i}\t{name}\t{name.split('_')[0]}\t{subtypes}\t{n_sub}\t{split_list[i-1]}\n")

    print(f"\nFiles written:")
    print(f"  Sequences : {out_seqs}  ({os.path.getsize(out_seqs) / 1e6:.1f} MB)")
    print(f"  Labels    : {out_labels}  ({os.path.getsize(out_labels) / 1e6:.1f} MB)")
    print(f"  Metadata  : {out_names}")

    # Print an example recombinant sequence with labels for sanity check
    for name in names:
        if name.startswith("recombinant"):
            example_idx = names.index(name)
            break
    print(f"\nExample sequence #{example_idx + 1}: {names[example_idx]}")
    seq_mm = np.load(out_seqs, mmap_mode='r')
    lbl_mm = np.load(out_labels, mmap_mode='r')
    print("Sequence (ATA):")
    print(''.join(chr(b) for b in seq_mm[example_idx]))
    lbl_row = np.unpackbits(lbl_mm[example_idx], axis=-1)[:, :n_subtypes]
    sample_subtypes = [pure_st_list[i] for i in range(n_subtypes) if lbl_row[:, i].any()]
    print(f"Subtypes present in this sequence: {', '.join(sample_subtypes)}")
