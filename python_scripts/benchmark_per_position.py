"""Compare sbtr regions with regions produced by jpHMM (and optionally with true regions)."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import re

from python_scripts.src.config import HXB2_ACC, PARTIAL_THR, build_hxb2_ata_maps
from python_scripts.figs import visualize_region_comparison

Regions = List[Tuple[int, int, str]]
RegionsDict = Dict[str, Regions]

MAX_INSERTION_LEN = 0.05                      # max fraction of a sequence that may be LTR/insertion
LTR_SBTR = {"5'LTR", "3'LTR"}                 # sbtr LTR labels
LTR_JPHMM = {"5'-Insertion", "3'-Insertion"}  # jpHMM LTR labels
LTR_ALL = LTR_SBTR | LTR_JPHMM | {"LTR"}
PURE_PREFIX = "p_"                            # pure (non-recombinant) sample names start with this
GAP_CHARS = (ord("-"), ord("."))

# Prediction label -> true labels it is also accepted for (one-directional). jpHMM has no pure E reference: its "E" is CRF01_AE, which is A1 in pol in our labelling. So jpHMM "E" on a true "A1" position is correct, but a predicted "A1" on a true "E" is not.
JPHMM_ACCEPTED = {"E": {"A1"}}
# sbtr (prediction) vs jpHMM (reference): jpHMM's "E" is CRF01_AE, whose pol is "A1" in sbtr's labelling, so an sbtr "A1" on a jpHMM "E" position counts as agreement. One-directional: an sbtr "E" on a jpHMM "A1" position is still a disagreement.
SBTR_VS_JPHMM_ACCEPTED = {"A1": {"E"}}
# Reference labels that accept any prediction: if the truth is "U" (unassigned) at a position, whatever the method called there is counted as correct. This happens in A5 samples only as CRF26_A5U includes segment with U regions.
TRUE_WILDCARDS = {"U"}
# compact mapout line: "<start><base> - <end><base> > <col_before>v<col_after>" (all 1-based)
_MAPOUT_RE = re.compile(r"^(\d+)[A-Za-z]*\s*-\s*(\d+)[A-Za-z]*\s*>\s*(\d+)\s*v\s*(\d+)$")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare results from various methods.")
    parser.add_argument("--sbtr_regions", type=str, required=True,
                        help="CSV file containing sbtr regions for each sample.")
    parser.add_argument("--jphmm_regions", type=str, required=True,
                        help="CSV file containing jpHMM regions for each sample.")
    parser.add_argument("--true_regions", type=str, required=False, default=None,
                        help="CSV file containing true regions for each sample.")
    parser.add_argument("--aligned_fasta", type=str, required=False, default=None,
                        help="FASTA of the input sequences in the ATA alignment, including the HXB2 "
                             f"row ({HXB2_ACC}). Enables 3-frame gene panels in the figures.")
    parser.add_argument("--compact_mapout", type=str, required=False, default=None,
                        help="MAFFT compact mapout (insertions removed by --keeplength) for the "
                             "sequences in --aligned_fasta. Needed to map raw positions to alignment "
                             "columns when sequences have insertions relative to the alignment.")
    parser.add_argument("--output_figs", action="store_true",
                        help="If set, output comparison figures (SVG).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save results.")
    return parser.parse_args()



# Region loading / expansion

def get_regions_dict(regions_csv: Optional[str]) -> RegionsDict:
    """
    Convert a CSV of sample_name,start,end,subtype,length into a dict of
    sample_name -> list of (start, end, subtype) tuples (1-based, inclusive).
    """
    regions_dict: RegionsDict = {}
    if regions_csv is None:
        return regions_dict
    df = pd.read_csv(regions_csv)
    for sample_name, group in df.groupby("sample_name", sort=False):
        regions_dict[str(sample_name)] = [
            (int(s), int(e), str(st))
            for s, e, st in zip(group["start"], group["end"], group["subtype"])
        ]
    return regions_dict


def _expand_regions(regions: Regions) -> np.ndarray:
    """
    Per-position subtype labels indexed by the real coordinates (position p -> index p-1).
    Positions not covered by any region are None. Overlaps: the later region wins.
    """
    if not regions:
        return np.empty(0, dtype=object)
    arr = np.full(max(e for _, e, _ in regions), None, dtype=object)
    for start, end, subtype in regions:
        arr[max(start, 1) - 1:end] = subtype
    return arr


def _mask_from_regions(regions: Regions, length: int) -> np.ndarray:
    """Boolean mask of positions (0-indexed) covered by the given regions."""
    mask = np.zeros(length, dtype=bool)
    for start, end, _ in regions:
        mask[max(start, 1) - 1:end] = True
    return mask


def is_pure_sample(sample_name: str) -> bool:
    return sample_name.startswith(PURE_PREFIX)



# Comparison
def _match_mask(pred: np.ndarray, true: np.ndarray,
                accepted: Optional[Dict[str, set]] = None,
                ref_wildcards: Optional[set] = None) -> np.ndarray:
    """
    Per-position correctness: exact match, or pred label accepted for the true label,
    or true label is a wildcard (any prediction is correct there).
    """
    match = pred == true
    if accepted:
        for pred_label, true_labels in accepted.items():
            is_pred = pred == pred_label
            for true_label in true_labels:
                match |= is_pred & (true == true_label)
    if ref_wildcards:
        for wildcard in ref_wildcards:
            match |= true == wildcard
    return match


def compare_regions(region_dict_1: RegionsDict,
                    region_dict_2: RegionsDict,
                    ltr_regions_dict: RegionsDict,
                    accepted: Optional[Dict[str, set]] = None,
                    ref_wildcards: Optional[set] = None,
                    ) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, int]]]:
    """
    Compare two dicts of regions and output a matching score (ratio of matching positions)
    for each sample, plus per-subtype confusion counts aggregated over all samples.

    region_dict_2 is the reference, region_dict_1 the prediction (1 count = 1 position).
    Scored positions = positions called in the reference and not in the merged LTR mask.
    Uncalled positions in the prediction count as mismatches.

    Returns:
        matching_scores: sample_name -> matching score (0.0 to 1.0)
        is_full: sample_name -> True if the sequence is full-length (>= PARTIAL_THR)
        is_pure: sample_name -> True if the sample is pure (non-recombinant)
        confusion: subtype -> {"TP", "FP", "FN", "TN"} aggregated over all scored positions
        accepted: optional {pred_label: {true_labels}} map. A position where region_dict_1 says
                pred_label and region_dict_2 says one of true_labels is counted as correct
                (one-directional). For the confusion counts, such positions are credited to the
                true label (TP), not counted as FP for pred_label / FN for the true label.
        ref_wildcards: reference labels at which any prediction is counted as correct (e.g. {"U"}).
    """
    matching_scores: Dict[str, float] = {}
    is_full: Dict[str, bool] = {}
    is_pure: Dict[str, bool] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    total_scored = 0

    for sample_name, pred_regions in region_dict_1.items():
        if sample_name not in region_dict_2:
            print(f"No regions for sample {sample_name} in second dict", flush=True)
            continue

        seq_1 = _expand_regions(pred_regions)
        seq_2 = _expand_regions(region_dict_2[sample_name])
        if len(seq_1) != len(seq_2):
            print(f"Length mismatch for sample {sample_name}: {len(seq_1)} vs {len(seq_2)}", flush=True)

        n = min(len(seq_1), len(seq_2))
        if n == 0:
            print(f"Empty regions for sample {sample_name}, skipping", flush=True)
            continue
        seq_1, seq_2 = seq_1[:n], seq_2[:n]

        ltr_mask = _mask_from_regions(ltr_regions_dict.get(sample_name, []), n)
        if ltr_mask.sum() / n > MAX_INSERTION_LEN:
            print(f"Not enough valid positions for {sample_name} (LTR/insertion > "
                  f"{MAX_INSERTION_LEN:.0%}), skipping", flush=True)
            continue

        valid = ~ltr_mask & pd.notna(seq_2)
        valid_positions = int(valid.sum())
        if valid_positions == 0:
            print(f"No scorable positions for {sample_name}, skipping", flush=True)
            continue

        match = _match_mask(seq_1, seq_2, accepted, ref_wildcards)
        total_match = int((match & valid).sum())
        matching_scores[sample_name] = total_match / valid_positions
        is_full[sample_name] = n >= PARTIAL_THR
        is_pure[sample_name] = is_pure_sample(sample_name)

        # Accepted equivalents are relabelled to the true label so they count as TP.
        pred = np.where(match, seq_2, seq_1)[valid]
        true = seq_2[valid]
        total_scored += valid_positions
        subtypes_here = (set(true.tolist()) | set(pred.tolist())) - {None}
        for subtype in subtypes_here:
            is_t = (true == subtype)
            is_p = (pred == subtype)
            c = confusion.setdefault(subtype, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
            c["TP"] += int(np.sum(is_t & is_p))
            c["FP"] += int(np.sum(~is_t & is_p))
            c["FN"] += int(np.sum(is_t & ~is_p))

    for c in confusion.values():
        c["TN"] = total_scored - c["TP"] - c["FP"] - c["FN"]

    return matching_scores, is_full, is_pure, confusion


def report_label_mismatch(dict_a: RegionsDict, dict_b: RegionsDict, name_a: str, name_b: str) -> None:
    """Print subtype labels used by only one of two methods (these always score as mismatches)."""
    labels_a = {st for regs in dict_a.values() for *_, st in regs} - LTR_ALL
    labels_b = {st for regs in dict_b.values() for *_, st in regs} - LTR_ALL
    only_a, only_b = sorted(labels_a - labels_b), sorted(labels_b - labels_a)
    if only_a or only_b:
        print(f"[info] Labels only in {name_a}: {only_a}; only in {name_b}: {only_b}", flush=True)



# Statistics


def _fmt(v) -> str:
    return "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def compute_stats(scores: List[float]) -> Tuple[float, float, float, float, float]:
    """Return avg, median, std (sample), ci95_low, ci95_high. std/CI are NaN for a single score."""
    scores = np.asarray(scores, dtype=float)
    avg = float(np.mean(scores))
    median = float(np.median(scores))
    if len(scores) > 1:
        std = float(np.std(scores, ddof=1))
        ci95 = 1.96 * std / np.sqrt(len(scores))
    else:
        std = ci95 = float("nan")
    return avg, median, std, avg - ci95, avg + ci95


def subset_scores(scores: Dict[str, float], flags: Dict[str, bool], wanted: Optional[bool] = None) -> List[float]:
    """
    Select score values, optionally filtered by a boolean flag dict.
    wanted=None -> all samples, True/False -> only samples whose flag equals wanted.
    """
    if wanted is None:
        return list(scores.values())
    return [score for sample_name, score in scores.items() if flags.get(sample_name) == wanted]



# Name matching / LTR handling


def match_jphmm_names(sbtr_regions_dict: RegionsDict, jphmm_regions_dict: RegionsDict) -> RegionsDict:
    """
    Build a version of jphmm_regions_dict keyed by the correct sbtr sample names,
    by matching each sbtr name to the jphmm name obtained after '.' -> '_' replacement.
    Warns on ambiguous (many-to-one) or missing matches.
    """
    name_mapping_dict: Dict[str, List[str]] = {}
    for sbtr_name in sbtr_regions_dict:
        name_mapping_dict.setdefault(sbtr_name.replace('.', '_'), []).append(sbtr_name)

    matched_dict: RegionsDict = {}
    for jphmm_name, regions in jphmm_regions_dict.items():
        sbtr_matches = name_mapping_dict.get(jphmm_name, [])
        if not sbtr_matches:
            print(f"[warn] No sbtr sample matches jpHMM name {jphmm_name}", flush=True)
            continue
        if len(sbtr_matches) > 1:
            print(f"[warn] Ambiguous match for jpHMM name {jphmm_name}: candidates {sbtr_matches}, skipping", flush=True)
            continue
        matched_dict[sbtr_matches[0]] = regions
    return matched_dict


def retrieve_LTR(regions_dict: RegionsDict, ltr_labels) -> RegionsDict:
    """
    Keep only the regions whose label is in ltr_labels. No length filtering is done here:
    samples with too much LTR/insertion are skipped later in compare_regions (MAX_INSERTION_LEN).
    """
    ltr_dict: RegionsDict = {}
    for sample_name, regions in regions_dict.items():
        ltr_regions = [(s, e, st) for s, e, st in regions if st in ltr_labels]
        if ltr_regions:
            ltr_dict[sample_name] = ltr_regions
    return ltr_dict


def merge_ltr_regions(sbtr_ltr_regions_dict: RegionsDict, jphmm_ltr_regions_dict: RegionsDict) -> RegionsDict:
    """
    Merge sbtr and jpHMM LTR regions by taking the union of their intervals.
    Overlapping or adjacent intervals are merged; merged intervals get the label "LTR".
    """
    merged_dict: RegionsDict = {}
    for sample_name in set(sbtr_ltr_regions_dict) | set(jphmm_ltr_regions_dict):
        regions = sbtr_ltr_regions_dict.get(sample_name, []) + jphmm_ltr_regions_dict.get(sample_name, [])
        if not regions:
            continue
        regions = sorted(regions, key=lambda x: (x[0], x[1]))
        merged = []
        current_start, current_end, _ = regions[0]
        for start, end, _ in regions[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end, "LTR"))
                current_start, current_end = start, end
        merged.append((current_start, current_end, "LTR"))
        merged_dict[sample_name] = merged
    return merged_dict



# Output writers


def compute_subtype_metrics(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Sensitivity and precision per subtype (NaN when the denominator is 0)."""
    metrics = {}
    for subtype, c in confusion.items():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        metrics[subtype] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        }
    return metrics


def write_subtype_rows(f, method_name: str, confusion: Dict[str, Dict[str, int]]) -> None:
    """One row per subtype with confusion counts and derived sensitivity/precision."""
    metrics = compute_subtype_metrics(confusion)
    for subtype in sorted(confusion):
        c, m = confusion[subtype], metrics[subtype]
        f.write(f"{method_name},{subtype},{c['TP']},{c['FP']},{c['FN']},{c['TN']},"
                f"{_fmt(m['sensitivity'])},{_fmt(m['precision'])}\n")


def write_general_row(f, method_name: str, scores: Dict[str, float],
                      is_full: Dict[str, bool], is_pure: Dict[str, bool]) -> None:
    """all / full / partial / pure / recombinant summary stats."""
    row = [method_name]
    for flags, wanted in ((is_full, None), (is_full, True), (is_full, False),
                          (is_pure, True), (is_pure, False)):
        subset = subset_scores(scores, flags, wanted)
        row += [_fmt(v) for v in compute_stats(subset)] if subset else ["NA"] * 5
    f.write(",".join(row) + "\n")


def write_per_sample(path: Path, columns: List[Tuple[str, Dict[str, float]]],
                     flag_sources: List[Tuple[Dict[str, bool], Dict[str, bool]]]) -> None:
    """
    One row per sample seen in any score dict. Missing scores are written as NA.
    flag_sources: (is_full, is_pure) pairs searched in order for each sample's flags.
    """
    samples = list(dict.fromkeys(s for _, scores in columns for s in scores))
    with open(path, "w") as f:
        f.write("sample_name," + ",".join(name for name, _ in columns) + ",is_full,is_pure\n")
        for s in samples:
            full = next((fl[s] for fl, _ in flag_sources if s in fl), "NA")
            pure = next((pu[s] for _, pu in flag_sources if s in pu), "NA")
            vals = [_fmt(scores[s]) if s in scores else "NA" for _, scores in columns]
            f.write(f"{s},{','.join(vals)},{full},{pure}\n")
    print(f"Wrote scores per sample in {path}", flush=True)



# Alignment context for gene-panel figures


def read_fasta(path: str) -> Dict[str, str]:
    """Minimal FASTA reader: id (first token of the header) -> sequence."""
    seqs, name, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks)
    return seqs


def _base_columns(aligned_seq: str) -> np.ndarray:
    """0-based alignment columns holding a base; entry p-1 is the column of sequence position p."""
    arr = np.frombuffer(aligned_seq.encode(), dtype=np.uint8)
    return np.flatnonzero(~np.isin(arr, GAP_CHARS))

def read_compact_mapout(path: str) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Parse a MAFFT compact mapout file into
    sample_name -> list of (ins_start, ins_end, col_before), all 1-based.
    ins_start..ins_end are raw-sequence positions removed from the aligned row;
    they sat between alignment columns col_before and col_before + 1.
    """
    maps: Dict[str, List[Tuple[int, int, int]]] = {}
    current = None
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                maps[current] = []
                continue
            m = _MAPOUT_RE.match(line)
            if m is None or current is None:
                raise ValueError(f"{path}:{lineno}: cannot parse line '{line}'")
            start, end, before, _after = (int(g) for g in m.groups())
            maps[current].append((start, end, before))
    return maps


def build_position_map(aligned_seq: str, insertions: List[Tuple[int, int, int]],
                       n_total: int) -> np.ndarray:
    """
    0-based alignment column of every raw sequence position (entry p-1 <-> position p).

    Raw positions inside a removed insertion map to the column just before the insertion.
    All other positions take the base columns of the aligned row, in order.
    Raises ValueError if the mapout and the aligned row are inconsistent.
    """
    base_cols = _base_columns(aligned_seq)
    ins = sorted(insertions)

    prev_end = 0
    for start, end, _ in ins:
        if start > end:
            raise ValueError(f"insertion {start}-{end} has start > end")
        if start <= prev_end:
            raise ValueError(f"overlapping insertions around position {start}")
        prev_end = end

    raw_len = base_cols.size + sum(end - start + 1 for start, end, _ in ins)
    if ins and ins[-1][1] > raw_len:
        raise ValueError(f"insertion ends at {ins[-1][1]} but the raw sequence would only have "
                         f"{raw_len} bases (mapout does not match the aligned row)")

    is_ins = np.zeros(raw_len, dtype=bool)
    for start, end, _ in ins:
        is_ins[start - 1:end] = True

    cols = np.empty(raw_len, dtype=np.int64)
    cols[~is_ins] = base_cols
    for start, end, before in ins:
        cols[start - 1:end] = min(max(before, 1), n_total) - 1   # clamp insertions at the ends

    # Columns must never go backwards along the raw sequence. If they do, the insertion
    # positions in the mapout don't agree with the aligned row (wrong file / wrong sample).
    if raw_len > 1 and np.any(np.diff(cols) < 0):
        bad = int(np.flatnonzero(np.diff(cols) < 0)[0]) + 1
        raise ValueError(f"position map goes backwards at raw position {bad} "
                         f"(mapout does not match the aligned row)")
    return cols

def load_alignment_context(aligned_fasta: str, sample_names,
                           mapout_path: Optional[str] = None) -> Optional[dict]:
    """
    Read the aligned FASTA, build hxb2_to_ata from the HXB2 row and, for each sample, the
    alignment column of every raw sequence position. If a compact mapout is given, insertions
    removed by --keeplength are put back so raw coordinates map correctly.
    Returns None if no HXB2 row is found.
    """
    seqs = read_fasta(aligned_fasta)
    hxb2_key = next((k for k in seqs if HXB2_ACC in k or "HXB2" in k.upper()), None)
    if hxb2_key is None:
        print(f"[warn] No HXB2 ({HXB2_ACC}) row in {aligned_fasta}: figures will not show gene panels", flush=True)
        return None

    hxb2_row = seqs[hxb2_key].upper().replace(".", "-")
    _, hxb2_to_ata = build_hxb2_ata_maps(hxb2_row)
    n_total = len(hxb2_row)

    insertions = read_compact_mapout(mapout_path) if mapout_path else {}
    if mapout_path:
        print(f"Loaded compact mapout for {len(insertions)} sequences", flush=True)

    def _lookup(d, name):
        return d[name] if name in d else d.get(name.replace(".", "_"))

    cols = {}
    missing_aln = missing_map = bad_map = 0
    for name in sample_names:
        seq = _lookup(seqs, name)
        if seq is None:
            missing_aln += 1
            continue
        if len(seq) != n_total:
            print(f"[warn] {name}: aligned length {len(seq)} != HXB2 row length {n_total}, "
                  f"no gene panel for this sample", flush=True)
            continue
        ins = _lookup(insertions, name) if mapout_path else []
        if ins is None:          # not in the mapout: assume no insertions (checked later on region ends)
            missing_map += 1
            ins = []
        try:
            cols[name] = build_position_map(seq, ins, n_total)
        except ValueError as err:
            bad_map += 1
            print(f"[warn] {name}: {err}, no gene panel for this sample", flush=True)

    if missing_aln:
        print(f"[warn] {missing_aln} samples not found in {aligned_fasta}, they get plain bar figures", flush=True)
    if missing_map:
        print(f"[info] {missing_map} samples not in {mapout_path}, assumed to have no insertions", flush=True)
    if bad_map:
        print(f"[warn] {bad_map} samples have a mapout inconsistent with their aligned row", flush=True)
    if not mapout_path:
        print("[info] No --compact_mapout: sequences with insertions will get plain bar figures", flush=True)
    print(f"Loaded alignment ({n_total} columns) for {len(cols)} samples", flush=True)
    return {"hxb2_to_ata": hxb2_to_ata, "n_total": n_total, "cols": cols}

def project_to_alignment(regions: Regions, cols: np.ndarray) -> Optional[Regions]:
    """
    Map (start, end, subtype) from sequence positions (1-based inclusive) to alignment
    columns (1-based inclusive). Returns None if a region falls outside the aligned sequence.
    """
    out = []
    for start, end, subtype in regions:
        if start < 1 or end > len(cols) or start > end:
            return None
        out.append((int(cols[start - 1]) + 1, int(cols[end - 1]) + 1, subtype))
    return out


def write_comparison_figure(sample_name: str, regions_list: List[Regions], labels_list: List[str],
                            suffix: str, fig_dir: Path, aln_ctx: Optional[dict]) -> None:
    """
    Stacked SVG comparison. Gene panels when the sample's aligned sequence is available and
    all region sets fit it; otherwise plain bars in sequence coordinates.
    """
    lengths = [max((e for _, e, _ in r), default=0) for r in regions_list]
    if len(set(lengths)) > 1:
        print(f"[warn] length mismatch for {sample_name}: "
              + ", ".join(f"{l}={n}" for l, n in zip(labels_list, lengths)), flush=True)
    seq_len = max(lengths)
    if seq_len <= 0:
        print(f"[warn] no regions to plot for {sample_name}", flush=True)
        return

    plot_regions, extra = regions_list, {"n_total": seq_len, "hxb2_to_ata": None}
    cols = aln_ctx["cols"].get(sample_name) if aln_ctx else None
    if cols is not None:
        projected = [project_to_alignment(r, cols) for r in regions_list]
        bad = [l for l, p in zip(labels_list, projected) if p is None]
        if bad:
            ends = {l: max((e for _, e, _ in r), default=0) for l, r in zip(labels_list, regions_list)}
            print(f"[warn] {sample_name}: position map covers {len(cols)} raw bases but regions end at "
                  f"{ends}, plotting without gene panel", flush=True)
        else:
            plot_regions = projected
            extra = {"n_total": aln_ctx["n_total"], "hxb2_to_ata": aln_ctx["hxb2_to_ata"]}

    visualize_region_comparison(
        seq_id=sample_name,
        regions_list=plot_regions,
        labels_list=labels_list,
        seq_length=seq_len,
        path=str(fig_dir / f"{sample_name}_{suffix}.svg"),
        **extra,
    )



# Main


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figs"
    if args.output_figs:
        print("output_figs set to True, printing figures", flush=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

    # Regions
    sbtr_regions_dict = get_regions_dict(args.sbtr_regions)
    jphmm_regions_dict = match_jphmm_names(sbtr_regions_dict, get_regions_dict(args.jphmm_regions))
    report_label_mismatch(sbtr_regions_dict, jphmm_regions_dict, "sbtr", "jpHMM")

    # Merged LTR mask (union of sbtr LTR calls and jpHMM insertion calls)
    merged_ltr_regions_dict = merge_ltr_regions(
        retrieve_LTR(sbtr_regions_dict, LTR_SBTR),
        retrieve_LTR(jphmm_regions_dict, LTR_JPHMM),
    )

    # Comparisons: method name -> (scores, is_full, is_pure, confusion)
    results = {"sbtr_jpHMM": compare_regions(sbtr_regions_dict, jphmm_regions_dict,
                                             merged_ltr_regions_dict,
                                             accepted=SBTR_VS_JPHMM_ACCEPTED)}
    per_sample_cols = {"sbtr_jpHMM": "ms_sbtr_jphmm"}

    compare_to_true = args.true_regions is not None
    true_regions_dict: RegionsDict = {}
    if compare_to_true:
        true_regions_dict = get_regions_dict(args.true_regions)
        report_label_mismatch(sbtr_regions_dict, true_regions_dict, "sbtr", "true")
        report_label_mismatch(jphmm_regions_dict, true_regions_dict, "jpHMM", "true")
        results["sbtr_True"] = compare_regions(sbtr_regions_dict, true_regions_dict,
                                               merged_ltr_regions_dict,
                                               ref_wildcards=TRUE_WILDCARDS)
        results["jpHMM_True"] = compare_regions(jphmm_regions_dict, true_regions_dict,
                                                merged_ltr_regions_dict,
                                                accepted=JPHMM_ACCEPTED,
                                                ref_wildcards=TRUE_WILDCARDS)
        per_sample_cols.update({"sbtr_True": "ms_sbtr_true", "jpHMM_True": "ms_jphmm_true"})
    else:
        print("True labels were not given, only comparing match between sbtr and jpHMM", flush=True)
        # jpHMM is the reference here: per-subtype metrics measure agreement with jpHMM, not accuracy.

    print(f"Compared {len(results['sbtr_jpHMM'][0])} samples (sbtr vs jpHMM)", flush=True)

    # Output files
    ms_per_sample_path = out_dir / "matching_scores_per_sample.csv"
    ms_general_path = out_dir / "matching_scores_general.csv"
    ms_per_subtype_path = out_dir / "matching_scores_per_subtype.csv"
    general_header = (
        "methods_compared,avg,median,std,ci95_low,ci95_high,"
        "avg_full,median_full,std_full,ci95_low_full,ci95_high_full,"
        "avg_partial,median_partial,std_partial,ci95_low_partial,ci95_high_partial,"
        "avg_pure,median_pure,std_pure,ci95_low_pure,ci95_high_pure,"
        "avg_recomb,median_recomb,std_recomb,ci95_low_recomb,ci95_high_recomb\n"
    )
    subtype_header = "methods_compared,subtype,TP,FP,FN,TN,sensitivity,precision\n"

    write_per_sample(
        ms_per_sample_path,
        columns=[(per_sample_cols[m], res[0]) for m, res in results.items()],
        flag_sources=[(res[1], res[2]) for res in results.values()],
    )

    with open(ms_general_path, "w") as f:
        f.write(general_header)
        for method, (scores, is_full, is_pure, _) in results.items():
            write_general_row(f, method, scores, is_full, is_pure)
    print(f"Wrote general scores in {ms_general_path}", flush=True)

    with open(ms_per_subtype_path, "w") as f:
        f.write(subtype_header)
        for method, (*_, confusion) in results.items():
            write_subtype_rows(f, method, confusion)
    print(f"Wrote per-subtype scores in {ms_per_subtype_path}", flush=True)

    # Figures
    if not args.output_figs:
        return
    if args.compact_mapout and not args.aligned_fasta:
        print("[warn] --compact_mapout is ignored without --aligned_fasta", flush=True)
    aln_ctx = (load_alignment_context(args.aligned_fasta, sbtr_regions_dict.keys(), args.compact_mapout)
               if args.aligned_fasta else None)
    if aln_ctx is None:
        print("No alignment context: figures are plain bars in sequence coordinates", flush=True)

    print("Printing composition comparison", flush=True)
    for sample_name, sbtr_regions in sbtr_regions_dict.items():
        regions, labels = [sbtr_regions], ["sbtr"]
        has_jphmm = sample_name in jphmm_regions_dict
        if has_jphmm:
            regions.append(jphmm_regions_dict[sample_name])
            labels.append("jpHMM")

        if compare_to_true:
            if sample_name not in true_regions_dict:
                print(f"No true regions for {sample_name}", flush=True)
                continue
            if not has_jphmm:
                print(f"[warn] No jpHMM regions for {sample_name}, plotting without it", flush=True)
            regions.append(true_regions_dict[sample_name])
            labels.append("True labels")
            suffix = "true_comp"
        else:
            if not has_jphmm:
                print(f"No jpHMM regions for {sample_name}", flush=True)
                continue
            suffix = "jphmm_comp"

        write_comparison_figure(sample_name, regions, labels, suffix, fig_dir, aln_ctx)


if __name__ == "__main__":
    main()