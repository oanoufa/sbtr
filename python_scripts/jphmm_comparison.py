import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
from src.figs import visualize_region_comparison

parser = argparse.ArgumentParser(
    description='Compare results from various methods.'
)
parser.add_argument("--sbtr_regions", type=str, required=True,
                    help="CSV file containing SBTR regions for each sample.")
parser.add_argument("--jphmm_regions", type=str, required=True,
                    help="CSV file containing jpHMM regions for each sample.")
parser.add_argument("--true_regions", type=str, required=False, default=None,
                    help="CSV file containing true regions for each sample.")
parser.add_argument("--output_figs",  action="store_true",
                    help="If true, output comparison figures")
parser.add_argument("--out_dir", type=str, required=True,
                    help="Directory to save results.")

args = parser.parse_args()

sbtr_regions = Path(args.sbtr_regions)
jphmm_regions = Path(args.jphmm_regions)
true_regions = args.true_regions
output_figs = args.output_figs
if output_figs:
    print("output_figs set to True, printing figures", flush=True)
out_dir = Path(args.out_dir)
sample_vis_dir = out_dir / "figs"
sample_vis_dir.mkdir(parents=True, exist_ok=True)

FULL_LENGTH_THRESHOLD = 7000
MAX_INSERTION_LEN = 0.80 # Percentage of the sequence length
LTR_SBTR = {"5'LTR", "3'LTR"} # SBTR LTR labels
LTR_JPHMM = {"5'-Insertion", "3'-Insertion"}

def get_regions_dict(regions_csv: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Convert a CSV of sample_name,start,end,subtype,length into a dict of sample_name -> list of (start, end, subtype) tuples.
    """
    regions_dict = {}
    if regions_csv is not None:
        df = pd.read_csv(regions_csv)
        for sample_name, group in df.groupby("sample_name", sort=False):
            regions_dict[sample_name] = list(zip(group["start"], group["end"], group["subtype"]))
    return regions_dict

def _expand_regions(regions: List[Tuple[int, int, str]]) -> np.ndarray:
    """
    Turn a list of (start, end, subtype) into a flat array of per-position subtype
    labels, concatenated in list order (same semantics as the original extend-loop).
    """
    if not regions:
        return np.empty(0, dtype=object)
    subtypes = np.array([subtype for _, _, subtype in regions], dtype=object)
    lengths = np.array([end - start + 1 for start, end, _ in regions])
    return np.repeat(subtypes, lengths)

def _mask_from_regions(regions: List[Tuple[int, int, str]], length: int) -> np.ndarray:
    """Boolean mask of positions (0-indexed) covered by the given regions."""
    mask = np.zeros(length, dtype=bool)
    for start, end, _ in regions:
        mask[start - 1:end] = True
    return mask

def compare_regions(region_dict_1: Dict[str, List[Tuple[int, int, str]]],
                    region_dict_2: Dict[str, List[Tuple[int, int, str]]],
                    ltr_regions_dict: Dict[str, List[Tuple[int, int, str]]]
                    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Compare two dicts of regions and output a matching score (ratio of matching positions) for each sample.

    Returns:
        matching_scores: sample_name -> matching_score (0.0 to 1.0)
        is_full: sample_name -> True if the sequence is full-length (> FULL_LENGTH_THRESHOLD), else False
    """
    matching_scores = {}
    is_full = {}
    is_pure = {}
    for sample_name in region_dict_1:
        if sample_name not in region_dict_2:
            print(f"No regions for sample {sample_name} in second dict", flush=True)
            continue

        # Reconstruct sequence representations (vectorized instead of per-base Python lists)
        seq_1 = _expand_regions(region_dict_1[sample_name])
        seq_2 = _expand_regions(region_dict_2[sample_name])

        # Sanity check for length match
        if len(seq_1) != len(seq_2):
            print(f"Length mismatch for sample {sample_name}: {len(seq_1)} vs {len(seq_2)}", flush=True)
        seq_len = len(seq_1)

        n = min(len(seq_1), len(seq_2))
        ltr_mask = _mask_from_regions(ltr_regions_dict.get(sample_name, []), n)
        valid = ~ltr_mask
        valid_positions = int(valid.sum())

        if ((seq_len - valid_positions) / seq_len) > MAX_INSERTION_LEN: # Check LTR prop
            print(f"Not enough valid positions for {sample_name}, skipping", flush=True)
            continue
        # Prevent DivisionByZero if every position is masked or sequences are empty
        elif valid_positions > 0:
            total_match = int(((seq_1[:n] == seq_2[:n]) & valid).sum())
            matching_score = total_match / valid_positions
            matching_scores[sample_name] = matching_score
            is_full[sample_name] = seq_len > FULL_LENGTH_THRESHOLD
            is_pure[sample_name] = 'p_' in sample_name

    return matching_scores, is_full, is_pure

def compute_stats(scores: List[float]) -> Tuple[float, float, float, float, float]:
    """Return avg, median, std, ci_low, ci_high for a list of scores."""
    avg = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    ci95 = 1.96 * std / np.sqrt(len(scores))
    return avg, median, std, avg - ci95, avg + ci95

def subset_scores(scores: Dict[str, float], is_full: Dict[str, bool], want_full: bool = None) -> List[float]:
    """
    Select score values from `scores`, optionally filtered by the is_full flag.
    want_full=None -> all samples, True -> full-length only, False -> partial only.
    """
    if want_full is None:
        return list(scores.values())
    return [score for sample_name, score in scores.items() if is_full.get(sample_name) == want_full]

def match_jphmm_names(sbtr_regions_dict: Dict[str, List[Tuple[int, int, str]]],
                      jphmm_regions_dict: Dict[str, List[Tuple[int, int, str]]]):
    """
    Build a version of jphmm_regions_dict keyed by the correct sbtr sample names,
    by matching each sbtr name to the jphmm name obtained after '.' -> '_' replacement.
    Warns on ambiguous (many-to-one) or missing matches.
    """
    # build a mapping between jphmm_name and sbtr_name
    name_mapping_dict = {}
    for sbtr_name in sbtr_regions_dict:
        mangled = sbtr_name.replace('.', '_')
        name_mapping_dict.setdefault(mangled, []).append(sbtr_name)

    matched_dict = {}
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


def retrieve_LTR(
    jphmm_regions_dict: Dict[str, List[Tuple[int, int, str]]],
    LTR_LABELS,
    ) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Retrieve only the LTR regions from the jpHMM regions dict, dropping any
    insertion call that's implausibly long (likely a jpHMM misclassification
    rather than a real LTR) so it gets scored as a mismatch instead of masked out.
    """
    ltr_dict = {}
    for sample_name, regions in jphmm_regions_dict.items():
        ltr_regions = []
        for start, end, subtype in regions:
            length = end - start + 1
            if subtype in LTR_LABELS:
                ltr_regions.append((start, end, subtype))
        if ltr_regions:
            ltr_dict[sample_name] = ltr_regions
    return ltr_dict

def merge_ltr_regions(
    sbtr_ltr_regions_dict: Dict[str, List[Tuple[int, int, str]]],
    jphmm_ltr_regions_dict: Dict[str, List[Tuple[int, int, str]]],
) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Merge SBTR and jpHMM LTR regions by taking the union of their intervals.

    For each sample, overlapping or adjacent LTR intervals are merged into a
    single interval. The merged intervals use the subtype "LTR".
    """
    merged_dict = {}

    sample_names = set(sbtr_ltr_regions_dict) | set(jphmm_ltr_regions_dict)

    for sample_name in sample_names:
        regions = (
            sbtr_ltr_regions_dict.get(sample_name, [])
            + jphmm_ltr_regions_dict.get(sample_name, [])
        )

        if not regions:
            continue

        # Sort by start coordinate, then end coordinate
        regions = sorted(regions, key=lambda x: (x[0], x[1]))

        merged = []
        current_start, current_end, _ = regions[0]

        for start, end, _ in regions[1:]:
            if start <= current_end + 1:
                # Overlapping or adjacent: extend the current interval
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end, "LTR"))
                current_start, current_end = start, end

        # Add final interval
        merged.append((current_start, current_end, "LTR"))

        merged_dict[sample_name] = merged

    return merged_dict

def write_general_row(f, method_name: str, scores: Dict[str, float], is_full: Dict[str, bool], is_pure: Dict[str, bool]):
    row = [method_name]
    for i, wanted_bool in enumerate((None, True, False, True, False)):
        if i < 4:
            dict_used = is_full
        else:
            dict_used = is_pure
        subset = subset_scores(scores, dict_used, wanted_bool)
        if subset:
            avg, median, std, ci_low, ci_high = compute_stats(subset)
            row += [f"{avg:.4f}", f"{median:.4f}", f"{std:.4f}", f"{ci_low:.4f}", f"{ci_high:.4f}"]
        else:
            row += ["NA", "NA", "NA", "NA", "NA"]
    f.write(",".join(row) + "\n")

if __name__ == "__main__":

    # jpHMM regions
    # Turn this csv of sample_name,start,end,subtype,length into a dict of sample_name -> list of (start, end) tuples
    jphmm_regions_dict = get_regions_dict(jphmm_regions)

    # sbtr regions
    sbtr_regions_dict = get_regions_dict(sbtr_regions)

    # Match jphmm names to sbtr names
    jphmm_regions_dict = match_jphmm_names(sbtr_regions_dict, jphmm_regions_dict)

    # ltr_regions from jphmm_regions
    jphmm_ltr_regions_dict = retrieve_LTR(jphmm_regions_dict, LTR_JPHMM)

    # ltr_regions from sbtr_regions
    sbtr_ltr_regions_dict = retrieve_LTR(sbtr_regions_dict, LTR_SBTR)

    # Merged ltr_regions
    merged_ltr_regions_dict = merge_ltr_regions(
        sbtr_ltr_regions_dict,
        jphmm_ltr_regions_dict,
    )

    if true_regions:
        compare_to_true = True
        true_regions_dict = get_regions_dict(true_regions)
        matching_scores_sbtr_true, is_full_sbtr_true, is_pure_sbtr_true = compare_regions(
            sbtr_regions_dict, true_regions_dict, merged_ltr_regions_dict)
        matching_scores_jphmm_true, is_full_jphmm_true, is_pure_jphmm_true = compare_regions(
            jphmm_regions_dict, true_regions_dict, merged_ltr_regions_dict)
    else:
        print(f"True labels were not given, only comparing match between sbtr and jphmm", flush=True)
        compare_to_true = False

    matching_scores_sbtr_jphmm, is_full_sbtr_jphmm, is_pure_sbtr_jphmm = compare_regions(
        sbtr_regions_dict, jphmm_regions_dict,merged_ltr_regions_dict)

    print(f"Comparing {len(matching_scores_sbtr_jphmm.keys())} samples", flush=True)
    # csv files path
    ms_per_sample_path = out_dir / "matching_scores_per_sample.csv"
    ms_general_path = out_dir / "matching_scores_general.csv"

    # header shared by both branches, extended with full/partial breakdown columns
    general_header = (
        "methods_compared,avg,median,std,ci95_low,ci95_high,"
        "avg_full,median_full,std_full,ci95_low_full,ci95_high_full,"
        "avg_partial,median_partial,std_partial,ci95_low_partial,ci95_high_partial,"
        "avg_pure,median_pure,std_pure,ci95_low_pure,ci95_high_pure,"
        "avg_recomb,median_recomb,std_recomb,ci95_low_recomb,ci95_high_recomb\n"
    )

    if not compare_to_true:
        with open(ms_per_sample_path, 'w') as f:
            f.write("sample_name,ms_sbtr_jphmm,is_full,is_pure\n")
            for sample_name, score in matching_scores_sbtr_jphmm.items():
                f.write(f"{sample_name},{score:.4f},{is_full_sbtr_jphmm[sample_name]},{is_pure_sbtr_jphmm[sample_name]}\n")
            print(f"Wrote scores per sample in {ms_per_sample_path}", flush=True)
        with open(ms_general_path, 'w') as f:
            f.write(general_header)
            write_general_row(f, "SBTR_jpHMM", matching_scores_sbtr_jphmm, is_full_sbtr_jphmm, is_pure_sbtr_jphmm)
            print(f"Wrote general scores in {ms_per_sample_path}", flush=True)
        # Visualization sbtr - jphmm
        if output_figs:
            print('Printing composition comparison', flush=True)
            for sample_name in sbtr_regions_dict:
                # Visualize the comparison between our model and jpHMM
                if sample_name in jphmm_regions_dict:
                    out_path = str(sample_vis_dir / f"{sample_name}_jphmm_comp.png")
                    sw_regions = sbtr_regions_dict[sample_name]
                    jp_regions = jphmm_regions_dict[sample_name]
                    max_len_sw = max(end for _, end, _ in sw_regions)
                    max_len_jp = max(end for _, end, _ in jp_regions)
                    max_len = max(max_len_sw, max_len_jp)
                    if max_len_sw != max_len_jp:
                        print(f"[warn] max_len mismatch for {sample_name}: SW={max_len_sw}, jpHMM={max_len_jp}", flush=True)
                    visualize_region_comparison(
                        seq_id=sample_name,
                        regions_list=[sw_regions, jp_regions],
                        labels_list=["Sliding window on probs", "jpHMM"],
                        seq_length=max_len,
                        path=out_path) # Compare our model and true dealigned regions
                else:
                    print(f"No jpHMM regions for {sample_name}", flush=True)

    # True regions
    if compare_to_true:
        with open(ms_per_sample_path, 'w') as f:
            f.write("sample_name,ms_sbtr_jphmm,ms_sbtr_true,ms_jphmm_true,is_full,is_pure\n")
            for sample_name in matching_scores_sbtr_jphmm:
                f.write(f"{sample_name},{matching_scores_sbtr_jphmm[sample_name]:.4f},"
                        f"{matching_scores_sbtr_true[sample_name]:.4f},"
                        f"{matching_scores_jphmm_true[sample_name]:.4f},"
                        f"{is_full_sbtr_jphmm[sample_name]},"
                        f"{is_pure_sbtr_jphmm[sample_name]}\n")
            print(f"Wrote scores per sample in {ms_per_sample_path}", flush=True)

        with open(ms_general_path, 'w') as f:
            f.write(general_header)
            write_general_row(f, "SBTR_jpHMM", matching_scores_sbtr_jphmm, is_full_sbtr_jphmm, is_pure_sbtr_jphmm)
            write_general_row(f, "SBTR_True", matching_scores_sbtr_true, is_full_sbtr_true, is_pure_sbtr_true)
            write_general_row(f, "jpHMM_True", matching_scores_jphmm_true, is_full_jphmm_true, is_pure_jphmm_true)
            print(f"Wrote general scores in {ms_per_sample_path}", flush=True)

        # Visualize the comparison between our model and true regions
        if output_figs:
            print('Printing composition comparison', flush=True)
            for sample_name in sbtr_regions_dict:
                if sample_name in true_regions_dict:
                    out_path = str(sample_vis_dir / f"{sample_name}_true_comp.png")
                    sw_regions = sbtr_regions_dict[sample_name]
                    jp_regions = jphmm_regions_dict[sample_name]
                    true_regions_sample = true_regions_dict[sample_name]
                    max_len_sw = max(end for _, end, _ in sw_regions)
                    max_len_true = max(end for _, end, _ in true_regions_sample)
                    max_len = max(max_len_sw, max_len_true)
                    if max_len_sw != max_len_true:
                        print(f"[warn] max_len mismatch for {sample_name}: SW={max_len_sw}, True={max_len_true}", flush=True)
                        print(f"Regions: SW={sw_regions},\nTrue={true_regions_sample}", flush=True)
                    visualize_region_comparison(
                        seq_id=sample_name,
                        regions_list=[sw_regions, jp_regions, true_regions_sample],
                        labels_list=["Sliding window on probs", "jpHMM", "True labels"],
                        seq_length=max_len,
                        path=out_path) # Compare our model and true dealigned regions
                else:
                    print(f"No true regions for {sample_name}", flush=True)