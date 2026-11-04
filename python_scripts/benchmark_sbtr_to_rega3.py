"""sbtr to rega3 paper type results"""

import re
import math
from pathlib import Path
import argparse
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from python_scripts.figs import visualize_region_comparison
from python_scripts.src import config

parser = argparse.ArgumentParser(
    description='Compare results from various methods.'
)
parser.add_argument("--sbtr_results", type=str, required=True,
                    help="CSV file containing SBTR results for each sample. Sample names must be formatted as <true_subtype>.<etc>.")
parser.add_argument("--tag", type=str, required=False, default="results",
                    help="Name of the output file.")
parser.add_argument("--output_figs", action="store_true",
                    help="If true, output comparison figures")
parser.add_argument("--out_dir", type=str, required=False, default=None,
                    help="Directory to save results. Default is the parent directory of sbtr_results.")
parser.add_argument("--seg_bd_csv", type=str,
                    default="/pasteur/helix/projects/mPath/oanoufa/sbtr/data/lanl_crf_segments_brokedown_aln.csv",
                    help="LANL CRF segments reference (alignment coordinate system).")
parser.add_argument("--ci_method", choices=["wilson", "clopper-pearson"], default="wilson",
                    help="95%% CI method for sensitivity/specificity.")

args = parser.parse_args()

sbtr_results = Path(args.sbtr_results)
tag = args.tag
output_figs = args.output_figs
if output_figs:
    print("output_figs set to True, printing figures", flush=True)
if args.out_dir is None:
    out_dir = sbtr_results.parent
else:
    out_dir = Path(args.out_dir)
sample_vis_dir = out_dir / "figs"
sample_vis_dir.mkdir(parents=True, exist_ok=True)

ST_TO_ID_DICT = config.ST_TO_ID_DICT

# Breakpoints reference (only consulted for CRF truths, i.e. true_label not in ST_TO_ID_DICT -- pure-subtype truths are unambiguous regardless of fragment length and are scored by exact match)
ATA_LEN = config.ATA_LEN
WILDCARD = "U"

CRF_NUM_RE = re.compile(r"^(\d+)_")
ACTIVE_POS_RE = re.compile(r"\s*(\d+)\s*-\s*(\d+)\s*$")


class Segment:
    __slots__ = ("start", "end", "subtypes")

    def __init__(self, start, end, subtypes):
        self.start, self.end, self.subtypes = start, end, subtypes

    def overlaps(self, start, end):
        return self.start <= end and self.end >= start


def normalize_component(label: str) -> str:
    return label.strip()


def load_breakpoints(path: str) -> Dict[str, List[Segment]]:
    df = pd.read_csv(path)
    segments_by_crf: Dict[str, List[Segment]] = {}
    for crf, group in df.groupby("crf"):
        segs = []
        for _, row in group.iterrows():
            components = frozenset(normalize_component(c) for c in str(row["subtype"]).split("/"))
            segs.append(Segment(int(row["start"]), int(row["end"]), components))
        segments_by_crf[crf] = segs
    return segments_by_crf


def process_pred_label(final_decision: str) -> Tuple[str, str]:
    """final_decision grammar:
        <pure|recombinant>.<composition>.<partial|full>.<like|assigned|unassigned>.<crf_list>
    Returns (status, composition). Composition is the label to score, for
    both pure and recombinant calls.
    """
    parts = final_decision.split('.', 4)
    status = parts[0]
    composition = parts[1] if len(parts) > 1 else None
    length = parts[2] if len(parts) > 2 else None
    assignment = parts[3] if len(parts) > 3 else None
    crf_list = parts[4] if len(parts) > 4 else None
    return status, composition, length, assignment, crf_list


def acceptable_truth_labels(sample_name: str, true_label: str,
                            segments_by_crf: Dict[str, List[Segment]],
                            warnings: List[str]) -> Tuple[set, bool]:
    key = sample_name.split('.')[0]
    if key not in segments_by_crf:
        warnings.append(f"{sample_name}: unrecognized CRF key '{key}' "
                        "- likely a URF, excluding sample from analysis")
        return set(), True

    accepted = {true_label}
    for seg in segments_by_crf[key]:
        accepted.update(seg.subtypes)
    accepted.discard(WILDCARD)
    return accepted, False


def compare_true_ref_crf(
    sample_name, active_positions, ref_best_crf: str, true_crf: str, segments_by_crf: Dict[str, List[Segment]],
    ) -> Tuple[set, bool, bool]:
    """Only called when true_label is a CRF and ref_best_crf is a different CRF"""

    m = ACTIVE_POS_RE.match(str(active_positions)) if pd.notna(active_positions) else None
    if m:
        start, end = int(m.group(1)), int(m.group(2))
    else:
        warnings.append(f"{sample_name}: missing/unparseable active_positions - using full window")
        start, end = 1, ATA_LEN

    ref_best_crf_seq = []
    true_crf_seq = []
    # Reconstruct both subtype sequences (BBBBBBBBCCCCCCCBBBBBB...) then cut to active positions and compare
    for seg in segments_by_crf[ref_best_crf]:
        ref_best_crf_seq.extend('/'.join(seg.subtypes) for i in range(seg.start, seg.end + 1))

    for seg in segments_by_crf[true_crf]:
        true_crf_seq.extend('/'.join(seg.subtypes) for i in range(seg.start, seg.end + 1))

    ref_best_crf_seq = ref_best_crf_seq[start - 1:end]
    true_crf_seq = true_crf_seq[start - 1:end]

    for i, (ref_subtype, true_subtype) in enumerate(zip(ref_best_crf_seq, true_crf_seq), start=start):
        # Check if both are lists of one element (can also be a position wiht uncertainty, e.g. 'A1/B' or 'AE/B')
        if '/' in ref_subtype or '/' in true_subtype:
            ref_subtypes = set(ref_subtype.split('/'))
            true_subtypes = set(true_subtype.split('/'))
            if not ref_subtypes.intersection(true_subtypes):
                print(f"{sample_name}: WRONG position {i} true {true_subtypes} != ref_best {ref_subtypes} ")
                return False

    print(f"{sample_name}: CORRECT {true_crf} matches {ref_best_crf} in active positions {active_positions}")
    return True


def process_row_results_df(
    row: pd.Series,
    segments_by_crf_bd: Dict[str, List[Segment]],
    warnings: List[str]) -> pd.Series:
    sample_name = row.sample_name
    crf_key = sample_name.split('.')[0]
    true_label = crf_key
    status, predicted, length, assignment, crf_list = process_pred_label(row.final_decision)

    if predicted is None:
        warnings.append(f"{sample_name}: unparseable final_decision '{row.final_decision}'")
        return pd.Series({
            "true_label": true_label, "predicted_label": None, "status": status,
            "accepted_labels": {true_label}, "correct": False, "excluded": False
        })

    if true_label in ST_TO_ID_DICT:
        accepted, excluded = {true_label}, False
    else:
        accepted, excluded = acceptable_truth_labels(
            sample_name, true_label, segments_by_crf_bd, warnings
        )

    if excluded:
        return pd.Series({
            "true_label": true_label, "predicted_label": predicted, "status": status,
            "accepted_labels": set(), "correct": False, "excluded": True
        })

    pred_crfs = set(crf_list.split('+')) if crf_list else set()

    if true_label in ST_TO_ID_DICT:
        correct = (status == 'pure' and predicted == true_label)
    elif crf_key in pred_crfs:
        correct = True
    elif status == 'pure' and length == 'partial':
        correct = predicted in accepted
    else:
        correct = False

    print(f"{sample_name}: {'CORRECT' if correct else 'WRONG'} - "
          f"{status}.{predicted}.{length} vs true {crf_key}, crfs={sorted(pred_crfs)[:5]}")

    return pd.Series({
        "true_label": true_label, "predicted_label": predicted, "status": status,
        "accepted_labels": accepted, "correct": correct, "excluded": False,
    })

# Confidence intervals
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def clopper_pearson_ci(successes, n, alpha=0.05):
    from scipy import stats
    if n == 0:
        return (float("nan"), float("nan"))
    lo = stats.beta.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return (lo, hi)


def compute_stats(results_df: pd.DataFrame, ci_fn) -> pd.DataFrame:
    sub_re = re.compile(r'^([AF])\d+$')
    norm = lambda l: sub_re.sub(r'\1', l) if isinstance(l, str) else l

    results_df['accepted_labels'] = results_df['accepted_labels'].apply(
        lambda s: {norm(l) for l in s})
    results_df['predicted_label'] = results_df['predicted_label'].map(norm)
    results_df['true_label'] = results_df['true_label'].map(norm)

    valid = results_df.dropna(subset=["accepted_labels", "predicted_label"])
    classes = sorted(set(valid.true_label))
    print(f'Printing classes {classes}')
    rows = []
    for subtype in classes:
        tp = fp = tn = fn = excluded = 0
        for r in valid.itertuples():
            if r.true_label == subtype:
                if r.correct:
                    tp += 1
                else:
                    fn += 1
            elif r.correct:
                # sample judged correct for its own class (e.g. 01_AE called pure A1 on a partial fragment): calling `subtype` here is not a false alarm for that subtype
                if r.predicted_label == subtype:
                    excluded += 1
                else:
                    tn += 1
            elif subtype in r.accepted_labels:
                # ambiguous window: calling `subtype` here isn't a real false alarm
                if r.predicted_label == subtype:
                    excluded += 1
                else:
                    tn += 1
            else:
                if r.predicted_label == subtype:
                    fp += 1
                else:
                    tn += 1

        sens = round(tp / (tp + fn) * 100, 1) if (tp + fn) else np.nan
        spec = round(tn / (tn + fp) * 100, 1) if (tn + fp) else np.nan

        sens_lo, sens_hi = np.round(np.array(ci_fn(tp, tp + fn)) * 100, 1) if (tp + fn) else (np.nan, np.nan)
        spec_lo, spec_hi = np.round(np.array(ci_fn(tn, tn + fp)) * 100, 1) if (tn + fp) else (np.nan, np.nan)

        rows.append({
            "class": subtype, "n": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": sens, "sens_CI_lo": sens_lo, "sens_CI_hi": sens_hi,
            "specificity": spec, "spec_CI_lo": spec_lo, "spec_CI_hi": spec_hi,
        })
    print(f"Total excluded samples: {excluded}")
    return pd.DataFrame(rows)


if __name__ == "__main__":

    segments_by_crf_bd = load_breakpoints(args.seg_bd_csv)
    sbtr_results_df = pd.read_csv(sbtr_results)

    warnings: List[str] = []
    processed = sbtr_results_df.apply(
        lambda row: process_row_results_df(row, segments_by_crf_bd, warnings), axis=1
    )
    results_df = pd.concat([sbtr_results_df, processed], axis=1)

    n_excluded = int(results_df["excluded"].sum())
    if n_excluded:
        print(f"Excluding {n_excluded} sample(s) with unrecognized/URF true labels "
              f"({sorted(results_df.loc[results_df['excluded'], 'true_label'].unique())})", flush=True)
    results_df = results_df[~results_df["excluded"]].copy()

    ci_fn = wilson_ci if args.ci_method == "wilson" else clopper_pearson_ci
    stats_df = compute_stats(results_df, ci_fn)
    is_crf = stats_df['class'].str.match(r'^\d')

    # Sort df
    stats_df = stats_df.iloc[
        stats_df.assign(_is_crf=is_crf, _lower=stats_df['class'].str.lower())
        .sort_values(by=['_is_crf', '_lower'])
        .index
    ]

    stats_path = out_dir / f"{tag}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(stats_df.to_string(index=False))

    if warnings:
        print(f"\n{len(warnings)} warnings:", flush=True)
        for w in warnings[:30]:
            print(f"  {w}")
