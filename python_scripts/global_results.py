"""sbtr global results"""

import re
import math
from pathlib import Path
import argparse
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from figs import visualize_region_comparison
from src import config

parser = argparse.ArgumentParser(
    description='Compare results from various methods.'
)
parser.add_argument("--sbtr_results", type=str, required=True,
                    help="CSV file containing SBTR results for each sample. Sample names must be formatted as <true_subtype>.<etc>.")
parser.add_argument("--output_figs", action="store_true",
                    help="If true, output comparison figures")
parser.add_argument("--out_dir", type=str, required=False, default=None,
                    help="Directory to save results. Default is the parent directory of sbtr_results.")
parser.add_argument("--breakpoints_csv", type=str,
                    default="/pasteur/helix/projects/mPath/oanoufa/sbtr/data/output/lanl_crf_segments_brokedown_aln.csv",
                    help="LANL CRF breakpoints reference (alignment coordinate system).")
parser.add_argument("--ci_method", choices=["wilson", "clopper-pearson"], default="wilson",
                    help="95%% CI method for sensitivity/specificity.")

args = parser.parse_args()

sbtr_results = Path(args.sbtr_results)
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

# Breakpoints reference (only consulted for CRF truths, i.e. true_label not
# in ST_TO_ID_DICT -- pure-subtype truths are unambiguous regardless of
# fragment length and are scored by exact match)
ATA_LEN = config.ATA_LEN
WILDCARD = "U"
# 'A'->A5 for CRF26 and the general 'A' ambiguity are already resolved
# upstream in scrape_and_parse_lanl_breakpoints; only 01_AE's legacy 'E'
# split needs handling here, and that's fully overridden below anyway.
LEGACY_LABEL_MAP = {"E": "AE"}

CRF_NUM_RE = re.compile(r"^(\d+)_")
ACTIVE_POS_RE = re.compile(r"\s*(\d+)\s*-\s*(\d+)\s*$")


class Segment:
    __slots__ = ("start", "end", "subtypes")

    def __init__(self, start, end, subtypes):
        self.start, self.end, self.subtypes = start, end, subtypes

    def overlaps(self, start, end):
        return self.start <= end and self.end >= start


def normalize_component(label: str) -> str:
    return LEGACY_LABEL_MAP.get(label.strip(), label.strip())


def load_breakpoints(path: str) -> Dict[str, List[Segment]]:
    df = pd.read_csv(path)
    segments_by_crf: Dict[str, List[Segment]] = {}
    for crf, group in df.groupby("crf"):
        segs = []
        for _, row in group.iterrows():
            components = frozenset(normalize_component(c) for c in str(row["subtype"]).split("/"))
            segs.append(Segment(int(row["start"]), int(row["end"]), components))
        segments_by_crf[crf] = segs

    # CRF01 (=CRF01_AE): no usable breakpoint in the analyzed region
    # treat as pure AE across the whole alignment (overrides parsed rows).
    # segments_by_crf["CRF01"] = [Segment(1, ATA_LEN, frozenset({"AE"}))]
    return segments_by_crf


def process_true_label(true_label: str) -> str:
    if '01_AE' in true_label:
        return 'AE'
    return true_label


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


def acceptable_truth_labels(sample_name: str, active_positions, true_label: str,
                             segments_by_crf: Dict[str, List[Segment]],
                             warnings: List[str]) -> Tuple[set, bool, bool]:
    """Only called when true_label is not a pure subtype (not in ST_TO_ID_DICT).
    Returns (accepted_subtypes, excluded)."""
    raw_token = sample_name.split('.')[0]
    key = raw_token

    if key not in segments_by_crf:
        warnings.append(f"{sample_name}: unrecognized CRF key '{key}' (from '{raw_token}') "
                         "- likely a URF, excluding sample from analysis")
        return set(), True

    m = ACTIVE_POS_RE.match(str(active_positions)) if pd.notna(active_positions) else None
    if m:
        start, end = int(m.group(1)), int(m.group(2))
    else:
        warnings.append(f"{sample_name}: missing/unparseable active_positions - using full window")
        start, end = 1, ATA_LEN

    accepted = {true_label}
    for seg in segments_by_crf[key]:
        if seg.overlaps(start, end):
            accepted.update(seg.subtypes)
            accepted.discard(WILDCARD)
    return accepted, False


def process_row_results_df(row: pd.Series, segments_by_crf: Dict[str, List[Segment]],
                            warnings: List[str]) -> pd.Series:
    sample_name = row.sample_name
    final_decision = row.final_decision
    active_positions = row.active_positions
    ref_best_crf = row.ref_best_crf.split('.')[0]

    true_label = process_true_label(sample_name.split('.')[0])
    status, predicted, length, assignment, crf_list = process_pred_label(final_decision)

    if predicted is None:
        warnings.append(f"{sample_name}: unparseable final_decision '{final_decision}'")
        return pd.Series({
            "true_label": true_label, "predicted_label": None, "status": status,
            "accepted_labels": {true_label}, "correct": False, "excluded": False
        })

    if true_label in ST_TO_ID_DICT:
        accepted, excluded = {true_label}, False
    else:
        accepted, excluded = acceptable_truth_labels(
            sample_name, active_positions, true_label, segments_by_crf, warnings
        )

    if excluded:
        return pd.Series({
            "true_label": true_label, "predicted_label": predicted, "status": status,
            "accepted_labels": set(), "correct": False, "excluded": True
        })

    if not accepted:
        warnings.append(f"{sample_name}: no accepted labels for true_label '{true_label}' "
                        f"with active_positions '{active_positions}'")
        return pd.Series({
            "true_label": true_label, "predicted_label": predicted, "status": status,
            "accepted_labels": set(), "correct": False, "excluded": False
        })

    # 2. Evaluate correctness
    correct = False

    if status == 'pure':
        # Pure predictions are correct if predicted subtype is in accepted set
        # (e.g., pred 'A1' in accepted {'AE', 'A1'} for partial 01_AE gag)
        correct = (predicted in accepted)
        print(f"{sample_name}: CORRECT {predicted} in {accepted}")

    if status == 'recombinant':
        if true_label in ST_TO_ID_DICT:
            correct = False
        
        else:
            # Parse predicted CRFs or composition constituents
            pred_crfs = crf_list.split('+') if crf_list else [ref_best_crf]

            # Check direct match on true label
            if true_label in pred_crfs:
                correct = True
                print(f"{sample_name}: CORRECT {true_label} in {pred_crfs}")
            else:
                # Check if any predicted CRF resolves to an identical subtype set 
                # in this active window (e.g., 32_06A6 is identical to 06_cpx on 2127-9719 in HXB2 coordinates)

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
    valid = results_df.dropna(subset=["accepted_labels", "predicted_label"])
    classes = sorted(set(valid.true_label))
    rows = []
    for cls in classes:
        tp = fp = tn = fn = excluded = 0
        for r in valid.itertuples():
            if r.true_label == cls:
                if r.correct:
                    tp += 1
                else:
                    fn += 1
            elif cls in r.accepted_labels:
                # ambiguous window: calling `cls` here isn't a real false alarm
                if r.predicted_label == cls:
                    excluded += 1
                else:
                    tn += 1
            else:
                if r.predicted_label == cls:
                    fp += 1
                else:
                    tn += 1

        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        sens_lo, sens_hi = ci_fn(tp, tp + fn) if (tp + fn) else (np.nan, np.nan)
        spec_lo, spec_hi = ci_fn(tn, tn + fp) if (tn + fp) else (np.nan, np.nan)

        rows.append({
            "class": cls, "n_truth": tp + fn, "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "n_excluded_ambiguous": excluded,
            "sensitivity": sens, "sens_CI_lo": sens_lo, "sens_CI_hi": sens_hi,
            "specificity": spec, "spec_CI_lo": spec_lo, "spec_CI_hi": spec_hi,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":

    segments_by_crf = load_breakpoints(args.breakpoints_csv)
    sbtr_results_df = pd.read_csv(sbtr_results)

    warnings: List[str] = []
    processed = sbtr_results_df.apply(
        lambda row: process_row_results_df(row, segments_by_crf, warnings), axis=1
    )
    results_df = pd.concat([sbtr_results_df, processed], axis=1)

    n_excluded = int(results_df["excluded"].sum())
    if n_excluded:
        print(f"Excluding {n_excluded} sample(s) with unrecognized/URF true labels "
              f"({sorted(results_df.loc[results_df['excluded'], 'true_label'].unique())})", flush=True)
    results_df = results_df[~results_df["excluded"]].copy()

    ci_fn = wilson_ci if args.ci_method == "wilson" else clopper_pearson_ci
    stats_df = compute_stats(results_df, ci_fn)

    stats_path = out_dir / "sens_spec_by_class.csv"
    stats_df.to_csv(stats_path, index=False)
    print(stats_df.to_string(index=False))

    if warnings:
        print(f"\n{len(warnings)} warnings:", flush=True)
        for w in warnings[:30]:
            print(f"  {w}")
