"""Turn jpHMM segment output into per class benchmark metrics."""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from python_scripts.src import config

# settings
COLLAPSE_SUBSUBTYPES = True   # F1, F2 become F and A1, A6 become A
# by default a sequence split in several segments with the same label is fine, flip this to True if you want those flagged too
FLAG_ANY_MULTISEGMENT = False

CRF_RE  = re.compile(r'^(?:CRF)?(\d{2,3}_[A-Za-z0-9]+?)(?:[_.]|$)')
PURE_RE = re.compile(r'^([A-Z]\d*)(?:[_.]|$)')
PURE_THR = config.PURE_THR

def normalize(label):
    """Clean up a subtype label so names and jpHMM calls look the same."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    lbl = str(label).strip()
    # drop the CRF prefix if there is one
    if lbl.upper().startswith("CRF"):
        lbl = lbl[3:].lstrip("_")
    # F1 becomes F etc
    if COLLAPSE_SUBSUBTYPES and re.fullmatch(r"[A-Z]\d+", lbl):
        lbl = lbl[0]
    return lbl


def truth_from_name(name):
    """Get the true subtype from the sequence name."""
    # try CRF first since 02_AG would otherwise not match anything useful
    m = CRF_RE.match(name)
    if m:
        return normalize(m.group(1))
    m = PURE_RE.match(name)
    if m:
        return normalize(m.group(1))
    return None


def class_sort_key(c):
    """Sort pure subtypes first, then CRFs by number."""
    m = re.match(r"^(\d+)_", c)
    return (1, int(m.group(1)), c) if m else (0, 0, c)


def wilson(k, n, z=1.959964):
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(T, P):
    """Count TP, FP, FN, TN per class from truth and call matrices."""
    # T and P are N x K booleans, P can have several True in a row for mixed calls
    tp = (T & P).sum(axis=0)
    fp = (~T & P).sum(axis=0)
    fn = (T & ~P).sum(axis=0)
    tn = (~T & ~P).sum(axis=0)
    return tp, fp, fn, tn


def safe_div(a, b):
    """Divide a by b and give NaN where b is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


if __name__ == "__main__":
    """Read jpHMM csv, flag weird sequences, compute and save metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("jphmm_csv")
    ap.add_argument("-o", "--out", default="results.csv")
    ap.add_argument("--per-seq", default="per_sequence.csv")
    ap.add_argument("--flagged", default="flagged_sequences.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.jphmm_csv)
    df["pred"] = df["subtype"].map(normalize)
    df = df.sort_values(["sample_name", "start"])

    # squash the segments so we get one row per sequence
    rows = []
    for name, g in df.groupby("sample_name", sort=False):
        truth = truth_from_name(name)
        labels = list(dict.fromkeys(g["pred"]))      # unique labels, order kept
        # how much of the sequence each label covers
        seg_len = g["end"] - g["start"] + 1
        cov = (seg_len.groupby(g["pred"]).sum() / seg_len.sum()).to_dict()
        segs = ";".join(f"{r.start}-{r.end}:{r.subtype}"
                        for r in g.itertuples())
        reason = ""
        if truth is None:
            reason = "unparseable_name"
        elif FLAG_ANY_MULTISEGMENT and len(g) > 1 and len(labels) == 1:
            reason = "multi_segment_same_subtype"
        true_cov = cov.get(truth, 0.0) if truth else np.nan
        # the label covering the most sequence apart from the true one
        others = {c: v for c, v in cov.items() if c != truth}
        top_other = max(others, key=others.get) if others else None
        rows.append(dict(sample_name=name, true=truth,
                         pred="/".join(labels), true_cov=true_cov,
                         top_other=top_other,
                         n_segments=len(g), segments=segs,
                         mixed=len(labels) > 1, flag=reason))
    seqs = pd.DataFrame(rows)
    seqs["correct"] = np.where(seqs["flag"] == "",
                               seqs["true_cov"] >= PURE_THR, np.nan)
    seqs.to_csv(args.per_seq, index=False)

    # flagged ones go to their own file and are left out of the metrics
    flagged = seqs[seqs["flag"] != ""]
    flagged.to_csv(args.flagged, index=False)
    ev = seqs[seqs["flag"] == ""].reset_index(drop=True)

    # above the threshold it's a clean hit, below it the true label is a miss and only the main wrong label gets the false call
    called = []
    for tr, cv, top in zip(ev["true"], ev["true_cov"], ev["top_other"]):
        if cv >= PURE_THR:
            called.append([tr])
        else:
            called.append([top])

    # every class seen in truth or in calls gets a row
    all_called = {c for labs in called for c in labs}
    classes = sorted(set(ev["true"]) | all_called, key=class_sort_key)
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    N = len(ev)

    # build the truth and call matrices
    T = np.zeros((N, K), dtype=bool)
    T[np.arange(N), ev["true"].map(idx).to_numpy()] = True
    P = np.zeros((N, K), dtype=bool)
    for i, labs in enumerate(called):
        P[i, [idx[c] for c in labs]] = True

    tp, fp, fn, tn = confusion(T, P)
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)

    # wilson confidence intervals
    sens_ci = np.array([wilson(tp[k], tp[k] + fn[k]) for k in range(K)])
    spec_ci = np.array([wilson(tn[k], tn[k] + fp[k]) for k in range(K)])
    sens_lo, sens_hi = sens_ci[:, 0], sens_ci[:, 1]
    spec_lo, spec_hi = spec_ci[:, 0], spec_ci[:, 1]

    out = pd.DataFrame({
        "class": classes,
        "n": tp + fn,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": sens * 100, "sens_CI_lo": sens_lo * 100,
        "sens_CI_hi": sens_hi * 100,
        "specificity": spec * 100, "spec_CI_lo": spec_lo * 100,
        "spec_CI_hi": spec_hi * 100,
    })
    out.to_csv(args.out, index=False, na_rep="NA", float_format="%.1f")

    rescued = int((ev["mixed"] & (ev["true_cov"] >= PURE_THR)).sum())
    print(f"Mixed calls         : {int(ev['mixed'].sum())} "
          f"({rescued} above threshold)", file=sys.stderr)
    print(f"Threshold           : {PURE_THR:.0%}", file=sys.stderr)
    # quick recap in the terminal
    print(f"Sequences total     : {len(seqs)}", file=sys.stderr)
    print(f"Evaluated           : {N}", file=sys.stderr)
    print(f"Flagged (excluded)  : {len(flagged)}", file=sys.stderr)
    for r, c in flagged["flag"].value_counts().items():
        print(f"   {r:28s}: {c}", file=sys.stderr)
    if N:
        print(f"Overall accuracy    : {ev['correct'].mean()*100:.1f}%",
              file=sys.stderr)
    print(f"Saved: {args.out}, {args.per_seq}, {args.flagged}", file=sys.stderr)