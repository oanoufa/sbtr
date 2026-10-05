"""Preprocessing: Deduplicate, remove too short sequences, keep exactly N columns of an alignment, favouring pure-subtype coverage."""
import argparse
import re
import sys
from collections import defaultdict

import numpy as np
from Bio import SeqIO

from python_scripts.src import config

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="aligned FASTA")
    p.add_argument("-o", "--output", required=True, help="trimmed aligned FASTA")
    p.add_argument("-w", "--pure-weight", type=float, default=0.8,
                   help="weight of pure-subtype coverage in [0,1]; "
                        "1 = CRFs only break ties (default: 0.8)")
    p.add_argument("--pooled", action="store_true",
                   help="pool all pure sequences instead of averaging per subtype")
    p.add_argument("-r", "--ref-id", default="K03455",
                   help="substring identifying the reference sequence (default: HXB2 "
                        "accession K03455); all columns where it has a residue are kept")
    p.add_argument("--no-dedup", action="store_true", help="skip redundancy removal")
    p.add_argument("--identity", type=float, default=0.98,
                   help="drop a sequence if >= this identity to a kept one of its group (default: 0.98)")
    p.add_argument("--min-overlap", type=float, default=0.8,
                   help="min fraction of the shorter sequence that must overlap for identity to count (default: 0.8)")
    p.add_argument("--min-per-group", type=int, default=10,
                   help="never reduce a subtype/CRF group below this many sequences (default: 10)")
    p.add_argument("--removed-out", default=None,
                   help="optional TSV: removed sequence, group, closest kept sequence, identity")
    return p.parse_args()


BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

GAP = ord("-")
CRF_RE = re.compile(r"^\d+_")
TARGET_LEN = config.ATA_LEN
HXB2_LEN = config.HXB2_LEN


def crop_to_hxb2_range(ids, arr, ref_idx, start, end):
    ref_cols = np.where(arr[ref_idx] != GAP)[0]     # alignment column of each HXB2 base
    if len(ref_cols) != HXB2_LEN:
        sys.exit(f"HXB2 row has {len(ref_cols)} residues, expected {HXB2_LEN}: "
                 f"coordinates would be wrong")
    if not 1 <= start <= end <= HXB2_LEN:
        sys.exit(f"Invalid HXB2 range {start}-{end}")
    lo, hi = ref_cols[start - 1], ref_cols[end - 1]
    sub = arr[:, lo:hi + 1]
    keep = np.isin(sub, BASES).any(axis=1)
    keep[ref_idx] = True
    print(f"Cropped to HXB2 {start}-{end}: {sub.shape[1]} of {arr.shape[1]} columns; "
          f"dropped {int((~keep).sum())} sequences with no residue in the window")
    return ([s for s, k in zip(ids, keep) if k], sub[keep], int(keep[:ref_idx].sum()))

def accession(seq_id):
    """LANL headers: SUBTYPE.COUNTRY.YEAR.NAME.ACCESSION"""
    return seq_id.split(".")[-1]


def drop_duplicate_accessions(ids, arr):
    first, keep, report, n_conflict = {}, [], [], 0
    for i, sid in enumerate(ids):
        acc = accession(sid)
        if acc not in first:
            first[acc] = i
            keep.append(i)
            continue
        j = first[acc]
        lab_i, lab_j = classify(sid)[1], classify(ids[j])[1]
        if lab_i != lab_j:
            n_conflict += 1
            if n_conflict <= 20:
                print(f"  label conflict for {acc}: kept {ids[j]}, dropped {sid}")
        report.append((sid, lab_i, ids[j], 1.0, "duplicate_accession"))
    print(f"Duplicate accessions: removed {len(report)} "
          f"({n_conflict} with conflicting subtype labels)")
    keep = np.array(keep)
    return [ids[i] for i in keep], arr[keep], report


def write_report(path, rows):
    with open(path, "w") as f:
        f.write("removed\tgroup\tclosest_kept\tidentity\treason\n")
        for rid, grp, closest, idt, reason in rows:
            f.write(f"{rid}\t{grp}\t{closest}\t{idt:.4f}\t{reason}\n")

def load_alignment(path):
    ids, seqs = [], []
    for rec in SeqIO.parse(path, "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq).upper())
    if len({len(s) for s in seqs}) != 1:
        sys.exit("Sequences have different lengths: input is not an alignment")
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    return ids, arr

def filter_by_min_positions(ids, arr, min_positions, gap_char='-'):
    """
    Removes sequences that do not have more than min_positions with actual data.
    
    Parameters:
    - ids: list of sequence IDs
    - arr: 2D uint8 NumPy array of sequences (from load_alignment)
    - min_positions: minimum integer count of non-gap positions required (exclusive: >)
    - gap_char: the character used for gaps (default '-')
    
    Returns:
    - filtered_ids: list of IDs that passed the filter
    - filtered_arr: 2D NumPy array containing only the passing sequences
    """
    gap_val = ord(gap_char)
    data_counts = (arr != gap_val).sum(axis=1)
    mask = data_counts > min_positions
    filtered_ids = [sid for sid, keep in zip(ids, mask) if keep]
    filtered_arr = arr[mask]
    removed_count = len(ids) - len(filtered_ids)
    print(f"Kept {len(filtered_ids)} sequences, removed {removed_count} sequences (<= {min_positions} positions).")

    return filtered_ids, filtered_arr

def write_alignment(path, ids, arr, keep):
    sub = arr[:, keep]
    with open(path, "w") as f:
        for sid, row in zip(ids, sub):
            f.write(f">{sid}\n{row.tobytes().decode()}\n")

def reference_columns(arr, ids, ref_id):
    hits = [i for i, sid in enumerate(ids) if ref_id in sid]
    if len(hits) != 1:
        sys.exit(f"Expected exactly 1 sequence matching '{ref_id}', found {len(hits)}: "
                 f"{[ids[i] for i in hits]}")
    print(f"Reference: {ids[hits[0]]}")
    return hits[0], arr[hits[0]] != GAP

# selection
def classify(seq_id):
    """Return (kind, label) from a LANL-style header: SUBTYPE.country.year..."""
    label = seq_id.split(".")[0]
    if CRF_RE.match(label) or label in {"CPX", "URF"}:
        return "crf", label
    if label == "U":
        return "other", label
    return "pure", label


def group_identity(sub):
    """Pairwise identity / overlap fraction over columns where both sequences have A/C/G/T."""
    present = np.isin(sub, BASES)
    P = present.astype(np.float32)
    matches = np.zeros((len(sub), len(sub)), dtype=np.float32)
    for b in BASES:
        X = (sub == b).astype(np.float32)
        matches += X @ X.T
    overlap = P @ P.T
    n = present.sum(axis=1)
    ident = np.divide(matches, overlap, out=np.zeros_like(matches), where=overlap > 0)
    cover = overlap / np.maximum(np.minimum.outer(n, n), 1)
    return ident, cover, n


def dedup_group(sub, ref_row, identity, min_overlap, min_keep):
    ident, cover, n = group_identity(sub)
    redundant = (ident >= identity) & (cover >= min_overlap)
    np.fill_diagonal(redundant, False)

    order = list(np.argsort(-n, kind="stable"))      # most complete first
    if ref_row is not None:                          # reference always kept
        order.remove(ref_row)
        order.insert(0, ref_row)

    kept, dropped = [], []
    for i in order:
        (dropped if kept and redundant[i, kept].any() else kept).append(i)

    n_back = 0
    while len(kept) < min_keep and dropped:          # restore the most divergent first
        sim = [ident[d, kept].max() for d in dropped]
        kept.append(dropped.pop(int(np.argmin(sim))))
        n_back += 1
    return kept, dropped, ident, n_back


def deduplicate(ids, arr, ref_idx, identity, min_overlap, min_keep, removed_out=None):
    groups = defaultdict(list)
    for i, sid in enumerate(ids):
        groups[classify(sid)[1]].append(i)

    keep_rows, report = [], []
    for label, rows in sorted(groups.items()):
        rows = np.array(rows)
        hit = np.where(rows == ref_idx)[0]
        ref_row = int(hit[0]) if len(hit) else None
        kept, dropped, ident, n_back = dedup_group(arr[rows], ref_row, identity, min_overlap, min_keep)
        keep_rows.extend(rows[kept])
        for d in dropped:
            j = int(np.argmax(ident[d, kept]))
            report.append((ids[rows[d]], label, ids[rows[kept[j]]],
                           float(ident[d, kept[j]]), "redundant"))
        note = f"  (+{n_back} restored to reach --min-per-group)" if n_back else ""
        # print(f"  {label:>10}: {len(rows):5d} -> {len(kept):5d}{note}")

    keep_rows = np.sort(np.array(keep_rows))
    print(f"Deduplication: {len(ids)} -> {len(keep_rows)} sequences")
    return ([ids[i] for i in keep_rows], arr[keep_rows],
            int(np.searchsorted(keep_rows, ref_idx)), report)

def compute_coverage(arr, ids, pooled):
    present = arr != GAP
    pure_by_subtype, crf_rows = defaultdict(list), []
    for i, sid in enumerate(ids):
        kind, label = classify(sid)
        if kind == "pure":
            pure_by_subtype[label].append(i)
        elif kind == "crf":
            crf_rows.append(i)
    if not pure_by_subtype:
        sys.exit("No pure-subtype sequence found: check the header format")

    if pooled:
        rows = [i for r in pure_by_subtype.values() for i in r]
        pure_cov = present[rows].mean(axis=0)
    else:
        pure_cov = np.mean([present[r].mean(axis=0) for r in pure_by_subtype.values()], axis=0)
    crf_cov = present[crf_rows].mean(axis=0) if crf_rows else np.zeros(arr.shape[1])
    overall = present.mean(axis=0)

    print(f"{len(pure_by_subtype)} pure subtypes, {len(crf_rows)} CRF sequences, "
          f"{len(ids)} total")
    return pure_cov, crf_cov, overall


def select_columns(pure_cov, crf_cov, overall, n_keep, w, protected):
    score = w * pure_cov + (1 - w) * crf_cov
    # lexsort: last key is primary -> protected first, then score, overall coverage, position
    order = np.lexsort((np.arange(len(score)), -overall, -score, ~protected))
    return np.sort(order[:n_keep])


if __name__ == "__main__":
    args = parse_args()
    ids, arr = load_alignment(args.input)
    ids, arr, report = drop_duplicate_accessions(ids, arr)
    ref_idx, _ = reference_columns(arr, ids, args.ref_id)

    ids, arr, ref_idx = crop_to_hxb2_range(ids, arr, ref_idx, config.HXB2_START, config.HXB2_END)
    if not args.no_dedup:
        ids, arr, ref_idx, rep = deduplicate(ids, arr, ref_idx, args.identity,
                                             args.min_overlap, args.min_per_group)
        report += rep
    if args.removed_out:
        write_report(args.removed_out, report)
    n_total = arr.shape[1]
    if not 0 < TARGET_LEN <= n_total:
        sys.exit(f"Target length must be in [1, {n_total}]")

    pure_cov, crf_cov, overall = compute_coverage(arr, ids, args.pooled)
    protected = arr[ref_idx] != GAP
    ins = ~protected
    print(f"{int(ins.sum())} insertion columns relative to HXB2; "
          f"{int((pure_cov[ins] >= 0.05).sum())} with pure coverage >= 5%")
    n_prot = int(protected.sum())
    if n_prot > TARGET_LEN:
        sys.exit(f"{n_prot} reference columns to protect exceed target length={TARGET_LEN}")
    print(f"Protected {n_prot} reference columns; {TARGET_LEN - n_prot} free slots")

    keep = select_columns(pure_cov, crf_cov, overall, TARGET_LEN, args.pure_weight, protected)
    assert protected[keep].sum() == n_prot  # every reference column survived

    dropped = np.setdiff1d(np.arange(n_total), keep)
    print(f"Kept {len(keep)}/{n_total} columns; "
          f"dropped {int((pure_cov[dropped] > 0).sum())} columns with some pure coverage "
          f"(max pure coverage among dropped: "
          f"{pure_cov[dropped].max() if len(dropped) else 0:.4f})")

    ids, arr = filter_by_min_positions(ids, arr, min_positions=8000)

    write_alignment(args.output, ids, arr, keep)