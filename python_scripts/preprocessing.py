"""LANL reference preparation: filter, deduplicate and trim the combined alignment to exactly
ATA_LEN columns (favouring pure-subtype coverage), split it into pure / CRF / combined reference
alignments, and build the CRF breakpoint label tracks."""

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict, Counter

import numpy as np
from Bio import SeqIO

from python_scripts.src import config
from python_scripts.src import preprocessing_aux as bp


def parse_args():
    out = f"{config.WORKSPACE_PATH}/data/output"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="combined aligned FASTA (SFL + LANL refs)")
    p.add_argument("--pure-out", default=f"{out}/HIV1_PURE_REF.fasta")
    p.add_argument("--crf-out", default=f"{out}/HIV1_CRF_REF.fasta")
    p.add_argument("--combined-out", default=f"{out}/HIV1_COMBINED_REF.fasta")
    p.add_argument("--pooled", action="store_true",
                   help="pool all pure sequences instead of averaging per subtype")
    p.add_argument("--no-dedup", action="store_true", help="skip redundancy removal")
    p.add_argument("--removed-out", default=None,
                   help="optional TSV: removed sequence, group, closest kept sequence, "
                        "identity (or covered fraction for too_short rows), reason")
    return p.parse_args()


WORKSPACE_PATH = config.WORKSPACE_PATH
OUT_DIR = f"{WORKSPACE_PATH}/data/output"
BP_HTML = f"{WORKSPACE_PATH}/data/input/HIV Circulating Recombinant Forms (CRFs)_26_10_2026.html"

BASES = np.frombuffer(b"ACGT", dtype=np.uint8)   # sequences are upper-cased at load time
GAP = ord("-")
CRF_RE = re.compile(r"^\d+_")
URF_RE = re.compile(r"^(?:\d{2,3}|[A-Z]\d?){2,}$")   # 01A1A6, 02BG, BC, A1D, ...
CRF01_AE_RE = re.compile(r"^(CRF)?01[_-]?AE$", re.IGNORECASE)
CRF26_A5U_RE = re.compile(r"^(CRF)?26[_-]?A5U?$", re.IGNORECASE)
YEAR_RE = re.compile(r"^(\d{2}|\d{4})$")

TARGET_LEN = config.ATA_LEN
HXB2_LEN = config.HXB2_LEN
HXB2_USED_LEN = config.HXB2_USED_LEN
ST_TO_ID_DICT = config.ST_TO_ID_DICT
WEIGHT = config.WEIGHT
HXB2_ACC = config.HXB2_ACC
CLUSTERING_IDENTITY = config.CLUSTERING_IDENTITY
MIN_OVERLAP = config.MIN_OVERLAP
MIN_PER_GROUP = config.MIN_PER_GROUP
MIN_POSITIONS = config.MIN_POSITIONS
RESCUE_MIN_POSITIONS = int(MIN_POSITIONS * 0.7)

# CRF01_AE and CRF26_A5U are written to the pure alignment as E / A5 and to the CRF alignment as 01_AE / 26_A5U.
PURE_LIKE_CRFS_ALSO_IN_CRF = True


# Headers and classification

def normalise_id(raw):
    """Canonical SUBTYPE.COUNTRY.YEAR.NAME.ACCESSION: strip a 'Ref.' prefix, A -> A1."""
    parts = raw.split(".")
    if len(parts) > 1 and parts[0].lower() == "ref":
        parts = parts[1:]
    if parts[0].upper() == "A":
        parts[0] = "A1"
    return ".".join(parts)


def classify(seq_id):
    """Return (kind, label) from a normalised header."""
    label = seq_id.split(".")[0]
    if CRF_RE.match(label) or label == "CPX":
        return "crf", label
    if label in ST_TO_ID_DICT:
        return "pure", label
    if label == "URF" or URF_RE.match(label):
        return "urf", label
    return "other", label


def pure_output_label(seq_id):
    """Label used in the pure alignment, or None if the sequence does not go there."""
    kind, label = classify(seq_id)
    if kind == "pure":
        return label
    if CRF01_AE_RE.match(label):
        return "E"
    if CRF26_A5U_RE.match(label):
        return "A5"
    return None


def has_valid_year(seq_id):
    parts = seq_id.split(".")
    return len(parts) >= 4 and bool(YEAR_RE.match(parts[2]))


def accession(seq_id):
    return seq_id.split(".")[-1]


def output_id(seq_id, label=None):
    """SUBTYPE.CC.YEAR.NAME.ACC -> Ref.SUBTYPE.CC.YY.NAME.ACC, optionally relabelled."""
    parts = seq_id.split(".")
    if label:
        parts[0] = label
    if len(parts) > 2 and len(parts[2]) == 4 and parts[2].isdigit():
        parts[2] = parts[2][2:]
    return ".".join(["Ref"] + parts)


def is_data(a):
    """True where the residue is A/C/G/T. Gaps, N and IUPAC codes never count as data."""
    return np.isin(a, BASES)


# I/O

def load_alignment(path):
    ids, seqs = [], []
    for rec in SeqIO.parse(path, "fasta"):
        ids.append(normalise_id(rec.id))
        seqs.append(str(rec.seq).upper())          # acgt -> ACGT
    if len({len(s) for s in seqs}) != 1:
        sys.exit("Sequences have different lengths: input is not an alignment")
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    return ids, arr


def write_fasta(path, records):
    """Atomic write of (id, uint8 row) pairs, one line per sequence."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".fasta")
    try:
        with os.fdopen(fd, "w") as f:
            for rid, row in records:
                f.write(f">{rid}\n{row.tobytes().decode()}\n")
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_report(path, rows):
    with open(path, "w") as f:
        f.write("removed\tgroup\tclosest_kept\tidentity\treason\n")
        for rid, grp, closest, idt, reason in rows:
            f.write(f"{rid}\t{grp}\t{closest}\t{idt:.4f}\t{reason}\n")


# Sequence filters

def label_filter_reason(seq_id):
    kind, label = classify(seq_id)
    if kind == "urf":
        return "urf"
    if kind == "other" or label == "CPX":          # never written to any output
        return "unused_label"
    if HXB2_ACC not in seq_id and pure_output_label(seq_id) and not has_valid_year(seq_id):
        return "no_year"                           # pure-alignment sequences only
    return None


def drop_by_label(ids, arr):
    reasons = [label_filter_reason(s) for s in ids]
    for reason in ("urf", "unused_label", "no_year"):
        c = Counter(classify(s)[1] for s, r in zip(ids, reasons) if r == reason)
        print(f"Removed {sum(c.values())} sequences ({reason}); labels: {dict(c.most_common(15))}")
    report = [(s, classify(s)[1], "", 0.0, r) for s, r in zip(ids, reasons) if r]
    keep = np.array([r is None for r in reasons])
    return [s for s, k in zip(ids, keep) if k], arr[keep], report


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


def find_reference(ids, ref_id):
    hits = [i for i, sid in enumerate(ids) if ref_id in sid]
    if len(hits) != 1:
        sys.exit(f"Expected exactly 1 sequence matching '{ref_id}', found {len(hits)}: "
                 f"{[ids[i] for i in hits]}")
    print(f"Reference: {ids[hits[0]]}")
    return hits[0]


def crop_to_hxb2_range(ids, arr, ref_idx, start, end):
    # HXB2 coordinates count every non-gap residue of the reference, so this uses != GAP, not is_data
    ref_cols = np.where(arr[ref_idx] != GAP)[0]
    if len(ref_cols) != HXB2_LEN:
        sys.exit(f"HXB2 row has {len(ref_cols)} residues, expected {HXB2_LEN}: "
                 f"coordinates would be wrong")
    if not 1 <= start <= end <= HXB2_LEN:
        sys.exit(f"Invalid HXB2 range {start}-{end}")
    if end - start + 1 != HXB2_USED_LEN:
        sys.exit(f"HXB2 range {start}-{end} has {end - start + 1} positions, "
                 f"config.HXB2_USED_LEN is {HXB2_USED_LEN}")
    lo, hi = ref_cols[start - 1], ref_cols[end - 1]
    sub = arr[:, lo:hi + 1]
    keep = is_data(sub).any(axis=1)
    keep[ref_idx] = True
    print(f"Cropped to HXB2 {start}-{end}: {sub.shape[1]} of {arr.shape[1]} columns; "
          f"dropped {int((~keep).sum())} sequences with no A/C/G/T in the window")
    return ([s for s, k in zip(ids, keep) if k], sub[keep], int(keep[:ref_idx].sum()))


def filter_by_min_positions(ids, arr, ref_idx, min_positions, min_keep_per_class=0,
                            rescue_min_positions=0, stage=""):
    """
    Remove sequences with <= min_positions A/C/G/T residues over the columns of `arr`.

    The reference row is always kept. If min_keep_per_class > 0, each class keeps at
    least that many sequences (classes smaller than that keep all they can): missing
    slots are filled with the best-covered removed sequences that still have
    > rescue_min_positions bases.

    Returns filtered ids, filtered arr, updated ref_idx and report rows.
    """
    counts = is_data(arr).sum(axis=1)
    mask = counts > min_positions
    mask[ref_idx] = True

    if min_keep_per_class > 0:
        classes = np.array([classify(sid)[1] for sid in ids])
        rescued = {}
        for cls in np.unique(classes):
            idx = np.flatnonzero(classes == cls)
            missing = min(min_keep_per_class, len(idx)) - int(mask[idx].sum())
            if missing <= 0:
                continue
            cand = idx[~mask[idx] & (counts[idx] > rescue_min_positions)]
            cand = cand[np.argsort(-counts[cand], kind="stable")[:missing]]
            mask[cand] = True
            if len(cand):
                rescued[cls] = len(cand)
        if rescued:
            print(f"[{stage}] Rescued per class (<= {min_positions} bases but kept): {rescued}")

    n_cols = arr.shape[1]
    report = [(sid, classify(sid)[1], "", counts[i] / n_cols, f"too_short_{stage}")
              for i, sid in enumerate(ids) if not mask[i]]
    per_class = Counter(r[1] for r in report)
    print(f"[{stage}] Length filter over {n_cols} columns: kept {int(mask.sum())}, "
          f"removed {len(report)} (<= {min_positions} A/C/G/T); "
          f"per subtype: {dict(per_class.most_common())}")

    return ([s for s, k in zip(ids, mask) if k], arr[mask],
            int(mask[:ref_idx].sum()), report)


# Deduplication

def group_identity(sub):
    """Pairwise identity / overlap fraction over columns where both sequences have A/C/G/T."""
    present = is_data(sub)
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


def deduplicate(ids, arr, ref_idx, identity, min_overlap, min_keep):
    groups = defaultdict(list)
    for i, sid in enumerate(ids):
        groups[classify(sid)[1]].append(i)

    keep_rows, report, restored = [], [], {}
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
        if n_back:
            restored[label] = n_back

    keep_rows = np.sort(np.array(keep_rows))
    print(f"Deduplication: {len(ids)} -> {len(keep_rows)} sequences")
    if restored:
        print(f"  restored to reach MIN_PER_GROUP: {restored}")
    return ([ids[i] for i in keep_rows], arr[keep_rows],
            int(np.searchsorted(keep_rows, ref_idx)), report)


# Column selection

def compute_coverage(arr, ids, pooled):
    present = is_data(arr)
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


# Outputs

def write_hxb2_mapping(hxb2_gapped):
    ata_to_hxb2, _ = config.build_hxb2_ata_maps(hxb2_gapped)
    with open(config.MAPPING_PATH, "w") as f:
        f.write("ata_pos,hxb2_pos\n")
        for ata_pos in range(len(hxb2_gapped)):
            f.write(f"{ata_pos},{ata_to_hxb2[ata_pos]}\n")
    print(f"HXB2 <-> ATA mapping written to {config.MAPPING_PATH}")


def split_and_write(ids, arr, ref_idx, pure_path, crf_path, combined_path):
    hxb2 = (output_id(ids[ref_idx]), arr[ref_idx])
    pure, crf = [], []
    for i, sid in enumerate(ids):
        if i == ref_idx:
            continue
        kind, _ = classify(sid)
        plabel = pure_output_label(sid)
        if plabel:
            pure.append((output_id(sid, plabel), arr[i]))
        if kind == "crf" and (plabel is None or PURE_LIKE_CRFS_ALSO_IN_CRF):
            crf.append((output_id(sid), arr[i]))

    pure.sort(key=lambda r: r[0].split(".")[1])
    crf.sort(key=lambda r: int(r[0].split(".")[1].split("_")[0]))

    write_fasta(pure_path, [hxb2] + pure)
    write_fasta(crf_path, [hxb2] + crf)
    write_fasta(combined_path, [hxb2] + pure + crf)

    counts = Counter(r[0].split(".")[1] for r in pure)
    print(f"Pure alignment written to     : {pure_path} ({len(pure)} sequences + HXB2)")
    print(f"CRF  alignment written to     : {crf_path} ({len(crf)} sequences + HXB2)")
    print(f"Combined alignment written to : {combined_path} ({len(pure) + len(crf)} sequences + HXB2)")
    print(f"Pure-subtype counts: { {st: counts.get(st, 0) for st in sorted(ST_TO_ID_DICT)} }")
    return hxb2[0]


def parse_breakpoints():
    bp_path = f"{OUT_DIR}/LANL_scrapped_bp.breakpoints"
    bp.scrape_and_parse_lanl_breakpoints(output_path=bp_path, html_path=BP_HTML)
    df_segments, df_breakpoints = bp.parse_breakpoints_file(bp_path)
    print(f"CRFs parsed       : {df_segments['crf'].nunique()}")
    print(f"Total segments    : {len(df_segments)}")
    print(f"Total breakpoints : {len(df_breakpoints)}")
    df_segments.to_csv(f"{OUT_DIR}/lanl_crf_segments_hxb2.csv", index=False)
    df_breakpoints.to_csv(f"{OUT_DIR}/lanl_crf_breakpoints_hxb2.csv", index=False)
    return df_segments, df_breakpoints


def build_label_tracks(df_segments, df_breakpoints, hxb2_gapped):
    print("Projecting HXB2-coordinate segments/breakpoints onto ATA alignment coordinates")
    hxb2_pos_to_aln = bp.build_hxb2_pos_to_aln_col(hxb2_gapped)
    first, last = config.HXB2_START, config.HXB2_START + HXB2_USED_LEN - 1
    seg_win = df_segments[(df_segments["end"] >= first) & (df_segments["start"] <= last)]
    bp_win = df_breakpoints[df_breakpoints["position"].between(first, last)]
    bp.convert_hxb2_df_to_aln(seg_win, hxb2_pos_to_aln, pos_cols=("start", "end")).to_csv(
        f"{OUT_DIR}/lanl_crf_segments_aln.csv", index=False)
    bp.convert_hxb2_df_to_aln(bp_win, hxb2_pos_to_aln, pos_cols=("position",)).to_csv(
        f"{OUT_DIR}/lanl_crf_breakpoints_aln.csv", index=False)

    print("Building CRF label sequences")
    label_seqs = bp.build_all_crf_label_sequences(df_segments, hxb2_gapped)
    print(f"  {len(label_seqs)} CRFs + pure subtypes | alignment length = {len(hxb2_gapped)}")
    seqs_path = f"{OUT_DIR}/lanl_crf_label_seqs.npz"
    np.savez_compressed(seqs_path, **label_seqs)
    print(f"  Label-sequence archive : {seqs_path}")

    df_aln = bp.label_sequences_to_segments_csv(label_seqs, config.CRF_SEGMENTS_BD_ATA)
    print(f"Alignment-coord segments : {config.CRF_SEGMENTS_BD_ATA} ({len(df_aln)} rows)")
    print(df_aln.head(20).to_string(index=False))


if __name__ == "__main__":
    args = parse_args()

    # Cheap and independent of the alignment: fail fast if the HTML snapshot is wrong
    df_segments, df_breakpoints = parse_breakpoints()

    ids, arr = load_alignment(args.input)
    ids, arr, report = drop_by_label(ids, arr)          # URF, unused labels, pure without year
    ids, arr, rep = drop_duplicate_accessions(ids, arr)
    report += rep

    ref_idx = find_reference(ids, HXB2_ACC)
    ids, arr, ref_idx = crop_to_hxb2_range(ids, arr, ref_idx, config.HXB2_START, config.HXB2_END)
    assert HXB2_ACC in ids[ref_idx]

    # Pass 1: whole cropped window. Necessary condition, since column selection can only lower counts.
    ids, arr, ref_idx, rep = filter_by_min_positions(
        ids, arr, ref_idx, MIN_POSITIONS, MIN_PER_GROUP, RESCUE_MIN_POSITIONS, stage="pre")
    report += rep

    if not args.no_dedup:
        ids, arr, ref_idx, rep = deduplicate(ids, arr, ref_idx, CLUSTERING_IDENTITY,
                                             MIN_OVERLAP, MIN_PER_GROUP)
        report += rep

    n_total = arr.shape[1]
    if not 0 < TARGET_LEN <= n_total:
        sys.exit(f"Target length must be in [1, {n_total}]")

    pure_cov, crf_cov, overall = compute_coverage(arr, ids, args.pooled)
    protected = arr[ref_idx] != GAP          # reference columns (coordinate system, not data)
    ins = ~protected
    print(f"{int(ins.sum())} insertion columns relative to HXB2; "
          f"{int((pure_cov[ins] >= 0.05).sum())} with pure coverage >= 5%")
    n_prot = int(protected.sum())
    if n_prot > TARGET_LEN:
        sys.exit(f"{n_prot} reference columns to protect exceed target length={TARGET_LEN}")
    print(f"Protected {n_prot} reference columns; {TARGET_LEN - n_prot} free slots")

    keep = select_columns(pure_cov, crf_cov, overall, TARGET_LEN, WEIGHT, protected)
    assert protected[keep].sum() == n_prot  # every reference column survived

    dropped = np.setdiff1d(np.arange(n_total), keep)
    print(f"Kept {len(keep)}/{n_total} columns; "
          f"dropped {int((pure_cov[dropped] > 0).sum())} columns with some pure coverage "
          f"(max pure coverage among dropped: "
          f"{pure_cov[dropped].max() if len(dropped) else 0:.4f})")

    arr = arr[:, keep]

    # Pass 2: exact check on the columns actually written
    ids, arr, ref_idx, rep = filter_by_min_positions(
        ids, arr, ref_idx, MIN_POSITIONS, MIN_PER_GROUP, RESCUE_MIN_POSITIONS, stage="post")
    report += rep

    if args.removed_out:
        write_report(args.removed_out, report)

    hxb2_gapped = arr[ref_idx].tobytes().decode()
    assert len(hxb2_gapped) == TARGET_LEN
    assert int((arr[ref_idx] != GAP).sum()) == HXB2_USED_LEN
    write_hxb2_mapping(hxb2_gapped)

    split_and_write(ids, arr, ref_idx, args.pure_out, args.crf_out, args.combined_out)
    build_label_tracks(df_segments, df_breakpoints, hxb2_gapped)