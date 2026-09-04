import re
import pandas as pd
import requests
import urllib3
import os
from bs4 import BeautifulSoup
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np
from Bio import SeqIO
import tempfile
from argparse import ArgumentParser

from src import config

workspace_path = config.WORKSPACE_PATH

parser = ArgumentParser()
parser.add_argument("--input",    type=str,
                    default=f"{workspace_path}/data/output/LANL_COMBINED.fasta",
                    help="Input combined LANL alignment (FASTA)")
parser.add_argument("--pure_out", type=str,
                    default=f"{workspace_path}/data/output/HIV1_PURE_REF.fasta",
                    help="Output pure-subtype reference alignment (FASTA)")
parser.add_argument("--crf_out",  type=str,
                    default=f"{workspace_path}/data/output/HIV1_CRF_REF.fasta",
                    help="Output CRF reference alignment (FASTA)")
parser.add_argument("--combined_out", type=str,
                    default=f"{workspace_path}/data/output/HIV1_COMBINED_REF.fasta",
                    help="Output combined reference alignment (FASTA)")

args = parser.parse_args()

lanl_subtype_ref_path = args.input
pure_ref_path    = args.pure_out
crf_ref_path     = args.crf_out
combined_ref_path = args.combined_out

urllib3.disable_warnings()

ST_TO_ID_DICT = config.ST_TO_ID_DICT
# ── Harmonising dict for breakpoints taken from LANL ──────────────────────────
CRF_CANONICAL = {
    '01':        'CRF01_AE',
    '01_AE':     'CRF01_AE',
    'CRF1' :     'CRF01_AE',
    'CRF01':     'CRF01_AE',
    'CRF_01':    'CRF01_AE',
    'CRF01_AE':  'CRF01_AE',
    '02':        'CRF02_AG',
    '02_AG':     'CRF02_AG',
    'CRF02':     'CRF02_AG',
    'CRF_02':    'CRF02_AG',
    'CRF02_AG':  'CRF02_AG',
    '06':        'CRF06_cpx',
    'CRF06':     'CRF06_cpx',
    'CRF_06':    'CRF06_cpx',
    '07':        'CRF07_BC',
    '07_BC':     'CRF07_BC',
    'CRF07':     'CRF07_BC',
    'CRF_07':    'CRF07_BC',
    'CRF07_BC':  'CRF07_BC',
    '08':        'CRF08_BC',
    '08_BC':     'CRF08_BC',
    'CRF08':     'CRF08_BC',
    'CRF_08':    'CRF08_BC',
    'CRF08_BC':  'CRF08_BC',
    'CRRF08_BC': 'CRF08_BC',
    '55':        'CRF55_01B',
    '55_01B':    'CRF55_01B',
    'CRF55':     'CRF55_01B',
    'CRF55_01B': 'CRF55_01B',
}

U_ALIASES = {'U', 'U1', 'Undetermined', 'Unsequenced', 'unknown', 'Undefined'}

# ──────────────────────────────────────────────────────────────────────────────
# CRF01_AE mosaic structure (HXB2 coordinates, 1-based)
# ──────────────────────────────────────────────────────────────────────────────
_CRF01_BREAKPOINTS: List[Tuple[int, str, str]] = [
    (5097, "A1",   "E"   ),
    (5321, "E",    "A1"  ),
    (5451, "A1",   "E"   ),
    (5651, "E",    "A1"  ),
    (6311, "A1",   "E/A1"),
    (6501, "E/A1", "E"   ),
    (8000, "E",    "E/A1"),
    (8379, "E/A1", "A1"  ),
    (9086, "A1",   "E"   ),
]

_CRF26_A5U_SEGMENTS: List[Tuple[int, int, str]] = [
    (   1, 1398, "A5"),
    (1399, 1659, "U"),
    (1660, 2378, "A5"),
    (2379, 2679, "U"),
    (2680, 4235, "A5"),
    (4236, 4514, "U"),
    (4515, 6692, "A5"),
    (6693, 6857, "U"),
    (6858, 7969, "A5"),
    (7970, 8372, "U"),
    (8373, 9719, "A5"),
]

_HXB2_LENGTH: int = 9719
_CRF_RE      = re.compile(r"^\d+_", re.IGNORECASE)
_CRF01_AE_RE = re.compile(r"^(CRF)?01[_-]?AE$",   re.IGNORECASE)
_CRF26_A5U_RE= re.compile(r"^(CRF)?26[_-]?A5U?$", re.IGNORECASE)
_HXB2_ID     = "HXB2_LAI_IIIB_BRU.K03455"
_PURE_ST: FrozenSet[str] = frozenset(ST_TO_ID_DICT.keys()) # | {'A', 'F'}

# ──────────────────────────────────────────────────────────────────────────────
# Subtype normalisation helpers (breakpoint file parsing)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_subtype(raw: str) -> str:
    if '/' in raw:
        parts = raw.split('/')
        return '/'.join(normalize_subtype(p) for p in parts)

    s = raw.strip()

    if s in U_ALIASES:
        return 'U'

    if s in CRF_CANONICAL:
        return CRF_CANONICAL[s]

    m = re.match(r'^(CRF|_?CRF)?[_-]?(\d+)(_\w+)?$', s, re.IGNORECASE)
    if m:
        prefix = m.group(1) or ''
        num    = m.group(2)
        suffix = m.group(3) or ''
        if prefix:
            return f"{prefix}{num}{suffix}"
        else:
            return f"CRF{num}{suffix}"

    return s


def resolve_u_in_compound(subtype, prev_st, next_st):
    parts  = subtype.split('/')
    if 'U' not in parts:
        return subtype

    non_u = [p for p in parts if p != 'U']

    if len(non_u) >= 2:
        return '/'.join(non_u)
    elif len(non_u) == 1:
        internal = non_u[0]
        external = next_st if parts[-1] == 'U' else prev_st
        if external and external != internal:
            return f"{internal}/{external}"
        else:
            return internal
    else:
        return None


def resolve_u_segments(segs):
    resolved = [s.copy() for s in segs]
    n = len(resolved)

    for i, seg in enumerate(resolved):
        if 'U' not in seg['subtype']:
            continue

        prev_st = next((resolved[j]['subtype'].split('/')[0] for j in range(i-1, -1, -1)
                        if 'U' not in resolved[j]['subtype']), None)
        next_st = next((resolved[j]['subtype'].split('/')[-1] for j in range(i+1, n)
                        if 'U' not in resolved[j]['subtype']), None)

        if seg['subtype'] == 'U':
            if prev_st is None and next_st is None:
                pass
            elif prev_st is None:
                resolved[i]['subtype'] = next_st
            elif next_st is None:
                resolved[i]['subtype'] = prev_st
            elif prev_st == next_st:
                resolved[i]['subtype'] = prev_st
            else:
                resolved[i]['subtype'] = f"{prev_st}/{next_st}"
        else:
            resolved[i]['subtype'] = resolve_u_in_compound(
                seg['subtype'], prev_st, next_st
            )
    return resolved


def join_neighbor_segments(resolved):
    if not resolved:
        return []
    resolved.sort(key=lambda x: x['start'])
    merged = [resolved[0].copy()]
    for current in resolved[1:]:
        prev = merged[-1]
        if (current['start'] == prev['end'] + 1 and
                current['subtype'] == prev['subtype']):
            prev['end'] = current['end']
        else:
            merged.append(current.copy())
    return merged


def check_crf_is_pure(resolved):
    if not resolved:
        return True
    first_st = resolved[0]['subtype']
    return all(seg['subtype'] == first_st for seg in resolved[1:])


def parse_breakpoints_file(filepath):
    segments    = []
    breakpoints = []
    current_crf  = None
    current_segs = []

    def flush(crf, segs):
        # segs = resolve_u_segments(segs)
        segs = join_neighbor_segments(segs)
        if not check_crf_is_pure(segs):
            for seg in segs:
                segments.append({
                    'crf':     crf,
                    'start':   seg['start'],
                    'end':     seg['end'],
                    'subtype': seg['subtype'],
                    'length':  seg['end'] - seg['start'],
                })
            for i in range(1, len(segs)):
                prev = segs[i - 1]
                curr = segs[i]
                if prev['subtype'] != curr['subtype']:
                    breakpoints.append({
                        'crf':          crf,
                        'position':     curr['start'],
                        'from_subtype': prev['subtype'],
                        'to_subtype':   curr['subtype'],
                    })
        else:
            print(f'{current_crf} is PURE after U handling and neighbour segment joining')

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or (line.startswith('#') and not line.startswith('>')):
                continue
            if line.startswith('>'):
                if current_crf and current_segs:
                    flush(current_crf, current_segs)
                m = re.match(r'>CRF_(\d+)', line)
                if m:
                    num = int(m.group(1))
                    current_crf  = f"CRF{num:02d}"
                    current_segs = []
                continue
            parts = line.split()
            if len(parts) >= 3 and current_crf:
                subtype = normalize_subtype(parts[2])
                try:
                    current_segs.append({
                        'start':   int(parts[0]),
                        'end':     int(parts[1]),
                        'subtype': subtype,
                    })
                except ValueError:
                    print(f"Couldn't parse {parts}")
                    continue

    if current_crf and current_segs:
        flush(current_crf, current_segs)

    df_segments    = pd.DataFrame(segments)
    df_breakpoints = pd.DataFrame(breakpoints)
    return df_segments, df_breakpoints


def scrape_and_parse_lanl_breakpoints(output_path):
    path = "https://www.hiv.lanl.gov/components/sequence/HIV/crfdb/crfs.comp"
    r = requests.get(path, verify=False)
    html_content = r.content.decode('utf-8')
    soup = BeautifulSoup(html_content, 'html.parser')
    L = soup.find_all("pre")
    r.close()
    segment_re   = re.compile(r'^(\d+)\s+(\d+)\s+(\S+)')
    blocks_written = 0

    with open(output_path, 'w') as out:
        out.write(
            "# Breakpoints from Los Alamos HIV sequence database:\n"
            "# https://www.hiv.lanl.gov/content/sequence/HIV/CRFs/breakpoints.html\n\n\n"
        )

        for pre in L:
            parent_tr = pre.find_parent('tr')
            if parent_tr is None:
                continue
            tgl_id = parent_tr.get('id', '')
            if not tgl_id.startswith('tgl_'):
                continue
            crf_name = tgl_id[4:]

            m_num = re.match(r'CRF(\d+)', crf_name, re.IGNORECASE)
            if m_num is None:
                continue
            num_str = m_num.group(1).zfill(2)

            parent_td   = pre.parent
            comment_parts = []
            for child in parent_td.children:
                if child is pre:
                    break
                text = child.get_text(separator=' ') if hasattr(child, 'get_text') else str(child)
                comment_parts.append(text)
            raw_comment  = re.sub(r'\s+', ' ', ' '.join(comment_parts)).strip()
            m_bp         = re.search(r'(Breakpoints?\s+.*)', raw_comment, re.IGNORECASE)
            comment_line = m_bp.group(1).strip() if m_bp else ''

            header_line = f">CRF_{num_str}"
            if comment_line:
                header_line += f" # {comment_line}"

            raw_lines = [str(br.previous_sibling or '').strip()
                         for br in pre.find_all('br')]
            if list(pre.children):
                last = list(pre.children)[-1]
                raw_lines.append(str(last).strip() if last else '')

                seg_lines = []
                for raw in raw_lines:
                    sm = segment_re.match(raw)
                    if sm:
                        label = sm.group(3)
                        if num_str == "26" and label == "A":
                            label = "A5"
                        seg_lines.append(f"{sm.group(1)}\t{sm.group(2)}\t{label}")

                if not seg_lines:
                    continue

                out.write(header_line + "\n")
                for sl in seg_lines:
                    out.write(sl + "\n")
                out.write("\n")
                blocks_written += 1
            else:
                print(f"No segment lines found for {crf_name}")

    print(f"Written {blocks_written} CRF blocks to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# HXB2 position masks
# ──────────────────────────────────────────────────────────────────────────────

def _build_pure_e_hxb2_mask() -> np.ndarray:
    boundaries = (
        [1]
        + [bp[0] for bp in _CRF01_BREAKPOINTS]
        + [_HXB2_LENGTH + 1]
    )
    labels = (
        [_CRF01_BREAKPOINTS[0][1]]
        + [bp[2] for bp in _CRF01_BREAKPOINTS]
    )
    mask = np.zeros(_HXB2_LENGTH + 2, dtype=bool)
    for i, label in enumerate(labels):
        if label == "E":
            mask[boundaries[i] : boundaries[i + 1]] = True
    return mask


def _build_crf26_a5u_a_mask() -> np.ndarray:
    mask = np.zeros(_HXB2_LENGTH + 2, dtype=bool)
    for start, end, label in _CRF26_A5U_SEGMENTS:
        if label == "A5":
            mask[start : end + 1] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Record classification helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_subtype(record_id: str) -> Optional[str]:
    parts = record_id.split(".")
    if len(parts) >= 2 and parts[0].lower() == "ref":
        st = parts[1]
    else:
        st = parts[0]
    # A should be A1
    if st.upper() == "A":
        return "A1"
    else:
        return st


def _is_pure_subtype(subtype: Optional[str]) -> bool:
    if not subtype:
        return False
    return subtype in _PURE_ST


def _is_crf(subtype: Optional[str]) -> bool:
    if not subtype:
        return False
    return bool(_CRF_RE.match(subtype))


def _is_crf01_ae(subtype: Optional[str]) -> bool:
    return bool(subtype and _CRF01_AE_RE.match(subtype))


def _is_crf26_a5u(subtype: Optional[str]) -> bool:
    return bool(subtype and _CRF26_A5U_RE.match(subtype))


# ──────────────────────────────────────────────────────────────────────────────
# CRF label-sequence construction
# ──────────────────────────────────────────────────────────────────────────────


def _crf_df_key(label: str, df_by_crf: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Return the df_by_crf dict key for a CRF subtype label, or None.

    'CRF01_AE' → 'CRF01'   'CRF07_BC' → 'CRF07'
    'CRF109_0107' → 'CRF109'
    """
    m = re.match(r'^CRF(\d+)', label, re.IGNORECASE)
    if m:
        key = f"CRF{int(m.group(1)):02d}"
        return key if key in df_by_crf else None
    return label if label in df_by_crf else None


def _pure_set_of(
    label: str,
    df_by_crf: Dict[str, pd.DataFrame],
    seen: FrozenSet[str],
    depth: int,
) -> FrozenSet[str]:
    """Flatten *label* to its constituent pure-subtype strings.

    Used to resolve compound labels that contain a CRF reference,
    e.g. 'B/CRF01_AE' → {'A', 'B', 'E'}.
    """
    if depth > 6:
        return frozenset({label}) if label in _PURE_ST else frozenset()

    if label in _PURE_ST:
        return frozenset({label})

    if '/' in label:
        out: Set[str] = set()
        for part in label.split('/'):
            out |= _pure_set_of(part.strip(), df_by_crf, seen, depth)
        return frozenset(out)

    key = _crf_df_key(label, df_by_crf)
    if key and key not in seen:
        out = set()
        for sub in df_by_crf[key]['subtype']:
            out |= _pure_set_of(str(sub), df_by_crf, seen | {key}, depth + 1)
        return frozenset(out)

    return frozenset()   # unresolvable (e.g. 'mix')


def _resolve_single_label(
    subtype: str,
    df_by_crf: Dict[str, pd.DataFrame],
    seen: FrozenSet[str],
    depth: int,
) -> str:
    """Resolve a terminal or compound label to a string of pure subtypes.

    Examples
    --------
    'A1'          → 'A'
    'E/A1'        → 'A/E'       (sorted)
    'B/CRF01_AE'  → 'A/B/E'
    'CRF02_AG/B'  → 'A/B/G'
    """
    if subtype in _PURE_ST:
        return subtype

    if '/' in subtype:
        pure: Set[str] = set()
        for part in subtype.split('/'):
            pure |= _pure_set_of(part.strip(), df_by_crf, seen, depth)
        return '/'.join(sorted(pure)) if pure else subtype

    return 'U'


def _build_hxb2_label_array(
    crf: str,
    df_by_crf: Dict[str, pd.DataFrame],
    hxb2_len: int,
    seen: FrozenSet[str],
    depth: int,
    cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build a 1-indexed (hxb2_len+1,) object array of subtype labels for *crf*.

    Index 0 is unused; valid positions are 1 … hxb2_len.
    Uncovered positions are ``'-'``.

    Expansion rules
    ---------------
    * Pure-subtype terminal (A1, B, E, …)  → unified string, e.g. A1 → 'A'
    * Compound label (B/C, E/A1, …)        → sorted pure-subtype string per
                                              position, e.g. 'E/A1' → 'A/E'
    * CRF reference (CRF01_AE, CRF07_BC)  → per-position labels copied from
                                              the referenced CRF's own array
                                              (intersected with [start, end])
    * 'mix' or unknown label               → replace with 'U'
    """
    if crf in cache:
        return cache[crf]

    labels = np.full(hxb2_len + 1, '-', dtype=object)

    if crf not in df_by_crf or depth > 6:
        cache[crf] = labels
        return labels

    new_seen = seen | {crf}

    for _, row in df_by_crf[crf].iterrows():
        s       = max(1, min(int(row['start']), hxb2_len))
        e       = max(1, min(int(row['end']),   hxb2_len))
        subtype = str(row['subtype'])

        if s > e:
            continue

        # ── terminal or compound (may contain CRF refs inside compound) ──
        if subtype in _PURE_ST or '/' in subtype:
            labels[s : e + 1] = _resolve_single_label(
                subtype, df_by_crf, new_seen, depth
            )
            continue

        # ── CRF reference: expand position-by-position ───────────────────
        key = _crf_df_key(subtype, df_by_crf)
        if key and key not in new_seen:
            ref = _build_hxb2_label_array(
                key, df_by_crf, hxb2_len, new_seen, depth + 1, cache
            )
            src  = ref[s : e + 1]
            mask = src != '-'
            if mask.any():
                dst       = labels[s : e + 1].copy()
                dst[mask] = src[mask]
                labels[s : e + 1] = dst
            continue

        # ── unresolvable (absent CRF) — keep as-is ────────────────
        labels[s : e + 1] = subtype

    cache[crf] = labels
    return labels


def build_all_crf_label_sequences(
    df_segments: pd.DataFrame,
    hxb2_gapped: str,
) -> Dict[str, np.ndarray]:
    """Build a per-alignment-column subtype-label array for every CRF.

    For each alignment column that corresponds to an HXB2 base the label is
    the expanded pure-subtype label at that HXB2 position.  Alignment columns
    that are gaps in HXB2 always carry ``'-'``, as do HXB2 positions not
    covered by any CRF segment.

    Parameters
    ----------
    df_segments : output of :func:`parse_breakpoints_file`; HXB2 1-based coords.
    hxb2_gapped : HXB2 row from the MSA (gapped sequence, same alignment that
                  will be used for matching).

    Returns
    -------
    ``{crf_name: np.ndarray(aln_len, dtype=object)}``

    Each value is an array of length ``len(hxb2_gapped)``.  Elements are either
    a pure-subtype string, a ``'/'``-joined compound of pure subtypes (at
    annotated transition regions), or ``'-'`` (gap / uncovered).
    """
    hxb2_arr = np.frombuffer(hxb2_gapped.encode(), dtype=np.uint8)

    # Alignment columns that carry an actual HXB2 base (col i → HXB2 pos i+1)
    hxb2_base_cols = np.where(hxb2_arr != ord('-'))[0]
    hxb2_len       = len(hxb2_base_cols)   # = 9719 for the standard alignment
    aln_len        = len(hxb2_gapped)

    df_by_crf: Dict[str, pd.DataFrame] = {
        str(crf): grp.reset_index(drop=True)
        for crf, grp in df_segments.groupby('crf')
    }


    cache: Dict[str, np.ndarray] = {}   # memorised HXB2-length label arrays

    label_seqs: Dict[str, np.ndarray] = {}
    for crf in df_by_crf:
        hxb2_labels = _build_hxb2_label_array(
            crf, df_by_crf, hxb2_len,
            seen  = frozenset({crf}),
            depth = 0,
            cache = cache,
        )

        # Project: hxb2_base_cols[0] ↔ HXB2 pos 1, [k] ↔ pos k+1
        aln_labels = np.full(aln_len, '-', dtype=object)
        aln_labels[hxb2_base_cols] = hxb2_labels[1 : hxb2_len + 1]
        # Fill gaps that are between segments of the same subtype with that subtype
        # AAA------AAA --> AAAAAAAAAAAA
        if np.any(aln_labels != '-'):
            # Find positions of all non-gap labels
            non_gap_mask = aln_labels != '-'
            non_gap_idx  = np.where(non_gap_mask)[0]

            # Walk consecutive pairs of labelled positions
            for left, right in zip(non_gap_idx[:-1], non_gap_idx[1:]):
                if aln_labels[left] == aln_labels[right]:
                    # All columns strictly between left and right carry the same label
                    aln_labels[left + 1 : right] = aln_labels[left]
        # Replace gaps with U
        aln_labels[aln_labels == '-'] = 'U'
        label_seqs[crf] = aln_labels

    # Sort label_seqs by CRF number (CRF01, CRF02, …) for consistent ordering in output files.
    label_seqs = dict(sorted(label_seqs.items(), key=lambda x: int(re.match(r'CRF(\d+)', x[0]).group(1))))

    # Add Pure subtypes as their own entries in label_seqs
    for pure in _PURE_ST:
        label_seqs[pure] = np.full(aln_len, pure, dtype=object)

    return label_seqs

def _merge_same_subtype_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive same-subtype segments within each CRF.

    Segments that are separated only by alignment-gap (``'-'``) positions and
    carry the same subtype label are merged into a single segment.  Segments
    with a different subtype in between are left as separate boundaries.

    Parameters
    ----------
    df : DataFrame with columns crf, start, end, subtype, length.
         Rows must be sorted by start within each CRF (as produced by
         :func:`label_sequences_to_segments_csv`).

    Returns
    -------
    DataFrame with the same columns; *length* = end − start + 1.
    """
    rows: List[Dict] = []
    for crf, grp in df.groupby('crf', sort=False):
        segs = grp.sort_values('start').to_dict('records')
        if not segs:
            continue
        merged = [segs[0].copy()]
        for seg in segs[1:]:
            prev = merged[-1]
            if seg['subtype'] == prev['subtype']:
                # Same subtype — extend the open segment
                prev['end']    = seg['end']
                prev['length'] = prev['end'] - prev['start'] + 1
            else:
                # Different subtype — start a new segment
                merged.append(seg.copy())
        rows.extend(merged)
    return pd.DataFrame(rows, columns=['crf', 'start', 'end', 'subtype', 'length'])

def label_sequences_to_segments_csv(
    label_seqs: Dict[str, np.ndarray],
    output_path: str,
) -> pd.DataFrame:
    """Run-length encode label sequences into a segment DataFrame and save.

    Consecutive segments that share the same subtype and are separated only by
    alignment-gap positions are merged into a single segment before saving.

    Parameters
    ----------
    label_seqs  : output of :func:`build_all_crf_label_sequences`.
    output_path : path for the output CSV.

    Returns
    -------
    DataFrame with columns: crf, start, end, subtype, length.
    Coordinates are 1-based alignment column indices (both endpoints inclusive).
    Gap positions (``'-'``) are excluded.
    """
    rows: List[Dict] = []
    for crf, labels in label_seqs.items():
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] == '-':
                i += 1
                continue
            lab = labels[i]
            j   = i + 1
            while j < n and labels[j] == lab:
                j += 1
            rows.append({
                'crf':     crf,
                'start':   i + 1,
                'end':     j,
                'subtype': lab,
                'length':  j - i,
            })
            i = j

    df = pd.DataFrame(rows, columns=['crf', 'start', 'end', 'subtype', 'length'])

    # ── Merge same-subtype segments that are only gap-separated ──────────
    # df = _merge_same_subtype_segments(df)

    df.to_csv(output_path, index=False)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Alignment preparation
# ──────────────────────────────────────────────────────────────────────────────

def validate_record_id(record_id: str) -> bool:
    """Check if a LANL record ID is in the expected format.
    Expected format: 'Ref.SUBTYPE.COUNTRY.YY.NAME.ACCESSION',
    e.g. 'Ref.A1.CD.87.P4039.MH705157'.

    Parameters
    ----------
    record_id : str
        The record ID to validate.

    Returns
    -------
    bool
        True if the ID matches the expected format, False otherwise.
    """

    # Record should be like 'Ref.A1.CD.87.P4039.MH705157' for instance.
    parts = record_id.split('.')
    # Should have at least 4 elements
    if len(parts) < 5:
        return False
    # Country code should be 2 letters in caps
    # if not re.match(r'^[A-Z]{2}$', parts[2]):
    #     return False
    # Year should be 2 digits
    if not re.match(r'^\d{2}$', parts[3]):
        return False
    return True
    
def prepare_pure_alignment(lanl_alignment_path: str,
                           pure_alignment_path: str,
                           crf_alignment_path: str,
                           combined_alignment_path: str,
                           ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Read LANL subtype reference alignment, separate pure subtypes from CRFs.
    Applies trimAl column filtering while preserving all HXB2-nucleotide columns.
    """
    # ── 1. Load original alignment (for HXB2 row only) ───────────────────
    original_records = list(SeqIO.parse(lanl_alignment_path, "fasta"))
    if not original_records:
        raise ValueError(f"No sequences found in '{lanl_alignment_path}'.")

    hxb2_original_seq: Optional[str] = None
    for rec in original_records:
        if _HXB2_ID in rec.id:
            hxb2_original_seq = str(rec.seq).upper()
            break
    if hxb2_original_seq is None:
        raise ValueError(
            f"HXB2 not found in the original alignment. "
            f"Expected a record whose ID contains: {_HXB2_ID}."
        )
    original_aln_len = len(hxb2_original_seq)

    # ── 2. Parse trimAl kept-column indices ───────────────────────────────
    colnumbering_path = os.path.join(
        config.WORKSPACE_PATH, "data", "output", "trimal_kept_columns.txt"
    )
    with open(colnumbering_path) as fh:
        line = fh.read().strip()
    # Strip the '#ColumnsMap\t' prefix
    line = re.sub(r'^#ColumnsMap\s*', '', line)
    trimal_kept = np.array(
        [int(x.strip()) for x in line.split(',') if x.strip()],
        dtype=np.int64,
    )

    # ── 3. Build final column mask ────────────────────────────────────────
    # Start with columns kept by trimAl
    final_mask = np.zeros(original_aln_len, dtype=bool)
    final_mask[trimal_kept] = True

    # Reintroduce any column where HXB2 had a nucleotide (non-gap)
    hxb2_arr       = np.frombuffer(hxb2_original_seq.encode(), dtype=np.uint8)
    hxb2_nongap    = hxb2_arr != ord('-')
    final_mask     = final_mask | hxb2_nongap

    # final_cols: sorted array of column indices to keep
    final_cols = np.where(final_mask)[0]

    print(f"  Original alignment columns  : {original_aln_len}")
    print(f"  trimAl kept columns         : {len(trimal_kept)}")
    print(f"  HXB2 nucleotide columns     : {int(hxb2_nongap.sum())}")
    print(f"  Final columns (union)       : {len(final_cols)}")
    print(f"  Columns removed             : {original_aln_len - len(final_cols)}")

    # ── 4. Load trimmed alignment and apply final column mask ─────────────
    trimmed_path = lanl_alignment_path + ".trimaled"
    trimmed_records = list(SeqIO.parse(trimmed_path, "fasta"))
    if not trimmed_records:
        raise ValueError(f"No sequences found in trimmed alignment '{trimmed_path}'.")

    # Build a dict from the original alignment for fast lookup
    # We need to slice from the ORIGINAL sequences using final_cols
    # (trimAl may have reordered nothing, but we re-slice from original
    #  to guarantee correct column correspondence)
    original_seq_dict: Dict[str, str] = {
        rec.id: str(rec.seq).upper() for rec in original_records
    }

    def _slice_seq(seq: str) -> str:
        """Slice a sequence to final_cols."""
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        return arr[final_cols].tobytes().decode()

    # ── 5. Locate HXB2 in the sliced alignment ────────────────────────────
    hxb2_ata: Optional[str] = None
    hxb2_id:  Optional[str] = None
    for rec in original_records:
        if _HXB2_ID in rec.id:
            hxb2_ata = _slice_seq(str(rec.seq).upper())
            hxb2_id  = rec.id
            break

    if hxb2_ata is None:
        raise ValueError("HXB2 not found after column slicing.")

    # ── 6. Build coordinate maps from the final HXB2 row ─────────────────
    ata_to_hxb2, hxb2_to_ata = config.build_hxb2_ata_maps(hxb2_ata)
    #  print the mapping as a csv with columns: ata_pos, hxb2_pos 
    mapping_path = Path(f"{WORKSPACE_PATH}/data/output/hxb2_ata_mapping.csv")
    if mapping_path.is_file():
        mapping_df = pd.read_csv(mapping_path)
        if len(mapping_df) == ata_len:
            return ata_to_hxb2, hxb2_to_ata
    
    with open(mapping_path, "w") as f:
        f.write("ata_pos,hxb2_pos\n")
        for ata_pos in range(ata_len):
            hxb2_pos = ata_to_hxb2[ata_pos]
            f.write(f"{ata_pos},{hxb2_pos}\n")
    print(f"  New ATA length              : {len(hxb2_ata)}")
    print(f"  HXB2 positions covered      : {int(hxb2_nongap.sum())}")

    # ── 7. Project per-HXB2-position masks onto new alignment columns ─────
    _PURE_E_HXB2:  np.ndarray = _build_pure_e_hxb2_mask()
    pure_e_cols:   np.ndarray = _PURE_E_HXB2[ata_to_hxb2]

    _CRF26_A_HXB2: np.ndarray = _build_crf26_a5u_a_mask()
    crf26_a_cols:  np.ndarray = _CRF26_A_HXB2[ata_to_hxb2]

    # ── 8. Classify and collect ───────────────────────────────────────────
    pure_result: Dict[str, str] = {}
    crf_result:  Dict[str, str] = {}
    accession_names_seen: Set[str] = set()
    pure_subtype_count: Dict[str, int] = {st: 0 for st in _PURE_ST}

    for rec in original_records:
        subtype  = _extract_subtype(rec.id)
        old_id   = rec.id
        new_id   = '.'.join(['Ref'] + [subtype] + old_id.split('.')[1:])
        rec.id   = new_id
        seq_str  = _slice_seq(original_seq_dict[old_id])

        accession_name = rec.id.split('.')[-1]
        if accession_name in accession_names_seen:
            print(f"Warning: Duplicate accession '{accession_name}' in '{rec.id}'. Skipping.")
            continue

        if old_id == hxb2_id or rec.id == 'Ref.' + hxb2_id:
            hxb2_id = rec.id        # update to rewritten ID
            pure_result[rec.id] = seq_str
            accession_names_seen.add(accession_name)
            continue

        if _is_pure_subtype(subtype):
            if not validate_record_id(rec.id):
                print(f"Warning: Record ID '{rec.id}' does not match expected format. Skipping.")
                continue
            pure_result[rec.id] = seq_str
            pure_subtype_count[subtype] += 1
            accession_names_seen.add(accession_name)
            continue

        if _is_crf01_ae(subtype):
            if not validate_record_id(rec.id):
                print(f"Warning: Record ID '{rec.id}' does not match expected format. Skipping.")
                continue
            arr = np.frombuffer(seq_str.encode(), dtype=np.uint8).copy()
            # arr[~pure_e_cols] = ord("N") # In the end we decided to keep the full sequence for CRF01_AE and consider it a pure subtype, as the A parts of AE also diverged enough to be considered a subsubtype of A.
            masked_seq = arr.tobytes().decode()
            new_id_e = str(rec.id).replace("01_AE", "AE")
            pure_result[new_id_e] = masked_seq
            pure_subtype_count['AE'] += 1
            accession_names_seen.add(accession_name)

        if _is_crf26_a5u(subtype):
            if not validate_record_id(rec.id):
                print(f"Warning: Record ID '{rec.id}' does not match expected format. Skipping.")
                continue
            arr = np.frombuffer(seq_str.encode(), dtype=np.uint8).copy()
            arr[~crf26_a_cols] = ord("N")
            masked_seq = arr.tobytes().decode()
            new_id_a = str(rec.id).replace("26_A5U", "A5").replace("26_A5", "A5")
            pure_result[new_id_a] = masked_seq
            pure_subtype_count['A5'] += 1
            accession_names_seen.add(accession_name)

        if _is_crf(subtype):
            crf_result[rec.id] = seq_str
            accession_names_seen.add(accession_name)

    # ── 9. Sort and write ─────────────────────────────────────────────────
    def _write_fasta(path: str, primary_id: str, primary_seq: str,
                     sequences: Dict[str, str]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(path)),
            suffix=".fasta"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                fh.write(f">{primary_id}\n{primary_seq}\n")
                for rec_id, seq in sequences.items():
                    if rec_id != primary_id:
                        fh.write(f">{rec_id}\n{seq}\n")
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    pure_result     = dict(sorted(pure_result.items(),
                                  key=lambda x: x[0].split('.')[1]))
    crf_result      = dict(sorted(crf_result.items(),
                                  key=lambda x: int(x[0].split('.')[1].split('_')[0])))
    combined_result = {**pure_result, **crf_result}

    _write_fasta(pure_alignment_path,     hxb2_id, pure_result[hxb2_id],     pure_result)
    _write_fasta(crf_alignment_path,      hxb2_id, pure_result[hxb2_id],     crf_result)
    _write_fasta(combined_alignment_path, hxb2_id, pure_result[hxb2_id],     combined_result)

    # Alphabetically sort the pure_subtype_count dictionary for consistent output
    pure_subtype_count = dict(sorted(pure_subtype_count.items()))

    print(f"Pure alignment written to     : {pure_alignment_path} ({len(pure_result)-1} sequences)")
    print(f"CRF  alignment written to     : {crf_alignment_path} ({len(crf_result)} sequences)")
    print(f"Combined alignment written to : {combined_alignment_path} ({len(combined_result)-1} sequences)")
    print(f"Pure-subtype counts: {pure_subtype_count}")

    return hxb2_id, pure_result, crf_result, combined_result

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Scrape and parse LANL breakpoints ─────────────────────────────
    bp_path = f"{workspace_path}/data/output/LANL_scrapped_bp.breakpoints"
    scrape_and_parse_lanl_breakpoints(bp_path)

    df_segments, df_breakpoints = parse_breakpoints_file(bp_path)

    print(f"CRFs parsed       : {df_segments['crf'].nunique()}")
    print(f"Total segments    : {len(df_segments)}")
    print(f"Total breakpoints : {len(df_breakpoints)}")

    df_segments.to_csv(
        f"{workspace_path}/data/output/lanl_crf_segments.csv",    index=False)
    df_breakpoints.to_csv(
        f"{workspace_path}/data/output/lanl_crf_breakpoints.csv", index=False)
    print("Saved lanl_crf_segments.csv and lanl_crf_breakpoints.csv\n")

    # ── 2. Prepare pure / CRF reference alignments ───────────────────────

    hxb2_id, pure_result, crf_result, combined_result = prepare_pure_alignment(
        lanl_subtype_ref_path, pure_ref_path, crf_ref_path, combined_ref_path,
    )

    # ── 3. Build per-position label sequences (HXB2 → alignment) ─────────
    hxb2_gapped = pure_result[hxb2_id]

    print("Building CRF label sequences …")
    label_seqs = build_all_crf_label_sequences(df_segments, hxb2_gapped)
    aln_len    = len(next(iter(label_seqs.values())))
    print(f"  {len(label_seqs)} CRFs + PURE subtypes  |  alignment length = {aln_len}")

    # Save the label arrays (intermediary, used by the matching script)
    seqs_path = f"{workspace_path}/data/output/lanl_crf_label_seqs.npz"
    np.savez_compressed(seqs_path, **label_seqs)
    print(f"  Label-sequence archive : {seqs_path}")

    # ── 4. Derive alignment-coord segment CSV from the label sequences ────
    csv_path = f"{workspace_path}/data/output/lanl_crf_segments_aln.csv"
    df_aln   = label_sequences_to_segments_csv(label_seqs, csv_path)
    print(f"\nAlignment-coord segments : {csv_path}  ({len(df_aln)} rows)")
    print(df_aln.head(20).to_string(index=False))