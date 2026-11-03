"""
Auxiliary functions for preprocessing sequences.
"""

import sys
from Bio import SeqIO
from Bio.Seq import Seq
from python_scripts.src import config
from pathlib import Path
import subprocess
import re
import pandas as pd
from bs4 import BeautifulSoup, Tag
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np

from python_scripts.src import config


ST_TO_ID_DICT = config.ST_TO_ID_DICT
# Harmonising dict for breakpoints taken from LANL
CRF_CANONICAL = {
    '01':        '01_AE',
    'CRF1' :     '01_AE',
    'CRF01':     '01_AE',
    'CRF_01':    '01_AE',
    'CRF01_AE':  '01_AE',
    '02':        '02_AG',
    'CRF02':     '02_AG',
    'CRF_02':    '02_AG',
    'CRF02_AG':  '02_AG',
    '06':        '06_cpx',
    'CRF06':     '06_cpx',
    'CRF_06':    '06_cpx',
    '07':        '07_BC',
    'CRF07':     '07_BC',
    'CRF_07':    '07_BC',
    'CRF07_BC':  '07_BC',
    '08':        '08_BC',
    'CRF08':     '08_BC',
    'CRF_08':    '08_BC',
    'CRF08_BC':  '08_BC',
    'CRRF08_BC': '08_BC',
    '55':        '55_01B',
    'CRF55':     '55_01B',
    'CRF55_01B': '55_01B',
}

U_ALIASES = {'U', 'U1', 'Undetermined', 'Unsequenced', 'unknown', 'Undefined'}
_HXB2_LENGTH = config.HXB2_LEN
_HXB2_USED_LEN = config.HXB2_USED_LEN
HXB2_FIRST_POS = config.HXB2_START
_CRF_RE      = re.compile(r"^\d+_", re.IGNORECASE)
_CRF01_AE_RE = re.compile(r"^(CRF)?01[_-]?AE$",   re.IGNORECASE)
_CRF26_A5U_RE= re.compile(r"^(CRF)?26[_-]?A5U?$", re.IGNORECASE)
_HXB2_ID     = "HXB2_LAI_IIIB_BRU.K03455"
_PURE_ST: FrozenSet[str] = frozenset(ST_TO_ID_DICT.keys()) # | {'A', 'F'}


# Subtype normalisation helpers (breakpoint file parsing)
def normalize_subtype(raw: str) -> str:
    if '/' in raw:
        parts = raw.split('/')
        return '/'.join(normalize_subtype(p) for p in parts)
    s = raw.strip()
    if s in U_ALIASES:
        return 'U'
    if s in CRF_CANONICAL:
        return CRF_CANONICAL[s]
    if s in ST_TO_ID_DICT:
        return s
    if s == 'A':
        return 'A1'
    if s == 'E':
        return 'E'
    if s == 'A5':
        return 'A5'
    if s == 'F':
        return '/'.join(normalize_subtype(p) for p in ['F1', 'F2'])
    if s == 'mix':
        return '07_BC/08_BC'
    if s == '0708_mix':
        return '07_BC/08_BC'

    # Match optional CRF prefix, digits, and optional suffix
    m = re.match(r'^(?:CRF)?[_-]?(\d+)(?:[_-]?(\w+))?$', s, re.IGNORECASE)
    if m:
        num = m.group(1).zfill(2)  # Zero-pad single digits (e.g., '3' -> '03')
        suffix = m.group(2)
        
        # Omit 'CRF' prefix and enforce XX_SUFFIX formatting
        if suffix:
            return f"{num}_{suffix}"
        return num
    print(f"Warning: unrecognised subtype label {s!r}, returning as-is", flush=True)
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
                m = re.match(r'>([^\s#]+)', line)
                if m:
                    current_crf  = m.group(1)
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

def _read_html(path):
    """Read a saved HTML file. Browsers may save it as UTF-8 or as the
    original ISO-8859-1, so try UTF-8 first and fall back to cp1252."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def scrape_and_parse_lanl_breakpoints(
    output_path,
    html_path="HIV_Circulating_Recombinant_Forms__CRFs_.html",
    verbose=True,
):
    soup = BeautifulSoup(_read_html(html_path), "html.parser")

    # Page version, so you know which snapshot the output came from
    m_date = re.search(r"last modified:\s*(.+)", soup.get_text())
    page_date = m_date.group(1).strip() if m_date else "unknown"

    rows = soup.find_all("tr", id=re.compile(r"^tgl_"))
    if not rows:
        raise RuntimeError(
            f"No CRF rows (tr id='tgl_*') found in {html_path}. "
            "Is this the right file / did the page layout change?"
        )

    segment_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\S+)")
    crf_re = re.compile(r"CRF_?(\d+)(?:[_-]?(\w+))?", re.IGNORECASE)

    blocks_written = 0
    skipped = {"hiv2": [], "bad_name": [], "no_segments": []}

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(
            "# Breakpoints from Los Alamos HIV sequence database:\n"
            "# https://www.hiv.lanl.gov/components/sequence/HIV/crfdb/crfs.comp\n"
            f"# Parsed from local copy: {Path(html_path).name}\n"
            f"# Page last modified: {page_date}\n\n\n"
        )

        for row in rows:
            crf_name = row["id"][4:]

            # HIV-2 CRFs use Mac239 coordinates, not HXB2, so skip them
            if crf_name.upper().startswith("HIV2"):
                skipped["hiv2"].append(crf_name)
                continue

            m_num = crf_re.match(crf_name)
            if m_num is None:
                skipped["bad_name"].append(crf_name)
                continue

            num_str = m_num.group(1).zfill(2)
            suffix = m_num.group(2)
            crf_key = f"{num_str}_{suffix}" if suffix else num_str
            crf_key = CRF_CANONICAL.get(crf_key, crf_key)

            pre = row.find("pre")
            if pre is None:
                skipped["no_segments"].append(crf_name)
                continue

            # Comment = the "Breakpoints from ..." note, i.e. the text
            # between the last <br> and the <pre>
            note_parts = []
            for child in pre.parent.children:
                if child is pre:
                    break
                if isinstance(child, Tag) and child.name == "br":
                    note_parts = []  # keep only what follows the last <br>
                    continue
                note_parts.append(
                    child.get_text(" ") if isinstance(child, Tag) else str(child)
                )
            comment_line = re.sub(r"\s+", " ", "".join(note_parts)).strip()

            # Segment lines (works with <br> or newline separators)
            for br in pre.find_all("br"):
                br.replace_with("\n")

            seg_lines = []
            for line in pre.get_text().splitlines():
                sm = segment_re.match(line)
                if sm:
                    label = sm.group(3)
                    if num_str == "26" and label == "A":
                        label = "A5"
                    seg_lines.append(f"{sm.group(1)}\t{sm.group(2)}\t{label}")

            if not seg_lines:
                skipped["no_segments"].append(crf_name)
                continue

            header_line = f">{crf_key}"
            if comment_line:
                header_line += f" # {comment_line}"

            out.write(header_line + "\n" + "\n".join(seg_lines) + "\n\n")
            blocks_written += 1

    print(f"Written {blocks_written} CRF blocks to {output_path} "
          f"(page last modified: {page_date})")
    if verbose:
        for reason, names in skipped.items():
            if names:
                print(f"  Skipped ({reason}): {', '.join(names)}")



# HXB2 <-> ATA alignment coordinate projection helpers
def build_hxb2_pos_to_aln_col(hxb2_gapped: str) -> np.ndarray:
    """Build a 1-indexed lookup: HXB2 position (1..hxb2_len) -> ATA alignment
    column (1-based).

    Index 0 is unused padding, so the returned array has length
    ``hxb2_len + 1`` and ``mapping[p]`` gives the alignment column for HXB2
    position ``p``.

    Parameters
    ----------
    hxb2_gapped : the (possibly gapped) HXB2 row from the final ATA alignment
                  (i.e. ``pure_result[hxb2_id]``).
    """
    arr = np.frombuffer(hxb2_gapped.encode(), dtype=np.uint8)
    base_cols = np.where(arr != ord('-'))[0]
    last_pos = HXB2_FIRST_POS + len(base_cols) - 1
    mapping = np.zeros(_HXB2_LENGTH + 1, dtype=np.int64)
    mapping[HXB2_FIRST_POS:last_pos + 1] = base_cols + 1
    mapping[1:HXB2_FIRST_POS] = 1            # cropped 5' region -> first column
    mapping[last_pos + 1:] = len(arr)        # cropped 3' region -> last column
    return mapping


def convert_hxb2_df_to_aln(
    df: pd.DataFrame,
    hxb2_pos_to_aln: np.ndarray,
    pos_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Project HXB2 1-based coordinate column(s) of *df* onto ATA alignment
    1-based column coordinates using *hxb2_pos_to_aln*
    (see :func:`build_hxb2_pos_to_aln_col`).

    If the resulting frame still has both 'start' and 'end' columns, 'length'
    is recomputed as ``end - start + 1`` to reflect the (potentially larger,
    due to insertion columns present in the alignment but absent from HXB2)
    alignment span.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    max_pos = len(hxb2_pos_to_aln) - 1
    for col in pos_cols:
        out[col] = out[col].apply(
            lambda p: int(hxb2_pos_to_aln[int(p)])
            if 0 <= int(p) <= max_pos else p
        )

    if 'start' in out.columns and 'end' in out.columns:
        out['length'] = out['end'] - out['start'] + 1

    return out

# CRF label-sequence construction
def _crf_df_key(label: str, df_by_crf: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Return the df_by_crf dict key for a CRF subtype label, or None.

    'CRF01_AE' -> 'CRF01'   'CRF07_BC' -> 'CRF07'
    'CRF109_0107' -> 'CRF109'
    """
    m = re.match(r'^(?:CRF)?(\d+)', label, re.IGNORECASE)
    if m:
        num_str = f"{int(m.group(1)):02d}"
        for key in df_by_crf:
            if key.startswith(num_str):
                return key
    return label if label in df_by_crf else None

def _pure_set_of(
    label: str,
    df_by_crf: Dict[str, pd.DataFrame],
    seen: FrozenSet[str],
    depth: int,
) -> FrozenSet[str]:
    """Flatten *label* to its constituent pure-subtype strings.

    Used to resolve compound labels that contain a CRF reference,
    e.g. 'B/CRF01_AE' -> {'A', 'B', 'E'}.
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

    if label == '0708_mix':
        return frozenset({'B', 'C'})

    if label == 'mix':
        # Only appears in 174_0708, hopefully it doesn't appear in another CRF in the future
        return frozenset({'B', 'C'})

    key = _crf_df_key(label, df_by_crf)
    if key and key not in seen:
        out = set()
        for sub in df_by_crf[key]['subtype']:
            out |= _pure_set_of(str(sub), df_by_crf, seen | {key}, depth + 1)
        return frozenset(out)

    return frozenset()

def _resolve_single_label(
    subtype: str,
    df_by_crf: Dict[str, pd.DataFrame],
    seen: FrozenSet[str],
    depth: int,
) -> str:
    """Resolve a terminal or compound label to a string of pure subtypes.

    Examples
    --------
    'A1'          -> 'A'
    'E/A1'        -> 'A/E'       (sorted)
    'B/CRF01_AE'  -> 'A/B/E'
    'CRF02_AG/B'  -> 'A/B/G'
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
    * Pure-subtype terminal (A1, B, E, ...)  -> unified string, e.g. A1 -> 'A'
    * Compound label (B/C, E/A1, ...)        -> sorted pure-subtype string per
                                              position, e.g. 'E/A1' -> 'A/E'
    * CRF reference (CRF01_AE, CRF07_BC)  -> per-position labels copied from
                                              the referenced CRF's own array
                                              (intersected with [start, end])
    * 'mix' or unknown label               -> replace with 'U'
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

        # terminal or compound (may contain CRF refs inside compound)
        if subtype in _PURE_ST or '/' in subtype:
            labels[s : e + 1] = _resolve_single_label(
                subtype, df_by_crf, new_seen, depth
            )
            continue

        # CRF reference: expand position-by-position
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

        # Unresolvable (absent CRF) - keep as-is.
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

    # Alignment columns that carry an actual HXB2 base
    hxb2_base_cols = np.where(hxb2_arr != ord('-'))[0]
    n_base         = len(hxb2_base_cols)     # = _HXB2_USED_LEN after cropping
    hxb2_len       = _HXB2_LENGTH            # label arrays stay in full HXB2 coordinates
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

        # Project: hxb2_base_cols[k] <-> HXB2 pos HXB2_FIRST_POS + k
        aln_labels = np.full(aln_len, '-', dtype=object)
        aln_labels[hxb2_base_cols] = hxb2_labels[HXB2_FIRST_POS : HXB2_FIRST_POS + n_base]
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
    label_seqs = dict(sorted(label_seqs.items(), key=lambda x: int(re.match(r'^(?:CRF)?(\d+)', x[0]).group(1))))
    # Add Pure subtypes as their own entries in label_seqs
    for pure in _PURE_ST:
        if not pure in ['E', 'A5']:
            label_seqs[pure] = np.full(aln_len, pure, dtype=object)

    return label_seqs

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

    df.to_csv(output_path, index=False)
    return df


def parse_compactmapout(
    compactmapout_path: Path,
    ) -> Dict[str, List[int]]:
    """
    Parse the MAFFT compactmapout file to get the mapping from aligned positions to original positions.
    Returns a dictionary where keys are sequence names and values are lists of original positions.
    
    The file is structured as follows:
    
    # Insertion in added sequence > Position in reference
    >r_K+G+C_2013
    2433c - 2444g > 7810v7811
    >r_A7+C+F2_2019
    8584a - 8590g > 10908v10909
    8605a - 8607a > 10966v10967
    8618g - 8626t > 10995v10996
    8756t - 8756t > 11220v11221

    The goal is to output a dictionary of sample_name:
    """
    mapping = {}
    with open(compactmapout_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                seq_name = line[1:].strip()
                mapping[seq_name] = []
            else:
                parts = line.strip().split('>')
                start_pos = parts[0].split('-')[0].strip()[:-1]  # Get the start position
                end_pos = parts[0].split('-')[1].strip()[:-1]    # Get the end position
                ata_pos = parts[1].strip().split('v')[0]  # Get the ATA position
                mapping[seq_name].append((int(ata_pos), int(start_pos), int(end_pos)))  # Store as a tuple of (ATA position, start, end)

    return mapping


def dealign_to_records(input_path: Path) -> List:
    """
    Automatically detects format (FASTA, FASTQ, PHYLIP, NEXUS, CLUSTAL),
    strips gaps/dots from alignments or unaligned sequences, and produces 
    query records ready to be aligned to the reference with MAFFT.
    """
    parsed_records = None
    format_used = None

    # Try unaligned sequence formats (FASTA, FASTQ)
    for fmt in ["fasta", "fastq"]:
        try:
            with open(input_path, "r") as handle:
                parsed = list(SeqIO.parse(handle, fmt))
                if parsed:
                    parsed_records = parsed
                    format_used = fmt.upper()
                    break
        except Exception:
            pass

    # Try alignment formats (PHYLIP, NEXUS, CLUSTAL) if unaligned failed
    if not parsed_records:
        for fmt in ["phylip", "phylip-relaxed", "nexus", "clustal"]:
            try:
                with open(input_path, "r") as handle:
                    alignment = AlignIO.read(handle, fmt)
                    if alignment:
                        # Convert MultipleSeqAlignment into a list of SeqRecords
                        parsed_records = list(alignment)
                        format_used = f"Alignment ({fmt.upper()})"
                        break
            except Exception:
                pass

    if not parsed_records:
        raise ValueError(
            f"Could not parse '{input_path}'. Please check that it is a valid "
            "FASTA, FASTQ, PHYLIP, NEXUS, or CLUSTAL file."
        )
    print(f"Loaded {len(parsed_records)} records using format: {format_used}")

    records = []
    for index, record in enumerate(parsed_records):
        # Skip HXB2 reference if it is the first record
        if index == 0 and "HXB2" in record.id:
            continue
        record.description = ""
        # Clean sequence: uppercase, remove alignment gaps (- and .), strip leading/trailing Ns
        seq_str = str(record.seq).upper().replace("-", "").replace(".", "").strip('N')
        if not seq_str:
            continue  # Skip if sequence becomes empty after cleaning
        record.seq = Seq(seq_str)
        record.id = record.id.replace("Ref.", "")
        records.append(record)
        
    return records


def load_reference_ids(
    reference_path: Path,
    ) -> Set[str]:
    """IDs present in the reference alignment, used to strip it back out post-MAFFT."""
    return {rec.id for rec in SeqIO.parse(str(reference_path), "fasta")}


def run_mafft_addfragments(
    query_records: List,
    reference_path: Path,
    mafft_bin: str,
    threads: int,
    tmp_dir: Path,
    ) -> Tuple[Path, Path]:
    """
    Align query_records onto reference_path with `mafft --addfragments`.
    Returns (aligned_fasta_path, compactmapout_path).
    """
    query_fasta = tmp_dir / "query.fasta"
    SeqIO.write(query_records, str(query_fasta), "fasta")

    aligned_fasta = tmp_dir / "aligned.fasta"
    map_file = Path(str(query_fasta) + ".map")  # mafft's --compactmapout naming convention

    cmd = [
        mafft_bin, "--auto", "--keeplength", "--compactmapout", "--quiet",
        "--thread", str(threads), "--addfragments", str(query_fasta), str(reference_path),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    with open(aligned_fasta, "w") as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        sys.exit(f"ERROR: mafft failed (exit {result.returncode}):\n{result.stderr}")
    if not aligned_fasta.exists() or aligned_fasta.stat().st_size == 0:
        sys.exit(f"ERROR: mafft produced an empty alignment.\nstderr:\n{result.stderr}")
    if not map_file.exists():
        sys.exit(f"ERROR: expected mafft compactmapout file not found at {map_file}")

    return aligned_fasta, map_file


def clean_for_alignment(rec):
    rec.seq = Seq(str(rec.seq).upper().replace("-", "").replace(".", "").replace("U", "T").strip("N"))
    rec.description = ""
    return rec

def align_to_ata(records, ref_path, mafft_bin, threads, tmp_dir):
    fasta, map_file = run_mafft_addfragments(records, ref_path, mafft_bin, threads, tmp_dir)
    ref_ids = load_reference_ids(ref_path)
    ali = [r for r in SeqIO.parse(fasta, "fasta") if r.id not in ref_ids]
    for r in ali:
        r.id = ''.join(r.id.split('|')[1:]) if 'ins:' in r.id else r.id
    return ali, parse_compactmapout(map_file)


def decode_sample(decoder, sample_name, preds_slice, is_real, hxb2_to_ata, compactmapout_entry=()):
    probs = preds_slice / (preds_slice.sum(axis=-1, keepdims=True) + 1e-9)
    return decoder.query(sample_name=sample_name, probs=probs,
                         compactmapout_entry=list(compactmapout_entry),
                         hxb2_to_ata=hxb2_to_ata, query_mask=is_real)