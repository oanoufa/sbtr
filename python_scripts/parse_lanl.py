import re
import pandas as pd
import requests
import urllib3
import sys
import os
from bs4 import BeautifulSoup
import config
from typing import Dict, List, Optional, Tuple
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import tempfile

from utils import build_hxb2_ata_maps


workspace_path = config.WORKSPACE_PATH

urllib3.disable_warnings()
# Harmonizing dict for the breakpoints taken from LANL
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
    'CRRF08_BC': 'CRF08_BC',  # typo in source file
    '55':        'CRF55_01B',
    '55_01B':    'CRF55_01B',
    'CRF55':     'CRF55_01B',
    'CRF55_01B': 'CRF55_01B',
}

U_ALIASES = {'U', 'U1', 'Undetermined', 'Unsequenced', 'unknown'}

# ──────────────────────────────────────────────────────────────────────────────
# CRF01_AE mosaic structure  (HXB2 coordinates, 1-based)
# Each tuple: (breakpoint_hxb2_pos, region_before, region_after)
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

_HXB2_LENGTH:    int  = 9719   # K03455 bp count
_CRF_RE = re.compile(r"^(\d|CRF)", re.IGNORECASE)
_SIV_RE = re.compile(r"^(CPZ|GOR)", re.IGNORECASE)
_CRF01_AE_RE = re.compile(r"^(CRF)?01[_-]?AE$", re.IGNORECASE)
_HXB2_ID = "K03455|HIVHXB2CG"

def normalize_subtype(raw: str) -> str:
    """
    Normalize a subtype string to a canonical form.
    Handles:
      - Known CRF aliases (CRF_01 / CRF01 / 01 / 01_AE → CRF01_AE, etc.)
      - U aliases (Undetermined, Unsequenced, unknown, U1 → U)
      - Compound subtypes (A/CRF_01 → A/CRF01_AE, each component normalized)
      - Typos (CRRF08_BC → CRF08_BC)
    """
    # Handle compound subtypes by normalizing each component
    if '/' in raw:
        parts = raw.split('/')
        return '/'.join(normalize_subtype(p) for p in parts)

    s = raw.strip()

    # U aliases
    if s in U_ALIASES:
        return 'U'

    # Exact match in canonical map
    if s in CRF_CANONICAL:
        return CRF_CANONICAL[s]

    # Generic CRF_NN or CRFNN pattern not in the map above
    # → normalize to CRFnn format, keeping any suffix (e.g. CRF_103 → CRF103)
    m = re.match(r'^(CRF|_?CRF)?[_-]?(\d+)(_\w+)?$', s, re.IGNORECASE)
    if m:
        prefix = m.group(1) or ''
        num = m.group(2)
        suffix = m.group(3) or ''
        if prefix:
            return f"{prefix}{num}{suffix}"  # Keep existing prefix
        else:
            return f"CRF{num}{suffix}"       # Add CRF only when missing


    return s

def resolve_u_in_compound(subtype, prev_st, next_st):
    """
    Resolve U within a compound subtype like 'J/U' or 'U/K' or 'J/U/K'.
    U is replaced by its non-U neighbor within the compound first,
    falling back to external prev_st/next_st only if needed.
    """
    parts = subtype.split('/')
    if 'U' not in parts:
        return subtype

    # Non-U parts within the compound itself
    non_u = [p for p in parts if p != 'U']

    if len(non_u) >= 2:
        # e.g. 'J/U/K' → 'J/K'
        return '/'.join(non_u)
    elif len(non_u) == 1:
        internal = non_u[0]
        # e.g. 'J/U' → neighbor is J internally, external is next_st
        # pick the external neighbor that differs from internal
        external = next_st if parts[-1] == 'U' else prev_st
        if external and external != internal:
            return f"{internal}/{external}"
        else:
            return internal
    else:
        # pure 'U' — handled by caller
        return None

def resolve_u_segments(segs):
    """
    Replace U segments:
    - At the start/end (no non-U neighbor on one side): replace with the nearest non-U subtype
    - In the middle between two different subtypes: replace with ST1/ST2
    """
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
            # Pure U segment
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
            # Compound subtype containing U e.g. 'J/U', 'U/K', 'J/U/K'
            resolved[i]['subtype'] = resolve_u_in_compound(
                seg['subtype'], prev_st, next_st
            )
    return resolved

def join_neighbor_segments(resolved):
    """Join consecutive segments with identical subtypes"""
    if not resolved:
        return []
    # Sort by position
    resolved.sort(key=lambda x: x['start'])
    merged = [resolved[0].copy()]
    for current in resolved[1:]:
        prev = merged[-1]
        # Check if neighbor + same subtype
        if (current['start'] == prev['end'] + 1 and 
            current['subtype'] == prev['subtype']):
            # Extend previous segment
            prev['end'] = current['end']
        else:
            # New segment
            merged.append(current.copy())
    return merged

def check_crf_is_pure(resolved):
    """Check if all segments have the same subtype after U resolution"""
    if not resolved:
        return True
    first_st = resolved[0]['subtype']
    for seg in resolved[1:]:
        if seg['subtype'] != first_st:
            return False
    return True

def parse_breakpoints_file(filepath):
    """
    Parses HIV1.breakpoints format into segments and breakpoints dataframes.
    Format:
        >CRF_01 # comment
        790    5096    A1
        5097   5320    E
        ...
    """
    segments    = []
    breakpoints = []
    current_crf  = None
    current_segs = []

    def flush(crf, segs):
        segs = resolve_u_segments(segs)
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
            if not line.strip() or (line.startswith('#') and not line.startswith('>') ):
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
    """
    ------------------------------------------------------------------ #
     Generate HIV1_updated_clean.breakpoints from the parsed <pre> blocks
    ------------------------------------------------------------------ #
    HTML structure (one <pre> per CRF):
    
      <tr id="tgl_CRF01_AE">          ← CRF canonical name in the tr id
        <td>
          ... description text ...
          <br/>
          Breakpoints are approximate, based on <a>...</a> Figure 6.  ← comment
          <pre>                                                         ← segments
            790 5096 A1<br/>
            5097 5320 E<br/>
            ...
          </pre>
        </td>
      </tr>
    
    The <pre> tag does NOT contain the CRF header; it only holds segment rows.
    The CRF number comes from the parent <tr id="tgl_CRFXX_YY">.
    The comment is assembled from the sibling nodes that precede <pre> inside
    the parent <td>, using only the sentence that starts with "Breakpoints".
    """
    
    path = "https://www.hiv.lanl.gov/components/sequence/HIV/crfdb/crfs.comp"
    r = requests.get(path, verify=False)
    html_content = r.content.decode('utf-8')
    soup = BeautifulSoup(html_content, 'html.parser')
    L = soup.find_all("pre")
    r.close()
    segment_re = re.compile(r'^(\d+)\s+(\d+)\s+(\S+)')
    blocks_written = 0

    with open(output_path, 'w') as out:
        out.write(
            "# Breakpoints from Los Alamos HIV sequence database:\n"
            "# https://www.hiv.lanl.gov/content/sequence/HIV/CRFs/breakpoints.html\n\n\n"
        )

        for pre in L:
            # ── 1. CRF canonical name from <tr id="tgl_CRFXX_YY"> ───────
            parent_tr = pre.find_parent('tr')
            if parent_tr is None:
                continue
            tgl_id = parent_tr.get('id', '')
            if not tgl_id.startswith('tgl_'):
                continue
            crf_name = tgl_id[4:]   # strip leading "tgl_"  →  e.g. "CRF01_AE"

            # Extract the zero-padded number for the >CRF_NN header format
            m_num = re.match(r'CRF(\d+)', crf_name, re.IGNORECASE)
            if m_num is None:
                continue
            num_str = m_num.group(1).zfill(2)   # "1" → "01", "107" → "107"

            # ── 2. Comment: siblings before <pre> inside the parent <td> ─
            parent_td = pre.parent
            comment_parts = []
            for child in parent_td.children:
                if child is pre:
                    break
                text = child.get_text(separator=' ') if hasattr(child, 'get_text') else str(child)
                comment_parts.append(text)
            raw_comment = re.sub(r'\s+', ' ', ' '.join(comment_parts)).strip()

            # Keep only the sentence starting with "Breakpoints"
            m_bp = re.search(r'(Breakpoints?\s+.*)', raw_comment, re.IGNORECASE)
            comment_line = m_bp.group(1).strip() if m_bp else ''

            header_line = f">CRF_{num_str}"
            if comment_line:
                header_line += f" # {comment_line}"

            # ── 3. Segment rows: split on <br> tags inside <pre> ─────────
            # Each NavigableString preceding a <br>, plus the final child,
            # is one "start end subtype" row.
            raw_lines = [str(br.previous_sibling or '').strip()
                         for br in pre.find_all('br')]
            # Some CRF have no breakpoints given on the web page
            if list(pre.children):
                last = list(pre.children)[-1]
                raw_lines.append(str(last).strip() if last else '')

                seg_lines = []
                for raw in raw_lines:
                    sm = segment_re.match(raw)
                    if sm:
                        seg_lines.append(f"{sm.group(1)}\t{sm.group(2)}\t{sm.group(3)}")

                if not seg_lines:
                    continue   # no segments — skip

                out.write(header_line + "\n")
                for sl in seg_lines:
                    out.write(sl + "\n")
                out.write("\n")
                blocks_written += 1
            else:
                print(f"No segment lines found for {crf_name}")

    print(f"Written {blocks_written} CRF blocks to {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Pure-E HXB2 mask – computed once at import
# ──────────────────────────────────────────────────────────────────────────────
def _build_pure_e_hxb2_mask() -> np.ndarray:
    """
    Boolean array of length (_HXB2_LENGTH + 2).
    Index p (1-based HXB2 coordinate) is True iff position p lies in a
    **pure-E** region (not A1, not E/A1) of the canonical CRF01_AE mosaic.

    Interval layout derived from breakpoints:
        boundaries = [1, bp0, bp1, …, bp_n, HXB2_LENGTH+1]
        labels     = [region_before_bp0, region_after_bp0, …, region_after_bpn]
    so labels[i] occupies [boundaries[i], boundaries[i+1]).
    """
    boundaries = (
        [1]
        + [bp[0] for bp in _CRF01_BREAKPOINTS]
        + [_HXB2_LENGTH + 1]
    )
    labels = (
        [_CRF01_BREAKPOINTS[0][1]]           # region *before* first break
        + [bp[2] for bp in _CRF01_BREAKPOINTS]
    )
    mask = np.zeros(_HXB2_LENGTH + 2, dtype=bool)
    for i, label in enumerate(labels):
        if label == "E":
            mask[boundaries[i] : boundaries[i + 1]] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Internal: subtype parsing / classification
# ──────────────────────────────────────────────────────────────────────────────
def _extract_subtype(record_id: str) -> Optional[str]:
    """
    Parse subtype field from a LANL FASTA record ID.

    ``'Ref.A1.AU.03.PS1044.DQ676872'`` → ``'A1'``
    ``'Ref.01_AE.TH.90.CM240.U54771'`` → ``'01_AE'``
    """
    parts = record_id.split(".")
    if len(parts) >= 2 and parts[0].lower() == "ref":
        return parts[1]
    return parts[0] if parts else None


def _is_pure_subtype(subtype: Optional[str]) -> bool:
    """True for pure subtypes (A1, B, …); False for anything else."""
    if not subtype:
        return False
    return not _CRF_RE.match(subtype) and not _SIV_RE.match(subtype)

def _is_crf(subtype: Optional[str]) -> bool:
    """True for any CRF."""
    if not subtype:
        return False
    return _CRF_RE.match(subtype)

def _is_crf01_ae(subtype: Optional[str]) -> bool:
    """True iff *subtype* denotes CRF01_AE (handles 01_AE and CRF01_AE)."""
    return bool(subtype and _CRF01_AE_RE.match(subtype))

def prepare_pure_alignment(lanl_alignment_path: str,
                           pure_alignment_path: str,
                           crf_alignment_path: str,
                           ) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Read LANL subtype reference alignment, separate pure subtypes from CRFs.

    Returns
    -------
    tuple[dict, dict]
        - pure_result  : {record_id: aligned_sequence} for pure subtypes + HXB2
        - crf_result   : {record_id: aligned_sequence} for CRFs only
        CRF01_AE entries have A1 / E/A1 columns masked to 'N' in both outputs.
    Writes two FASTA files in-place:
        - pure_alignment_path         (pure subtypes)
        - crf_alignment_path.crf     (CRFs)
    """
    # ── 1. Load ───────────────────────────────────────────────────────────
    records = list(SeqIO.parse(lanl_alignment_path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in '{lanl_alignment_path}'.")

    # ── 2. Locate HXB2 row ────────────────────────────────────────────────
    hxb2_ata: Optional[str] = None
    for rec in records:
        if rec.id == _HXB2_ID:
            hxb2_ata = str(rec.seq).upper()
            hxb2_id  = rec.id
            break

    if hxb2_ata is None:
        raise ValueError(
            "HXB2 not found in the alignment. "
            f"Expected a record whose ID contains HXB2 ID: {_HXB2_ID}."
        )

    # ── 3. Column → HXB2 position map ────────────────────────────────────
    ata_to_hxb2: np.ndarray = build_hxb2_ata_maps(hxb2_ata)

    # ── 4. Project per-HXB2-position pure-E mask onto alignment columns ───
    _PURE_E_HXB2: np.ndarray = _build_pure_e_hxb2_mask()
    pure_e_cols: np.ndarray = _PURE_E_HXB2[ata_to_hxb2]

    # ── 5. Classify and collect ───────────────────────────────────────────
    pure_result: Dict[str, str] = {}
    crf_result:  Dict[str, str] = {}

    for rec in records:
        subtype = _extract_subtype(rec.id)
        seq_str = str(rec.seq).upper()

        if rec.id == hxb2_id:
            pure_result[rec.id] = seq_str
            continue

        if _is_pure_subtype(subtype):
            pure_result[rec.id] = seq_str

        if _is_crf01_ae(subtype):
            arr = np.frombuffer(seq_str.encode(), dtype=np.uint8).copy()
            arr[~pure_e_cols] = ord("N")
            masked_seq = arr.tobytes().decode()
            new_id = str(rec.id).replace("01_AE", "E")
            pure_result[new_id] = masked_seq   # pure E columns go to pure alignment

        if _is_crf(subtype):
            crf_result[rec.id] = seq_str

        # Other unrecognised entries are silently dropped

    # ── 6. Write pure alignment ───────────────────────────────────────────
    def _write_fasta(path: str, primary_id: str, primary_seq: str, sequences: Dict[str, str]) -> None:
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


    # ── 7. Write alignments (HXB2 included as coordinate reference) ────
    _write_fasta(pure_alignment_path, hxb2_id, pure_result[hxb2_id], pure_result)
    _write_fasta(crf_alignment_path, hxb2_id, pure_result[hxb2_id], crf_result)

    print(f"Pure alignment written to: {pure_alignment_path} ({len(pure_result)-1} sequences)")
    print(f"CRF  alignment written to: {crf_alignment_path} ({len(crf_result)} sequences)")

    return pure_result, crf_result


if __name__ == "__main__":

    output_path = f"{workspace_path}/data/output/LANL_scrapped_bp.breakpoints"
    scrape_and_parse_lanl_breakpoints(output_path)
    df_segments, df_breakpoints = parse_breakpoints_file(output_path)

    print(f"CRFs parsed       : {df_segments['crf'].nunique()}")
    print(f"Total segments    : {len(df_segments)}")
    print(f"Total breakpoints : {len(df_breakpoints)}")

    df_segments.to_csv(f"{workspace_path}/data/output/lanl_crf_segments.csv",    index=False)
    df_breakpoints.to_csv(f"{workspace_path}/data/output/lanl_crf_breakpoints.csv", index=False)
    print("\nSaved crf_segments.csv and crf_breakpoints.csv")


    subtype_ref_alignment_path = f"{workspace_path}/data/output/HIV1_SUBTYPE_REF.fasta"
    pure_ref_path = f"{workspace_path}/data/input/HIV1_PURE_REF.fasta"
    crf_ref_path = f"{workspace_path}/data/output/HIV1_CRF_REF.fasta"

    final_pure_alignment, final_crf_alignment = prepare_pure_alignment(subtype_ref_alignment_path,
                                                                       pure_ref_path,
                                                                       crf_ref_path,
                                                                       )
    