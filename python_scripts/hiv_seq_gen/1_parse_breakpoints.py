import re
import pandas as pd
import requests
import urllib3

workspace_path = "/workspaces/mpath/oanoufa"

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
    from bs4 import BeautifulSoup
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


if __name__ == "__main__":

    output_path = f"{workspace_path}/data/LANL/crf/LANL_scrapped_bp.breakpoints"
    scrape_and_parse_lanl_breakpoints(output_path)

    corrected_path = f"{workspace_path}/data/LANL/crf/LANL_scrapped_bp.breakpoints"
    df_segments, df_breakpoints = parse_breakpoints_file(corrected_path)

    print(f"CRFs parsed       : {df_segments['crf'].nunique()}")
    print(f"Total segments    : {len(df_segments)}")
    print(f"Total breakpoints : {len(df_breakpoints)}")

    df_segments.to_csv(f"{workspace_path}/data/LANL/crf/lanl_crf_segments.csv",    index=False)
    df_breakpoints.to_csv(f"{workspace_path}/data/LANL/crf/lanl_crf_breakpoints.csv", index=False)
    print("\nSaved crf_segments.csv and crf_breakpoints.csv")