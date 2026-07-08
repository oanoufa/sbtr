"""
Auxiliary functions used in several scripts
"""

from typing import Tuple
import numpy as np
import config
from pathlib import Path
import pandas as pd

WORKSPACE_PATH = config.WORKSPACE_PATH


def build_hxb2_ata_maps(hxb2_ata_seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build bidirectional maps between alignment columns and HXB2 positions.

    Parameters
    ----------
    hxb2_ata_seq : str
        HXB2 row in the multiple-sequence alignment (``'-'`` = gap column).

    Returns
    -------
    ata_to_hxb2 : np.ndarray, shape (aln_len,), dtype int32
        ata_to_hxb2[col] = HXB2 position (1-based) of alignment column ``col``.
        Gap columns carry forward the position of the nearest preceding base;
        columns before the first HXB2 base carry 0.

    hxb2_to_ata : np.ndarray, shape (hxb2_len + 1,), dtype int32
        hxb2_to_ata[pos] = alignment column index of HXB2 position ``pos``
        (1-based).  Index 0 is unused (set to 0).
    """
    seq      = np.frombuffer(hxb2_ata_seq.encode(), dtype=np.uint8)
    is_base  = seq != ord("-")
    ata_pos  = np.where(is_base)[0]          # aln cols where HXB2 has a base
    ata_len  = len(hxb2_ata_seq)
    hxb2_len = ata_pos.size

    # ── forward: alignment column → HXB2 position (gap-filled) ──────────
    ata_to_hxb2          = np.zeros(ata_len, dtype=np.int32)
    ata_to_hxb2[ata_pos] = np.arange(1, hxb2_len + 1)
    for i in range(1, ata_len):
        if ata_to_hxb2[i] == 0:
            ata_to_hxb2[i] = ata_to_hxb2[i - 1]

    # ── reverse: HXB2 position → alignment column (exact) ───────────────
    hxb2_to_ata     = np.zeros(hxb2_len + 1, dtype=np.int32)  # index 0 unused
    hxb2_to_ata[1:] = ata_pos
    
    # ── print the mapping as a csv with columns: ata_pos, hxb2_pos ──────
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

    return ata_to_hxb2, hxb2_to_ata