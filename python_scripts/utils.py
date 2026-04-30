"""
Auxiliary functions used in several scripts
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helper: alignment-column → HXB2-position map 
# ──────────────────────────────────────────────────────────────────────────────
def build_hxb2_ata_maps(hxb2_ata_seq: str) -> np.ndarray:
    """
    Map every alignment column to a HXB2 position (1-based).

    Gap columns carry forward the position of the nearest preceding base;
    columns before the first HXB2 base carry position 0.

    Parameters
    ----------
    hxb2_ata_seq : str
        HXB2 row in the alignment (``'-'`` = gap column).

    Returns
    -------
    np.ndarray of shape (aln_len,), dtype int32
    """
    seq             = np.frombuffer(hxb2_ata_seq.encode(), dtype=np.uint8)
    is_base         = seq != ord("-")
    ata_pos         = np.where(is_base)[0]
    ata_len         = len(hxb2_ata_seq)

    ata_to_hxb2            = np.zeros(ata_len, dtype=np.int32)
    ata_to_hxb2[ata_pos]   = np.arange(1, ata_pos.size + 1)
    for i in range(1, ata_len):
        if ata_to_hxb2[i] == 0:
            ata_to_hxb2[i] = ata_to_hxb2[i - 1]
    return ata_to_hxb2