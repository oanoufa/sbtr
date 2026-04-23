#!/bin/bash
source /usr/local/conda/etc/profile.d/conda.sh
conda activate oanoufa_39

# Ensure the script uses the environment's python
python /workspaces/mpath/oanoufa/python_scripts/hiv_seq_gen/0_aio_hiv_seq_gen.py --n_seq 200000 --rp 0.8 --n_workers 16