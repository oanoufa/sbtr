#!/bin/bash
source /usr/local/conda/etc/profile.d/conda.sh
conda activate oanoufa_39

# Ensure the script uses the environment's python
python /workspaces/mpath/oanoufa/python_scripts/hiv_seq_gen/3_hiv_seq_gen_opti.py --n_seq 5000 --rp 0.8 --n_workers 16