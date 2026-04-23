#!/bin/bash
source /usr/local/conda/etc/profile.d/conda.sh
conda activate oanoufa_39

# Ensure the script uses the environment's python
python /workspaces/mpath/oanoufa/python_scripts/hiv_seq_gen/2_gen_blueprints_opti.py --n_bp 500000