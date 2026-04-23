#!/bin/bash
source /usr/local/conda/etc/profile.d/conda.sh
conda activate oanoufa_311

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=INTEL

# Ensure the script uses the environment's python
python /workspaces/mpath/oanoufa/python_scripts/nt_v3_hiv_st/training.py