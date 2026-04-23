#!/bin/bash

ref=/workspaces/mpath/oanoufa/data/SBTR_DATA/LANL_SBTR_ALIGNMENT.fasta
seqs=/workspaces/mpath/oanoufa/data/LANL/curated_sequences/HIV1_COMPLETE_SEQ.fasta
output=/workspaces/mpath/oanoufa/data/SBTR_DATA/LANL_SBTR_MUTATION_RATE_ALIGNMENT.fasta
mafft --auto --keeplength --thread 32 --addfragments $seqs $ref > $output
