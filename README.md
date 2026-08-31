# SBTR
HIV-1 Deep Learning based SuBTypeR

Subtyping is done per position. This fine-grain scale allows for the detection of novel recombinant and precise breakpoints while maintaining a fast inference.

Usage:

docker run --rm \
  -e HF_TOKEN=hf_xxxxx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /users/mpath/oanoufa/HIV_PROJECT/data/cuban:/data/in \
  -v /users/mpath/oanoufa/HIV_PROJECT/output_sbtr:/data/out \
  sbtr:cpu \
  --seq /data/in/HIV_POL_SEQ_2013-2018_AGG.txt \
  --mafft_bin mafft \
  --tag test1 \
  --out_dir /data/out \
  --wto r \
  --num_cpu 4

