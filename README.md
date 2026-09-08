
# sbtr: HIV-1 Deep Learning-based SuBTypeR

<img src="figs/readme/sbtr.png" width="130" align="left">

**sbtr** is a novel genomic language model tool designed for fine-grain HIV-1 subtyping per nucleotide position. By predicting subtypes at high spatial resolution, sbtr detects novel recombinant forms and identifies precise recombination breakpoints rapidly.

<br clear="left">

## How it works

sbtr processes input sequences through an automated end-to-end pipeline:
1. **Dealign & Align**: Input FASTA sequences or existing alignments are dealigned and aligned against an internal HIV-1 reference alignment using MAFFT.
2. **Language Model Inference**: Aligned sequences pass through a pre-trained genomic language model that uses [Nucleotide Transformer v3](https://huggingface.co/spaces/InstaDeepAI/ntv3) as a backbone. The model outputs predictions as a 2D array of size (num_subtypes, alignment_length). A value close to 1 in cell i, j means a prediction of subtype i at position j.
3. **Subtype Classification**: Predictions are compared against model outputs from a reference bank built with 3 sequences of each Circulating Recombinant Forms (CRFs). The comparison is a simple intersection between the input sequence prediction array and each prediction array in the bank. sbtr finally generates per-position and global subtype assignments. An example is given in the readme.



## Installation & setup

sbtr runs inside isolated container environments (Docker or Apptainer/Singularity) to manage CUDA and MAFFT dependencies.

### 1. Retrieve the container

**Docker:**

```bash
docker pull ghcr.io/oanoufa/sbtr:latest
```

**Apptainer:**

```bash
apptainer pull sbtr.sif docker://ghcr.io/oanoufa/sbtr:latest
```



## Usage

> **Note:** A Hugging Face token must be specified (`HF_TOKEN`) to download model weights from InstaDeepAI/NTv3_650M_pre. This token must come from an account with access to the model repository. This can be done at this address [https://huggingface.co/InstadeepAI/NTv3_650M_pre](https://huggingface.co/InstadeepAI/NTv3_650M_pre).

> **Note:** Using gpu is highly recommended if possible. The backbone model contains 650M parameters, running on a computer with high memory is also recommended.

### Running with Docker

```bash
docker run --rm --shm-size=2g \
  -e HF_TOKEN=hf_xxxxxx \
  -v /path/to/data/input:/data/in \
  -v /path/to/data/output:/data/out \
  sbtr \
  --seq /data/in/sequences.fasta \
  --mafft_bin mafft \
  --tag my_sequences \
  --out_dir /data/out \
  --wto r \
  --num_cpu ${N_WORKERS} \
  --gpu \
  --batch_size 8
```


### Running with Apptainer

```bash
apptainer run --nv \
  --env HF_TOKEN=hf_xxxxxx \
  --bind /path/to/tmp:/tmp \
  --bind /path/to/data/input:/data/in \
  --bind /path/to/data/output:/data/out \
  /pasteur/helix/projects/mPath/oanoufa/sbtr/sbtr.sif \
  --seq /data/in/sequences.fasta \
  --mafft_bin mafft \
  --tag my_sequences \
  --out_dir /data/out \
  --wto r \
  --num_cpu ${N_WORKERS} \
  --gpu \
  --batch_size 8
```


## Command line arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--seq` | `str` | *Required* | Path to input FASTA file or alignment of sequences. |
| `--out_dir` | `str` | `"./sbtr_output"` | Output directory destination. |
| `--tag` | `str` | `"sbtr"` | Text appended to generated output file names. |
| `--mafft_bin` | `str` | `"mafft"` | Path to MAFFT executable (assumed on `PATH`). |
| `--gpu` | `flag` | `False` | Enable CUDA GPU acceleration. |
| `--num_cpu` | `int` | `1` | Number of CPUs for concurrent processing. |
| `--batch_size` | `int` | `1` | Forward pass batch size (increase for GPU runs). |
| `--wto` | `str` | `""` | Optional outputs to write (see details below). |

### Output flag (`--wto`)
The tool always outputs `results_<tag>.csv` and `summary_<tag>.json`. You can request additional outputs by concatenating any combination of these letters to `--wto`:

* `f`: Generate plots/figures showing per-sequence predictions *(adds runtime)*.
* `r`: Output a region csv containing `(start, end, subtype)` breakpoints.
* `p`: Export raw prediction scores as a NumPy (`.npy`) file.
* `a`: Save model attention masks.

*Example:* `--wto fr` outputs both the prediction figures and the genomic regions csv.


## Outputs

* `results_<tag>.csv`: Final per-position subtype predictions.
* `summary_<tag>.json`: Run metadata and summary statistics on the input batch of sequences.
* `regions_dealigned_<tag>.csv` *(optional)*: Genomic coordinates and assigned subtypes.
* `--out_dir/figs/` *(optional)*: Graphical visualisations of sequence subtype profiles.


## Example output

Below is an example run on `65_cpx.CN.11.ANHUI_HF104.KC183778`, a CRF65_cpx recombinant (subtypes 01_AE / B / C). We consider that `--wto` was set to `fr` to generate both figures and region outputs.

**1. Results csv**

The results csv contains one row per sample in the input sequence file. This row details the global results on the sample:
* `composition` is a string of the subtypes present in the sequence, separated by `_`. Here `AE_B_C`.
* `dominant_subtype` is the subtype in majority in the sequence. Here `C`.
* `dominant_fraction` is the proportion of `dominant_subtype` in the sequence. Here `0.6100`.
* `ref_best_crf` is the closest matching CRF sequence in the reference bank. Here `65_cpx.CN.10.YNFL02.KC870028`.
* `ref_best_score` is the matching score of `ref_best_crf`. Here `0.9288`.
* `ref_top5` is the top 5 of the closest CRFs to the input sequence. They are given with their score as `<crf>:<score>`. Here `65_cpx:0.9288  106_cpx:0.6668  85_BC:0.6428  110_BC:0.6297  82_cpx:0.6237`.
* `final_decision` is a summary of all columns and sbtr's global opinion on the sequence. It looks like `<pure/recombinant>.<composition>.<full/partial>.<assigned/like/unassigned>.<crf>`. A sequence will be *pure* if only one subtype is detected, *full* if its length is higher or equal than 7000bp, *unassigned* if `ref_best_score` is under `0.5`, *like* if several CRFs have scores within `0.02` of `ref_best_score`, *assigned* if `ref_best_crf` sits alone at the top.

The line corresponding to our example is shown below.

```csv
sample_name,composition,dominant_subtype,dominant_fraction,ref_best_crf,ref_best_score,ref_top5,final_decision
65_cpx.CN.11.ANHUI_HF104.KC183778,AE_B_C,C,0.6100,65_cpx.CN.10.YNFL02.KC870028,0.9288,65_cpx:0.9288  106_cpx:0.6668  85_BC:0.6428  110_BC:0.6297  82_cpx:0.6237,recombinant.AE+B+C.full.assigned.65_cpx
```

**2. Summary json**

The summary json contains global information on the input batch of sequences. For the sake of the example, the summary JSON presented here is the result of a run on a batch containing both the example and three other CRF sequences.

```json
{
  "n_sequences": 4,
  "composition_prevalence": {
    "AE_C_B": 0.5,
    "B_C": 0.25,
    "B_F1": 0.25
  },
  "dominant_subtype_prevalence": {
    "C": 0.75,
    "B": 0.25
  },
  "pure_vs_recombinant": {
    "recombinant": 1.0
  },
  "full_vs_partial": {
    "full": 0.75,
    "partial": 0.25
  },
  "recombinant_assigned_vs_unassigned": {
    "assigned": 1.0
  },
  "crf_prevalence_among_assigned": {
    "65_cpx": 0.5,
    "64_BC": 0.25,
    "70_BF1": 0.25
  },
  "crf_prevalence_among_like_and_assigned": {
    "65_cpx": 0.5,
    "64_BC": 0.25,
    "70_BF1": 0.25
  },
  "dominant_fraction_stats": {
    "mean": 0.7046,
    "std": 0.1396,
    "min": 0.5971,
    "50%": 0.6558,
    "max": 0.9098
  },
  "low_confidence_fraction": 0.0,
  "repeated_best_ref": {
    "65_cpx.CN.11.ANHUI_HF104.KC183778": 2
  }
}
```

**3. Regions csv**

The region output applies when `r` is specified in `--wto`. It is a single csv file for all the input sequences containing their subtype structure. Here are the entries that would appear for our example sequence. The structure given here is unaligned, meaning it goes from the first position of the input sequence to its last position. It is not aligned to the reference genome HXB2 nor to our reference alignment.

```csv
sample_name,start,end,subtype,length
65_cpx.CN.11.ANHUI_HF104.KC183778,1,505,AE,505
65_cpx.CN.11.ANHUI_HF104.KC183778,506,514,U,9
65_cpx.CN.11.ANHUI_HF104.KC183778,515,796,B,282
65_cpx.CN.11.ANHUI_HF104.KC183778,797,3062,C,2266
65_cpx.CN.11.ANHUI_HF104.KC183778,3063,3345,B,283
65_cpx.CN.11.ANHUI_HF104.KC183778,3346,4146,C,801
65_cpx.CN.11.ANHUI_HF104.KC183778,4147,4352,AE,206
65_cpx.CN.11.ANHUI_HF104.KC183778,4353,4988,C,636
65_cpx.CN.11.ANHUI_HF104.KC183778,4989,5198,AE,210
65_cpx.CN.11.ANHUI_HF104.KC183778,5199,5394,B,196
65_cpx.CN.11.ANHUI_HF104.KC183778,5395,5777,AE,383
65_cpx.CN.11.ANHUI_HF104.KC183778,5778,6917,C,1140
65_cpx.CN.11.ANHUI_HF104.KC183778,6918,7618,B,701
65_cpx.CN.11.ANHUI_HF104.KC183778,7619,7811,AE,193
65_cpx.CN.11.ANHUI_HF104.KC183778,7812,7888,C,77
65_cpx.CN.11.ANHUI_HF104.KC183778,7889,8184,AE,296
65_cpx.CN.11.ANHUI_HF104.KC183778,8185,8343,B,159
65_cpx.CN.11.ANHUI_HF104.KC183778,8344,8762,C,419
65_cpx.CN.11.ANHUI_HF104.KC183778,8763,8930,3'LTR,168
```


**4. Figure**

> **Note:** Choosing to output figures (`f` in `--wto`) will increase runtime, especially for large input files. It is not recommended for large batches of sequences (> 100). Simpler figures can be generated by removing the *sample_name* and *length* columns from the region CSV and using the Recombinant HIV-1 Drawing Tool available at [LANL HIV DB](https://www.hiv.lanl.gov/content/sequence/DRAW_CRF/recom_mapper.html).
 

One figure for each sequence is output when `f` is specified in `--wto`. It is composed of four panels arranged vertically. The whole figure is set in the all-to-all alignment perspective, so the x-axis is always the genome positions, ranging from 1 to 11561 (length of the alignment).

The top panel shows the loss mask on the sequence, displaying non-N positions. Under it, a panel recalls the positions and frames of genes along HIV-1 genome.

The third panel is the crucial element of the figure. This heatmap shows the model's subtype predictions at each position. As the legend on the right shows, a bright yellow block corresponds to a confident subtype prediction.

The last panel corresponds to the predictions after applying a sliding window. This gives a final 1D list of prediction at single-nucleotide resolution.

<img src="figs/readme/65_cpx.CN.11.ANHUI_HF104.KC183778_preds.png" width="800">

**Comparison with jpHMM**

As a comparison, we show here the known subtype composition of CRF65_cpx across the HIV-1 genome (`gag`, `pol`, `env`, accessory genes), used here as ground truth. The structure comes from the [original paper that identified this CRF](https://doi.org/10.1089/aid.2013.0233). The Figure was built using LANL Recombinant HIV-1 Drawing Tool.

<img src="figs/readme/65_cpx_paper.png" width="800">

This 1D representation compares our results to the output of jpHMM. jpHMM was used with a reference file built using the python script `build_jphmm_ref.py`. It uses the same subtypes as us. This figure is unaligned to the reference genome HXB2 so there is a shift of around 600 - 700 positions (missing 5'LTR) between the ground truth blocks and ours.

<img src="figs/readme/65_cpx.CN.11.ANHUI_HF104.KC183778_jphmm_comp.png" width="800">

We notice that sbtr's sliding-window subtype calls closely track the reference structure and recover breakpoints that jpHMM misses such as the B block in the middle of the *pol* gene and the C block in between the two 01_AE blocks at the end of `env`. However, we also notice that our model found two new blocks: a B at the beginning of the genome, after the 01_AE block and a 01_AE block at the beginning of `vif`.

Those two new blocks were further inspected.




## Repository breakdown

Using a genomic language model means training a model on sequence data. In the case of HIV-1, real sequence data is extremely biased towards known subtypes. `data/input` contains all real sequences that were used to train our model. We used LANL HIV-1 2023 full genome reference alignment and HIV-1 2022 full genome Super Filtered Web Alignment. Information on those alignments can be found on [LANL website](https://www.hiv.lanl.gov/content/sequence/NEWALIGN/align.html). The distributions of the sequences from both these alignments per subtype is shown below. We also display the sequence year to evaluate the temporal diversity.

<img src="figs/readme/ref_ali_dist.png" width="800">

The process used to augment this set to our training set is described precisely in our manuscript. The final distribution after augmentation is shown in the figure below. Our intention here is to keep track of the global prevalence of subtypes, as more common subtypes have greater within-subtype diversity, while still letting the model train on enough sequences of each subtype, even the rarest ones.

<img src="figs/readme/gen_ali_dist.png" width="800">

The FASTA files in `data/input/diversity` corresponds to all full length genome (>7000nt) sequences in LANL HIV Database for subtypes A, B, C, D, F, G and CRF01_AE. They were used to compute mutation rates per site thanks to [IQ-TREE](https://iqtree.github.io).


All scripts in `python_scripts` were used for preprocessing, data generation, results comparison, or are called by `sbtr.py` when the model is ran.

