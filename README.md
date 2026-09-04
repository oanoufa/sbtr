# sbtr: HIV-1 Deep Learning-based SuBTypeR

**sbtr** is a novel genomic language model tool designed for fine-grain HIV-1 subtyping per nucleotide position. By predicting subtypes at high spatial resolution, sbtr detects novel recombinant forms and identifies precise recombination breakpoints rapidly.

---

## How It Works

sbtr processes input sequences through an automated end-to-end pipeline:
1. **Dealign & Align**: Input FASTA sequences or existing alignments are dealigned and aligned against an internal HIV-1 reference using MAFFT.
2. **Language Model Inference**: Aligned sequences pass through a pre-trained genomic language model.
3. **Subtype Classification**: Predictions are compared against model outputs from a reference bank of Circulating Recombinant Forms (CRFs) to generate final per-position and global subtype assignments.

---

## Installation & Setup

sbtr runs inside isolated container environments (Docker or Apptainer/Singularity) to manage CUDA and MAFFT dependencies.

### 1. Build the Container

**Docker:**
```bash
docker build -t sbtr:latest .
```

**Apptainer:**
```bash
apptainer build sbtr.sif docker://oanoufa/sbtr:latest
```

---

## Usage

> **Note:** sbtr requires a Hugging Face token (`HF_TOKEN`) to download model weights on the first run.

### Running with Docker

```bash
docker run --rm --shm-size=2g \
  -e HF_TOKEN=hf_xxxxxx \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/data/input:/data/in \
  -v /path/to/data/output:/data/out \
  sbtr \
  --seq /data/in/sequences.fasta \
  --out_dir /data/out \
  --tag my_run \
  --wto fr \
  --num_cpu 4
```

### Running with Apptainer

```bash
apptainer run \
  --env HF_TOKEN=hf_xxxxxx \
  --bind ~/.cache/huggingface:/root/.cache/huggingface \
  --bind /path/to/data/input:/data/in \
  --bind /path/to/data/output:/data/out \
  sbtr.sif \
  --seq /data/in/sequences.fasta \
  --out_dir /data/out \
  --tag my_run \
  --wto fr \
  --num_cpu 4
```

---

## Command Line Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--seq` | `str` | *Required* | Path to input FASTA file or alignment. |
| `--out_dir` | `str` | `"./sbtr_output"` | Output directory destination. |
| `--tag` | `str` | `"sbtr"` | Text appended to generated output file names. |
| `--mafft_bin` | `str` | `"mafft"` | Path to MAFFT executable (assumed on `PATH`). |
| `--gpu` | `flag` | `False` | Enable CUDA GPU acceleration. |
| `--num_cpu` | `int` | `1` | Number of CPUs for concurrent processing. |
| `--batch_size` | `int` | `1` | Forward pass batch size (increase for GPU runs). |
| `--wto` | `str` | `""` | Optional outputs to write (see details below). |

### Output Flags (`--wto`)
The tool always outputs `results_<tag>.csv` and `summary_<tag>.json`. You can request additional outputs by concatenating any combination of these letters to `--wto`:

* `f`: Generate plots/figures showing per-sequence predictions *(adds runtime)*.
* `r`: Output a region CSV containing `(start, end, subtype)` breakpoints.
* `p`: Export raw prediction scores as a NumPy (`.npy`) file.
* `a`: Save model attention masks.

*Example:* `--wto fr` outputs both the prediction figures and the genomic regions CSV.

---

## Outputs

* `results_<tag>.csv`: Final per-position subtype predictions.
* `summary_<tag>.json`: Run metadata and summary statistics.
* `regions_dealigned_<tag>.csv` *(optional)*: Genomic coordinates and assigned subtypes.
* `figures/` *(optional)*: Graphical visualisations of sequence subtype profiles.
