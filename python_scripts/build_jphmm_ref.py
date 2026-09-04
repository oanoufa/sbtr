"""Convert HIV FASTA sequences into a jpHMM reference file."""

import argparse
import random
from collections import defaultdict
from src import config

parser = argparse.ArgumentParser(
    description="Convert HIV FASTA sequences to jpHMM reference format with subtype sampling."
)
parser.add_argument(
    "-i", "--input", required=True, help="Path to input FASTA file"
)
parser.add_argument(
    "-o", "--output", required=True, help="Path to output jpHMM reference file"
)
parser.add_argument(
    "-n",
    "--max-seqs",
    type=int,
    default=100,
    help="Max sequences per subtype (default: 100)",
)

args = parser.parse_args()

seed = config.MODEL_CONFIG["seed"]

def parse_fasta(fasta_path):
    """Parses a FASTA file and returns a list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                    current_seq = []
                current_header = line[1:]  # strip the leading '>'
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))

    return sequences


def extract_subtype(header):
    """Extracts the subtype (2nd element split by '.') from the FASTA header.

    Example: 'Ref.A1.CD.87.PBS888.MH705133' -> 'A1'
    """
    parts = header.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Header '{header}' does not contain at least two '.'-delimited fields."
        )
    return parts[1]


def format_jphmm_reference(
    input_file, output_file, max_per_subtype=100, seed=None
):
    if seed is not None:
        random.seed(seed)

    print(f"Reading input FASTA: {input_file}")
    sequences = parse_fasta(input_file)

    # Group sequences by subtype
    subtypes = defaultdict(list)
    for header, seq in sequences:
        try:
            st = extract_subtype(header)
            subtypes[st].append((header, seq))
        except ValueError as e:
            print(f"Warning: Skipping sequence due to header format issue: {e}")

    print(
        f"Found {len(sequences)} total sequences across {len(subtypes)} subtypes."
    )

    # Sample up to max_per_subtype per group and write output
    with open(output_file, "w") as out:
        for st, seq_list in subtypes.items():
            # Write group header
            out.write(f">>{st}\n")

            # Downsample if necessary
            if len(seq_list) > max_per_subtype:
                selected = random.sample(seq_list, max_per_subtype)
                print(
                    f"  Subtype '{st}': randomly selected {max_per_subtype} out of {len(seq_list)} sequences"
                )
            else:
                selected = seq_list
                print(
                    f"  Subtype '{st}': keeping all {len(seq_list)} sequences"
                )

            # Write sequence entries
            for header, seq in selected:
                out.write(f">{header}\n")
                out.write(f"{seq}\n")

    print(f"Successfully generated jpHMM reference file: {output_file}")


if __name__ == "__main__":
    format_jphmm_reference(args.input, args.output, args.max_seqs, seed)
