from Bio import SeqIO
from pathlib import Path
import pandas as pd
from Bio.SeqIO.FastaIO import FastaWriter
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Turn an alignment into a FASTA file of sequences unaligned and normalized.')
parser.add_argument('--i', type=str,
                    help='Input alignment',
                    default='/pasteur/helix/projects/mPath/oanoufa/sbtr/data/input/HIV1_PURE_REF.fasta',
                    )
parser.add_argument('--o', type=str,
                    help='Output file (default: same as input with _seq suffix)',
                    default=None,
                    )
args = parser.parse_args()

# Get the arguments
input_alignment = Path(args.i)
if args.o is None:
    output_file = input_alignment.with_name(input_alignment.stem + "_seq.fasta")
else:
    output_file = Path(args.o)

if __name__ == "__main__":
    """
    Turn an alignment into sequences (strip gaps)
    """
    print(f"Writing sequences to {output_file}")
    with open(input_alignment, "r") as alignment:
        with open(output_file, "w") as sequences:
            writer = FastaWriter(sequences, wrap=100000)
            for index, record in tqdm(enumerate(SeqIO.parse(alignment, "fasta")),
                                      desc="Extracting sequences from alignment..."):
                # Skip ref if it is HXB2
                if index == 0 and "HXB2" in record.id:
                    continue
                else:
                    record.description = ""
                    seq = record.seq.upper()
                    seq = seq.replace('-', "")
                    seq = seq.strip('N')
                    record.seq = seq
                    record.id = str(record.id).replace("Ref.", "")
                    writer.write_record(record)
