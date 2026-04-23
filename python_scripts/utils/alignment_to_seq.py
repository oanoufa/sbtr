from Bio import SeqIO
from pathlib import Path
import pandas as pd
from Bio.SeqIO.FastaIO import FastaWriter
import pycountry
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Turn an alignment into a FASTA file of sequences unaligned and normalized.')
parser.add_argument('--i', type=str,
                    help='Input alignment',
                    default=None,
                    )
args = parser.parse_args()

# Get the arguments
input_alignment = Path(args.i)

if __name__ == "__main__":
    """
    Turn an alignment into sequences (strip gaps)
    """
    output = str(input_alignment.parent) + '/' + str(input_alignment.stem) + '_seq' + '.fasta'
    print(f"Writing sequences to {output}")
    with open(input_alignment, "r") as alignment:
        with open(output, "w") as sequences:
            writer = FastaWriter(sequences, wrap=100000)
            for index, record in tqdm(enumerate(SeqIO.parse(alignment, "fasta")), desc="Extracting sequences from alignment..."):
                # Skip ref
                if index != 0:
                    record.description = ""
                    seq = record.seq.upper()
                    seq = seq.replace('-', "")
                    seq = seq.strip('N')
                    record.seq = seq
                    writer.write_record(record)
