"""Parse jpHMM output into a flat regions CSV.

The input contains a sequence header followed by breakpoint rows:

    >seq_name (bw=1e-20)
    start_position    end_position    predicted_subtype
    ...

The output contains sample_name, start, end, subtype, and length columns.
"""

import argparse
import csv
import re
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("input", help="Path to jpHMM output file")
parser.add_argument("output", help="Path to output regions CSV")

args = parser.parse_args()

# Matches: >seq_name (bw=1e-20)   -- captures seq_name, tolerant of extra whitespace
HEADER_RE = re.compile(r"^>(.+?)\s*\(bw=")

def parse_jphmm(lines):
    """Yield (sample_name, start, end, subtype) tuples from jpHMM output lines."""
    current_sample = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if not line.strip():
            continue

        # Comment / metadata lines (format description, chosen parameters, etc.)
        if line.startswith("#"):
            continue

        # Sequence header line
        if line.startswith(">"):
            match = HEADER_RE.match(line)
            if not match:
                print(f"Warning: could not parse header line, skipping: {line!r}", file=sys.stderr)
                current_sample = None
                continue
            current_sample = match.group(1).strip()
            continue

        # Region data line: start \t end \t subtype
        if current_sample is None:
            print(f"Warning: region line found before any header, skipping: {line!r}", file=sys.stderr)
            continue

        fields = line.split("\t")
        if len(fields) != 3:
            # Fall back to arbitrary whitespace splitting, in case tabs got mangled
            fields = line.split()
        if len(fields) != 3:
            print(f"Warning: could not parse region line, skipping: {line!r}", file=sys.stderr)
            continue

        start_str, end_str, subtype = fields
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            print(f"Warning: non-integer start/end, skipping: {line!r}", file=sys.stderr)
            continue

        subtype = subtype.strip()

        yield current_sample, start, end, subtype


if __name__ == "__main__":
    with open(args.input, "r") as f:
        rows = list(parse_jphmm(f))

    if not rows:
        print("Warning: no regions parsed - output CSV will be empty.", file=sys.stderr)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "start", "end", "subtype", "length"])
        for sample_name, start, end, subtype in rows:
            writer.writerow([sample_name, start, end, subtype, end - start + 1])

    print(f"Wrote {len(rows)} regions to {args.output}")