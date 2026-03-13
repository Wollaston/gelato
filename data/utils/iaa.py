from utils import read_conll_data, compute_metrics
import argparse
# Calculate IAA

# read in data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two files.")
    parser.add_argument("file1", type=str, help="Path to the first file")
    parser.add_argument("file2", type=str, help="Path to the second file")
    parser.add_argument("--io", action="store_true", help="Score in IO mode")

    args = parser.parse_args()

    c1 = read_conll_data(args.file1)
    c2 = read_conll_data(args.file2)
    #print(c1, c2)

    print(compute_metrics((c1[1], c2[1]), args.io))

# F1

# Overlap

# Overlap Outside O

