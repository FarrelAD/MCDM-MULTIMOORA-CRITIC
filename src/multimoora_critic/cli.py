import argparse
import sys
from typing import List, Optional
import numpy as np
import pandas as pd

from multimoora_critic.solver import multimoora_critic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimoora-critic",
        description="Multi-Criteria Decision Making using MULTIMOORA with CRITIC weighting.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run evaluation on built-in demo dataset.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to CSV file containing the decision matrix.",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        type=int,
        help="Space-separated criteria types: 1 for benefit, -1 for cost (e.g. --criteria 1 1 -1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to CSV file where results will be saved.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="fancy_grid",
        help="Table format style for stdout (default: fancy_grid). Options: fancy_grid, grid, github, plain.",
    )
    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.demo and not parsed_args.input:
        parser.print_help()
        sys.stderr.write("\nError: Either --demo or --input PATH must be specified.\n")
        return 1

    if parsed_args.demo:
        matrix = np.array([
            [250.0, 16.0, 12.0],
            [200.0, 20.0, 8.0],
            [300.0, 18.0, 11.0],
            [270.0, 17.0, 10.0],
        ])
        criteria = np.array([1, 1, -1])
        alt_names = None
    else:
        if not parsed_args.criteria:
            sys.stderr.write("Error: --criteria is required when using --input CSV.\n")
            return 1

        try:
            df_in = pd.read_csv(parsed_args.input)
        except FileNotFoundError:
            sys.stderr.write(f"Error: Input file not found at '{parsed_args.input}'.\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"Error reading CSV file '{parsed_args.input}': {e}\n")
            return 1

        # Check if first column is non-numeric (alternative names)
        first_col = df_in.iloc[:, 0]
        if not pd.api.types.is_numeric_dtype(first_col):
            alt_names = first_col.astype(str).tolist()
            matrix_df = df_in.iloc[:, 1:]
        else:
            alt_names = None
            matrix_df = df_in

        try:
            matrix = matrix_df.to_numpy(dtype=float)
        except ValueError as e:
            sys.stderr.write(f"Error converting CSV data to numeric matrix: {e}\n")
            return 1

        criteria = np.array(parsed_args.criteria)

    try:
        result = multimoora_critic(matrix, criteria)
    except ValueError as e:
        sys.stderr.write(f"Error during calculation: {e}\n")
        return 1

    # Output to stdout
    print("-" * 50)
    print("MCDM: MULTIMOORA with CRITIC Weighting")
    print("-" * 50)
    print("\nCalculated CRITIC Weights:")
    for i, w in enumerate(result.weights):
        print(f"Criterion C{i + 1}: {w:.4f}")

    print("\nRanking Results Summary:")
    print(result.to_table(alternative_names=alt_names, tablefmt=parsed_args.format))

    names = alt_names or result._default_alternative_names()
    best_name = names[result.best_alternative_index]
    print(f"\nConclusion: The best alternative is {best_name}")

    if parsed_args.output:
        try:
            res_df = result.to_dataframe(alternative_names=alt_names)
            res_df.to_csv(parsed_args.output, index=False)
            print(f"\nResults successfully saved to '{parsed_args.output}'.")
        except Exception as e:
            sys.stderr.write(f"Error writing output file '{parsed_args.output}': {e}\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
