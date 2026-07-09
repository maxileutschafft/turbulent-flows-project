"""Generate a screening design for surrogate hyperparameter tuning.

By default this emits a DSD-sized plan for the repository's recommended GNO
hyperparameters, plus the objective metrics that should be used to score each
training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.screening import (
    definitive_screening_design,
    recommended_gno_factors,
    write_screening_csv,
    write_screening_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/screening_design.csv"),
        help="Destination CSV path for the screening plan.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional destination for the full screening plan as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to break ties in the row selection.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on the number of three-level candidates explored.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    design = definitive_screening_design(
        recommended_gno_factors(),
        seed=args.seed,
        max_candidates=args.max_candidates,
    )
    csv_path = write_screening_csv(design, args.output)
    print(f"Wrote screening CSV -> {csv_path}")
    if args.json is not None:
        json_path = write_screening_json(design, args.json)
        print(f"Wrote screening JSON -> {json_path}")
    print(f"Runs: {design.runs}")
    print("Objective metrics:")
    for metric in design.objective.primary_metrics:
        print(f"  - {metric}")
    for metric in design.objective.secondary_metrics:
        print(f"  - {metric} (secondary)")


if __name__ == "__main__":
    main()