"""
main.py
-------
Entry point for the ElderGuard ML pipeline.

Usage
-----
    python src/main.py
    python src/main.py --smote       # enable SMOTE oversampling

"""

import sys
import argparse

# Ensure project root is on path when running from root
sys.path.insert(0, ".")

from src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="ElderGuard Activity Level Prediction Pipeline"
    )
    parser.add_argument(
        "--smote", action="store_true", default=False,
        help="Apply SMOTE oversampling to handle class imbalance"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(apply_imbalance_handling=args.smote)
