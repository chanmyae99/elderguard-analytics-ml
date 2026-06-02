"""
main.py
-------
Entry point for running the ElderGuard ML pipeline.
"""

from src.pipeline import Pipeline


def main():
    pipeline = Pipeline(
        apply_imbalance_handling=True
    )

    pipeline.run()


if __name__ == "__main__":
    main()