"""
main.py
-------
Entry point for running the ElderGuard ML pipeline.
"""

from src.pipeline import Pipeline


def main():
    # Configure pipeline execution settings
    pipeline = Pipeline(
        apply_imbalance_handling=False,
    )

    # Execute the complete ML workflow
    pipeline.run()


if __name__ == "__main__":
    # Run only when executed directly, not when imported as a module
    main()