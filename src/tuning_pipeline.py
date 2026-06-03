def _save_tuning_results(self, search, filename):
    """
    Save tuning results for documentation and model justification.
    """

    import os
    import pandas as pd

    metrics_dir = os.path.join(
        REPORT_DIR,
        "metrics"
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        metrics_dir,
        filename
    )

    results_df = pd.DataFrame(search.cv_results_)

    results_df[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].to_csv(output_path, index=False)

    print(
        f"[training_service] Saved tuning results: "
        f"{output_path}"
    )