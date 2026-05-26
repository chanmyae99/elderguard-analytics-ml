"""
pipeline.py
-----------
End-to-end machine learning pipeline for ElderGuard Analytics.

Reads configuration from config.yaml, then executes:
  1. Load raw data from SQLite
  2. Clean and preprocess
  3. Feature engineering
  4. Train / test split
  5. Build model (set via config.yaml → models.active_model)
  6. Train with cross-validation
  7. Evaluate on held-out test set
  8. Save model artefacts to saved_model/

Usage
-----
    python pipeline.py                           # uses config.yaml
    python pipeline.py --config custom.yaml      # custom config
    python pipeline.py --model xgboost           # override model

"""

import argparse
import sys
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")
from data_loader      import load_data
from preprocessor     import Preprocessor
from feature_engineer import add_features
from model_factory    import build_model
from trainer          import train
from evaluator        import evaluate
from model_io         import save_model


def load_config(path: str) -> dict:
    """Load and return the YAML configuration file as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ElderGuard Activity Level Prediction Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to the YAML config file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override active model name (must match a key in config models:)"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    # Command-line --model flag takes priority over config file
    active_model_name = args.model or cfg["models"]["active_model"]

    data_cfg   = cfg["data"]
    prep_cfg   = cfg["preprocessing"]
    feat_cfg   = cfg.get("feature_engineering", {})
    model_cfg  = cfg["models"][active_model_name]
    eval_cfg   = cfg["evaluation"]
    output_cfg = cfg["output"]

    print(f"\n{'='*55}")
    print(f"  ElderGuard Analytics — ML Pipeline")
    print(f"  Active model: {active_model_name}")
    print(f"{'='*55}")

    # ── Step 1: Load ──────────────────────────────────────────
    print("\n[Step 1] Loading data ...")
    df_raw = load_data(data_cfg["db_path"], data_cfg["table_name"])

    # ── Step 2: Clean ─────────────────────────────────────────
    print("\n[Step 2] Cleaning data ...")
    preprocessor = Preprocessor(prep_cfg)
    df_clean = preprocessor.clean(df_raw)

    # ── Step 3: Feature engineering ───────────────────────────
    print("\n[Step 3] Engineering features ...")
    if feat_cfg.get("enabled", False):
        df_clean = add_features(df_clean)
    else:
        print("[feature_engineer] Disabled — skipping.")

    # ── Step 4: Split ─────────────────────────────────────────
    print("\n[Step 4] Splitting train / test ...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=df_clean["Activity Level"],
    )
    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    # ── Step 5: Encode (fit on train only) ────────────────────
    print("\n[Step 5] Encoding features ...")
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test,  y_test  = preprocessor.transform(test_df)
    class_names = list(preprocessor.target_encoder.classes_)
    print(f"  Features: {X_train.shape[1]}  |  Classes: {class_names}")

    # ── Step 6: Build model ───────────────────────────────────
    print(f"\n[Step 6] Building model: {active_model_name} ...")
    model = build_model(active_model_name, model_cfg, data_cfg["random_state"])

    # ── Step 7: Train ─────────────────────────────────────────
    print("\n[Step 7] Training ...")
    train_result = train(
        model, X_train, y_train,
        cv_folds=eval_cfg["cv_folds"],
        scoring=eval_cfg["scoring"],
        random_state=data_cfg["random_state"],
    )

    # ── Step 8: Evaluate ──────────────────────────────────────
    print("\n[Step 8] Evaluating on test set ...")
    evaluate(
        train_result["model"], X_test, y_test,
        class_names=class_names,
        train_result=train_result,
        output_dir=output_cfg["model_dir"],
        report_filename=output_cfg["report_filename"],
    )

    # ── Step 9: Save ──────────────────────────────────────────
    print("\n[Step 9] Saving model ...")
    save_model(
        train_result["model"],
        output_dir=output_cfg["model_dir"],
        filename=output_cfg["model_filename"],
    )

    print(f"\n{'='*55}")
    print("  Pipeline complete. Outputs in saved_model/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
