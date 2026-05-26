#!/bin/bash
# ============================================================
# run.sh — ElderGuard Analytics Pipeline Runner
#
# Usage:
#   ./run.sh                            # default config, random_forest
#   ./run.sh --model xgboost            # override model
#   ./run.sh --model logistic_regression
#   ./run.sh --config custom.yaml       # custom config file
# ============================================================

set -e  # exit immediately on error

echo "================================================"
echo "  ElderGuard Analytics — ML Pipeline"
echo "================================================"

# Pass all arguments through to pipeline.py
python pipeline.py "$@"

echo ""
echo "================================================"
echo "  Done. Check saved_model/ for outputs."
echo "================================================"
