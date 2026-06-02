g#!/bin/bash
# ============================================================
# run.sh — ElderGuard Analytics Pipeline Runner
#
# Usage:
#   ./run.sh              # run pipeline normally
#   ./run.sh --smote      # run with SMOTE oversampling
# ============================================================

set -e  # stop immediately if any command fails

echo "================================================"
echo "  ElderGuard Analytics — ML Pipeline"
echo "================================================"

# Install dependencies
pip install -r requirements.txt

# Run the pipeline, pass any arguments through (e.g. --smote)
python src/main.py "$@"

echo ""
echo "================================================"
echo "  Done. Check reports/ and saved_model/"
echo "================================================"
