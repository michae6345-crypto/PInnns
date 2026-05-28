#!/usr/bin/env bash
# =============================================================
# setup.sh  —  One-time Lambda environment setup
# Run this once after cloning the repo on a new Lambda instance.
# =============================================================

set -e

echo "Installing Python dependencies..."
pip install torch numpy matplotlib tqdm pandas scipy --break-system-packages -q

echo "Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  PyTorch: {torch.__version__}')
else:
    print('  WARNING: No GPU detected — runs will be slow on CPU')
"

echo "Verifying repo files..."
REQUIRED=("train_pinn.py" "run_all.sh" "aggregate_results.py" "util.py" "model_dict.py")
MISSING=()
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
        MISSING+=("$f")
    fi
done

if [ ${#MISSING[@]} -ne 0 ]; then
    echo "  WARNING: Missing files: ${MISSING[*]}"
    echo "  Make sure util.py and model_dict.py are in the repo."
else
    echo "  All required files present."
fi

echo ""
echo "Setup complete. Run experiments with:"
echo "  bash run_all.sh                 # reaction only"
echo "  bash run_all.sh reaction wave   # two PDEs"
echo "  bash run_all.sh all             # all PDEs"
