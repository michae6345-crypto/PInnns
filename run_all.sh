#!/usr/bin/env bash
# =============================================================
# run_all.sh  —  TMC-PINN experiment launcher
# =============================================================
# USAGE:
#   bash run_all.sh              # reaction only (default)
#   bash run_all.sh reaction wave
#   bash run_all.sh all          # all 4 PDEs (28 total runs)
#
# All 7 conditions use the same epoch count for a fair comparison.
# 2000 epochs = sufficient for L-BFGS convergence on reaction.
# Adam conditions may not fully converge — this is expected and
# noted in the paper methods section as a fixed compute budget.
#
# Estimated time on H100 (reaction, all 7 conditions): ~60-90 min total
#
# SETTINGS — only edit these:
DEVICE="cuda:0"
MODEL="PINN"
TOTAL_EPOCHS=2000
SWITCH_EPOCH=1000    # halfway point, same for all switching runs
OUT_DIR="./results"
# =============================================================

set -e

if [ "$1" = "all" ]; then
    PDELIST=("reaction" "wave" "ac" "convection")
elif [ $# -eq 0 ]; then
    PDELIST=("reaction")
else
    PDELIST=("$@")
fi

echo ""
echo "============================================="
echo "  TMC-PINN Experiment Runner"
echo "  PDEs    : ${PDELIST[*]}"
echo "  Epochs  : $TOTAL_EPOCHS (all conditions — fixed budget)"
echo "  Switch  : epoch $SWITCH_EPOCH"
echo "  Device  : $DEVICE"
echo "  Output  : $OUT_DIR"
echo "============================================="

run() {
    local PDE=$1
    local DS=$2
    local OS=$3
    local DW=$4
    local OW=$5

    if [ -n "$DW" ]; then
        LABEL="${DS}${OS}_to_${DW}${OW}"
        EXTRA="--dtype_switch $DW --optim_switch $OW --switch_epoch $SWITCH_EPOCH"
    else
        LABEL="${DS}${OS}"
        EXTRA=""
    fi

    echo ""
    echo "--- [$PDE] $LABEL ---"
    python train_pinn.py \
        --pde          "$PDE" \
        --model        "$MODEL" \
        --device       "$DEVICE" \
        --total_epochs "$TOTAL_EPOCHS" \
        --out_dir      "$OUT_DIR" \
        --dtype_start  "$DS" \
        --optim_start  "$OS" \
        $EXTRA
}

for PDE in "${PDELIST[@]}"; do
    echo ""
    echo "============================================="
    echo "  PDE: $PDE"
    echo "============================================="

    # 4 static baselines
    run $PDE fp64 lbfgs
    run $PDE fp64 adam
    run $PDE fp32 lbfgs
    run $PDE fp32 adam

    # 3 switching conditions
    run $PDE fp32 adam  fp64 adam
    run $PDE fp32 lbfgs fp64 lbfgs
    run $PDE fp32 adam  fp64 lbfgs   # key novel condition

done

echo ""
echo "============================================="
echo "  All runs complete!"
echo "  Results: $OUT_DIR"
echo ""
echo "  Generate paper figures:"
echo "    python aggregate_results.py --results_dir $OUT_DIR --out_dir ./paper_figures"
echo ""
echo "  IMPORTANT — back up before Lambda shuts down:"
echo "    zip -r results_backup_\$(date +%Y%m%d_%H%M).zip $OUT_DIR"
echo "============================================="
