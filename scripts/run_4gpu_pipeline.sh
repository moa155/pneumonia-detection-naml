#!/usr/bin/env bash
# ----------------------------------------------------------------------
# End-to-end pipeline for a 4-GPU pod (4x H100).
#
# Trains all FOUR models in parallel (one per GPU):
#   GPU0: FCOS              (Adam, our baseline recipe)
#   GPU1: RetinaNet         (Adam)
#   GPU2: Faster R-CNN      (Adam)
#   GPU3: FCOS paper SGD    (SGD+momentum, paper's ablation)
#
# Then sequentially: cache predictions -> run robustness analyses.
#
# Usage:
#   bash scripts/run_4gpu_pipeline.sh                  # 40 epochs, defaults
#   EPOCHS=20 bash scripts/run_4gpu_pipeline.sh        # override epochs
#   bash scripts/run_4gpu_pipeline.sh --skip-train     # skip stage 1
# ----------------------------------------------------------------------
set -uo pipefail

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-3}"
VAL_FREQUENCY="${VAL_FREQUENCY:-4}"
EARLY_STOPPING="${EARLY_STOPPING:-0}"
N_BOOT="${N_BOOT:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SEED="${SEED:-42}"

# Paper recipe overrides for fcos_paper
PAPER_LR="${PAPER_LR:-5e-3}"
PAPER_WD="${PAPER_WD:-1.6e-4}"
PAPER_MOM="${PAPER_MOM:-0.9}"

SKIP_TRAIN=0; SKIP_CACHE=0; SKIP_ANALYSES=0
for arg in "$@"; do
    case "$arg" in
        --skip-train)    SKIP_TRAIN=1 ;;
        --skip-cache)    SKIP_CACHE=1 ;;
        --skip-analyses) SKIP_ANALYSES=1 ;;
        -h|--help) grep -E "^# ?" "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown flag: $arg"; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p results checkpoints

banner() { echo; echo "===================================================================="; echo "  $1"; echo "  $(date '+%Y-%m-%d %H:%M:%S')"; echo "===================================================================="; }

# --- Pre-flight ---
banner "Pre-flight"
[ -f data/stage_2_train_labels.csv ] || { echo "ERROR: RSNA dataset missing — run scripts/cloud_setup.sh first."; exit 3; }

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "  GPUs detected: $N_GPUS"
if [ "$N_GPUS" -lt 4 ]; then
    echo "  WARNING: this script expects 4 GPUs. With fewer GPUs, models will share devices and may OOM."
fi

case "$REPO_ROOT" in
    /workspace/*) echo "  Project on /workspace — survives pod stop. OK." ;;
    *)            echo "  WARNING: project is NOT under /workspace. Stop is unsafe. Continuing anyway." ;;
esac

T_START=$(date +%s)

# --- Stage 1: 4 trainings in parallel ---
if [ "$SKIP_TRAIN" -eq 0 ]; then
    banner "Stage 1/3: train 4 models in parallel (${EPOCHS} epochs)"

    COMMON_FLAGS=" --mode train --epochs $EPOCHS --batch-size $BATCH_SIZE
        --image-size $IMAGE_SIZE --freeze-epochs $FREEZE_EPOCHS
        --val-frequency $VAL_FREQUENCY --early-stopping $EARLY_STOPPING
        --scheduler cosine --bf16
        --num-workers $NUM_WORKERS --prefetch-factor $PREFETCH_FACTOR
        --seed $SEED"

    declare -a PIDS=()

    # GPU0: FCOS (Adam)
    ( CUDA_VISIBLE_DEVICES=0 python -u main.py \
            $COMMON_FLAGS \
            --model fcos --lr $LR --optimizer adam \
            --device cuda:0 \
            > results/train_fcos.log 2>&1 ) &
    PIDS+=($!)

    # GPU1: RetinaNet (Adam)
    ( CUDA_VISIBLE_DEVICES=1 python -u main.py \
            $COMMON_FLAGS \
            --model retinanet --lr $LR --optimizer adam \
            --device cuda:0 \
            > results/train_retinanet.log 2>&1 ) &
    PIDS+=($!)

    # GPU2: Faster R-CNN (Adam)
    ( CUDA_VISIBLE_DEVICES=2 python -u main.py \
            $COMMON_FLAGS \
            --model faster_rcnn --lr $LR --optimizer adam \
            --device cuda:0 \
            > results/train_faster_rcnn.log 2>&1 ) &
    PIDS+=($!)

    # GPU3: FCOS paper SGD ablation
    ( CUDA_VISIBLE_DEVICES=3 python -u main.py \
            $COMMON_FLAGS \
            --model fcos --lr $PAPER_LR --optimizer sgd --momentum $PAPER_MOM \
            --checkpoint-suffix _paper \
            --device cuda:0 \
            > results/train_fcos_paper.log 2>&1 ) &
    PIDS+=($!)

    echo "  Launched 4 training PIDs: ${PIDS[*]}"
    echo "  Monitor any: tail -f results/train_<model>.log"
    echo "  Status:      bash scripts/status.sh"

    # Wait for all 4 to finish (or fail)
    FAILED=()
    NAMES=("fcos" "retinanet" "faster_rcnn" "fcos_paper")
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            FAILED+=("${NAMES[$i]}")
        fi
    done

    if [ "${#FAILED[@]}" -gt 0 ]; then
        echo "ERROR: training failed for: ${FAILED[*]} — check results/train_<model>.log"
        exit 4
    fi

    echo "  All 4 trainings finished cleanly."
else
    banner "Stage 1/3: SKIPPED (--skip-train)"
fi

# --- Stage 2: cache predictions ---
if [ "$SKIP_CACHE" -eq 0 ]; then
    banner "Stage 2/3: cache predictions (TTA + Soft-NMS)"
    CUDA_VISIBLE_DEVICES=0 python -u scripts/cache_predictions.py 2>&1 | tee results/pipeline_cache.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "Cache stage FAILED — see results/pipeline_cache.log"; exit 5; }
else
    banner "Stage 2/3: SKIPPED (--skip-cache)"
fi

# --- Stage 3: analyses ---
if [ "$SKIP_ANALYSES" -eq 0 ]; then
    banner "Stage 3/3: robustness analyses (n_boot=$N_BOOT)"
    python -u scripts/run_analyses.py --n-boot "$N_BOOT" --seed "$SEED" \
        --models fcos fcos_paper retinanet faster_rcnn \
        2>&1 | tee results/pipeline_analyses.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "Analyses stage FAILED — see results/pipeline_analyses.log"; exit 6; }
else
    banner "Stage 3/3: SKIPPED (--skip-analyses)"
fi

T_END=$(date +%s)
ELAPSED_MIN=$(( (T_END - T_START) / 60 ))

banner "DONE in ${ELAPSED_MIN} min"

echo "Artefacts:"
echo "  Checkpoints     : checkpoints/{fcos,fcos_paper,retinanet,faster_rcnn}_best.pth"
echo "  Histories       : results/{fcos,fcos_paper,retinanet,faster_rcnn}_history.json"
echo "  Headline plots  : results/{training_loss,val_ap_over_epochs,ap_comparison,...}.{png,pdf}"
echo "  Predictions     : results/predictions/{fcos,fcos_paper,retinanet,faster_rcnn}_preds.pt"
echo "  Robustness JSON : results/all_metrics_v2.json"
echo "  Robustness LaTeX: results/analyses.tex"
echo
echo "Next:"
echo "  Status check  : bash scripts/status.sh"
echo "  Bundle to /ws : tar --exclude='results/predictions' -czf /workspace/results.tgz results/"
echo "  Download via Jupyter Lab (port 8888) → file browser → results.tgz"
