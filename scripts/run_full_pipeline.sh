#!/usr/bin/env bash
# ----------------------------------------------------------------------
# End-to-end pipeline: train all three detectors, cache predictions,
# run robustness analyses. One command, suitable for cloud (H100) runs.
#
# Usage:
#   bash scripts/run_full_pipeline.sh                 # 40 epochs, defaults
#   EPOCHS=20 bash scripts/run_full_pipeline.sh       # override epochs
#   bash scripts/run_full_pipeline.sh --skip-train    # use existing checkpoints
#   bash scripts/run_full_pipeline.sh --skip-cache    # use existing preds
#
# All stages tee their output to results/pipeline_<stage>.log so a failed
# step does not lose the prior step's diagnostics.
# ----------------------------------------------------------------------
set -euo pipefail

# --- Defaults (override via env vars) ---
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

# --- Parse flags ---
SKIP_TRAIN=0
SKIP_CACHE=0
SKIP_ANALYSES=0
EXTRA_TRAIN_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --skip-train)    SKIP_TRAIN=1 ;;
        --skip-cache)    SKIP_CACHE=1 ;;
        --skip-analyses) SKIP_ANALYSES=1 ;;
        --no-bf16)       EXTRA_TRAIN_FLAGS="$EXTRA_TRAIN_FLAGS --no-bf16" ;;
        --fp32)          EXTRA_TRAIN_FLAGS="$EXTRA_TRAIN_FLAGS --no-amp" ;;
        --max-samples=*) EXTRA_TRAIN_FLAGS="$EXTRA_TRAIN_FLAGS --max-samples ${arg#--max-samples=}" ;;
        -h|--help)
            grep -E "^# " "$0" | sed 's/^# \?//'; exit 0 ;;
        *)
            echo "Unknown flag: $arg (try --help)"; exit 2 ;;
    esac
done

# --- Locate repo root (scripts/ is a child of root) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results checkpoints

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() {
    echo
    echo "===================================================================="
    echo "  $1"
    echo "  $(stamp)"
    echo "===================================================================="
}

# --- Pre-flight: data + GPU sanity ---
banner "Pre-flight"

if [ ! -f data/stage_2_train_labels.csv ]; then
    echo "ERROR: data/stage_2_train_labels.csv missing — pull the RSNA dataset first." >&2
    exit 3
fi
if [ ! -d data/stage_2_train_images_png ] || [ -z "$(ls -A data/stage_2_train_images_png 2>/dev/null)" ]; then
    echo "INFO: stage_2_train_images_png missing/empty — running preprocess (DICOM → PNG)..."
    python -m src.preprocess
fi

python - <<'PY'
import torch, os, sys
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.cuda.is_available()}  (devices={torch.cuda.device_count()})")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA — this script is meant for H100 / cloud runs.")
PY

echo
echo "  Hyperparameters:"
echo "    EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  LR=$LR  IMAGE_SIZE=$IMAGE_SIZE"
echo "    FREEZE_EPOCHS=$FREEZE_EPOCHS  VAL_FREQUENCY=$VAL_FREQUENCY  EARLY_STOPPING=$EARLY_STOPPING"
echo "    N_BOOT=$N_BOOT  NUM_WORKERS=$NUM_WORKERS  SEED=$SEED"
echo "    SKIP_TRAIN=$SKIP_TRAIN  SKIP_CACHE=$SKIP_CACHE  SKIP_ANALYSES=$SKIP_ANALYSES"
echo "    EXTRA_TRAIN_FLAGS='$EXTRA_TRAIN_FLAGS'"

T_START=$(date +%s)

# --- Stage 1: Train + evaluate + headline plots ---
if [ "$SKIP_TRAIN" -eq 0 ]; then
    banner "Stage 1/3: Train all 3 detectors ($EPOCHS epochs)"
    python -u main.py \
        --mode full \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --image-size "$IMAGE_SIZE" \
        --freeze-epochs "$FREEZE_EPOCHS" \
        --val-frequency "$VAL_FREQUENCY" \
        --early-stopping "$EARLY_STOPPING" \
        --scheduler cosine \
        --bf16 \
        --num-workers "$NUM_WORKERS" \
        --prefetch-factor "$PREFETCH_FACTOR" \
        --seed "$SEED" \
        $EXTRA_TRAIN_FLAGS \
        2>&1 | tee results/pipeline_train.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "Train stage FAILED — see results/pipeline_train.log"; exit 4; }
else
    banner "Stage 1/3: SKIPPED (--skip-train)"
fi

# --- Stage 2: Cache per-image predictions for downstream analyses ---
if [ "$SKIP_CACHE" -eq 0 ]; then
    banner "Stage 2/3: Cache predictions (TTA + Soft-NMS)"
    python -u scripts/cache_predictions.py 2>&1 | tee results/pipeline_cache.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "Cache stage FAILED — see results/pipeline_cache.log"; exit 5; }
else
    banner "Stage 2/3: SKIPPED (--skip-cache)"
fi

# --- Stage 3: Robustness analyses + LaTeX fragment for the report ---
if [ "$SKIP_ANALYSES" -eq 0 ]; then
    banner "Stage 3/3: Robustness analyses (n_boot=$N_BOOT)"
    python -u scripts/run_analyses.py --n-boot "$N_BOOT" --seed "$SEED" 2>&1 | tee results/pipeline_analyses.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "Analyses stage FAILED — see results/pipeline_analyses.log"; exit 6; }
else
    banner "Stage 3/3: SKIPPED (--skip-analyses)"
fi

T_END=$(date +%s)
ELAPSED_MIN=$(( (T_END - T_START) / 60 ))

banner "DONE in ${ELAPSED_MIN} min"

echo "Artefacts:"
echo "  Checkpoints       : checkpoints/{fcos,fcos_paper,retinanet,faster_rcnn}_best.pth"
echo "  Headline metrics  : results/all_metrics.json"
echo "  Headline plots    : results/{training_loss,val_ap_over_epochs,ap_comparison,"
echo "                       ar_comparison,pr_curve,classification_metrics,ap_vs_iou,"
echo "                       epoch_times,detection_samples}.{png,pdf}"
echo "  Cached preds      : results/predictions/{fcos,fcos_paper,retinanet,faster_rcnn}_preds.pt"
echo "  Robustness JSON   : results/all_metrics_v2.json"
echo "  Robustness LaTeX  : results/analyses.tex  (\\input{}'d by report.tex)"
echo "  Robustness plots  : results/{froc,calibration,ap_bootstrap}.{png,pdf}"
echo
echo "Next steps on your laptop:"
echo "  1. Bundle on pod  : tar --exclude='results/predictions' -czf /workspace/results.tgz results/"
echo "  2. Download via Jupyter Lab (port 8888) — direct SCP from ssh.runpod.io is not supported."
echo "  3. Pull checkpoints: scp -P <tcp-port> root@<tcp-host>:/.../checkpoints/*_best.pth checkpoints_pod/"
echo "                       (or use the litter.catbox.moe URLs in results/backup_urls.txt)"
echo "  4. Extract on Mac : tar xzf ~/Downloads/results.tgz -C results_pod/"
echo "  5. Recompile report + presentation; commit."
