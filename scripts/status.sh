#!/usr/bin/env bash
# ----------------------------------------------------------------------
# One-shot status check for the pneumonia detection pipeline on a pod.
#
# Prints, in order:
#   - storage location (must be /workspace to survive pod stop)
#   - data + dataset status
#   - checkpoints present (per model, with epoch + AP@0.5)
#   - training history files present (per model, with epoch count)
#   - predictions cache (per model)
#   - analysis outputs present (plots, tables, json)
#   - running training/analysis processes + GPU usage
#   - disk usage
#   - last 5 lines of backup URL log
#
# Run any time:
#   bash scripts/status.sh
# ----------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m" "$*"; }
red() { printf "\033[31m%s\033[0m" "$*"; }
yellow() { printf "\033[33m%s\033[0m" "$*"; }

# ------- 1. Storage location -------
bold "=== 1. Storage location ==="
case "$REPO_ROOT" in
    /workspace/*) echo "  Project on $(green '/workspace') — persistent across pod stop. OK."  ;;
    *)            echo "  Project on $(red "$REPO_ROOT") — NOT /workspace. Data will be lost on pod stop. Run scripts/bootstrap_pod.sh." ;;
esac
df -h "$REPO_ROOT" | tail -1 | awk '{printf "  Filesystem: %s  Used: %s / %s (%s)\n", $1, $3, $2, $5}'
echo

# ------- 2. Dataset -------
bold "=== 2. Dataset ==="
if [ -d data/stage_2_train_images_png ]; then
    PNG_COUNT=$(ls data/stage_2_train_images_png 2>/dev/null | wc -l)
    if [ "$PNG_COUNT" -ge 26000 ]; then
        echo "  PNGs: $(green "$PNG_COUNT") ready in data/stage_2_train_images_png"
    else
        echo "  PNGs: $(yellow "$PNG_COUNT/26684") — preprocess incomplete"
    fi
elif [ -d data/stage_2_train_images ]; then
    DICOM_COUNT=$(ls data/stage_2_train_images 2>/dev/null | wc -l)
    echo "  DICOMs: $(yellow "$DICOM_COUNT") downloaded, $(red "PNGs missing") — run python -m src.preprocess"
else
    echo "  $(red "No dataset found") — run scripts/cloud_setup.sh"
fi
echo

# ------- 3. Checkpoints -------
bold "=== 3. Checkpoints ==="
for m in fcos fcos_paper retinanet faster_rcnn; do
    f="checkpoints/${m}_best.pth"
    if [ -f "$f" ]; then
        info=$(python3 -c "
import torch, sys
try:
    ck = torch.load('$f', map_location='cpu', weights_only=False)
    print(f\"epoch={ck.get('epoch','?'):>2}  AP@0.5={ck.get('ap50',0)*100:5.2f}%  size={$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")//1048576}MB\")
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
" 2>/dev/null)
        printf "  %-15s $(green '✓') %s\n" "${m}_best" "$info"
    else
        printf "  %-15s $(red '✗') missing\n" "${m}_best"
    fi
done
echo

# ------- 4. Training histories -------
bold "=== 4. Training histories (per-epoch curves) ==="
for m in fcos fcos_paper retinanet faster_rcnn; do
    f="results/${m}_history.json"
    if [ -f "$f" ]; then
        info=$(python3 -c "
import json
h = json.load(open('$f'))
n = len(h.get('train_losses', []))
vals = [m for m in h.get('val_metrics', []) if m]
max_ap = max((m.get('AP@0.5', 0) for m in vals), default=0) * 100
print(f'epochs={n:>2}  val_evals={len(vals):>2}  max_val_AP={max_ap:5.2f}%')
" 2>/dev/null)
        printf "  %-15s $(green '✓') %s\n" "${m}_history" "$info"
    else
        printf "  %-15s $(red '✗') missing\n" "${m}_history"
    fi
done
echo

# ------- 5. Predictions cache -------
bold "=== 5. Predictions cache ==="
PRED_DIR=results/predictions
if [ -d "$PRED_DIR" ]; then
    for m in fcos fcos_paper retinanet faster_rcnn; do
        f="$PRED_DIR/${m}_preds.pt"
        if [ -f "$f" ]; then
            size_mb=$(( $(stat -c %s "$f" 2>/dev/null || stat -f %z "$f") / 1048576 ))
            printf "  %-25s $(green '✓') %sMB\n" "${m}_preds.pt" "$size_mb"
        else
            printf "  %-25s $(red '✗') missing\n" "${m}_preds.pt"
        fi
    done
    if [ -f "$PRED_DIR/targets.pt" ]; then
        printf "  %-25s $(green '✓')\n" "targets.pt + val_index.json"
    fi
else
    echo "  $(red "$PRED_DIR not found") — run scripts/cache_predictions.py"
fi
echo

# ------- 6. Analyses output -------
bold "=== 6. Analyses output ==="
for f in results/all_metrics.json results/all_metrics_v2.json results/analyses.tex \
         results/ap_bootstrap.png results/calibration.png results/froc.png \
         results/val_ap_over_epochs.png results/detection_samples.png; do
    if [ -f "$f" ]; then
        printf "  %-40s $(green '✓')\n" "$f"
    else
        printf "  %-40s $(red '✗') missing\n" "$f"
    fi
done
echo

# ------- 7. Running processes -------
bold "=== 7. Running processes ==="
if ps -eo pid,etime,%cpu,%mem,state,command >/dev/null 2>&1; then
    PS_FMT="pid,etime,%cpu,%mem,state,command"
else
    PS_FMT="pid,etime,%cpu,%mem,state,args"
fi
ps_out=$(ps -eo $PS_FMT 2>/dev/null | grep -E "(python3?.*main\.py|run_analyses|run_full_pipeline|cache_predictions|watchdog_backup)" | grep -v grep)
if [ -n "$ps_out" ]; then
    echo "$ps_out" | head -10
else
    echo "  $(yellow 'No training/analysis process running.')"
fi
echo

# ------- 8. GPU usage -------
bold "=== 8. GPU usage ==="
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r idx name util mem_used mem_total temp; do
        printf "  GPU%s %s: util=%s mem=%s/%s temp=%s\n" "$idx" "$name" "$util" "$mem_used" "$mem_total" "$temp"
    done
else
    echo "  nvidia-smi not available"
fi
echo

# ------- 9. Backup URL log -------
bold "=== 9. Last 5 successful watchdog backups ==="
if [ -f results/backup_urls.txt ]; then
    grep -E "https://litter\.catbox\.moe" results/backup_urls.txt | tail -5 | awk '{printf "  %s  %s  %s  %s\n", $1, $2, $3, $4}'
else
    echo "  $(yellow 'No backup_urls.txt') — watchdog never ran. Start it with: bash scripts/watchdog_backup.sh &"
fi
echo

# ------- 10. Workspace usage -------
bold "=== 10. Workspace disk usage ==="
if [ -d /workspace ]; then
    du -sh /workspace 2>/dev/null | awk '{printf "  /workspace: %s used\n", $1}'
fi
du -sh "$REPO_ROOT" 2>/dev/null | awk '{printf "  Repo:       %s used\n", $1}'
