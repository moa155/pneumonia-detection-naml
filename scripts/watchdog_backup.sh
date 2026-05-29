#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Periodic backup of training artefacts to transfer.sh (or similar
# anonymous file hosts). Survives pod eviction: even if the pod dies,
# the URLs printed by this script remain downloadable for 14 days.
#
# Usage:
#   bash scripts/watchdog_backup.sh &              # run in background
#   bash scripts/watchdog_backup.sh --once         # single shot
#
# The script writes one line per upload to results/backup_urls.txt:
#   <ISO timestamp>  <stage>  <size>  <URL>
#
# Stages:
#   results-small  : results/*.json, *.png, *.pdf, *.tex, history files (~50 MB)
#   checkpoints    : *_best.pth                                  (~450 MB)
#   predictions    : results/predictions/*.pt                    (~100 MB)
#
# Defaults:
#   INTERVAL_SEC=600   (10 min between backups of the small stuff)
#   HOST=https://transfer.sh
# ----------------------------------------------------------------------
set -uo pipefail

ONCE=0
for arg in "$@"; do
    case "$arg" in
        --once) ONCE=1 ;;
        -h|--help) grep -E "^#( |$)" "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown flag: $arg"; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL_SEC="${INTERVAL_SEC:-600}"
LITTERBOX_TTL="${LITTERBOX_TTL:-72h}"    # 1h, 12h, 24h, 72h
LOG="results/backup_urls.txt"
mkdir -p results

upload() {
    local label="$1" tgz="$2"
    [ -f "$tgz" ] || return 0
    local size_h
    size_h=$(du -h "$tgz" | cut -f1)
    local fname
    fname="$(basename "$tgz")"
    # litterbox.catbox.moe: anonymous, up to 1GB, returns plain URL on success.
    local url
    url=$(curl --silent --max-time 900 \
        -F "reqtype=fileupload" \
        -F "time=$LITTERBOX_TTL" \
        -F "fileToUpload=@${tgz}" \
        https://litterbox.catbox.moe/resources/internals/api.php 2>/dev/null)
    if [ -z "$url" ] || [[ "$url" != http* ]]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $label  $size_h  UPLOAD_FAILED  ${url:-empty}" | tee -a "$LOG"
        return 1
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $label  $size_h  $url" | tee -a "$LOG"
}

snapshot_small() {
    local out=/tmp/results-small.tgz
    tar -czf "$out" \
        --exclude='results/predictions' \
        --exclude='results/*.log' \
        results/ 2>/dev/null
    upload "results-small" "$out"
}

snapshot_checkpoints() {
    local out=/tmp/checkpoints-best.tgz
    if ls checkpoints/*_best.pth >/dev/null 2>&1; then
        tar -czf "$out" checkpoints/*_best.pth 2>/dev/null
        upload "checkpoints" "$out"
    fi
}

snapshot_predictions() {
    local out=/tmp/predictions.tgz
    if [ -d results/predictions ] && ls results/predictions/*.pt >/dev/null 2>&1; then
        tar -czf "$out" results/predictions/ 2>/dev/null
        upload "predictions" "$out"
    fi
}

snapshot_all() {
    snapshot_small
    # Heavy stuff only on cycles where it exists (cheap if missing)
    [ -f checkpoints/fcos_best.pth ] && snapshot_checkpoints
    [ -f results/predictions/targets.pt ] && snapshot_predictions
}

echo "watchdog_backup: interval=${INTERVAL_SEC}s  host=litterbox.catbox.moe  ttl=$LITTERBOX_TTL  log=$LOG"
echo "watchdog_backup: started at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

if [ "$ONCE" -eq 1 ]; then
    snapshot_all
    exit 0
fi

while true; do
    snapshot_all
    # Exit cleanly when pipeline.out shows the DONE marker (one last backup)
    if grep -q "DONE in " pipeline.out 2>/dev/null; then
        echo "watchdog_backup: pipeline DONE detected, final backup..." | tee -a "$LOG"
        snapshot_all
        echo "watchdog_backup: exiting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
        exit 0
    fi
    sleep "$INTERVAL_SEC"
done
