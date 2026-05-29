#!/usr/bin/env bash
# ----------------------------------------------------------------------
# One-shot cloud setup for the pneumonia detection rerun.
#
# Idempotent — every step checks if it's already done and skips.
#
# Usage:
#   # Option A: pass Kaggle credentials via env vars (cleanest)
#   KAGGLE_USERNAME=youruser KAGGLE_KEY=yourkey bash scripts/cloud_setup.sh
#
#   # Option B: write ~/.kaggle/kaggle.json yourself, then:
#   bash scripts/cloud_setup.sh
#
#   # Add --train to also launch the training pipeline at the end
#   bash scripts/cloud_setup.sh --train
#
# Steps:
#   1. Verify smoke check passes
#   2. Set up ~/.kaggle/kaggle.json (if KAGGLE_USERNAME/KAGGLE_KEY set)
#   3. Download + unzip RSNA dataset (skip if already there)
#   4. Convert DICOM → PNG (skip if already done)
#   5. Optionally launch the full training pipeline in background
# ----------------------------------------------------------------------
set -euo pipefail

LAUNCH_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --train)   LAUNCH_TRAIN=1 ;;
        -h|--help) grep -E "^#( |$)" "$0" | sed 's/^# \?//'; exit 0 ;;
        *)         echo "Unknown flag: $arg (try --help)"; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

banner() { echo; echo "=== $1 ==="; echo; }

# ---------- 1. Smoke check ----------
banner "Step 1/5: smoke check"
if ! python3 scripts/smoke_check.py; then
    echo "Smoke check failed — fix before continuing." >&2
    exit 3
fi

# ---------- 2. Kaggle credentials ----------
banner "Step 2/5: Kaggle credentials"
mkdir -p ~/.kaggle

# Modern Kaggle uses a single API token (KGAT_*) stored in ~/.kaggle/access_token
# OR legacy {"username":..., "key":...} in ~/.kaggle/kaggle.json.
if [ -n "${KAGGLE_TOKEN:-}" ]; then
    echo "$KAGGLE_TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
    rm -f ~/.kaggle/kaggle.json
    echo "  Wrote ~/.kaggle/access_token from KAGGLE_TOKEN env var (modern format)."
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    rm -f ~/.kaggle/access_token
    echo "  Wrote ~/.kaggle/kaggle.json from env vars (legacy format)."
elif [ -f ~/.kaggle/access_token ] || [ -f ~/.kaggle/kaggle.json ]; then
    echo "  Existing Kaggle credentials found — keeping them."
else
    echo "ERROR: no Kaggle credentials." >&2
    echo "       Re-run as:  KAGGLE_TOKEN=KGAT_... bash $0   (modern, recommended)" >&2
    echo "       Or:         KAGGLE_USERNAME=foo KAGGLE_KEY=bar bash $0   (legacy)" >&2
    exit 4
fi

# Install kaggle CLI if missing and verify auth.
# IMPORTANT: this section disables pipefail temporarily because `kaggle
# competitions list | head -3` always produces SIGPIPE on head closing,
# which under `set -euo pipefail` would exit the script silently.
python3 -c "import kaggle" 2>/dev/null || pip install --quiet kaggle
set +e
set +o pipefail
KAGGLE_OUT="$(kaggle competitions list 2>&1 | head -10)"
KAGGLE_RC=$?
set -e
set -o pipefail
if echo "$KAGGLE_OUT" | grep -qiE "401|unauthorized|authentication required|not authenticated"; then
    echo "ERROR: Kaggle credentials rejected — token may be expired or in the wrong format." >&2
    echo "       Output was:" >&2
    echo "$KAGGLE_OUT" | sed 's/^/         /' >&2
    exit 5
fi
echo "  Kaggle auth OK (rc=$KAGGLE_RC)."

# ---------- 3. Dataset ----------
banner "Step 3/5: RSNA dataset"
mkdir -p data
if [ -f data/stage_2_train_labels.csv ] && [ -d data/stage_2_train_images ] && \
   [ "$(ls data/stage_2_train_images 2>/dev/null | wc -l)" -ge 26000 ]; then
    echo "  Dataset already present ($(ls data/stage_2_train_images | wc -l) images) — skipping download."
else
    cd data
    if [ ! -f rsna-pneumonia-detection-challenge.zip ]; then
        kaggle competitions download -c rsna-pneumonia-detection-challenge
    else
        echo "  zip already downloaded — skipping."
    fi
    echo "  Unzipping (this takes a couple of minutes)..."
    unzip -q -o rsna-pneumonia-detection-challenge.zip
    cd ..
    echo "  $(ls data/stage_2_train_images | wc -l) DICOM images extracted."
fi

# ---------- 4. DICOM → PNG ----------
banner "Step 4/5: DICOM → PNG"
PNG_DIR=data/stage_2_train_images_png
if [ -d "$PNG_DIR" ] && [ "$(ls "$PNG_DIR" 2>/dev/null | wc -l)" -ge 26000 ]; then
    echo "  PNG directory already populated ($(ls "$PNG_DIR" | wc -l) images) — skipping preprocess."
else
    python3 -m src.preprocess
    echo "  Preprocess complete: $(ls "$PNG_DIR" | wc -l) PNGs."
fi

# ---------- 4b. Persistence check ----------
banner "Step 4b/5: persistence check"
case "$REPO_ROOT" in
    /workspace/*)
        echo "  Project lives on /workspace — survives pod stop. OK."
        ;;
    *)
        echo "  WARNING: project is NOT under /workspace ($REPO_ROOT)."
        echo "  Run scripts/bootstrap_pod.sh first if you want everything to"
        echo "  survive a pod stop. Continuing anyway."
        ;;
esac

# ---------- 5. Optional: launch training + watchdog ----------
banner "Step 5/5: training pipeline + offsite backups"
if [ "$LAUNCH_TRAIN" -eq 1 ]; then
    echo "  Launching scripts/run_full_pipeline.sh in background..."
    rm -f pipeline.out
    nohup bash scripts/run_full_pipeline.sh > pipeline.out 2>&1 &
    PID=$!
    disown
    echo "  Training PID: $PID"

    echo "  Launching scripts/watchdog_backup.sh (offsite backups every 10 min)..."
    nohup bash scripts/watchdog_backup.sh > watchdog.out 2>&1 &
    WPID=$!
    disown
    echo "  Watchdog PID: $WPID  (URLs will be appended to results/backup_urls.txt)"
    echo
    echo "  Monitor training:   tail -f $(pwd)/pipeline.out"
    echo "  Check overall state: bash scripts/status.sh"
    echo "  Wait for the line: 'DONE in <N> min' (ETA ~3-4h on 3x H100)."
else
    echo "  Setup complete. To launch training + backups:"
    echo "      nohup bash scripts/run_full_pipeline.sh > pipeline.out 2>&1 & disown"
    echo "      nohup bash scripts/watchdog_backup.sh > watchdog.out 2>&1 & disown"
    echo "      tail -f pipeline.out"
    echo
    echo "  Status at any time: bash scripts/status.sh"
    echo "  Or re-run this script with --train to start both automatically."
fi

echo
echo "=== ALL SETUP STEPS DONE ==="
