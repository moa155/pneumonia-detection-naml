#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Bootstrap a fresh RunPod container with the project in /workspace.
#
# /workspace is the only path that survives RunPod's "Stop" action
# (the rest of the filesystem is overlayfs and gets wiped on stop/restart).
# This script makes sure the repo, data and outputs all live on /workspace
# so a Stop-then-Start cycle never loses work again.
#
# Usage on a fresh pod:
#   curl -sL https://raw.githubusercontent.com/moa155/NAML/main/Project_Pneumonia_Detection/scripts/bootstrap_pod.sh | bash
#   # or, if you already cloned somewhere:
#   bash scripts/bootstrap_pod.sh
#
# What it does:
#   1. If /workspace/Project_Pneumonia_Detection is missing, git-clone into it.
#   2. Symlink /NAML/Project_Pneumonia_Detection -> the workspace clone so
#      existing scripts/paths keep working.
#   3. Hand off to scripts/cloud_setup.sh for kaggle + dataset + preprocess.
# ----------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/moa155/NAML.git}"
SUBDIR="Project_Pneumonia_Detection"
WORKSPACE_DIR="/workspace/$SUBDIR"
LEGACY_LINK="/NAML/$SUBDIR"

banner() { echo; echo "=== $1 ==="; echo; }

# ---------- 1. Ensure repo is in /workspace ----------
banner "1/3: project on /workspace (survives pod stop)"

if [ ! -d "$WORKSPACE_DIR/.git" ]; then
    mkdir -p /workspace
    if [ -d /workspace/NAML/.git ]; then
        echo "  Found existing /workspace/NAML — using its $SUBDIR."
    else
        echo "  Cloning $REPO_URL into /workspace/NAML..."
        git clone --depth 1 "$REPO_URL" /workspace/NAML
    fi
    # Move/symlink the project subdirectory to /workspace/$SUBDIR for convenience
    if [ -d "/workspace/NAML/$SUBDIR" ]; then
        ln -sfn "/workspace/NAML/$SUBDIR" "$WORKSPACE_DIR"
    fi
fi
echo "  Project at: $WORKSPACE_DIR"

# ---------- 2. Backward-compat /NAML symlink ----------
banner "2/3: /NAML compat symlink"
mkdir -p /NAML
if [ ! -L "$LEGACY_LINK" ] && [ ! -d "$LEGACY_LINK" ]; then
    ln -sfn "$WORKSPACE_DIR" "$LEGACY_LINK"
    echo "  Linked $LEGACY_LINK -> $WORKSPACE_DIR"
elif [ -L "$LEGACY_LINK" ]; then
    echo "  $LEGACY_LINK already a symlink — leaving it."
else
    echo "  $LEGACY_LINK exists as a real directory — leaving it (rename it if you want the symlink)."
fi

# ---------- 3. Defer to cloud_setup.sh ----------
banner "3/3: handing off to cloud_setup.sh"
cd "$WORKSPACE_DIR"
echo "  Now run:"
echo "      KAGGLE_TOKEN=KGAT_... bash scripts/cloud_setup.sh --train"
echo "  (or set KAGGLE_USERNAME/KAGGLE_KEY for the legacy format)"
