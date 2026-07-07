#!/usr/bin/env python3
# The classic comparison plots run_analyses.py skips. Needs history JSONs + cached preds.
# Run from the project root.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.visualize as viz
from src.evaluate import compute_metrics

ORDER = ["fcos", "fcos_paper", "retinanet", "faster_rcnn"]


def maybe_build_all_metrics(out_dir: Path, preds_by_model, targets):
    """Build old-style all_metrics.json (precision/recall arrays) for the legacy plots."""
    out_path = out_dir / "all_metrics.json"
    am = {}
    if out_path.exists():
        try:
            am = json.load(open(out_path))
        except Exception:
            am = {}
    changed = False
    for name in ORDER:
        if name not in preds_by_model:
            continue
        if name in am and "AP@0.5" in am[name]:
            continue
        print(f"  computing old-style metrics for {name}...")
        m = compute_metrics(preds_by_model[name], targets)
        am[name] = m
        changed = True
    if changed:
        with open(out_path, "w") as f:
            json.dump(am, f)
        print(f"  Saved: {out_path}")
    else:
        print(f"  all_metrics.json already complete — skipping")


def cleanup(preds_list, targets, youden_thr, iou_thresh=0.5):
    """Filter dets for the sample figure: keep top-n_gt on positives, Youden-threshold on negatives."""
    out = []
    for p, t in zip(preds_list, targets):
        n_gt = len(t["boxes"])
        boxes, scores = p["boxes"], p["scores"]
        if len(boxes) == 0:
            out.append(p)
            continue
        keep = nms(boxes, scores, iou_thresh)
        if n_gt > 0:
            keep = keep[:n_gt]
        else:
            keep = keep[scores[keep] > youden_thr]
        out.append({"boxes": boxes[keep], "scores": scores[keep], "labels": p["labels"][keep]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pred-dir", default="results/predictions")
    parser.add_argument("--image-dir", default="data/stage_2_train_images_png")
    args = parser.parse_args()

    out = Path(args.results_dir)
    pred = Path(args.pred_dir)
    img_dir = Path(args.image_dir)

    # history-driven plots
    histories = {}
    for n in ORDER:
        p = out / f"{n}_history.json"
        if p.exists():
            histories[n] = json.load(open(p))
        else:
            print(f"  WARN: {p} missing — skipping its curve")
    print(f"Loaded histories: {list(histories.keys())}")

    if histories:
        viz.plot_training_losses(histories, out)
        viz.plot_learning_rates(histories, out)
        viz.plot_val_ap_over_epochs(histories, out)
        viz.plot_epoch_times(histories, out)

    # prediction-driven plots
    targets_path = pred / "targets.pt"
    if not targets_path.exists():
        print(f"  ERROR: {targets_path} missing — cannot make prediction plots.")
        return
    targets = torch.load(targets_path, weights_only=False)

    preds_by_model = {}
    for n in ORDER:
        p = pred / f"{n}_preds.pt"
        if p.exists():
            preds_by_model[n] = torch.load(p, weights_only=False)
        else:
            print(f"  WARN: {p} missing — skipping {n}")
    print(f"Loaded preds: {list(preds_by_model.keys())}")

    if preds_by_model:
        viz.plot_ap_vs_iou(preds_by_model, targets, out)
        maybe_build_all_metrics(out, preds_by_model, targets)
        try:
            am = json.load(open(out / "all_metrics.json"))
            all_metrics = {k: am[k] for k in ORDER if k in am}  # canonical order
            viz.plot_ap_comparison(all_metrics, out)
            viz.plot_ar_comparison(all_metrics, out)
            viz.plot_pr_curves(all_metrics, out)
            viz.plot_classification_metrics(all_metrics, out)
            print("  Saved: ap_comparison/ar_comparison/pr_curve/classification_metrics")
        except Exception as e:
            print(f"  WARN: legacy plots skipped ({e})")

    # detection samples: needs raw PNGs + Youden thresholds from metrics_v2
    metrics_v2 = out / "all_metrics_v2.json"
    if not metrics_v2.exists() or not img_dir.exists():
        print("  WARN: detection_samples skipped (need all_metrics_v2.json + raw PNGs)")
        return

    am = json.load(open(metrics_v2))
    val_ids = json.load(open(pred / "val_index.json"))
    youden = {n: am[n]["holdout"]["threshold"] for n in ORDER if n in am}

    preds_clean = {n: cleanup(preds_by_model[n], targets, youden[n]) for n in preds_by_model}

    # 3 positives (2 GT boxes each) + 1 negative
    pos2 = [i for i, t in enumerate(targets) if len(t["boxes"]) == 2][:30]
    neg = [i for i, t in enumerate(targets) if len(t["boxes"]) == 0]
    chosen = [pos2[0], pos2[5], pos2[12], neg[3]]
    print(f"Detection-sample indices: {chosen}")

    images = []
    for i in chosen:
        pid = val_ids[i]
        img = np.array(Image.open(img_dir / f"{pid}.png").convert("L"), dtype=np.float32) / 255.0
        images.append(torch.from_numpy(np.stack([img, img, img], axis=0)))

    sub_targets = [targets[i] for i in chosen]
    sub_preds = {n: [preds_clean[n][i] for i in chosen] for n in preds_clean}
    thresholds = {n: 0.0 for n in preds_clean}  # already filtered above

    viz.plot_detection_samples(
        images, sub_targets, sub_preds, out,
        num_samples=len(images), per_model_threshold=thresholds,
    )
    print("ALL MISSING PLOTS GENERATED.")


if __name__ == "__main__":
    main()
