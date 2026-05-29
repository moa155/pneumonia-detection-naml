"""COCO-style evaluation metrics for object detection.

Computes AP, AR at various IoU thresholds, matching the metrics reported
in the paper (Table 2 and Table 3).

Also provides:
  - Gaussian Soft-NMS (Bodla et al., 2017) for improved detection
  - Optimal patient-level threshold via ROC/Youden's J statistic
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor

    Returns:
        (N, M) IoU matrix
    """
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    return inter / (union + 1e-6)


# -----------------------------------------------------------------------
# Gaussian Soft-NMS (Bodla et al., "Soft-NMS", ICCV 2017)
# -----------------------------------------------------------------------

def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian Soft-NMS: decays overlapping scores instead of hard removal.

    Unlike standard NMS which removes overlapping detections, Soft-NMS
    decays their scores based on IoU overlap using a Gaussian penalty.
    This preserves closely-spaced detections (common in pneumonia where
    multiple lesions can overlap) while still suppressing duplicates.

    Args:
        boxes: (N, 4) tensor in xyxy format
        scores: (N,) confidence scores
        labels: (N,) class labels
        sigma: Gaussian decay parameter (lower = more aggressive suppression)
        score_threshold: Minimum score to keep after decay

    Returns:
        Filtered (boxes, scores, labels) tensors
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    dets = boxes.clone()
    sc = scores.clone()
    labs = labels.clone()
    N = len(dets)

    for i in range(N):
        # Find the highest-scoring detection among remaining
        max_idx = i + sc[i:].argmax()

        # Swap to front
        dets[i], dets[max_idx] = dets[max_idx].clone(), dets[i].clone()
        sc[i], sc[max_idx] = sc[max_idx].item(), sc[i].item()
        labs[i], labs[max_idx] = labs[max_idx].item(), labs[i].item()

        if i < N - 1:
            # Compute IoU of current box with all subsequent
            ious = torchvision.ops.box_iou(dets[i : i + 1], dets[i + 1 :])[0]
            # Gaussian decay: exp(-IoU^2 / sigma)
            decay = torch.exp(-(ious ** 2) / sigma)
            sc[i + 1 :] *= decay

    # Filter by score threshold
    keep = sc > score_threshold
    return dets[keep], sc[keep], labs[keep]


def apply_soft_nms_to_predictions(
    predictions: List[Dict],
    sigma: float = 0.5,
    score_threshold: float = 0.05,
) -> List[Dict]:
    """Apply Gaussian Soft-NMS to a list of per-image predictions."""
    for pred in predictions:
        if len(pred["boxes"]) > 0:
            pred["boxes"], pred["scores"], pred["labels"] = soft_nms(
                pred["boxes"],
                pred["scores"],
                pred["labels"],
                sigma=sigma,
                score_threshold=score_threshold,
            )
    return predictions


# -----------------------------------------------------------------------
# AP / AR computation
# -----------------------------------------------------------------------

def compute_ap_at_iou(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AP and AR at a single IoU threshold.

    Uses the standard VOC/COCO matching: each ground-truth box is matched
    to at most one prediction (highest IoU, greedy).

    Returns dict with keys: AP, precision, recall (arrays), num_gt, num_pred.
    """
    all_scores = []
    all_tp = []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by score descending
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched_gt = set()

        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i].item())

            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue

            # Compute IoU with all GT boxes
            ious = compute_iou_matrix(pred_boxes[i : i + 1], gt_boxes)[0]
            best_iou_val, best_idx_val = ious.max(0)
            best_iou_val = best_iou_val.item()
            best_idx_val = best_idx_val.item()

            if best_iou_val >= iou_threshold and best_idx_val not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_idx_val)
            else:
                all_tp.append(0)

    if total_gt == 0:
        return {"AP": 0.0, "num_gt": 0, "num_pred": len(all_scores)}

    # Sort all predictions globally by score
    indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[indices]
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_recall = precisions[recalls >= t]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    ap /= 101

    return {
        "AP": float(ap),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "num_gt": total_gt,
        "num_pred": len(all_scores),
    }


def compute_ar(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    max_dets: int = 100,
) -> float:
    """Compute Average Recall at a given IoU with a max detection limit."""
    total_recalled = 0
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # Keep top-K by score
        order = pred_scores.argsort(descending=True)[:max_dets]
        pred_boxes = pred_boxes[order]

        ious = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_gt = set()

        for i in range(len(pred_boxes)):
            row_ious = ious[i]
            best_idx = row_ious.argmax().item()
            if row_ious[best_idx].item() >= iou_threshold and best_idx not in matched_gt:
                matched_gt.add(best_idx)
                total_recalled += 1

    return total_recalled / max(total_gt, 1)


# -----------------------------------------------------------------------
# Optimal patient-level threshold via ROC
# -----------------------------------------------------------------------

def find_optimal_patient_threshold(
    predictions: List[Dict],
    targets: List[Dict],
) -> Tuple[float, float]:
    """Find optimal patient-level threshold via ROC / Youden's J statistic.

    For each patient, the max detection score is used as the patient-level
    score. The optimal threshold maximizes (sensitivity + specificity - 1),
    giving the best trade-off between detecting pneumonia and avoiding
    false alarms.

    Returns:
        (optimal_threshold, roc_auc)
    """
    patient_scores = []
    patient_labels = []

    for pred, gt in zip(predictions, targets):
        has_gt = len(gt["boxes"]) > 0
        max_score = pred["scores"].max().item() if len(pred["scores"]) > 0 else 0.0
        patient_scores.append(max_score)
        patient_labels.append(1 if has_gt else 0)

    patient_scores = np.array(patient_scores)
    patient_labels = np.array(patient_labels)

    # Check for degenerate cases
    if len(np.unique(patient_labels)) < 2:
        return 0.3, 0.0

    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(patient_labels, patient_scores)
        roc_auc = auc(fpr, tpr)

        # Youden's J statistic: optimal balance of sensitivity and specificity
        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        optimal_threshold = float(thresholds[optimal_idx])

        return optimal_threshold, float(roc_auc)
    except ImportError:
        # Fallback: use fixed threshold
        return 0.3, 0.0


# -----------------------------------------------------------------------
# Full metrics computation
# -----------------------------------------------------------------------

def _size_buckets_from_areas(all_areas: np.ndarray, scheme: str) -> Tuple[float, float]:
    """Return (small/medium boundary, medium/large boundary) for an AP-size scheme.

    - "coco": the standard 32^2, 96^2 COCO cutoffs. On RSNA at 1024px these put
      >99% of ground-truth boxes in "large", so AP_M is essentially noise.
    - "rsna": tertile boundaries computed from the actual area distribution
      (33rd and 67th percentiles), producing roughly balanced small/medium/large
      buckets that are meaningful for chest-X-ray opacities.
    """
    if scheme == "coco":
        return 32.0 ** 2, 96.0 ** 2
    if scheme == "rsna":
        if len(all_areas) == 0:
            return 32.0 ** 2, 96.0 ** 2
        return float(np.percentile(all_areas, 33)), float(np.percentile(all_areas, 67))
    raise ValueError(f"Unknown size scheme: {scheme!r}")


def _filter_targets_by_size(targets: List[Dict], lo: float, hi: float) -> List[Dict]:
    """Return a copy of `targets` with only boxes whose area is in [lo, hi)."""
    out = []
    for gt in targets:
        areas = gt["area"] if len(gt["boxes"]) > 0 else torch.zeros(0)
        mask = (areas >= lo) & (areas < hi)
        out.append({
            "boxes": gt["boxes"][mask] if len(gt["boxes"]) > 0 else gt["boxes"],
            "labels": gt["labels"][mask] if len(gt["labels"]) > 0 else gt["labels"],
            "area": areas[mask] if len(areas) > 0 else areas,
            "iscrowd": gt["iscrowd"][mask] if len(gt["iscrowd"]) > 0 else gt["iscrowd"],
        })
    return out


def compute_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    patient_threshold: Optional[float] = None,
    threshold_holdout: float = 0.0,
    size_scheme: str = "coco",
) -> Dict[str, float]:
    """Compute the full set of detection metrics reported in the paper.

    Args:
        predictions: per-image model outputs (boxes, scores, labels).
        targets: per-image ground-truth (boxes, area, ...).
        patient_threshold: if None, an optimal Youden-J threshold is found.
        threshold_holdout: if >0, an in-distribution fraction of the input is
            held out and used *only* to find the Youden threshold; all reported
            detection and classification metrics are computed on the remaining
            patients. Set to e.g. 0.5 to eliminate the threshold-on-test
            data-leakage issue when there is no separate test split.
            Default 0.0 reproduces the original single-set behaviour.
        size_scheme: "coco" for the standard 32^2 / 96^2 cutoffs; "rsna" for
            tertile cutoffs from the actual area distribution (recommended for
            meaningful AP_M / AP_L on RSNA, where COCO puts 99% of boxes in
            "large").

    Returns dict with:
        AP@0.5, AP@0.5:0.95, AP_M, AP_L, AR@10, AR_M, AR_L,
        patient_accuracy, patient_precision, patient_recall, patient_f1,
        optimal_threshold, roc_auc, size_scheme, threshold_holdout
    """
    # --- Optional threshold-holdout split (deterministic, by index) ---
    if threshold_holdout > 0.0:
        n = len(predictions)
        cal_n = max(1, int(round(threshold_holdout * n)))
        cal_pred, eval_pred = predictions[:cal_n], predictions[cal_n:]
        cal_tgt, eval_tgt = targets[:cal_n], targets[cal_n:]
        # Threshold from the held-out calibration split, metrics from the rest.
        optimal_thresh, _ = find_optimal_patient_threshold(cal_pred, cal_tgt)
        predictions, targets = eval_pred, eval_tgt
        # ROC AUC on the evaluation split (calibration-free ranking metric).
        _, roc_auc = find_optimal_patient_threshold(predictions, targets)
    else:
        optimal_thresh, roc_auc = find_optimal_patient_threshold(predictions, targets)

    # --- Object detection metrics ---
    result_50 = compute_ap_at_iou(predictions, targets, iou_threshold=0.5)
    ap50 = result_50["AP"]

    # AP@[0.5:0.95] (COCO standard)
    ap_sum = 0.0
    for iou_t in np.arange(0.5, 1.0, 0.05):
        ap_sum += compute_ap_at_iou(predictions, targets, iou_threshold=iou_t)["AP"]
    ap_5095 = ap_sum / 10

    # Size-based AP. COCO buckets are inappropriate for RSNA chest X-rays
    # (99% of opacity boxes land in "large"); see report Section 6. Pass
    # size_scheme="rsna" to use percentile-based, dataset-adapted buckets.
    all_areas = np.concatenate([
        gt["area"].numpy() for gt in targets if len(gt["boxes"]) > 0
    ]) if any(len(gt["boxes"]) > 0 for gt in targets) else np.array([])
    medium_lo, large_lo = _size_buckets_from_areas(all_areas, size_scheme)

    tgt_m = _filter_targets_by_size(targets, medium_lo, large_lo)
    tgt_l = _filter_targets_by_size(targets, large_lo, float("inf"))
    pred_m = predictions  # predictions are not size-filtered (any size may match)
    pred_l = predictions

    ap_m = compute_ap_at_iou(pred_m, tgt_m, iou_threshold=0.5)["AP"]
    ap_l = compute_ap_at_iou(pred_l, tgt_l, iou_threshold=0.5)["AP"]

    # Recall metrics
    ar_10 = compute_ar(predictions, targets, iou_threshold=0.5, max_dets=10)
    ar_m = compute_ar(pred_m, tgt_m, iou_threshold=0.5, max_dets=100)
    ar_l = compute_ar(pred_l, tgt_l, iou_threshold=0.5, max_dets=100)

    # --- Patient-level threshold to apply ---
    if patient_threshold is None:
        patient_threshold = optimal_thresh

    # --- Patient-level classification ---
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, gt in zip(predictions, targets):
        has_gt = len(gt["boxes"]) > 0
        has_pred = len(pred["boxes"]) > 0 and (pred["scores"] > patient_threshold).any()

        if has_gt and has_pred:
            tp += 1
        elif has_gt and not has_pred:
            fn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "AP@0.5": ap50,
        "AP@0.5:0.95": ap_5095,
        "AP_M": ap_m,
        "AP_L": ap_l,
        "AR@10": ar_10,
        "AR_M": ar_m,
        "AR_L": ar_l,
        "patient_accuracy": accuracy,
        "patient_precision": precision,
        "patient_recall": recall,
        "patient_f1": f1,
        "optimal_threshold": optimal_thresh,
        "roc_auc": roc_auc,
        "precisions": result_50.get("precisions", []),
        "recalls": result_50.get("recalls", []),
        "size_scheme": size_scheme,
        "size_bucket_boundaries": [medium_lo, large_lo],
        "threshold_holdout": threshold_holdout,
    }
