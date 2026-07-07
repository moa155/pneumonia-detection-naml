# COCO-style detection metrics (AP/AR) + soft-NMS + patient-level threshold

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU, xyxy -> (N, M)."""
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    return inter / (union + 1e-6)


# Soft-NMS (Bodla et al., ICCV 2017)

def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decay overlapping scores (Gaussian) instead of hard-removing them.

    Keeps close-by boxes, which happens a lot with overlapping opacities.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    dets = boxes.clone()
    sc = scores.clone()
    labs = labels.clone()
    N = len(dets)

    for i in range(N):
        # pick best remaining, swap it to front
        max_idx = i + sc[i:].argmax()
        dets[i], dets[max_idx] = dets[max_idx].clone(), dets[i].clone()
        sc[i], sc[max_idx] = sc[max_idx].item(), sc[i].item()
        labs[i], labs[max_idx] = labs[max_idx].item(), labs[i].item()

        if i < N - 1:
            ious = torchvision.ops.box_iou(dets[i : i + 1], dets[i + 1 :])[0]
            decay = torch.exp(-(ious ** 2) / sigma)  # gaussian penalty
            sc[i + 1 :] *= decay

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


# AP / AR

def compute_ap_at_iou(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """AP at one IoU. Greedy VOC/COCO matching (1 pred per GT)."""
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

        # sort by score desc
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched_gt = set()

        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores[i].item())

            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue

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

    # global sort by score
    indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[indices]
    fp = 1 - tp

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 101-point interpolation (COCO)
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
    """Average Recall at given IoU, capped at max_dets."""
    total_recalled = 0
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # top-K by score
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


# Patient-level threshold via ROC

def find_optimal_patient_threshold(
    predictions: List[Dict],
    targets: List[Dict],
) -> Tuple[float, float]:
    """Patient threshold maximising Youden's J. Returns (thr, auc)."""
    patient_scores = []
    patient_labels = []

    for pred, gt in zip(predictions, targets):
        has_gt = len(gt["boxes"]) > 0
        max_score = pred["scores"].max().item() if len(pred["scores"]) > 0 else 0.0
        patient_scores.append(max_score)
        patient_labels.append(1 if has_gt else 0)

    patient_scores = np.array(patient_scores)
    patient_labels = np.array(patient_labels)

    # only one class present -> can't build ROC
    if len(np.unique(patient_labels)) < 2:
        return 0.3, 0.0

    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(patient_labels, patient_scores)
        roc_auc = auc(fpr, tpr)

        j_scores = tpr - fpr  # Youden's J
        optimal_idx = j_scores.argmax()
        optimal_threshold = float(thresholds[optimal_idx])

        return optimal_threshold, float(roc_auc)
    except ImportError:
        return 0.3, 0.0  # no sklearn, fall back to fixed thr


# Full metrics

def _size_buckets_from_areas(all_areas: np.ndarray, scheme: str) -> Tuple[float, float]:
    """Size-bucket boundaries. coco = 32^2/96^2, rsna = area tertiles.

    On RSNA the COCO cutoffs dump ~99% of boxes into "large", so rsna
    percentile buckets are the ones that actually mean something.
    """
    if scheme == "coco":
        return 32.0 ** 2, 96.0 ** 2
    if scheme == "rsna":
        if len(all_areas) == 0:
            return 32.0 ** 2, 96.0 ** 2
        return float(np.percentile(all_areas, 33)), float(np.percentile(all_areas, 67))
    raise ValueError(f"Unknown size scheme: {scheme!r}")


def _filter_targets_by_size(targets: List[Dict], lo: float, hi: float) -> List[Dict]:
    """Copy of targets keeping only boxes with area in [lo, hi)."""
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
    """All detection + patient-level metrics from the report.

    threshold_holdout>0 splits off a calibration slice to pick the Youden
    threshold, so it isn't tuned on the same data we report on.
    size_scheme picks coco vs rsna (tertile) size buckets.
    """
    # optional holdout split (deterministic, by index): pick thr on cal, report on rest
    if threshold_holdout > 0.0:
        n = len(predictions)
        cal_n = max(1, int(round(threshold_holdout * n)))
        cal_pred, eval_pred = predictions[:cal_n], predictions[cal_n:]
        cal_tgt, eval_tgt = targets[:cal_n], targets[cal_n:]
        optimal_thresh, _ = find_optimal_patient_threshold(cal_pred, cal_tgt)
        predictions, targets = eval_pred, eval_tgt
        _, roc_auc = find_optimal_patient_threshold(predictions, targets)  # auc on eval
    else:
        optimal_thresh, roc_auc = find_optimal_patient_threshold(predictions, targets)

    # detection metrics
    result_50 = compute_ap_at_iou(predictions, targets, iou_threshold=0.5)
    ap50 = result_50["AP"]

    # AP@[0.5:0.95]
    ap_sum = 0.0
    for iou_t in np.arange(0.5, 1.0, 0.05):
        ap_sum += compute_ap_at_iou(predictions, targets, iou_threshold=iou_t)["AP"]
    ap_5095 = ap_sum / 10

    # size-based AP (see report Sec 6 for why coco buckets are useless here)
    all_areas = np.concatenate([
        gt["area"].numpy() for gt in targets if len(gt["boxes"]) > 0
    ]) if any(len(gt["boxes"]) > 0 for gt in targets) else np.array([])
    medium_lo, large_lo = _size_buckets_from_areas(all_areas, size_scheme)

    tgt_m = _filter_targets_by_size(targets, medium_lo, large_lo)
    tgt_l = _filter_targets_by_size(targets, large_lo, float("inf"))
    pred_m = predictions  # preds unfiltered, any size can match
    pred_l = predictions

    ap_m = compute_ap_at_iou(pred_m, tgt_m, iou_threshold=0.5)["AP"]
    ap_l = compute_ap_at_iou(pred_l, tgt_l, iou_threshold=0.5)["AP"]

    ar_10 = compute_ar(predictions, targets, iou_threshold=0.5, max_dets=10)
    ar_m = compute_ar(pred_m, tgt_m, iou_threshold=0.5, max_dets=100)
    ar_l = compute_ar(pred_l, tgt_l, iou_threshold=0.5, max_dets=100)

    if patient_threshold is None:
        patient_threshold = optimal_thresh

    # patient-level classification
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
