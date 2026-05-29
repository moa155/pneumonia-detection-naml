"""Weighted Box Fusion (WBF) ensemble for detection predictions.

Reference:
    Solovyev, Wang, Gabruseva, "Weighted boxes fusion: Ensembling boxes from
    different object detection models", Image and Vision Computing (2021).

Given per-image predictions from K models, produces a single ensembled
prediction set by clustering overlapping boxes and computing weighted averages.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchvision


def _box_iou(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """IoU between one box (4,) and a set of boxes (N, 4). Returns (N,)."""
    if others.numel() == 0:
        return torch.zeros(0)
    return torchvision.ops.box_iou(box.unsqueeze(0), others)[0]


def weighted_boxes_fusion_single(
    boxes_per_model: Sequence[torch.Tensor],
    scores_per_model: Sequence[torch.Tensor],
    labels_per_model: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WBF for a single image.

    Args:
        boxes_per_model:  list of (N_m, 4) xyxy tensors, one per model.
        scores_per_model: list of (N_m,) score tensors.
        labels_per_model: list of (N_m,) label tensors.
        weights: per-model weights (default: all 1.0).
        iou_thr: IoU threshold for clustering overlapping boxes.
        skip_box_thr: drop boxes with score below this before fusion.

    Returns:
        (fused_boxes, fused_scores, fused_labels) with shapes
        (M, 4), (M,), (M,).
    """
    n_models = len(boxes_per_model)
    if weights is None:
        weights = [1.0] * n_models
    assert len(weights) == n_models

    weight_sum = float(sum(weights))

    # Collect (box, score*weight, label, model_idx) tuples above threshold
    entries = []
    for m_idx, (boxes, scores, labels) in enumerate(
        zip(boxes_per_model, scores_per_model, labels_per_model)
    ):
        w = float(weights[m_idx])
        for j in range(len(boxes)):
            s = float(scores[j])
            if s < skip_box_thr:
                continue
            entries.append((boxes[j], s * w, int(labels[j]), m_idx))

    if not entries:
        return (
            torch.zeros((0, 4)),
            torch.zeros(0),
            torch.zeros(0, dtype=torch.long),
        )

    # Sort by weighted score desc
    entries.sort(key=lambda e: -e[1])

    # Cluster by label + IoU
    clusters: List[List[tuple]] = []
    for entry in entries:
        box, wscore, label, m_idx = entry
        placed = False
        for cluster in clusters:
            if cluster[0][2] != label:
                continue
            iou = float(_box_iou(box, cluster[0][0].unsqueeze(0))[0])
            if iou >= iou_thr:
                cluster.append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])

    # Fuse each cluster following Solovyev et al. (2021):
    #   box   = score-weighted average of constituent boxes
    #   score = (sum of weighted scores / sum of weights) * (cluster_size / n_models)
    # The coverage factor penalises clusters missed by some of the models.
    fused_boxes_l: List[torch.Tensor] = []
    fused_scores_l: List[float] = []
    fused_labels_l: List[int] = []
    for cluster in clusters:
        total_weighted_score = sum(e[1] for e in cluster)
        weighted_box = torch.zeros(4)
        for box, wscore, _, _ in cluster:
            weighted_box = weighted_box + box * wscore
        weighted_box = weighted_box / max(total_weighted_score, 1e-8)
        coverage = min(len(cluster), n_models) / n_models
        fused_score = (total_weighted_score / weight_sum) * coverage
        fused_scores_l.append(min(float(fused_score), 1.0))
        fused_boxes_l.append(weighted_box)
        fused_labels_l.append(cluster[0][2])

    return (
        torch.stack(fused_boxes_l),
        torch.tensor(fused_scores_l, dtype=torch.float32),
        torch.tensor(fused_labels_l, dtype=torch.long),
    )


def ensemble_predictions(
    predictions_by_model: Dict[str, List[Dict]],
    weights: Optional[Dict[str, float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
) -> List[Dict]:
    """Apply WBF across all images for an ensemble of models.

    Args:
        predictions_by_model: {model_name: [pred_dict_per_image]}.
        weights: {model_name: weight}, default uniform.
        iou_thr: clustering IoU threshold.
        skip_box_thr: score threshold per input box.

    Returns:
        List of ensemble prediction dicts matching input structure.
    """
    names = list(predictions_by_model.keys())
    if weights is None:
        weights = {n: 1.0 for n in names}
    w_list = [weights[n] for n in names]

    num_images = len(next(iter(predictions_by_model.values())))
    ensembled: List[Dict] = []
    for i in range(num_images):
        boxes_list = [predictions_by_model[n][i]["boxes"].cpu() for n in names]
        scores_list = [predictions_by_model[n][i]["scores"].cpu() for n in names]
        labels_list = [predictions_by_model[n][i]["labels"].cpu() for n in names]

        fb, fs, fl = weighted_boxes_fusion_single(
            boxes_list, scores_list, labels_list,
            weights=w_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
        )
        ensembled.append({"boxes": fb, "scores": fs, "labels": fl})

    return ensembled
