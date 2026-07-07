# Weighted Box Fusion ensemble (Solovyev et al., 2021)

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
    """WBF on one image. Each input is a per-model list of tensors."""
    n_models = len(boxes_per_model)
    if weights is None:
        weights = [1.0] * n_models
    assert len(weights) == n_models

    weight_sum = float(sum(weights))

    # gather (box, score*weight, label, model_idx) above threshold
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

    entries.sort(key=lambda e: -e[1])  # weighted score desc

    # cluster by label + IoU
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

    # fuse: box = score-weighted avg, score scaled by coverage (how many
    # models hit this cluster), so boxes only 1 model found get penalised
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
    """Run WBF over every image. In: {model: [per-image preds]}."""
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
