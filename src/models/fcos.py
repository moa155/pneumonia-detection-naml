"""FCOS: Anchor-free detector — the paper's proposed method.

This implements the anchor-free detection framework from:
  Wu et al., "Pneumonia detection based on RSNA dataset and anchor-free
  deep learning detector", Scientific Reports (2024).

Architecture:
  - ResNet-50 backbone
  - Feature Pyramid Network (FPN) with 5 levels (strides 8–128)
  - Two-branch detection head (center classification + scale regression)
  - Focal loss for class imbalance
  - GroupNorm in head for training stability (matching v2 detector quality)

Fine-tuned from COCO-pretrained backbone + FPN for faster convergence.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSHead, FCOSRegressionHead
from torchvision.ops import sigmoid_focal_loss


class PaperFCOSHead(FCOSHead):
    """FCOS head with paper-matching losses (Wu et al., 2024).

    Differences from torchvision's default FCOSHead:
      - Box regression: Smooth L1 loss (instead of GIoU loss)
      - Center-ness: Sigmoid focal loss (instead of BCE)
      - Classification: Unchanged (already sigmoid focal loss)
    """

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        cls_logits = head_outputs["cls_logits"]       # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]   # [N, HWA, 1]

        # --- Build per-image GT targets (same as torchvision) ---
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros(
                    (len(matched_idxs_per_image),)
                )
                gt_boxes_targets = targets_per_image["boxes"].new_zeros(
                    (len(matched_idxs_per_image), 4)
                )
            else:
                gt_classes_targets = targets_per_image["labels"][
                    matched_idxs_per_image.clip(min=0)
                ]
                gt_boxes_targets = targets_per_image["boxes"][
                    matched_idxs_per_image.clip(min=0)
                ]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets)
        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        anchors = torch.stack(anchors)

        # Foreground mask
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # --- Classification loss (unchanged: sigmoid focal loss) ---
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # --- Box regression loss: Smooth L1 (paper) instead of GIoU ---
        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)
        loss_bbox_reg = F.smooth_l1_loss(
            bbox_regression[foregroud_mask],
            bbox_reg_targets[foregroud_mask],
            reduction="sum",
        )

        # --- Center-ness loss: sigmoid focal loss (paper) instead of BCE ---
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(
                bbox_reg_targets.size()[:-1]
            )
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = sigmoid_focal_loss(
            pred_centerness[foregroud_mask],
            gt_ctrness_targets[foregroud_mask],
            reduction="sum",
        )

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }


def build_fcos(num_classes: int = 2, pretrained_backbone: bool = True,
               min_size: int = 512, max_size: int = 512):
    """Build FCOS model with ResNet-50 FPN backbone.

    When pretrained_backbone=True, loads COCO-pretrained weights to get
    a pretrained backbone + FPN. The detection heads are then replaced
    with GroupNorm versions (torchvision has no fcos_resnet50_fpn_v2).
    """
    if pretrained_backbone:
        # Load COCO-pretrained model (gets pretrained backbone + FPN)
        model = fcos_resnet50_fpn(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace head with PaperFCOSHead (Smooth L1 + focal center-ness).
    # Sub-heads use GroupNorm for stable training with small batch sizes.
    in_channels = model.backbone.out_channels  # 256 for FPN
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]  # 1 for FCOS
    norm_layer = partial(nn.GroupNorm, 32)

    paper_head = PaperFCOSHead(in_channels, num_anchors, num_classes)
    paper_head.classification_head = FCOSClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer,
    )
    paper_head.regression_head = FCOSRegressionHead(
        in_channels, num_anchors, norm_layer=norm_layer,
    )
    model.head = paper_head

    model.score_thresh = 0.1  # paper: "confidence score threshold … set to 0.1"
    model.nms_thresh = 0.5
    return model
