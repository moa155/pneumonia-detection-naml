"""RetinaNet: One-stage anchor-based detector (comparison method).

RetinaNet (Lin et al., 2017) is included as a comparison method from the
paper's Table 3.  It uses:
  - ResNet-50 + FPN backbone
  - Anchor-based detection with focal loss
  - Separate classification and regression sub-networks

Fine-tuned from COCO-pretrained weights for faster convergence.
"""

import math

import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2


def build_retinanet(num_classes: int = 2, pretrained_backbone: bool = True,
                    min_size: int = 512, max_size: int = 512):
    """Build RetinaNet with ResNet-50 FPN v2 backbone.

    When pretrained_backbone=True, loads full COCO-pretrained weights
    and replaces only the final classification layer for the target
    number of classes. Keeps pretrained backbone + FPN + shared head
    conv layers.
    """
    if pretrained_backbone:
        # Load full COCO-pretrained model
        model = retinanet_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
        # Replace only the final classification conv for our num_classes
        # (keeps the 4 shared conv layers pretrained)
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head.num_classes = num_classes
        cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
                               kernel_size=3, stride=1, padding=1)
        nn.init.normal_(cls_logits.weight, std=0.01)
        nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        model.head.classification_head.cls_logits = cls_logits
    else:
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    model.score_thresh = 0.05
    model.nms_thresh = 0.5
    return model
