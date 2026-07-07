"""RetinaNet (Lin et al. 2017): one-stage anchor-based baseline."""

import math

import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn_v2


def build_retinanet(num_classes: int = 2, pretrained_backbone: bool = True,
                    min_size: int = 512, max_size: int = 512):
    """RetinaNet + ResNet-50 FPN v2."""
    if pretrained_backbone:
        model = retinanet_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
        # replace only the final cls conv, keep the 4 shared convs pretrained
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
