"""Faster R-CNN (Ren et al. 2015): two-stage anchor-based baseline."""

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int = 2, pretrained_backbone: bool = True,
                      min_size: int = 512, max_size: int = 512):
    """Faster R-CNN + ResNet-50 FPN v2."""
    if pretrained_backbone:
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
        # swap only the box predictor for our num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    return model
