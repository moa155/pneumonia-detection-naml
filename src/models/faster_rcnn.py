"""Faster R-CNN: Two-stage anchor-based detector (comparison method).

Faster R-CNN (Ren et al., 2015) is included as a comparison method from the
paper's Table 3.  It uses:
  - ResNet-50 + FPN backbone
  - Region Proposal Network (RPN) for candidate generation
  - ROI pooling + classification/regression heads

Fine-tuned from COCO-pretrained weights for faster convergence.
"""

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int = 2, pretrained_backbone: bool = True,
                      min_size: int = 512, max_size: int = 512):
    """Build Faster R-CNN with ResNet-50 FPN v2 backbone.

    When pretrained_backbone=True, loads full COCO-pretrained weights
    (backbone + FPN + heads) and replaces only the box predictor for
    the target number of classes. Much faster convergence than
    training from scratch.
    """
    if pretrained_backbone:
        # Load full COCO-pretrained model (backbone + FPN + heads)
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
        # Replace only the final box predictor for our number of classes
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
