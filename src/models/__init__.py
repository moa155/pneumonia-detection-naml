"""Model factory for pneumonia detection."""

from src.models.fcos import build_fcos
from src.models.retinanet import build_retinanet
from src.models.faster_rcnn import build_faster_rcnn

MODEL_REGISTRY = {
    "fcos": build_fcos,
    "retinanet": build_retinanet,
    "faster_rcnn": build_faster_rcnn,
}


def build_model(name: str, num_classes: int = 2, pretrained_backbone: bool = True,
                min_size: int = 512, max_size: int = 512):
    """Build a detection model by name.

    Args:
        name: One of 'fcos', 'retinanet', 'faster_rcnn'.
        num_classes: Number of classes (including background).
        pretrained_backbone: Whether to use ImageNet-pretrained backbone.
        min_size: Minimum image size for internal resize.
        max_size: Maximum image size for internal resize.

    Returns:
        A torchvision detection model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        min_size=min_size,
        max_size=max_size,
    )
