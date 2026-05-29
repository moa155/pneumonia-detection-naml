"""Detection transforms with medical imaging augmentations.

Augmentation pipeline optimized for chest X-ray pneumonia detection:
  - CLAHE: Enhances local contrast in X-ray images (medical imaging standard)
  - RandomBrightnessContrast: Simulates exposure/gain variation
  - RandomGamma: Simulates display gamma differences
  - HorizontalFlip + VerticalFlip: Paper uses H+V flip + mirror for 8x expansion
  - RandomResizedCrop: Scale jitter for robustness
  - ShiftScaleRotate: Minor position/scale/rotation changes (patient positioning)
  - GaussNoise/GaussianBlur: Simulates acquisition noise and defocus
  - GridDistortion/ElasticTransform: Simulates anatomical variation
  - CoarseDropout: Regularization via random occlusion

Uses albumentations for robust bbox-consistent augmentation with automatic
coordinate transformation for geometric augmentations.
"""

from typing import Dict, Tuple

import numpy as np
import torch

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
    _ALBU_V2 = int(A.__version__.split(".")[0]) >= 2
except ImportError:
    HAS_ALBUMENTATIONS = False
    _ALBU_V2 = False


class DetectionTransform:
    """Wraps an albumentations pipeline for (image, target) -> (tensor, target).

    Handles conversion between the dataset's (numpy HWC float32, target dict)
    format and albumentations' expected input (uint8 HWC + bbox list).
    """

    def __init__(self, album_transform):
        self.transform = album_transform

    def __call__(self, image: np.ndarray, target: Dict) -> Tuple[torch.Tensor, Dict]:
        # Convert float32 [0,1] -> uint8 [0,255] for albumentations
        if image.dtype in (np.float32, np.float64):
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Extract boxes and labels for albumentations
        boxes = target["boxes"].numpy().tolist() if len(target["boxes"]) > 0 else []
        labels = target["labels"].numpy().tolist() if len(target["labels"]) > 0 else []

        # Apply augmentation pipeline
        result = self.transform(image=image_uint8, bboxes=boxes, labels=labels)

        # Convert uint8 HWC -> float32 CHW tensor (avoids quantization round-trip)
        img_np = result["image"]
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1).copy()).float().div_(255.0)

        # Reconstruct target from augmented bboxes
        out_boxes = result["bboxes"]
        out_labels = result["labels"]

        if len(out_boxes) > 0:
            boxes_t = torch.tensor(out_boxes, dtype=torch.float32)
            labels_t = torch.tensor(out_labels, dtype=torch.int64)
            areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": target["image_id"],
                "area": areas,
                "iscrowd": torch.zeros(len(out_boxes), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": target["image_id"],
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }

        return img_tensor, target


class _FallbackToTensor:
    """Fallback transform when albumentations is not available."""

    def __call__(self, image: np.ndarray, target: Dict) -> Tuple[torch.Tensor, Dict]:
        if image.dtype in (np.float32, np.float64):
            img_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        else:
            img_tensor = torch.from_numpy(
                image.transpose(2, 0, 1).copy()
            ).float().div_(255.0)
        return img_tensor, target


def _build_train_pipeline():
    """Build albumentations training pipeline for chest X-ray detection."""
    transforms = []

    # --- Medical imaging specific ---
    transforms.append(
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
    )

    # --- Photometric augmentations ---
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4
        )
    )
    transforms.append(
        A.RandomGamma(gamma_limit=(80, 120), p=0.2)
    )

    # --- Geometric augmentations (paper: H+V flip + mirror for 8x expansion) ---
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.VerticalFlip(p=0.5))

    # --- Random resized crop (scale jitter) ---
    if _ALBU_V2:
        transforms.append(
            A.RandomResizedCrop(
                size=(512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3,
            )
        )
    else:
        transforms.append(
            A.RandomResizedCrop(
                height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3,
            )
        )

    if _ALBU_V2:
        transforms.append(
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                # albumentations 2.x: `mode` removed; default is BORDER_CONSTANT.
                fill=0,
                p=0.3,
            )
        )
    else:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,
                p=0.3,
            )
        )

    # --- Noise and blur ---
    transforms.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(p=1.0),
            ],
            p=0.2,
        )
    )

    # --- Elastic/grid distortion (simulates anatomical variation) ---
    if _ALBU_V2:
        transforms.append(
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, p=1.0),
                ],
                p=0.15,
            )
        )
    else:
        transforms.append(
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=1.0),
                ],
                p=0.15,
            )
        )

    # --- Regularization (random occlusion) ---
    if _ALBU_V2:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.15,
            )
        )
    else:
        transforms.append(
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.15,
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=100,         # Drop boxes smaller than 10x10 pixels
            min_visibility=0.3,   # Drop boxes less than 30% visible after crop
        ),
    )


def _build_val_pipeline():
    """Build albumentations validation pipeline (no augmentation)."""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
    )


def get_train_transforms(use_augmentation: bool = True) -> DetectionTransform:
    """Training transforms with medical imaging augmentations.

    Uses albumentations for CLAHE, rotation, contrast, noise, elastic
    distortion — all critical for chest X-ray detection performance.
    Falls back to simple tensor conversion if albumentations unavailable.
    """
    if not HAS_ALBUMENTATIONS:
        return _FallbackToTensor()

    if use_augmentation:
        pipeline = _build_train_pipeline()
    else:
        pipeline = _build_val_pipeline()

    return DetectionTransform(pipeline)


def get_val_transforms() -> DetectionTransform:
    """Validation/test transforms (no augmentation)."""
    if not HAS_ALBUMENTATIONS:
        return _FallbackToTensor()
    return DetectionTransform(_build_val_pipeline())
