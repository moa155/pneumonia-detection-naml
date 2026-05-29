"""RSNA Pneumonia Detection Challenge dataset loader.

Supports both DICOM (.dcm) and preprocessed PNG (.png) images.
PNG loading is ~10-50x faster; use `python -m src.preprocess` first.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RSNAPneumoniaDataset(Dataset):
    """Dataset for the RSNA Pneumonia Detection Challenge.

    Each sample is a chest X-ray image with bounding-box annotations
    for pneumonia-positive regions.  Images without pneumonia have empty
    annotation sets.

    The dataset is available at:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    """

    def __init__(
        self,
        image_dir: str,
        annotations_df: pd.DataFrame,
        transforms: Optional[Callable] = None,
    ):
        self.transforms = transforms

        # Determine image format: prefer PNG over DICOM
        image_dir = Path(image_dir)
        png_dir = image_dir.parent / "stage_2_train_images_png"
        if png_dir.exists() and any(png_dir.iterdir()):
            self.image_dir = png_dir
            self.use_png = True
        else:
            self.image_dir = image_dir
            self.use_png = False

        # Group bounding boxes by patient ID using groupby (O(N) instead of O(N*M))
        self.patient_ids = annotations_df["patientId"].unique().tolist()

        self.annotations: Dict[str, List[List[float]]] = {}
        grouped = annotations_df.groupby("patientId")
        for pid, group in grouped:
            boxes = []
            for _, row in group.iterrows():
                if row["Target"] == 1 and not np.isnan(row["x"]):
                    x, y, w, h = row["x"], row["y"], row["width"], row["height"]
                    boxes.append([x, y, x + w, y + h])  # xyxy format
            self.annotations[pid] = boxes

        # Ensure all patient_ids have an entry (some may not be in grouped)
        for pid in self.patient_ids:
            if pid not in self.annotations:
                self.annotations[pid] = []

    def __len__(self) -> int:
        return len(self.patient_ids)

    def get_positive_mask(self) -> List[bool]:
        """Return boolean list: True for patients with pneumonia boxes.

        Used by WeightedRandomSampler to oversample positive patients.
        """
        return [len(self.annotations[pid]) > 0 for pid in self.patient_ids]

    def _load_image(self, patient_id: str) -> np.ndarray:
        """Load image and return as float32 HWC array in [0, 1]."""
        try:
            if self.use_png:
                path = self.image_dir / f"{patient_id}.png"
                from PIL import Image
                img = np.array(Image.open(path), dtype=np.float32) / 255.0
            else:
                import pydicom
                path = self.image_dir / f"{patient_id}.dcm"
                dcm = pydicom.dcmread(str(path))
                img = dcm.pixel_array.astype(np.float32)
                pmin, pmax = img.min(), img.max()
                if pmax - pmin > 0:
                    img = (img - pmin) / (pmax - pmin)
                else:
                    img = np.zeros_like(img)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {self.image_dir / patient_id}.*")
        except Exception as e:
            raise RuntimeError(f"Failed to load image for patient {patient_id}: {e}")

        # Single-channel -> 3-channel
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        pid = self.patient_ids[idx]
        image = self._load_image(pid)

        # Build target dict
        boxes = self.annotations[pid]
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones(len(boxes), dtype=torch.int64)  # class 1 = pneumonia
            areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def load_rsna_dataframes(
    labels_csv: str, detail_csv: Optional[str] = None
) -> pd.DataFrame:
    """Load and merge RSNA label CSVs.

    Returns a DataFrame with columns:
        patientId, x, y, width, height, Target [, class]
    """
    df = pd.read_csv(labels_csv)
    if detail_csv is not None and os.path.exists(detail_csv):
        detail_df = pd.read_csv(detail_csv)
        detail_df = detail_df.drop_duplicates(subset=["patientId"])
        df = df.merge(detail_df, on="patientId", how="left")
    return df


def collate_fn(batch):
    """Custom collate for variable-size bounding boxes."""
    return tuple(zip(*batch))
