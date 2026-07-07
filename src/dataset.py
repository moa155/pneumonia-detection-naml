# RSNA dataset loader. Uses PNG if preprocessed (much faster), else DICOM.

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RSNAPneumoniaDataset(Dataset):
    """Chest X-rays + pneumonia boxes (empty box set = negative)."""

    def __init__(
        self,
        image_dir: str,
        annotations_df: pd.DataFrame,
        transforms: Optional[Callable] = None,
    ):
        self.transforms = transforms

        # prefer PNG dir if it exists
        image_dir = Path(image_dir)
        png_dir = image_dir.parent / "stage_2_train_images_png"
        if png_dir.exists() and any(png_dir.iterdir()):
            self.image_dir = png_dir
            self.use_png = True
        else:
            self.image_dir = image_dir
            self.use_png = False

        # group boxes per patient (groupby is O(N), not O(N*M))
        self.patient_ids = annotations_df["patientId"].unique().tolist()

        self.annotations: Dict[str, List[List[float]]] = {}
        grouped = annotations_df.groupby("patientId")
        for pid, group in grouped:
            boxes = []
            for _, row in group.iterrows():
                if row["Target"] == 1 and not np.isnan(row["x"]):
                    x, y, w, h = row["x"], row["y"], row["width"], row["height"]
                    boxes.append([x, y, x + w, y + h])  # xyxy
            self.annotations[pid] = boxes

        # make sure every patient has an entry
        for pid in self.patient_ids:
            if pid not in self.annotations:
                self.annotations[pid] = []

    def __len__(self) -> int:
        return len(self.patient_ids)

    def get_positive_mask(self) -> List[bool]:
        """True per positive patient (for WeightedRandomSampler oversampling)."""
        return [len(self.annotations[pid]) > 0 for pid in self.patient_ids]

    def _load_image(self, patient_id: str) -> np.ndarray:
        """Load image as float32 HWC in [0, 1]."""
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

        # grayscale -> 3ch
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        pid = self.patient_ids[idx]
        image = self._load_image(pid)

        boxes = self.annotations[pid]
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones(len(boxes), dtype=torch.int64)  # 1 = pneumonia
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
    """Load labels csv, optionally merge the class-detail csv."""
    df = pd.read_csv(labels_csv)
    if detail_csv is not None and os.path.exists(detail_csv):
        detail_df = pd.read_csv(detail_csv)
        detail_df = detail_df.drop_duplicates(subset=["patientId"])
        df = df.merge(detail_df, on="patientId", how="left")
    return df


def collate_fn(batch):
    """Collate keeping variable box counts (no stacking)."""
    return tuple(zip(*batch))
