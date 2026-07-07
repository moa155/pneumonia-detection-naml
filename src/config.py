"""Experiment config."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    # paths
    data_dir: str = "data"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # RSNA challenge files
    train_csv: str = "stage_2_train_labels.csv"
    detail_csv: str = "stage_2_detailed_class_info.csv"
    train_images_dir: str = "stage_2_train_images"

    # dataset
    val_split: float = 0.2
    seed: int = 42
    max_samples: Optional[int] = None  # None = all patients

    # training
    batch_size: int = 4
    num_workers: int = -1  # -1 -> auto
    num_epochs: int = 40
    learning_rate: float = 1e-3
    lr_milestones: tuple = (12, 16)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # model
    num_classes: int = 2  # bg + pneumonia
    pretrained_backbone: bool = True

    # detection
    nms_threshold: float = 0.5
    score_threshold: float = 0.05
    patient_threshold: float = 0.3  # patient-level cutoff

    # augmentation
    use_augmentation: bool = True
    image_min_size: int = 512
    image_max_size: int = 512

    # performance
    force_device: Optional[str] = None  # None = auto
    use_amp: bool = True  # CUDA only
    use_bf16: bool = False  # bf16 instead of fp16 (Ampere+)
    num_threads: int = 0
    use_compile: bool = True  # torch.compile, PyTorch 2.x
    prefetch_factor: int = 4

    # resume / efficiency
    resume: bool = False
    val_frequency: int = 4  # validate every N epochs
    early_stopping_patience: int = 5  # 0 = off

    # advanced training
    freeze_backbone_epochs: int = 3  # freeze early ResNet layers, 0 = off
    use_ema: bool = True
    ema_decay: float = 0.999  # EMA decay
    scheduler_type: str = "cosine"  # cosine | step
    gradient_accumulation: int = 1
    multi_scale: bool = False

    # advanced eval
    use_tta: bool = True  # hflip TTA
    use_soft_nms: bool = True

    # advanced data
    use_weighted_sampler: bool = True
    positive_sample_weight: float = 3.0  # oversample positives 3x

    # optimizer / ablation
    optimizer_type: str = "adam"       # adam | sgd (paper recipe)
    momentum: float = 0.9
    checkpoint_suffix: str = ""        # e.g. "_paper"

    @property
    def device(self) -> torch.device:
        if self.force_device is not None:
            return torch.device(self.force_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def effective_num_workers(self) -> int:
        if self.num_workers >= 0:
            return self.num_workers
        cpu_count = os.cpu_count() or 1
        return min(4, cpu_count)

    @property
    def pin_memory(self) -> bool:
        return self.device.type in ("cuda", "mps")

    @property
    def images_path(self) -> Path:
        return Path(self.data_dir) / self.train_images_dir

    @property
    def labels_path(self) -> Path:
        return Path(self.data_dir) / self.train_csv

    @property
    def detail_labels_path(self) -> Path:
        return Path(self.data_dir) / self.detail_csv
