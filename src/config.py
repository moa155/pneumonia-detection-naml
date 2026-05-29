"""Configuration for pneumonia detection experiments."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # Dataset files (RSNA Pneumonia Detection Challenge)
    train_csv: str = "stage_2_train_labels.csv"
    detail_csv: str = "stage_2_detailed_class_info.csv"
    train_images_dir: str = "stage_2_train_images"

    # --- Dataset ---
    val_split: float = 0.2
    seed: int = 42
    max_samples: Optional[int] = None  # Limit number of patients (None = use all)

    # --- Training ---
    batch_size: int = 4
    num_workers: int = -1  # -1 = auto-detect based on CPU cores
    num_epochs: int = 20
    learning_rate: float = 1e-3  # paper uses SGD with lr=0.001
    lr_milestones: tuple = (12, 16)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # --- Model ---
    num_classes: int = 2  # background + pneumonia
    pretrained_backbone: bool = True

    # --- Detection ---
    nms_threshold: float = 0.5
    score_threshold: float = 0.05
    patient_threshold: float = 0.3  # confidence threshold for patient-level classification

    # --- Data augmentation ---
    use_augmentation: bool = True
    image_min_size: int = 512
    image_max_size: int = 512

    # --- Performance ---
    force_device: Optional[str] = None  # None = auto-detect, "cpu", "cuda", "mps"
    use_amp: bool = True  # Automatic Mixed Precision (CUDA only)
    use_bf16: bool = False  # Use BFloat16 instead of Float16 (Ampere+ GPUs)
    num_threads: int = 0  # OpenMP threads (0 = auto)
    use_compile: bool = True  # torch.compile() for 20-40% speedup (PyTorch 2.x)
    prefetch_factor: int = 4  # DataLoader prefetch (batches per worker)

    # --- Resume & Efficiency ---
    resume: bool = False  # Resume training from last checkpoint
    val_frequency: int = 2  # Validate every N epochs (1=every epoch)
    early_stopping_patience: int = 5  # Stop after N validations without improvement (0=disabled)

    # --- Advanced Training ---
    freeze_backbone_epochs: int = 3  # Freeze early ResNet layers for N epochs (0=disabled)
    use_ema: bool = True  # Exponential Moving Average of model weights
    ema_decay: float = 0.999  # EMA decay factor
    scheduler_type: str = "cosine"  # "cosine" or "step" (paper uses step decay)
    gradient_accumulation: int = 1  # Effective batch = batch_size * accumulation
    multi_scale: bool = False  # Random multi-scale training [448..576]

    # --- Advanced Evaluation ---
    use_tta: bool = True  # Test-time augmentation (horizontal flip)
    use_soft_nms: bool = True  # Gaussian Soft-NMS instead of hard NMS

    # --- Advanced Data ---
    use_weighted_sampler: bool = True  # Oversample positive patients
    positive_sample_weight: float = 3.0  # Weight multiplier for positive patients

    # --- Optimizer + paper-recipe ablation ---
    optimizer_type: str = "adam"       # "adam" (default) or "sgd" (FCOS paper)
    momentum: float = 0.9              # SGD momentum
    checkpoint_suffix: str = ""        # Appended to checkpoint filenames (e.g. "_paper")

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
        # Auto-detect: use min(4, cpu_count) for good default
        cpu_count = os.cpu_count() or 1
        return min(4, cpu_count)

    @property
    def pin_memory(self) -> bool:
        """Enable pin_memory for CUDA and MPS devices."""
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
