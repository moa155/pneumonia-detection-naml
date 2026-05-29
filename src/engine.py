"""Training and evaluation engine for detection models.

Implements:
  - Backbone freezing for stable fine-tuning
  - Exponential Moving Average (EMA) of model weights
  - Cosine annealing and step LR schedules with warmup
  - Gradient accumulation for larger effective batch sizes
  - Multi-scale training for scale robustness
  - Test-Time Augmentation (TTA) with horizontal flip
  - Gaussian Soft-NMS for improved recall
  - Channels-last memory format for GPU speedup
"""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------
# Exponential Moving Average
# -----------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters as an exponential moving
    average. The EMA weights provide better generalization than the raw
    training weights (smoother optimization trajectory).

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            loss.backward(); optimizer.step()
            ema.update(model)
        # For evaluation:
        ema.apply_shadow(model)
        evaluate(model, ...)
        ema.restore(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        """Update shadow weights with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module):
        """Replace model parameters with EMA shadow (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original model parameters (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# -----------------------------------------------------------------------
# Backbone freezing
# -----------------------------------------------------------------------

def _freeze_backbone(model: torch.nn.Module):
    """Freeze early ResNet stages (conv1, bn1, layer1, layer2).

    Early layers learn generic low-level features (edges, textures) that
    transfer perfectly from COCO/ImageNet. Freezing them:
      - Prevents catastrophic forgetting of pretrained features
      - Reduces memory usage and computation by ~30-40%
      - Stabilizes training with small batch sizes
    """
    body = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "body"):
        body = model.backbone.body
    if body is None:
        return 0

    frozen_count = 0
    for name, param in body.named_parameters():
        if any(name.startswith(prefix) for prefix in ("conv1", "bn1", "layer1", "layer2")):
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def _unfreeze_backbone(model: torch.nn.Module):
    """Unfreeze all backbone parameters."""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = True


# -----------------------------------------------------------------------
# Model threshold manipulation (for Soft-NMS)
# -----------------------------------------------------------------------

def _set_model_thresholds(model, score_thresh=0.05, nms_thresh=0.5):
    """Set score and NMS thresholds for different model architectures."""
    if hasattr(model, "roi_heads"):  # Faster R-CNN
        model.roi_heads.score_thresh = score_thresh
        model.roi_heads.nms_thresh = nms_thresh
    else:  # FCOS, RetinaNet
        model.score_thresh = score_thresh
        model.nms_thresh = nms_thresh


def _get_model_thresholds(model):
    """Get current score and NMS thresholds."""
    if hasattr(model, "roi_heads"):
        return model.roi_heads.score_thresh, model.roi_heads.nms_thresh
    return model.score_thresh, model.nms_thresh


# -----------------------------------------------------------------------
# torch.compile
# -----------------------------------------------------------------------

def _try_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Attempt torch.compile(); return original model on failure."""
    from torchvision.models.detection import FasterRCNN, FCOS, RetinaNet

    if isinstance(model, (FasterRCNN, FCOS, RetinaNet)):
        print("  torch.compile() skipped (detection models use dynamic shapes)")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile() enabled (reduce-overhead mode)")
        return compiled
    except Exception as e:
        print(f"  torch.compile() unavailable ({e}), using eager mode")
        return model


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    gradient_accumulation: int = 1,
    multi_scale: bool = False,
    ema: Optional["ModelEMA"] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Train for one epoch, returning average losses."""
    model.train()
    running_losses = {}
    num_batches = 0
    use_amp = amp_dtype is not None and device.type == "cuda"
    accum_steps = max(1, gradient_accumulation)

    # Multi-scale training: random resize per batch
    multi_scale_sizes = [448, 480, 512, 544, 576]

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", file=sys.stdout)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]

        # Skip batches with invalid bounding boxes
        valid = True
        for t in targets:
            if t["boxes"].numel() > 0:
                if (
                    (t["boxes"][:, 2] <= t["boxes"][:, 0]).any()
                    or (t["boxes"][:, 3] <= t["boxes"][:, 1]).any()
                    or torch.isnan(t["boxes"]).any()
                    or torch.isinf(t["boxes"]).any()
                ):
                    valid = False
                    break
        if not valid:
            continue

        # Multi-scale: randomly change model's internal resize
        if multi_scale:
            scale = random.choice(multi_scale_sizes)
            model.transform.min_size = (scale,)
            model.transform.max_size = scale

        # Forward pass with optional mixed precision
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # Scale loss for gradient accumulation
            losses = losses / accum_steps

        if not math.isfinite(losses.item() * accum_steps):
            print(f"WARNING: non-finite loss {losses.item() * accum_steps:.4f}, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # Optimizer step every accum_steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(data_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Update EMA after each optimizer step (not per epoch)
            if ema is not None:
                ema.update(model)

        # Accumulate losses (unscaled)
        actual_loss = losses.item() * accum_steps
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()
        running_losses["total_loss"] = running_losses.get("total_loss", 0.0) + actual_loss
        num_batches += 1

        pbar.set_postfix(loss=f"{actual_loss:.4f}")

    # Reset multi-scale to default for validation
    if multi_scale:
        model.transform.min_size = (512,)
        model.transform.max_size = 512

    # Average
    if num_batches == 0:
        print("WARNING: All batches skipped (non-finite losses). Returning zero losses.")
        return {"total_loss": 0.0}
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    return avg_losses


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Run inference and collect predictions + ground truth.

    Returns:
        all_predictions: list of dicts with keys
            image_id, boxes (xyxy), scores, labels
        all_targets: list of dicts with keys
            image_id, boxes (xyxy), labels, area, iscrowd
    """
    model.eval()
    all_predictions = []
    all_targets = []
    amp_enabled = amp_dtype is not None and device.type == "cuda"

    for images, targets in tqdm(data_loader, desc="Evaluating", file=sys.stdout):
        images = [img.to(device, non_blocking=True) for img in images]

        with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(images)

        for output, target in zip(outputs, targets):
            img_id = target["image_id"].item()

            pred = {
                "image_id": img_id,
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            }
            all_predictions.append(pred)

            gt = {
                "image_id": img_id,
                "boxes": target["boxes"],
                "labels": target["labels"],
                "area": target["area"],
                "iscrowd": target["iscrowd"],
            }
            all_targets.append(gt)

    return all_predictions, all_targets


@torch.inference_mode()
def evaluate_with_tta(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate with Test-Time Augmentation (horizontal flip).

    Runs inference on both original and horizontally flipped images,
    then combines predictions and applies NMS to remove duplicates.
    Typically improves AP by 1-3%.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    amp_enabled = amp_dtype is not None and device.type == "cuda"

    for images, targets in tqdm(data_loader, desc="Evaluating (TTA)", file=sys.stdout):
        images_device = [img.to(device, non_blocking=True) for img in images]

        # Original predictions
        with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
            orig_outputs = model(images_device)

        # Flipped predictions
        flipped_images = [img.flip(-1) for img in images_device]
        with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
            flip_outputs = model(flipped_images)

        for orig_out, flip_out, img, target in zip(
            orig_outputs, flip_outputs, images_device, targets
        ):
            w = img.shape[-1]
            img_id = target["image_id"].item()

            # Flip boxes back to original coordinates
            flip_boxes = flip_out["boxes"].clone()
            if len(flip_boxes) > 0:
                flip_boxes[:, [0, 2]] = w - flip_boxes[:, [2, 0]]

            # Combine predictions from both passes
            if len(orig_out["boxes"]) > 0 and len(flip_boxes) > 0:
                all_boxes = torch.cat([orig_out["boxes"], flip_boxes])
                all_scores = torch.cat([orig_out["scores"], flip_out["scores"]])
                all_labels = torch.cat([orig_out["labels"], flip_out["labels"]])
            elif len(orig_out["boxes"]) > 0:
                all_boxes = orig_out["boxes"]
                all_scores = orig_out["scores"]
                all_labels = orig_out["labels"]
            elif len(flip_boxes) > 0:
                all_boxes = flip_boxes
                all_scores = flip_out["scores"]
                all_labels = flip_out["labels"]
            else:
                all_boxes = orig_out["boxes"]
                all_scores = orig_out["scores"]
                all_labels = orig_out["labels"]

            # Apply NMS on combined predictions to remove duplicates
            if len(all_boxes) > 0:
                keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=0.5)
                all_boxes = all_boxes[keep]
                all_scores = all_scores[keep]
                all_labels = all_labels[keep]

            pred = {
                "image_id": img_id,
                "boxes": all_boxes.cpu(),
                "scores": all_scores.cpu(),
                "labels": all_labels.cpu(),
            }
            all_predictions.append(pred)

            gt = {
                "image_id": img_id,
                "boxes": target["boxes"],
                "labels": target["labels"],
                "area": target["area"],
                "iscrowd": target["iscrowd"],
            }
            all_targets.append(gt)

    return all_predictions, all_targets


import torchvision


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    use_tta: bool = False,
    use_soft_nms: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate with optional TTA and Soft-NMS.

    This is the main evaluation entry point that orchestrates all
    advanced evaluation features.
    """
    # Relax model thresholds when using Soft-NMS (let more proposals through)
    orig_score_thresh, orig_nms_thresh = _get_model_thresholds(model)
    if use_soft_nms:
        _set_model_thresholds(model, score_thresh=0.01, nms_thresh=0.7)

    # Run evaluation (with or without TTA)
    if use_tta:
        predictions, targets = evaluate_with_tta(model, data_loader, device, use_amp, amp_dtype=amp_dtype)
    else:
        predictions, targets = evaluate(model, data_loader, device, use_amp, amp_dtype=amp_dtype)

    # Apply Soft-NMS post-processing
    if use_soft_nms:
        from src.evaluate import apply_soft_nms_to_predictions
        predictions = apply_soft_nms_to_predictions(predictions, sigma=0.5, score_threshold=0.05)
        # Restore original thresholds
        _set_model_thresholds(model, orig_score_thresh, orig_nms_thresh)

    return predictions, targets


# -----------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------

def _save_full_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    best_ap: float,
    history: Dict,
    ema: Optional[ModelEMA] = None,
):
    """Save full training state for resuming."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_ap": best_ap,
        "history": history,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    torch.save(state, path)


def _load_full_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    ema: Optional[ModelEMA] = None,
) -> Tuple[int, float, Dict]:
    """Load full training state. Returns (start_epoch, best_ap, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema is not None and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])
    start_epoch = ckpt["epoch"]
    best_ap = ckpt.get("best_ap", -1.0)
    history = ckpt.get("history", {
        "train_losses": [], "val_metrics": [], "epoch_times": [], "learning_rates": []
    })
    return start_epoch, best_ap, history


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    model_name: str,
) -> Dict:
    """Full training loop with all advanced techniques.

    Features:
      - Backbone freezing (early ResNet layers frozen for first N epochs)
      - EMA (exponential moving average of weights for better generalization)
      - Cosine annealing LR schedule with warmup
      - Gradient accumulation for effective larger batch size
      - Multi-scale training for scale robustness
      - TTA + Soft-NMS during validation
      - Early stopping based on AP improvement
      - Full checkpointing for crash recovery

    Returns a history dict with per-epoch losses and metrics.
    """
    device = config.device
    model.to(device)

    # Channels-last memory format for GPU speedup (10-30% faster convolutions)
    if device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
            print("  Channels-last memory format enabled")
        except Exception:
            pass

    # OpenMP threads for CPU parallelism
    if hasattr(config, "num_threads") and config.num_threads > 0:
        torch.set_num_threads(config.num_threads)

    # --- Checkpoint suffix (lets us train e.g. fcos_paper without overwriting fcos) ---
    ckpt_suffix = getattr(config, "checkpoint_suffix", "")
    if ckpt_suffix:
        model_name = f"{model_name}{ckpt_suffix}"

    # Mixed precision: choose dtype and scaler
    use_amp = hasattr(config, "use_amp") and config.use_amp and device.type == "cuda"
    use_bf16 = getattr(config, "use_bf16", False) and device.type == "cuda"
    amp_dtype: Optional[torch.dtype] = None
    scaler = None

    if use_amp:
        if use_bf16:
            # BFloat16: no loss scaling needed, requires Ampere+ (sm_80)
            amp_dtype = torch.bfloat16
            print(f"  Using BFloat16 AMP on {device} (no GradScaler)")
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler(device.type)
            print(f"  Using Float16 AMP on {device}")

    # torch.compile() for PyTorch 2.x speedup
    use_compile = getattr(config, "use_compile", False)
    if use_compile and device.type == "cuda":
        model = _try_compile(model)

    # --- Backbone freezing ---
    freeze_epochs = getattr(config, "freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        frozen_count = _freeze_backbone(model)
        total_params = sum(1 for p in model.parameters())
        print(f"  Backbone frozen: {frozen_count}/{total_params} params frozen for {freeze_epochs} epochs")

    # --- Optimizer: Adam (default) or SGD (paper-style for FCOS) ---
    params = [p for p in model.parameters() if p.requires_grad]
    fused = device.type == "cuda"
    optimizer_type = getattr(config, "optimizer_type", "adam").lower()
    if optimizer_type == "sgd":
        momentum = getattr(config, "momentum", 0.9)
        optimizer = torch.optim.SGD(
            params, lr=config.learning_rate,
            momentum=momentum, weight_decay=config.weight_decay,
        )
        print(f"  Optimizer: SGD lr={config.learning_rate} momentum={momentum}")
    else:
        optimizer = torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay,
            fused=fused,
        )
        print(f"  Optimizer: Adam lr={config.learning_rate}")

    # --- LR Schedule ---
    scheduler_type = getattr(config, "scheduler_type", "cosine")
    warmup_epochs = min(2, max(1, config.num_epochs // 5))

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )

    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs - warmup_epochs,
            eta_min=1e-6,
        )
    else:
        # Step decay (paper default)
        milestones = list(config.lr_milestones)
        if config.num_epochs < max(milestones):
            ratio = config.num_epochs / 20.0
            milestones = sorted(set(max(1, int(m * ratio)) for m in milestones))
        milestones = [m for m in milestones if m > warmup_epochs]
        if not milestones:
            milestones = [max(1, config.num_epochs - 1)]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=config.lr_gamma
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    print(f"  LR schedule: warmup({warmup_epochs}ep) + {scheduler_type}")

    # --- EMA ---
    use_ema = getattr(config, "use_ema", False)
    ema_decay = getattr(config, "ema_decay", 0.999)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"  EMA enabled (decay={ema_decay})")

    # --- Other config ---
    grad_accum = getattr(config, "gradient_accumulation", 1)
    multi_scale = getattr(config, "multi_scale", False)

    if grad_accum > 1:
        print(f"  Gradient accumulation: {grad_accum}x (effective batch={config.batch_size * grad_accum})")
    if multi_scale:
        print("  Multi-scale training: [448, 480, 512, 544, 576]")

    history = {
        "train_losses": [],
        "val_metrics": [],
        "epoch_times": [],
        "learning_rates": [],
    }

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    start_epoch = 0

    # --- Resume from checkpoint ---
    resume = getattr(config, "resume", False)
    resume_path = ckpt_dir / f"{model_name}_resume.pth"
    if resume and resume_path.exists():
        print(f"  Resuming from {resume_path}")
        start_epoch, best_ap, history = _load_full_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device, ema
        )
        print(f"  Resumed at epoch {start_epoch}, best AP@0.5={best_ap:.4f}")
    elif resume:
        best_path = ckpt_dir / f"{model_name}_best.pth"
        if best_path.exists():
            print(f"  No resume checkpoint, but found {best_path} — starting fresh training")

    # --- Early stopping state ---
    patience = getattr(config, "early_stopping_patience", 0)
    val_freq = getattr(config, "val_frequency", 1)
    no_improve_count = 0

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        t0 = time.time()

        # --- Unfreeze backbone after freeze_epochs ---
        if freeze_epochs > 0 and epoch == start_epoch + freeze_epochs + 1:
            print(f"  Unfreezing backbone at epoch {epoch}")
            _unfreeze_backbone(model)
            # Rebuild optimizer with all parameters (backbone gets lower LR)
            backbone_params = [
                p for n, p in model.named_parameters()
                if "backbone.body" in n and p.requires_grad
            ]
            other_params = [
                p for n, p in model.named_parameters()
                if "backbone.body" not in n and p.requires_grad
            ]
            if optimizer_type == "sgd":
                momentum = getattr(config, "momentum", 0.9)
                optimizer = torch.optim.SGD(
                    [
                        {"params": backbone_params, "lr": config.learning_rate * 0.1},
                        {"params": other_params, "lr": config.learning_rate},
                    ],
                    momentum=momentum, weight_decay=config.weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(
                    [
                        {"params": backbone_params, "lr": config.learning_rate * 0.1},
                        {"params": other_params, "lr": config.learning_rate},
                    ],
                    weight_decay=config.weight_decay,
                    fused=fused,
                )
            # Rebuild scheduler for remaining epochs
            remaining = config.num_epochs - epoch + 1
            if scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-6
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=max(1, remaining // 2), gamma=0.1
                )
            # Reinitialize EMA with new parameters
            if use_ema:
                ema = ModelEMA(model, decay=ema_decay)
            print(f"  Optimizer rebuilt: backbone LR={config.learning_rate * 0.1:.6f}, heads LR={config.learning_rate:.6f}")

        # --- Train ---
        avg_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler,
            gradient_accumulation=grad_accum,
            multi_scale=multi_scale,
            ema=ema,
            amp_dtype=amp_dtype,
        )
        history["train_losses"].append(avg_losses)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # --- Validate (every val_frequency epochs, or last epoch) ---
        is_last_epoch = epoch == config.num_epochs
        should_validate = (epoch % val_freq == 0) or is_last_epoch

        if should_validate:
            # Use EMA weights for evaluation
            if ema is not None:
                ema.apply_shadow(model)

            # During training: use plain evaluate (no TTA/Soft-NMS) for speed.
            # TTA and Soft-NMS are only used at final evaluation time.
            predictions, targets = evaluate(
                model, val_loader, device, use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            from src.evaluate import compute_metrics
            pt = getattr(config, "patient_threshold", 0.3)
            metrics = compute_metrics(predictions, targets, patient_threshold=pt)
            history["val_metrics"].append(metrics)

            if ema is not None:
                ema.restore(model)

            ap50 = metrics.get("AP@0.5", 0.0)

            # Save best checkpoint (with EMA weights if available)
            if ap50 > best_ap:
                best_ap = ap50
                no_improve_count = 0
                if ema is not None:
                    ema.apply_shadow(model)
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "ap50": ap50},
                    ckpt_dir / f"{model_name}_best.pth",
                )
                if ema is not None:
                    ema.restore(model)
            else:
                no_improve_count += 1
        else:
            # Carry forward the most recent non-None val metric (skipped epochs append None).
            last_metric = next(
                (m for m in reversed(history["val_metrics"]) if m is not None), None
            )
            ap50 = last_metric.get("AP@0.5", 0.0) if last_metric is not None else 0.0
            history["val_metrics"].append(None)

        scheduler.step()
        elapsed = time.time() - t0
        history["epoch_times"].append(elapsed)

        val_str = f"AP@0.5={ap50:.4f}" if should_validate else "AP@0.5=skipped"
        print(
            f"[{model_name}] Epoch {epoch}/{config.num_epochs}  "
            f"loss={avg_losses['total_loss']:.4f}  "
            f"{val_str}  "
            f"lr={optimizer.param_groups[0]['lr']:.6f}  "
            f"time={elapsed:.1f}s"
        )

        # Save resume checkpoint on validation epochs only (avoids I/O overhead)
        if should_validate or is_last_epoch:
            _save_full_checkpoint(
                resume_path, epoch, model, optimizer, scheduler, scaler, best_ap, history, ema
            )
            # Also write a partial history.json so it survives if training is
            # killed before the final save in line ~830. Cheap (history is
            # small per epoch) and means watchdog backups always include the
            # latest curve. Without this, an interrupted run loses everything.
            partial_out = Path(config.output_dir)
            partial_out.mkdir(parents=True, exist_ok=True)
            with open(partial_out / f"{model_name}_history.json", "w") as f:
                json.dump(_make_serializable(history), f, indent=2)

        # --- Early stopping ---
        if patience > 0 and no_improve_count >= patience and should_validate:
            print(f"  Early stopping: no improvement for {patience} validations")
            break

    # Save final checkpoint (with EMA weights if available)
    if ema is not None:
        ema.apply_shadow(model)
    torch.save(
        {"epoch": config.num_epochs, "model_state_dict": model.state_dict()},
        ckpt_dir / f"{model_name}_final.pth",
    )
    if ema is not None:
        ema.restore(model)

    # Clean up resume checkpoint
    if resume_path.exists():
        resume_path.unlink()

    # Save history
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = _make_serializable(history)
    with open(out_dir / f"{model_name}_history.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return history


def _make_serializable(obj):
    """Recursively convert torch tensors and numpy arrays for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int64, np.int32)):
        return float(obj)
    return obj
