#!/usr/bin/env python3
"""Pre-flight smoke check before launching the cloud training run.

Verifies that all imports work, GPUs are visible, albumentations transforms
build and apply, models build, and analysis modules import. Exits 0 if all
checks pass, 1 otherwise.

Usage:
    python scripts/smoke_check.py
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

checks = []
print(f"Python: {sys.version.split()[0]}")

# 1) Torch + CUDA
try:
    import torch
    import torchvision
    import numpy as np
    print(f"  torch={torch.__version__}  torchvision={torchvision.__version__}  numpy={np.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}  devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
    checks.append(("torch+cuda", True))
except Exception as e:
    print(f"  FAIL torch: {e}")
    traceback.print_exc()
    checks.append(("torch+cuda", False))

# 2) Albumentations (likely 1.x vs 2.x breakage)
try:
    import albumentations as A
    print(f"  albumentations={A.__version__}")
    from src.transforms import get_train_transforms, get_val_transforms
    t_train = get_train_transforms(True)
    t_val = get_val_transforms()
    # DetectionTransform signature is (image: np.ndarray, target: Dict) -> (Tensor, Dict)
    dummy = np.random.rand(512, 512, 3).astype(np.float32)
    dummy_target = {
        "boxes": torch.tensor([[10., 10., 100., 100.]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([0]),
        "area": torch.tensor([8100.]),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    img_t, tgt_t = t_train(dummy, dummy_target)
    print(f"  train transform OK (image shape {tuple(img_t.shape)}, {len(tgt_t['boxes'])} boxes back)")
    img_t, tgt_t = t_val(dummy, dummy_target)
    print(f"  val transform OK (image shape {tuple(img_t.shape)})")
    checks.append(("albumentations", True))
except Exception as e:
    print(f"  FAIL albumentations: {type(e).__name__}: {e}")
    traceback.print_exc()
    checks.append(("albumentations", False))

# 3) Models (architecture only, no checkpoint)
try:
    from src.models import build_model
    for n in ("fcos", "retinanet", "faster_rcnn"):
        m = build_model(n, num_classes=2, pretrained_backbone=True, min_size=512, max_size=512)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {n}: {n_params/1e6:.1f}M params")
    checks.append(("models", True))
except Exception as e:
    print(f"  FAIL models: {e}")
    traceback.print_exc()
    checks.append(("models", False))

# 4) Evaluate + analysis modules
try:
    from src.evaluate import compute_metrics
    from src.analysis import bootstrap_ap50, learnt_aggregator
    print(f"  evaluate + analysis modules OK")
    checks.append(("eval/analysis", True))
except Exception as e:
    print(f"  FAIL eval/analysis: {e}")
    traceback.print_exc()
    checks.append(("eval/analysis", False))

# 5) Ensemble
try:
    from src.ensemble import ensemble_predictions
    print(f"  ensemble OK")
    checks.append(("ensemble", True))
except Exception as e:
    print(f"  FAIL ensemble: {e}")
    traceback.print_exc()
    checks.append(("ensemble", False))

# 6) Pipeline script syntactic check
try:
    import subprocess
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_full_pipeline.sh")],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  run_full_pipeline.sh syntax OK")
        checks.append(("pipeline.sh", True))
    else:
        print(f"  FAIL pipeline.sh: {r.stderr}")
        checks.append(("pipeline.sh", False))
except Exception as e:
    print(f"  FAIL pipeline.sh: {e}")
    checks.append(("pipeline.sh", False))

print()
print("=" * 50)
all_ok = all(ok for _, ok in checks)
for name, ok in checks:
    marker = "OK  " if ok else "FAIL"
    print(f"  {marker}  {name}")
print("=" * 50)
print("ALL GREEN -- safe to proceed" if all_ok else "FAILURES -- fix before training")
sys.exit(0 if all_ok else 1)
