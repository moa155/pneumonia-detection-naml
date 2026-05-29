#!/usr/bin/env python3
"""Run inference once on the val split and cache predictions + targets to disk.

This unblocks all downstream analyses (bootstrap CIs, FROC, calibration,
learnt patient aggregator, threshold-holdout protocol, RSNA-bucket AP) without
having to re-run model inference each time.

Outputs (under results/predictions/):
    {model}_preds.pt   - list of {"boxes", "scores", "labels"} per image
    targets.pt         - list of {"boxes", "labels", "area"} per image
    val_index.json     - patient ID order for reproducibility

Usage:
    python scripts/cache_predictions.py
    python scripts/cache_predictions.py --no-tta            # faster, lower AP
    python scripts/cache_predictions.py --models fcos       # one model only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.engine import evaluate_model
from src.models import build_model
import main as M


def cache_one(name: str, val_loader, config, use_tta: bool, use_soft_nms: bool, out_dir: Path):
    ckpt_path = Path(config.checkpoint_dir) / f"{name}_best.pth"
    if not ckpt_path.exists():
        ckpt_path = Path(config.checkpoint_dir) / f"{name}_final.pth"
    if not ckpt_path.exists():
        print(f"  [{name}] no checkpoint found — skipping")
        return None

    print(f"  [{name}] loading {ckpt_path.name}")
    # Strip a trailing `_paper` (or any user suffix) to map back to the base
    # architecture name when building the model.
    arch_name = name
    for sfx in ("_paper",):
        if arch_name.endswith(sfx):
            arch_name = arch_name[: -len(sfx)]
            break
    model = build_model(
        arch_name, num_classes=config.num_classes, pretrained_backbone=False,
        min_size=config.image_min_size, max_size=config.image_max_size,
    )
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)

    t0 = time.time()
    preds, tgts = evaluate_model(
        model, val_loader, config.device,
        use_amp=False,           # MPS doesn't use CUDA AMP
        use_tta=use_tta,
        use_soft_nms=use_soft_nms,
    )
    dt = time.time() - t0
    print(f"  [{name}] inference done in {dt/60:.1f} min ({len(preds)} images)")

    # Compact: keep only what downstream code needs, on CPU
    compact_preds = [{
        "boxes": p["boxes"].detach().cpu(),
        "scores": p["scores"].detach().cpu(),
        "labels": p["labels"].detach().cpu(),
    } for p in preds]
    torch.save(compact_preds, out_dir / f"{name}_preds.pt")
    print(f"  [{name}] saved {out_dir / (name + '_preds.pt')}")

    # Free GPU/MPS memory
    del model
    if config.device.type == "cuda":
        torch.cuda.empty_cache()
    elif config.device.type == "mps":
        torch.mps.empty_cache()
    return tgts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["fcos", "fcos_paper", "retinanet", "faster_rcnn"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-soft-nms", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Train-side batch size; val loader uses 2x this.")
    args = ap.parse_args()

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_samples=args.max_samples,
        force_device=args.device,
        use_tta=not args.no_tta,
        use_soft_nms=not args.no_soft_nms,
        batch_size=args.batch_size,
        num_workers=0,
        prefetch_factor=2,
    )

    print(f"Device: {config.device}")
    print(f"TTA={config.use_tta}  SoftNMS={config.use_soft_nms}")
    print(f"max_samples={config.max_samples}")

    _, val_loader, val_dataset = M.build_data_loaders(config)

    out_dir = Path(config.output_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist val patient order for reproducibility
    (out_dir / "val_index.json").write_text(json.dumps(val_dataset.patient_ids))

    targets_saved = False
    for name in args.models:
        tgts = cache_one(name, val_loader, config, config.use_tta, config.use_soft_nms, out_dir)
        if tgts is not None and not targets_saved:
            compact_tgts = [{
                "boxes": t["boxes"].detach().cpu(),
                "labels": t["labels"].detach().cpu(),
                "area": t["area"].detach().cpu(),
                "iscrowd": t["iscrowd"].detach().cpu(),
            } for t in tgts]
            torch.save(compact_tgts, out_dir / "targets.pt")
            print(f"Saved targets to {out_dir / 'targets.pt'}")
            targets_saved = True

    print("Done.")


if __name__ == "__main__":
    main()
