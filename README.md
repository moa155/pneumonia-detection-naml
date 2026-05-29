# Pneumonia Detection: Anchor-Free vs. Anchor-Based Object Detection

**NAML (Numerical Analysis for Machine Learning)** course project — Politecnico di Milano, academic year 2025/26.

**Authors:** Mohamed Z. M. Mandour (`mohamedzeyad.mandour@mail.polimi.it`) and Elena Nuttini (`elena.nuttini@mail.polimi.it`).

This project reproduces and extends the method from:

> Wu et al., *“Pneumonia detection based on RSNA dataset and anchor-free deep learning detector”*, **Scientific Reports** **14**, 1929 (2024). DOI: [10.1038/s41598-024-52156-7](https://doi.org/10.1038/s41598-024-52156-7).

Four detector variants are implemented and compared for pneumonia localisation on chest X-rays from the **RSNA Pneumonia Detection Challenge**: anchor-free FCOS under our modern Adam recipe, the same FCOS retrained under the paper's original SGD recipe (controlled ablation), and the two anchor-based baselines RetinaNet and Faster R-CNN.

---

## TL;DR

All detectors surpass the paper’s reported FCOS AP@0.5 of 28.5%. Anchor-based detectors win on AP@0.5; switching FCOS from Adam to the paper’s SGD recipe brings FCOS to within statistical noise of RetinaNet (overlapping 95% bootstrap CIs). A Weighted Box Fusion ensemble of FCOS and RetinaNet matches Faster R-CNN on AP@0.5 while delivering the best patient-level F1 at the fixed τ=0.3 threshold.

| Model | AP@0.5 | ROC AUC | Patient F1 @ 0.3 | vs paper AP |
|---|---|---|---|---|
| Wu et al. 2024 (paper FCOS) | 28.5 | — | — | baseline |
| Our FCOS (Adam) | **35.3** | 86.5 | 40.2 | **+6.8** |
| Our FCOS (paper SGD ablation) | **39.8** | 88.1 | 43.8 | **+11.3** |
| Our RetinaNet | **41.2** | 89.1 | 49.7 | **+12.7** |
| Our Faster R-CNN | **42.9** | 88.9 | 53.2 | **+14.4** |
| **WBF ensemble (FCOS + RetinaNet)** | **42.0** | 87.5 | **63.0** | **+13.5** |

Full discussion, mathematical derivations, training recipe, and ablations are in [`report/report.pdf`](report/report.pdf).

---

## Methods

| # | Model | Type | Role |
|---|-------|------|------|
| 1 | **FCOS** | Anchor-free, one-stage | Paper's proposed method. Per-pixel prediction + center-ness branch + focal loss on FPN. |
| 2 | **FCOS (paper SGD)** | Anchor-free, one-stage | Same FCOS head, retrained with the paper's exact SGD recipe as an ablation. |
| 3 | **RetinaNet** | Anchor-based, one-stage | Comparison from paper Table 3. 9 anchors/location, focal loss, GIoU regression (v2). |
| 4 | **Faster R-CNN** | Anchor-based, two-stage | Comparison from paper Table 3. RPN proposals + ROI Align + per-class head (v2). |

All four share the same **ResNet-50** backbone + **Feature Pyramid Network** so that performance differences isolate the detection head / paradigm choice.

### Training pipeline (differences from the paper)

- 40 epochs, batch 32 (FCOS/RetinaNet) and 16 → 32 (Faster R-CNN), image size 512, Adam peak `lr=1e-3` (with 2-epoch warm-up + cosine annealing), weight decay `1e-4`. The `fcos_paper` ablation instead uses **SGD** (momentum 0.9, lr `1e-2`, multi-step decay) reproducing the paper's exact recipe.
- **BF16 mixed precision** on H100 Tensor Cores (no gradient scaler needed).
- **Cosine LR annealing** with a 2-epoch linear warm-up.
- **Exponential Moving Average (EMA)** of the weights, decay 0.999.
- **Medical-imaging augmentation** (CLAHE, RandomBrightnessContrast, Gamma, Affine, GridDistortion, CoarseDropout, horizontal flip — no vertical flip).
- **Weighted sampler** with 3× oversampling of positive patients.
- **Backbone frozen for the first 3 epochs** to stabilise early training.
- **Validation every 4 epochs** (10 validations over the 40-epoch run) with early-stopping patience of 5 — not triggered in the final run.
- **Test-time augmentation** (horizontal flip averaging) + **Gaussian Soft-NMS** (σ = 0.5) at evaluation.
- **Weighted Box Fusion (WBF)** ensemble of FCOS + RetinaNet with IoU = 0.55.

### Evaluation metrics

- COCO-style AP at IoU 0.5, AP@[0.5:0.95], AP/AR by object size.
- Patient-level accuracy, precision, recall, F1 at fixed confidence 0.3.
- ROC AUC and Youden-optimal threshold per model (threshold-free ranking quality).

---

## Dataset

**RSNA Pneumonia Detection Challenge** (Kaggle, 2018) — 26,684 chest X-rays (1024×1024 DICOM), ≈22% positive.

### Download (Kaggle CLI)

```bash
pip install kaggle
# Put kaggle.json in ~/.kaggle/ or export KAGGLE_API_TOKEN=KGAT_xxxxxxx
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/
unzip data/rsna-pneumonia-detection-challenge.zip -d data/
```

### Preprocess DICOM → PNG (recommended, ~10–50× faster loading)

```bash
python -m src.preprocess --data-dir data/
```

The dataset loader automatically prefers `stage_2_train_images_png/` if it exists.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.1, torchvision ≥ 0.16. CUDA GPU recommended; Apple Silicon MPS and CPU are supported.

---

## Reproducing the experiments

```bash
# Full pipeline (train 4 models + evaluate + plot)
python main.py --mode full --data-dir data --batch-size 32 --epochs 40 --image-size 512 --bf16 --num-workers 8

# Individual steps
python main.py --mode train    --model all --data-dir data
python main.py --mode evaluate --model all --data-dir data
python main.py --mode compare  --data-dir data    # runs eval + ensemble + all plots
python main.py --mode visualize --data-dir data   # per-image detection overlays
```

On 4× NVIDIA H100 80 GB SXM (RunPod), the full 4-model training (40 epochs each) finishes in **≈3 hours** with one model per GPU. The exact recipe used for the report is in `scripts/run_4gpu_pipeline.sh`:

```bash
# Manual per-GPU launch (what was actually run for the report)
python main.py --mode train --model fcos        --device cuda:0 --batch-size 32 --epochs 40 --bf16 &
python main.py --mode train --model fcos_paper  --device cuda:1 --batch-size 32 --epochs 40 --bf16 --optimizer sgd &
python main.py --mode train --model retinanet   --device cuda:2 --batch-size 32 --epochs 40 --bf16 &
python main.py --mode train --model faster_rcnn --device cuda:3 --batch-size 32 --epochs 40 --bf16 &
wait
```

### Notebook variants

- `Pneumonia_Detection_Kaggle.ipynb` — Kaggle T4 × 2 notebook (free tier).
- `Pneumonia_Detection_Colab.ipynb` — Google Colab notebook.

### Finalising the report from results

After `main.py --mode compare` has written `results/all_metrics.json`, run the analysis scripts in `scripts/` to (re)generate cached predictions, bootstrap CIs, FROC/calibration/sample plots, and the auto-generated tables:

```bash
python scripts/cache_predictions.py        # writes results/predictions/*.pt
python scripts/run_analyses.py             # bootstrap CIs, FROC, calibration, analyses.tex
python scripts/generate_missing_plots.py   # legacy comparison plots from cached preds
```

Then compile the LaTeX from the `report/` and `presentation/` directories (e.g. `pdflatex report.tex` run twice).

---

## Outputs

### `results/`

| File | Description |
|------|-------------|
| `training_loss.{png,pdf}` | Training loss curves for all four detector variants. |
| `val_ap_over_epochs.{png,pdf}` | Validation AP@0.5 per epoch. |
| `ap_comparison.{png,pdf}` | Grouped bar chart of AP metrics. |
| `ar_comparison.{png,pdf}` | Grouped bar chart of AR metrics. |
| `pr_curve.{png,pdf}` | Precision–Recall curves at IoU = 0.5. |
| `ap_vs_iou.{png,pdf}` | AP as a function of IoU threshold (0.5 → 0.95). |
| `classification_metrics.{png,pdf}` | Patient-level Accuracy / Precision / Recall / F1. |
| `detection_samples.{png,pdf}` | Ground-truth vs. per-model predictions on sample images. |
| `epoch_times.{png,pdf}` | Average epoch training time per model. |
| `learning_rate.png` | Learning-rate schedule (warm-up + cosine). |
| `froc.{png,pdf}` | FROC curves (sensitivity vs FP/image) with CPM operating points. |
| `calibration.{png,pdf}` | Reliability diagrams of patient-level max-scores; ECE in the legend. |
| `ap_bootstrap.{png,pdf}` | AP@0.5 with 95% bootstrap CIs (per-patient resampling, n=500). |
| `analyses.tex` | Auto-generated LaTeX tables: AP+CI / size-bucket / CPM and patient-level. |
| `all_metrics.json` | Legacy machine-readable metrics (PR-curve points etc.). |
| `all_metrics_v2.json` | New bootstrap-CI + calibration + FROC metrics. |
| `{model}_history.json` | Per-epoch training history (loss, LR, val metrics, timing). |

### `checkpoints/`

| File | Description |
|------|-------------|
| `{model}_best.pth` | Best checkpoint (highest val AP@0.5), uses EMA weights. |
| `{model}_final.pth` | Checkpoint after the last epoch. |

Not committed to git due to size (~120–165 MB per model). The 4 trained checkpoints from the final 40-epoch run are published as **[GitHub Release v1.0](https://github.com/moa155/pneumonia-detection-naml/releases/tag/v1.0)**:

```bash
gh release download v1.0 -R moa155/pneumonia-detection-naml -D checkpoints/
```

Or regenerate locally with `python main.py --mode train --model all`.

---

## Project layout

```
pneumonia-detection-naml/
├── main.py                         # CLI entry point (train / evaluate / compare / visualize / full)
├── regenerate_plots.py             # Re-evaluate + re-plot from existing checkpoints
├── requirements.txt                # Python dependencies
├── README.md                       # this file
├── Pneumonia_Detection_Kaggle.ipynb  # Kaggle notebook (T4 × 2, free)
├── Pneumonia_Detection_Colab.ipynb   # Google Colab notebook
├── src/
│   ├── config.py                   # Configuration dataclass
│   ├── dataset.py                  # RSNA dataset loader (DICOM + PNG)
│   ├── transforms.py               # Detection-aware medical-imaging augmentations
│   ├── analysis.py                 # Bootstrap, FROC, calibration, learnt aggregator
│   ├── models/
│   │   ├── fcos.py                 # FCOS head (paper method)
│   │   ├── retinanet.py            # RetinaNet v2 head (anchor-based, one-stage)
│   │   └── faster_rcnn.py          # Faster R-CNN v2 head (anchor-based, two-stage)
│   ├── engine.py                   # Training loop + EMA + cosine/warmup + TTA + Soft-NMS
│   ├── evaluate.py                 # COCO-style AP/AR + ROC AUC + optimal threshold
│   ├── ensemble.py                 # Weighted Box Fusion (Solovyev et al., 2021)
│   ├── visualize.py                # Plotting + LaTeX table generation
│   └── preprocess.py               # DICOM → PNG converter (multiprocessing)
├── scripts/
│   ├── cache_predictions.py        # Cache per-model val predictions to disk (.pt)
│   ├── run_analyses.py             # Bootstrap CIs, FROC, calibration, paired tests
│   ├── generate_missing_plots.py   # Legacy comparison plots from cached predictions
│   ├── run_4gpu_pipeline.sh        # Parallel 4-GPU training driver (RunPod)
│   ├── run_full_pipeline.sh        # Single-GPU end-to-end driver
│   ├── bootstrap_pod.sh            # Clone repo onto /workspace for persistence
│   ├── cloud_setup.sh              # RunPod / Vast.ai bootstrap (Kaggle creds, pip deps)
│   ├── status.sh                   # Read-only training/disk/backup status report
│   ├── watchdog_backup.sh          # Periodic upload of artefacts to litterbox (offsite)
│   ├── smoke_check.py              # Fast sanity check of dataset + model + 1 train step
│   └── finalize_tex.py             # Populate report/presentation tables from metrics JSON
├── report/
│   ├── report.tex                  # Full report (LaTeX source)
│   └── report.pdf                  # Compiled output
├── presentation/
│   ├── presentation.tex            # Beamer slide deck (LaTeX source)
│   └── presentation.pdf            # Compiled output
├── docs/
│   ├── Pneumonia_detection.pdf     # Reference paper (Wu et al., 2024)
│   └── logo_polimi.png             # Politecnico di Milano logo (title pages)
├── results/                        # Committed: plots, metrics, history JSONs, cached predictions
├── data/                           # RSNA dataset (gitignored — download separately)
└── checkpoints/                    # Trained weights (gitignored — see GitHub Release v1.0)
```

---

## Paper reference

```bibtex
@article{wu2024pneumonia,
  title={Pneumonia detection based on {RSNA} dataset and anchor-free deep learning detector},
  author={Wu, Linghua and Zhang, Jing and Wang, Yilin and Ding, Rong and Cao, Yueqin
          and Liu, Guiqin and Liufu, Changsheng and Xie, Baowei and Kang, Shanping
          and Liu, Rui and Li, Wenle and Guan, Furen},
  journal={Scientific Reports},
  volume={14},
  pages={1929},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-024-52156-7}
}
```

---

## License & attribution

Code released for academic purposes as part of the NAML course at Politecnico di Milano. The RSNA dataset is distributed by the Radiological Society of North America under its own license terms; see the [Kaggle competition page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) for details.

Implementation based on `torchvision` reference detectors (FCOS, RetinaNet v2, Faster R-CNN v2). Weighted Box Fusion follows the formulation of Solovyev et al.\ (2021). TTA + Soft-NMS follow Bodla et al.\ (2017) and standard practice.
