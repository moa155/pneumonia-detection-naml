"""Post-hoc analyses that run from cached predictions.

These analyses are how we turn the single training run we have into a more
defensible scientific result without re-training:

  * Paired patient-level bootstrap CIs on AP@0.5 and patient F1.
  * Three-way split: Youden threshold on a calibration half, F1 reported on
    a held-out test half — eliminates threshold-on-test data leakage.
  * RSNA-percentile size-bucket AP (instead of the COCO buckets that put 99%
    of opacities in "large").
  * Free-Response ROC (FROC) curve — the canonical localisation metric in
    medical-image detection.
  * Reliability diagram / Expected Calibration Error on patient scores.
  * Learnt patient-level aggregator: replace the naive "any box > tau" rule
    with a small logistic regression on (max, mean, count, top-k mean,
    spatial spread) features.

All inputs are the cached `predictions/*.pt` files produced by
`scripts/cache_predictions.py`; no model checkpoint or GPU is required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.evaluate import compute_ap_at_iou, find_optimal_patient_threshold


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------

def load_cached(
    pred_dir: Path, model_names: Sequence[str]
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Load cached predictions + shared targets from `pred_dir`."""
    pred_dir = Path(pred_dir)
    preds: Dict[str, List[Dict]] = {}
    for name in model_names:
        path = pred_dir / f"{name}_preds.pt"
        if path.exists():
            preds[name] = torch.load(path, weights_only=False)
    targets = torch.load(pred_dir / "targets.pt", weights_only=False)
    return preds, targets


# -----------------------------------------------------------------------
# Paired patient-level bootstrap
# -----------------------------------------------------------------------

def bootstrap_ap50(
    predictions: List[Dict], targets: List[Dict],
    n_boot: int = 1000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Patient-level bootstrap of AP@0.5: returns (point estimate, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(predictions)
    point = compute_ap_at_iou(predictions, targets, iou_threshold=0.5)["AP"]
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_b = [predictions[i] for i in idx]
        t_b = [targets[i] for i in idx]
        samples[b] = compute_ap_at_iou(p_b, t_b, iou_threshold=0.5)["AP"]
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def bootstrap_patient_f1(
    predictions: List[Dict], targets: List[Dict],
    threshold: float, n_boot: int = 2000, seed: int = 0,
) -> Tuple[float, float, float]:
    """Patient-level bootstrap of F1 at a fixed threshold."""
    rng = np.random.default_rng(seed)
    n = len(predictions)
    # Per-patient (has_gt, has_pred) so the bootstrap is cheap.
    has_gt = np.array([len(t["boxes"]) > 0 for t in targets])
    max_sc = np.array([
        (p["scores"].max().item() if len(p["scores"]) > 0 else 0.0)
        for p in predictions
    ])
    has_pred = max_sc > threshold

    def _f1(idx):
        gt, pr = has_gt[idx], has_pred[idx]
        tp = int((gt & pr).sum()); fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-8)

    point = _f1(np.arange(n))
    samples = np.array([_f1(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(point), float(lo), float(hi)


# -----------------------------------------------------------------------
# Three-way split: threshold on calibration half, eval on test half
# -----------------------------------------------------------------------

def threshold_holdout_metrics(
    predictions: List[Dict], targets: List[Dict],
    cal_fraction: float = 0.5, seed: int = 0,
) -> Dict[str, float]:
    """Eliminate threshold-on-test leakage by splitting val patients.

    Returns AP@0.5 (unchanged by the split), the Youden threshold found on
    the calibration half, and patient F1/precision/recall evaluated on the
    held-out test half at that threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(round(cal_fraction * n))
    cal_idx, eval_idx = idx[:cut], idx[cut:]
    cal_p = [predictions[i] for i in cal_idx]
    cal_t = [targets[i] for i in cal_idx]
    eval_p = [predictions[i] for i in eval_idx]
    eval_t = [targets[i] for i in eval_idx]

    tau, _ = find_optimal_patient_threshold(cal_p, cal_t)
    _, roc_eval = find_optimal_patient_threshold(eval_p, eval_t)

    has_gt = np.array([len(t["boxes"]) > 0 for t in eval_t])
    max_sc = np.array([
        (p["scores"].max().item() if len(p["scores"]) > 0 else 0.0)
        for p in eval_p
    ])
    has_pred = max_sc > tau
    tp = int((has_gt & has_pred).sum())
    fp = int((~has_gt & has_pred).sum())
    fn = int((has_gt & ~has_pred).sum())
    tn = int((~has_gt & ~has_pred).sum())
    total = tp + fp + tn + fn
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(total, 1)

    ap50 = compute_ap_at_iou(eval_p, eval_t, iou_threshold=0.5)["AP"]
    return {
        "threshold": float(tau),
        "AP@0.5_test": float(ap50),
        "patient_accuracy_test": acc,
        "patient_precision_test": prec,
        "patient_recall_test": rec,
        "patient_f1_test": f1,
        "roc_auc_test": float(roc_eval),
        "cal_n": int(cut), "test_n": int(n - cut),
    }


# -----------------------------------------------------------------------
# RSNA-percentile AP by size bucket
# -----------------------------------------------------------------------

def ap_by_rsna_buckets(
    predictions: List[Dict], targets: List[Dict],
) -> Dict[str, float]:
    """AP@0.5 stratified by RSNA-percentile size buckets (small/med/large)."""
    areas = np.concatenate([
        t["area"].numpy() for t in targets if len(t["boxes"]) > 0
    ]) if any(len(t["boxes"]) > 0 for t in targets) else np.array([0.0])
    p33, p67 = float(np.percentile(areas, 33)), float(np.percentile(areas, 67))

    def _filter(lo, hi):
        out = []
        for t in targets:
            a = t["area"]
            m = (a >= lo) & (a < hi)
            out.append({
                "boxes": t["boxes"][m] if len(t["boxes"]) > 0 else t["boxes"],
                "labels": t["labels"][m] if len(t["labels"]) > 0 else t["labels"],
                "area": a[m] if len(a) > 0 else a,
                "iscrowd": t["iscrowd"][m] if len(t["iscrowd"]) > 0 else t["iscrowd"],
            })
        return out

    return {
        "AP_S_rsna": float(compute_ap_at_iou(predictions, _filter(0, p33), iou_threshold=0.5)["AP"]),
        "AP_M_rsna": float(compute_ap_at_iou(predictions, _filter(p33, p67), iou_threshold=0.5)["AP"]),
        "AP_L_rsna": float(compute_ap_at_iou(predictions, _filter(p67, float("inf")), iou_threshold=0.5)["AP"]),
        "p33_area": p33, "p67_area": p67,
    }


# -----------------------------------------------------------------------
# FROC: average sensitivity vs. false positives per image
# -----------------------------------------------------------------------

def froc_curve(
    predictions: List[Dict], targets: List[Dict],
    iou_thr: float = 0.5,
    fp_points: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> Dict[str, np.ndarray]:
    """Compute the FROC curve: sensitivity (TPR) as a function of FP/image.

    Returns dict with arrays `fp_per_image`, `sensitivity` (matched length),
    `score_threshold` (per-point), and the standard CPM = mean sensitivity
    over `fp_points`.
    """
    from src.evaluate import compute_iou_matrix
    n_images = len(predictions)
    total_gt = 0
    all_score, all_tp = [], []
    for pred, gt in zip(predictions, targets):
        total_gt += len(gt["boxes"])
        if len(pred["boxes"]) == 0:
            continue
        order = pred["scores"].argsort(descending=True)
        pb = pred["boxes"][order]; ps = pred["scores"][order]
        matched = set()
        if len(gt["boxes"]) > 0:
            ious = compute_iou_matrix(pb, gt["boxes"])
        else:
            ious = None
        for i in range(len(pb)):
            all_score.append(ps[i].item())
            if ious is None:
                all_tp.append(0); continue
            row = ious[i]
            best_v, best_idx = row.max(0)
            if best_v.item() >= iou_thr and best_idx.item() not in matched:
                all_tp.append(1); matched.add(best_idx.item())
            else:
                all_tp.append(0)
    if total_gt == 0 or not all_score:
        return {"fp_per_image": np.array([]), "sensitivity": np.array([]),
                "score_threshold": np.array([]), "cpm": 0.0}
    order = np.argsort(-np.array(all_score))
    tp_arr = np.array(all_tp)[order]
    fp_arr = 1 - tp_arr
    score_arr = np.array(all_score)[order]
    tp_cum = np.cumsum(tp_arr); fp_cum = np.cumsum(fp_arr)
    sens = tp_cum / total_gt
    fp_per_img = fp_cum / n_images
    # CPM: mean sensitivity at the standard FP rates by step interpolation.
    cpm_vals = []
    for fp_target in fp_points:
        mask = fp_per_img <= fp_target
        cpm_vals.append(float(sens[mask].max()) if mask.any() else 0.0)
    return {
        "fp_per_image": fp_per_img, "sensitivity": sens,
        "score_threshold": score_arr,
        "cpm": float(np.mean(cpm_vals)),
        "fp_points": np.array(fp_points), "cpm_values": np.array(cpm_vals),
    }


# -----------------------------------------------------------------------
# Calibration: reliability diagram + ECE on patient max-scores
# -----------------------------------------------------------------------

def calibration(
    predictions: List[Dict], targets: List[Dict], n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Reliability diagram + Expected Calibration Error on patient max-scores."""
    scores = np.array([
        (p["scores"].max().item() if len(p["scores"]) > 0 else 0.0)
        for p in predictions
    ])
    labels = np.array([1 if len(t["boxes"]) > 0 else 0 for t in targets])
    edges = np.linspace(0, 1, n_bins + 1)
    bin_conf = np.zeros(n_bins); bin_acc = np.zeros(n_bins); bin_cnt = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (scores > lo) & (scores <= hi) if b > 0 else (scores >= lo) & (scores <= hi)
        if m.any():
            bin_cnt[b] = m.sum()
            bin_conf[b] = scores[m].mean()
            bin_acc[b] = labels[m].mean()
    total = bin_cnt.sum()
    ece = (bin_cnt / max(total, 1) * np.abs(bin_acc - bin_conf)).sum()
    return {
        "bin_edges": edges, "bin_conf": bin_conf, "bin_acc": bin_acc,
        "bin_count": bin_cnt, "ece": float(ece),
    }


# -----------------------------------------------------------------------
# Learnt patient-level aggregator
# -----------------------------------------------------------------------

@dataclass
class AggregatorResult:
    val_f1: float; val_precision: float; val_recall: float; val_accuracy: float
    threshold: float; cv_mean_f1: float; cv_std_f1: float


def _patient_features(p: Dict) -> np.ndarray:
    """Per-patient feature vector for the learnt aggregator."""
    s = p["scores"].numpy() if hasattr(p["scores"], "numpy") else np.asarray(p["scores"])
    b = p["boxes"].numpy() if hasattr(p["boxes"], "numpy") else np.asarray(p["boxes"])
    if len(s) == 0:
        return np.zeros(7, dtype=np.float32)
    s_sorted = np.sort(s)[::-1]
    top3 = s_sorted[:3]
    pad = np.zeros(3); pad[:len(top3)] = top3
    spatial = b[:, [0, 1]].std(0).mean() if len(b) > 1 else 0.0
    return np.array([
        float(s.max()), float(s.mean()), float(s.sum()),
        float(len(s)), float(pad[0]), float(pad[1]), float(spatial),
    ], dtype=np.float32)


def learnt_aggregator(
    predictions: List[Dict], targets: List[Dict],
    n_folds: int = 5, seed: int = 0,
) -> AggregatorResult:
    """Cross-validated logistic regression on per-patient features.

    Replaces the "any box > tau" rule with a 7-feature classifier
    (max, mean, sum, count, top-3 scores, spatial std). All numbers are
    out-of-fold so there is no train-on-test leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.stack([_patient_features(p) for p in predictions])
    y = np.array([1 if len(t["boxes"]) > 0 else 0 for t in targets])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_f1, oof_proba, oof_label, oof_idx = [], [], [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 C=1.0, solver="lbfgs").fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        oof_proba.append(proba); oof_label.append(y[te]); oof_idx.append(te)
        # Per-fold F1 at Youden threshold from train
        train_proba = clf.predict_proba(X[tr])[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y[tr], train_proba)
        j = tpr - fpr; tau_f = float(thr[j.argmax()])
        pred_f = proba > tau_f
        tp = int(((y[te] == 1) & pred_f).sum())
        fp = int(((y[te] == 0) & pred_f).sum())
        fn = int(((y[te] == 1) & ~pred_f).sum())
        prec_f = tp / max(tp + fp, 1); rec_f = tp / max(tp + fn, 1)
        fold_f1.append(2 * prec_f * rec_f / max(prec_f + rec_f, 1e-8))

    proba_all = np.concatenate(oof_proba); lab_all = np.concatenate(oof_label)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(lab_all, proba_all)
    j = tpr - fpr; tau = float(thr[j.argmax()])
    pred = proba_all > tau
    tp = int(((lab_all == 1) & pred).sum()); fp = int(((lab_all == 0) & pred).sum())
    fn = int(((lab_all == 1) & ~pred).sum()); tn = int(((lab_all == 0) & ~pred).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return AggregatorResult(
        val_f1=f1, val_precision=prec, val_recall=rec, val_accuracy=acc,
        threshold=tau, cv_mean_f1=float(np.mean(fold_f1)),
        cv_std_f1=float(np.std(fold_f1)),
    )


# -----------------------------------------------------------------------
# Convenience: paired test of two AP@0.5 on the same patients
# -----------------------------------------------------------------------

def paired_ap_test(
    preds_a: List[Dict], preds_b: List[Dict], targets: List[Dict],
    n_boot: int = 1000, seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap of AP@0.5_a - AP@0.5_b on the same patients.

    Returns (mean diff, lo95, hi95, p_value_two_sided).
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    point_a = compute_ap_at_iou(preds_a, targets, iou_threshold=0.5)["AP"]
    point_b = compute_ap_at_iou(preds_b, targets, iou_threshold=0.5)["AP"]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pa = [preds_a[j] for j in idx]; pb = [preds_b[j] for j in idx]
        tt = [targets[j] for j in idx]
        diffs[i] = (compute_ap_at_iou(pa, tt, 0.5)["AP"]
                    - compute_ap_at_iou(pb, tt, 0.5)["AP"])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(point_a - point_b), float(lo), float(hi), float(p_two)
