#!/usr/bin/env python3
"""Run all post-hoc analyses from cached predictions.

Produces:
  results/all_metrics_v2.json         — full metric table for the report
  results/froc.png / froc.pdf
  results/calibration.png / calibration.pdf
  results/ap_bootstrap.png / ap_bootstrap.pdf
  results/analyses.tex                 — LaTeX fragments to \\input{} from the
                                         report (tables of new numbers)

Usage:
  python scripts/run_analyses.py
  python scripts/run_analyses.py --pred-dir results/predictions --n-boot 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import (
    load_cached, bootstrap_ap50, bootstrap_patient_f1,
    threshold_holdout_metrics, ap_by_rsna_buckets, froc_curve,
    calibration, learnt_aggregator, paired_ap_test,
)
from src.ensemble import ensemble_predictions

MODEL_NAMES = ["fcos", "fcos_paper", "retinanet", "faster_rcnn"]
MODEL_LABELS = {
    "fcos": "FCOS", "fcos_paper": "FCOS (paper SGD)",
    "retinanet": "RetinaNet",
    "faster_rcnn": "Faster R-CNN", "ensemble": "Ensemble (FCOS+Retina, WBF)",
}
MODEL_COLORS = {
    "fcos": "#2196F3", "fcos_paper": "#0D47A1",
    "retinanet": "#FF9800",
    "faster_rcnn": "#4CAF50", "ensemble": "#9C27B0",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--cal-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Override default model list (e.g. fcos fcos_paper retinanet faster_rcnn)")
    args = ap.parse_args()
    model_list = args.models if args.models else MODEL_NAMES

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    preds, targets = load_cached(pred_dir, model_list)
    print(f"Loaded {list(preds.keys())} preds, {len(targets)} targets")

    if "fcos" in preds and "retinanet" in preds:
        print("Building FCOS+RetinaNet ensemble via WBF...")
        preds["ensemble"] = ensemble_predictions(
            {k: preds[k] for k in ("fcos", "retinanet")},
            iou_thr=0.55, skip_box_thr=0.01,
        )

    results = {}
    for name, pred_list in preds.items():
        print(f"\n=== {name.upper()} ===")
        boot_ap = bootstrap_ap50(pred_list, targets, n_boot=args.n_boot, seed=args.seed)
        print(f"  AP@0.5 = {boot_ap[0]*100:.2f}% [95% CI {boot_ap[1]*100:.2f}, {boot_ap[2]*100:.2f}]")

        # threshold-holdout protocol (no train-on-test leakage)
        holdout = threshold_holdout_metrics(
            pred_list, targets, cal_fraction=args.cal_fraction, seed=args.seed,
        )
        print(f"  Holdout protocol: tau={holdout['threshold']:.3f}, "
              f"F1_test={holdout['patient_f1_test']*100:.2f}%, "
              f"AP_test={holdout['AP@0.5_test']*100:.2f}%")

        # bootstrap F1 at the held-out threshold, on the *full* val (point est)
        boot_f1 = bootstrap_patient_f1(
            pred_list, targets, threshold=holdout["threshold"],
            n_boot=args.n_boot * 4, seed=args.seed,
        )
        print(f"  F1 @ holdout-tau = {boot_f1[0]*100:.2f}% [95% CI {boot_f1[1]*100:.2f}, {boot_f1[2]*100:.2f}]")

        rsna = ap_by_rsna_buckets(pred_list, targets)
        print(f"  RSNA buckets: AP_S={rsna['AP_S_rsna']*100:.2f}%, "
              f"AP_M={rsna['AP_M_rsna']*100:.2f}%, "
              f"AP_L={rsna['AP_L_rsna']*100:.2f}%  "
              f"(boundaries: {rsna['p33_area']:.0f}, {rsna['p67_area']:.0f} px²)")

        froc = froc_curve(pred_list, targets)
        print(f"  FROC CPM = {froc['cpm']*100:.2f}%")

        calib = calibration(pred_list, targets, n_bins=10)
        print(f"  ECE = {calib['ece']*100:.2f}%")

        agg = learnt_aggregator(pred_list, targets, n_folds=5, seed=args.seed)
        print(f"  Learnt aggregator OOF F1 = {agg.val_f1*100:.2f}% "
              f"(CV per-fold {agg.cv_mean_f1*100:.2f}±{agg.cv_std_f1*100:.2f}%)")

        results[name] = {
            "ap50": boot_ap[0], "ap50_lo": boot_ap[1], "ap50_hi": boot_ap[2],
            "f1_holdout_point": boot_f1[0],
            "f1_holdout_lo": boot_f1[1], "f1_holdout_hi": boot_f1[2],
            "holdout": holdout, "rsna_buckets": rsna,
            "froc_cpm": froc["cpm"],
            "froc_curve": {"fp": froc["fp_per_image"].tolist(),
                           "sens": froc["sensitivity"].tolist()},
            "calibration": {
                "ece": calib["ece"],
                "bin_edges": calib["bin_edges"].tolist(),
                "bin_conf": calib["bin_conf"].tolist(),
                "bin_acc": calib["bin_acc"].tolist(),
                "bin_count": calib["bin_count"].tolist(),
            },
            "aggregator": {
                "oof_f1": agg.val_f1, "oof_precision": agg.val_precision,
                "oof_recall": agg.val_recall, "oof_accuracy": agg.val_accuracy,
                "threshold": agg.threshold,
                "cv_mean_f1": agg.cv_mean_f1, "cv_std_f1": agg.cv_std_f1,
            },
        }

    # Paired tests between top single models
    if "retinanet" in preds and "faster_rcnn" in preds:
        diff, lo, hi, p = paired_ap_test(
            preds["faster_rcnn"], preds["retinanet"], targets,
            n_boot=args.n_boot, seed=args.seed,
        )
        results["paired_test_frcnn_vs_retina"] = {
            "diff": diff, "lo95": lo, "hi95": hi, "p_two_sided": p,
        }
        print(f"\nPaired AP@0.5 Faster R-CNN − RetinaNet: "
              f"{diff*100:+.2f} pts [{lo*100:+.2f}, {hi*100:+.2f}], p={p:.3f}")

    # Save JSON
    (out_dir / "all_metrics_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir / 'all_metrics_v2.json'}")

    # ---- Plots ----
    plot_froc(results, out_dir)
    plot_calibration(results, out_dir)
    plot_ap_bootstrap(results, out_dir)

    # ---- LaTeX fragment for the report ----
    write_latex(results, out_dir / "analyses.tex")
    print(f"Saved {out_dir / 'analyses.tex'}")


def plot_froc(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        if "froc_curve" not in r: continue
        fp = np.array(r["froc_curve"]["fp"])
        sens = np.array(r["froc_curve"]["sens"])
        if len(fp) == 0: continue
        ax.plot(fp, sens, label=f"{MODEL_LABELS.get(name, name)} (CPM={r['froc_cpm']*100:.1f})",
                color=MODEL_COLORS.get(name), linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50)
    ax.set_xlabel("False positives per image (log)")
    ax.set_ylabel("Sensitivity (recall @ IoU=0.5)")
    ax.set_title("FROC curves on RSNA validation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "froc.png", dpi=150)
    fig.savefig(out_dir / "froc.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'froc.png'}")


def plot_calibration(results, out_dir):
    items = [(n, r) for n, r in results.items() if "calibration" in r and r["calibration"]]
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4), squeeze=False)
    for ax, (name, r) in zip(axes[0], items):
        c = r["calibration"]
        edges = np.array(c["bin_edges"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        acc = np.array(c["bin_acc"]); cnt = np.array(c["bin_count"])
        mask = cnt > 0
        ax.bar(centers[mask], acc[mask], width=0.08,
               color=MODEL_COLORS.get(name), alpha=0.75, label="empirical")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted max score")
        ax.set_ylabel("Empirical TPR")
        ax.set_title(f"{MODEL_LABELS.get(name, name)}\nECE = {c['ece']*100:.1f}%", fontsize=10)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    fig.savefig(out_dir / "calibration.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'calibration.png'}")


def plot_ap_bootstrap(results, out_dir):
    names = [n for n in results if "ap50" in results[n]]
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(names))
    points = np.array([results[n]["ap50"] for n in names]) * 100
    los = np.array([results[n]["ap50_lo"] for n in names]) * 100
    his = np.array([results[n]["ap50_hi"] for n in names]) * 100
    err = np.stack([points - los, his - points])
    colors = [MODEL_COLORS.get(n, "#888") for n in names]
    ax.bar(xs, points, color=colors, alpha=0.85)
    ax.errorbar(xs, points, yerr=err, fmt="none", ecolor="black", capsize=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS.get(n, n) for n in names], rotation=20, ha="right")
    ax.set_ylabel("AP@0.5 (%) with 95% bootstrap CI")
    ax.set_title("Detection AP@0.5 with patient-level bootstrap CIs (n=500)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "ap_bootstrap.png", dpi=150)
    fig.savefig(out_dir / "ap_bootstrap.pdf")
    plt.close(fig)
    print(f"Saved {out_dir / 'ap_bootstrap.png'}")


def write_latex(results, path):
    """Write a LaTeX fragment with the new tables for the report."""
    rows_ap = []
    for n, r in results.items():
        if "ap50" not in r: continue
        rows_ap.append(
            f"{MODEL_LABELS.get(n, n)} & "
            f"{r['ap50']*100:.1f} [{r['ap50_lo']*100:.1f}, {r['ap50_hi']*100:.1f}] & "
            f"{r['rsna_buckets']['AP_S_rsna']*100:.1f} & "
            f"{r['rsna_buckets']['AP_M_rsna']*100:.1f} & "
            f"{r['rsna_buckets']['AP_L_rsna']*100:.1f} & "
            f"{r['froc_cpm']*100:.1f} \\\\"
        )

    rows_pt = []
    for n, r in results.items():
        if "holdout" not in r: continue
        h = r["holdout"]; a = r["aggregator"]
        rows_pt.append(
            f"{MODEL_LABELS.get(n, n)} & "
            f"{h['threshold']:.3f} & "
            f"{h['patient_f1_test']*100:.1f} & "
            f"{r['f1_holdout_point']*100:.1f} [{r['f1_holdout_lo']*100:.1f}, {r['f1_holdout_hi']*100:.1f}] & "
            f"{a['oof_f1']*100:.1f} & "
            f"{r['calibration']['ece']*100:.1f} \\\\"
        )

    paired_block = ""
    if "paired_test_frcnn_vs_retina" in results:
        p = results["paired_test_frcnn_vs_retina"]
        sig = ("statistically significant" if p["p_two_sided"] < 0.05
               else "not statistically significant")
        paired_block = (
            "\\paragraph{Paired bootstrap of the top two anchor-based detectors.}\n"
            f"On the same validation patients, Faster R-CNN exceeds RetinaNet on AP@0.5 by "
            f"${p['diff']*100:+.2f}$ points with a $95\\%$ paired-bootstrap CI of "
            f"$[{p['lo95']*100:+.2f}, {p['hi95']*100:+.2f}]$ and a two-sided $p$-value of "
            f"${p['p_two_sided']:.3f}$. The difference is therefore "
            f"{sig} at the $5\\%$ level on this split.\n"
        )

    content = (
        "% Auto-generated by scripts/run_analyses.py — do not edit by hand.\n"
        "\\begin{table}[H]\n"
        "\\centering\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Detection AP@0.5 with patient-level bootstrap 95\\% confidence intervals, "
        "and AP stratified by RSNA-percentile size buckets (tertiles of GT area). "
        "FROC CPM is the mean sensitivity at $\\{0.125, 0.25, 0.5, 1, 2, 4, 8\\}$ FP/image.}\n"
        "\\label{tab:robust_detection}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & AP@0.5 [95\\% CI] & $AP_S^{\\mathrm{R}}$ & $AP_M^{\\mathrm{R}}$ & $AP_L^{\\mathrm{R}}$ & CPM \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_ap) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\begin{table}[H]\n"
        "\\centering\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{Patient-level classification under a rigorous protocol. "
        "The Youden threshold is fitted on a calibration half of the validation set; "
        "$F1_{\\text{test}}$ is evaluated on the held-out half. "
        "$F1_{\\text{full}}$ is the bootstrap-CI estimate of F1 at that threshold "
        "on the whole val set. The Learnt-Agg.\\ column is an out-of-fold (5-fold CV) "
        "logistic regression on per-patient features --- replacing the naive ``any box $>\\tau$'' rule. "
        "ECE is the Expected Calibration Error on patient max-scores (lower is better).}\n"
        "\\label{tab:robust_patient}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & $\\tau^{\\star}_{\\text{cal}}$ & $F1_{\\text{test}}$ & "
        "$F1_{\\text{full}}$ [95\\% CI] & Learnt-Agg. & ECE \\\\\n"
        "\\midrule\n"
        + "\n".join(rows_pt) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        + paired_block
    )
    path.write_text(content)


if __name__ == "__main__":
    main()
