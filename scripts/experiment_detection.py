"""Detection-quality experiments on a TRAINING validation split only.

Methodology (no test-set leakage):
- KDDTrain+ is split stratified 80/20 (seed 42) into fit/validation.
- Preprocessing is fitted on the fit split only.
- Candidate Stage B configurations (class weighting, decision-threshold
  calibration) are trained on the fit split and compared on validation.
- Stage A (Isolation Forest) is trained on NORMAL fit-split records only;
  threshold candidates are compared on validation.
- KDDTest+ is NEVER read by this script. The winning configuration is
  later retrained on all training data and evaluated once on KDDTest+
  by train.py.

Usage: python scripts/experiment_detection.py
Output: results/metrics/detection_experiment.json + console summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
from src import data_loader, evaluation
from src.preprocessing import Preprocessor

SEED = config.RANDOM_SEED
CLASSES = config.ATTACK_CATEGORIES  # Normal, DoS, Probe, R2L, U2R


def log(msg: str) -> None:
    print(f"[experiment] {msg}", flush=True)


def per_class_recalls(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for c in CLASSES:
        mask = y_true == c
        out[c] = round(float((y_pred[mask] == c).mean()), 4) if mask.any() else None
    return out


def calibrate_thresholds(probs: np.ndarray, y_true: np.ndarray,
                         classes: list[str]) -> tuple[dict, float]:
    """Coordinate-descent search for per-class decision multipliers.

    Decision rule: argmax_c (p_c / t_c). t_c > 1 makes class c harder to
    predict; t_c < 1 makes it easier. Maximises macro-F1 on the given
    (validation) data. Returns ({class: multiplier}, best macro-F1).
    """
    grid = [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
    t = np.ones(len(classes))

    def macro_f1(thr: np.ndarray) -> float:
        pred = np.asarray(classes)[(probs / thr).argmax(axis=1)]
        return f1_score(y_true, pred, labels=classes, average="macro",
                        zero_division=0)

    best = macro_f1(t)
    for _ in range(3):  # sweeps
        improved = False
        for j in range(len(classes)):
            for g in grid:
                cand = t.copy()
                cand[j] = g
                score = macro_f1(cand)
                if score > best + 1e-6:
                    best, t, improved = score, cand, True
        if not improved:
            break
    return {c: round(float(v), 3) for c, v in zip(classes, t)}, round(best, 4)


def main() -> int:
    t0 = time.perf_counter()
    log("Loading KDDTrain+ ONLY (KDDTest+ is held out).")
    df, quality = data_loader.load_nsl_kdd_file(config.TRAIN_FILE)
    y = df[config.TARGET_COLUMN].to_numpy()

    fit_df, val_df, y_fit, y_val = train_test_split(
        df, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f"Fit split: {len(fit_df)} | Validation split: {len(val_df)}")
    log(f"Validation class counts: "
        f"{dict(zip(*np.unique(y_val, return_counts=True)))}")

    pre = Preprocessor().fit(fit_df)
    X_fit = pre.transform(fit_df)
    X_val = pre.transform(val_df)
    log(f"Encoded features: {X_fit.shape[1]}")

    results: dict = {
        "methodology": (
            "Stratified 80/20 split of KDDTrain+ (seed 42). Preprocessing "
            "fitted on fit split only. All candidates trained on fit split "
            "and evaluated on the validation split. KDDTest+ never used."),
        "seed": SEED,
        "n_fit": int(len(fit_df)),
        "n_validation": int(len(val_df)),
        "validation_class_counts": {
            str(c): int(n) for c, n in zip(*np.unique(y_val, return_counts=True))},
        "stage_b": {},
        "stage_a": {},
    }

    # ---------------------------------------------------------- Stage B
    candidates = []

    rf_base = dict(config.RANDOM_FOREST_PARAMS)
    candidates.append(("rf_baseline_balanced_subsample", "rf",
                       {**rf_base, "class_weight": "balanced_subsample"}, None))
    candidates.append(("rf_balanced", "rf",
                       {**rf_base, "class_weight": "balanced"}, None))
    # Extra emphasis on the rare attack classes beyond balanced weighting.
    candidates.append(("rf_balanced_minority_boost", "rf",
                       {**rf_base,
                        "class_weight": {c: w for c, w in zip(
                            CLASSES, _boosted_class_weights(y_fit))}}, None))

    xgb_base = dict(config.XGBOOST_PARAMS)
    candidates.append(("xgb_baseline_unweighted", "xgb", xgb_base, None))
    candidates.append(("xgb_balanced_sample_weight", "xgb", xgb_base, "balanced"))
    candidates.append(("xgb_balanced_minority_boost", "xgb", xgb_base, "boost"))

    for name, kind, params, weight_mode in candidates:
        log(f"Training {name} ...")
        start = time.perf_counter()
        if kind == "rf":
            model = RandomForestClassifier(**params)
            model.fit(X_fit, y_fit)
            probs = model.predict_proba(X_val)
            classes = [str(c) for c in model.classes_]
        else:
            model = XGBClassifier(**params)
            from sklearn.preprocessing import LabelEncoder
            enc = LabelEncoder().fit(y_fit)
            y_enc = enc.transform(y_fit)
            sw = None
            if weight_mode == "balanced":
                sw = compute_sample_weight("balanced", y=y_enc)
            elif weight_mode == "boost":
                sw = compute_sample_weight("balanced", y=y_enc)
                mult = {c: m for c, m in zip(CLASSES, _boost_multipliers())}
                sw = sw * np.array([mult[c] for c in y_fit])
            model.fit(X_fit, y_enc, sample_weight=sw)
            probs = model.predict_proba(X_val)
            classes = [str(c) for c in enc.classes_]
        elapsed = time.perf_counter() - start

        preds = np.asarray(classes)[probs.argmax(axis=1)]
        m = evaluation.classification_metrics(y_val, preds,
                                              probabilities=probs, classes=classes)
        entry = {
            "train_seconds": round(elapsed, 1),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "per_class_recall": per_class_recalls(y_val, preds),
            "per_class_f1": {c: m["per_class"][c]["f1"] for c in m["per_class"]},
            "roc_auc_ovr": m.get("roc_auc_ovr"),
        }
        results["stage_b"][name] = entry
        log(f"  acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
            f"recall={entry['per_class_recall']} ({elapsed:.0f}s)")

        # Threshold calibration on the SAME validation probabilities.
        thr, best_f1 = calibrate_thresholds(probs, y_val, classes)
        calibrated_preds = np.asarray(classes)[
            (probs / np.array([thr[c] for c in classes])).argmax(axis=1)]
        cm = evaluation.classification_metrics(y_val, calibrated_preds)
        results["stage_b"][name + "__calibrated"] = {
            "threshold_multipliers": thr,
            "accuracy": cm["accuracy"],
            "macro_f1": cm["macro_f1"],
            "weighted_f1": cm["weighted_f1"],
            "per_class_recall": per_class_recalls(y_val, calibrated_preds),
            "per_class_f1": {c: cm["per_class"][c]["f1"] for c in cm["per_class"]},
        }
        log(f"  calibrated: macroF1={cm['macro_f1']:.4f} multipliers={thr} "
            f"recall={results['stage_b'][name + '__calibrated']['per_class_recall']}")

    # ---------------------------------------------------------- Stage A
    normal_fit = X_fit[y_fit == "Normal"]
    y_val_attack = y_val != "Normal"
    log(f"Stage A: fitting Isolation Forest variants on "
        f"{len(normal_fit)} normal fit-split records ...")
    if_variants = {
        "if_current_200_ms0.8": dict(config.ISOLATION_FOREST_PARAMS),
        "if_300_ms256": {**config.ISOLATION_FOREST_PARAMS,
                         "n_estimators": 300, "max_samples": 256},
        "if_200_ms1.0": {**config.ISOLATION_FOREST_PARAMS, "max_samples": 1.0},
    }
    for name, params in if_variants.items():
        start = time.perf_counter()
        iforest = IsolationForest(**params)
        iforest.fit(normal_fit)
        train_s = time.perf_counter() - start
        normal_scores = iforest.score_samples(normal_fit)
        val_scores = iforest.score_samples(X_val)
        auc = float(roc_auc_score(y_val_attack.astype(int), -val_scores))
        entry = {"train_seconds": round(train_s, 1),
                 "validation_roc_auc": round(auc, 4),
                 "thresholds": {}}
        # Candidate thresholds: unsupervised quantiles of normal fit scores
        # plus validation-optimised points (supervised on validation only).
        cand_thresholds = {
            f"normal_quantile_{q:.2f}": float(np.quantile(normal_scores, q))
            for q in (0.01, 0.02, 0.05, 0.10)
        }
        order = np.argsort(val_scores)
        sorted_scores = val_scores[order]
        best_f1, best_thr = -1.0, None
        best_rec90_prec, best_rec90_thr = -1.0, None
        # Sweep unique-ish thresholds (subsample grid for speed).
        grid_idx = np.unique(np.linspace(0, len(sorted_scores) - 1, 400).astype(int))
        for idx in grid_idx:
            thr = float(sorted_scores[idx])
            pred = val_scores <= thr
            f1 = f1_score(y_val_attack, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
            rec = recall_score(y_val_attack, pred, zero_division=0)
            if rec >= 0.90:
                prec = precision_score(y_val_attack, pred, zero_division=0)
                if prec > best_rec90_prec:
                    best_rec90_prec, best_rec90_thr = prec, thr
        cand_thresholds["val_max_f1"] = best_thr
        cand_thresholds["val_recall_ge_0.90_max_precision"] = best_rec90_thr
        for tname, thr in cand_thresholds.items():
            if thr is None:
                entry["thresholds"][tname] = None
                continue
            am = evaluation.anomaly_metrics(y_val_attack, val_scores, thr)
            entry["thresholds"][tname] = {
                "threshold": round(thr, 5),
                "precision": am["precision"], "recall": am["recall"],
                "fpr": am["false_positive_rate"],
                "f1": round(f1_score(y_val_attack, val_scores <= thr,
                                     zero_division=0), 4),
            }
        results["stage_a"][name] = entry
        log(f"  {name}: val ROC-AUC={auc:.4f} | "
            + " | ".join(
                f"{tn}: R={v['recall']:.3f} P={v['precision']:.3f} FPR={v['fpr']:.3f}"
                for tn, v in entry["thresholds"].items() if v))

    results["total_seconds"] = round(time.perf_counter() - t0, 1)
    out_path = config.METRICS_DIR / "detection_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Saved {out_path} ({results['total_seconds']}s total).")

    # ------------------------------------------------- winner by val macro-F1
    ranked = sorted(
        ((n, e["macro_f1"]) for n, e in results["stage_b"].items()),
        key=lambda kv: kv[1], reverse=True)
    log("Stage B ranking by validation macro-F1:")
    for n, s in ranked:
        log(f"  {s:.4f}  {n}")
    return 0


def _boost_multipliers() -> list[float]:
    """Extra multipliers on top of balanced weights (Normal..U2R order)."""
    return [1.0, 1.0, 1.0, 2.0, 4.0]


def _boosted_class_weights(y: np.ndarray) -> list[float]:
    """Balanced class weights x minority boost, as a class-weight vector."""
    counts = np.array([(y == c).sum() for c in CLASSES], dtype=float)
    balanced = len(y) / (len(CLASSES) * np.maximum(counts, 1))
    return (balanced * np.array(_boost_multipliers())).tolist()


if __name__ == "__main__":
    raise SystemExit(main())