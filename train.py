"""End-to-end reproducible training and evaluation pipeline.

Usage:  python train.py [--skip-eval] [--shap-sample N]

Steps:
1. Load and validate KDDTrain+.txt / KDDTest+.txt.
2. Fit preprocessing on TRAINING data only (no leakage).
3. Train Isolation Forest on normal training traffic only.
4. Train Random Forest and XGBoost on labelled training traffic.
5. Save models + preprocessing artefacts.
6. Evaluate on KDDTest+ (anomaly + classification metrics, real numbers).
7. Compute global SHAP importance (clearly labelled as global).
8. Generate alerts for the test set and compute triage metrics.
9. Record resource-efficiency measurements (CPU, RAM, latency, sizes).
10. Write results/metrics/*.json and results/reports/training_report.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import data_loader  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402
from src.anomaly_detection import AnomalyDetector  # noqa: E402
from src.attack_classification import AttackClassifier  # noqa: E402
from src.alert_generator import AlertGenerator, AlertStore  # noqa: E402
from src.severity_engine import SeverityEngine  # noqa: E402
from src.explainability import Explainer  # noqa: E402
from src.resource_monitor import ResourceMonitor  # noqa: E402
from src import evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")

# Transparent, unsupervised threshold selection using NORMAL TRAINING
# data only: flag the most anomalous config.ANOMALY_NORMAL_FPR_QUANTILE
# fraction of normal training traffic (nominal FPR on normal traffic).
# The 10% recall-oriented operating point was chosen on a training
# validation split (scripts/experiment_detection.py). No attack labels
# and no test data are used for this choice.
NORMAL_FPR_QUANTILE = config.ANOMALY_NORMAL_FPR_QUANTILE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train and save models without test evaluation.")
    parser.add_argument("--shap-sample", type=int,
                        default=config.SHAP_GLOBAL_SAMPLE_SIZE,
                        help="Sample size for global SHAP importance.")
    args = parser.parse_args()

    monitor = ResourceMonitor()
    report: dict = {"random_seed": config.RANDOM_SEED}

    # ---------------------------------------------------------------- load
    with monitor.timed("load_training_file"):
        train_df, train_report = data_loader.load_nsl_kdd_file(
            config.TRAIN_FILE)
    with monitor.timed("load_testing_file"):
        test_df, test_report = data_loader.load_nsl_kdd_file(config.TEST_FILE)
    report["data_quality"] = {
        "train": train_report.to_dict(),
        "test": test_report.to_dict(),
    }
    log.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))
    log.info("Train label distribution: %s",
             train_df[config.TARGET_COLUMN].value_counts().to_dict())

    # --------------------------------------------------------- preprocess
    with monitor.timed("preprocessing"):
        pre = Preprocessor().fit(train_df)
        X_train = pre.transform(train_df)
        X_test = pre.transform(test_df)
    y_train_cat = train_df[config.TARGET_COLUMN].to_numpy()
    y_test_cat = test_df[config.TARGET_COLUMN].to_numpy()
    pre.save()
    log.info("Encoded feature matrix: %s", X_train.shape)

    # ------------------------------------------------- Stage A: anomalies
    normal_mask = y_train_cat == "Normal"
    X_normal = X_train[normal_mask]
    log.info("Training Isolation Forest on %d NORMAL records only.",
             len(X_normal))
    detector = AnomalyDetector()
    with monitor.timed("train_isolation_forest"):
        detector.fit(X_normal)
    # Threshold = quantile of NORMAL training scores (unsupervised; no
    # attack labels or test data involved). See NORMAL_FPR_QUANTILE.
    normal_scores = detector.score(X_normal)
    detector.threshold = float(
        np.quantile(normal_scores, NORMAL_FPR_QUANTILE))
    log.info("Anomaly threshold set to %.4f (flags the most anomalous %.0f%% "
             "of normal training traffic).",
             detector.threshold, NORMAL_FPR_QUANTILE * 100)
    detector.save()
    log.info("Isolation Forest trained in %.1fs.", detector.train_time_seconds)

    # ------------------------------------------- Stage B: classification
    classifiers: dict[str, AttackClassifier] = {}
    for name in ("random_forest", "xgboost"):
        clf = AttackClassifier(name)
        with monitor.timed(f"train_{name}"):
            clf.fit(X_train, y_train_cat)
        clf.save()
        classifiers[name] = clf
        log.info("%s trained in %.1fs on %d records.",
                 name, clf.train_time_seconds, clf.n_train_samples)

    if args.skip_eval:
        monitor.records.update(monitor.model_file_sizes())
        monitor.save()
        log.info("Models saved. Evaluation skipped (--skip-eval).")
        return 0

    # ------------------------------------------------------------ evaluate
    with monitor.timed("anomaly_inference_test"):
        anomaly_res = detector.predict(X_test)
    y_test_is_attack = y_test_cat != "Normal"
    known_test = y_test_cat != "Unknown"
    report["anomaly_detection"] = evaluation.anomaly_metrics(
        y_test_is_attack[known_test], anomaly_res.scores[known_test],
        detector.threshold,
    )
    log.info("Anomaly detection (KDDTest+): %s", {
        k: report["anomaly_detection"][k]
        for k in ("roc_auc", "precision", "recall", "false_positive_rate")
    })

    report["attack_classification"] = {}
    for name, clf in classifiers.items():
        with monitor.timed(f"{name}_inference_test"):
            res = clf.predict(X_test)
        mask = known_test & np.isin(y_test_cat, config.ATTACK_CATEGORIES)
        report["attack_classification"][name] = (
            evaluation.classification_metrics(
                y_test_cat[mask], res.predictions[mask],
                probabilities=res.probabilities[mask], classes=res.classes,
            )
        )
        m = report["attack_classification"][name]
        log.info("%s (KDDTest+): acc=%.4f macroF1=%.4f weightedF1=%.4f",
                 name, m["accuracy"], m["macro_f1"], m["weighted_f1"])
        log.info("  strongest=%s weakest=%s",
                 m.get("strongest_category", {}).get("name"),
                 m.get("weakest_category", {}).get("name"))
        report.setdefault("prediction_latency", {})
        monitor.measure_prediction_latency(
            clf.predict, X_test[:5000], n_repeats=3,
            key=f"{name}_prediction_latency",
        )
    monitor.measure_prediction_latency(
        detector.predict, X_test[:5000], n_repeats=3,
        key="isolation_forest_prediction_latency",
    )

    # --------------------------------------------------- global SHAP (A/B)
    primary_name = max(
        report["attack_classification"],
        key=lambda n: report["attack_classification"][n]["macro_f1"],
    )
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_idx = rng.choice(len(X_test),
                            size=min(args.shap_sample, len(X_test)),
                            replace=False)
    explainer = Explainer(classifiers[primary_name], pre.feature_names_)
    with monitor.timed("global_shap_computation"):
        shap_df, shap_seconds = explainer.global_importance(
            X_test[sample_idx], use_cache=False)
    report["global_shap"] = {
        "model": primary_name,
        "sample_size": int(len(sample_idx)),
        "compute_seconds": round(shap_seconds, 3),
        "top_15": shap_df.head(15).to_dict(orient="records"),
        "note": ("GLOBAL feature importance (mean |SHAP| over the sample). "
                 "This is NOT a per-alert explanation."),
    }
    log.info("Global SHAP top-5: %s", shap_df.head(5)["feature"].tolist())

    # local SHAP latency (one record, real measurement)
    loc = explainer.explain_record(X_test[sample_idx[0]])
    report["local_shap_latency_ms_example"] = round(
        loc.explanation_time_ms, 2)

    # ----------------------------------------------------- alerts + triage
    primary_res = classifiers[primary_name].predict(X_test)
    generator = AlertGenerator(SeverityEngine())
    with monitor.timed("alert_generation"):
        alerts = generator.generate(
            anomaly_res, primary_res,
            record_indices=np.arange(len(test_df)),
            true_labels=test_df[config.LABEL_COLUMN].to_numpy(),
        )
    store = AlertStore()
    store.clear()
    store.save(alerts, merge=False)
    report["triage"] = evaluation.triage_metrics(store.load())
    report["triage"]["primary_classifier"] = primary_name
    log.info("Generated %d alerts from %d test records.",
             len(alerts), len(test_df))

    # ---------------------------------------------------- targets & actuals
    report["targets_vs_actuals"] = evaluation.compare_with_targets(report)

    # -------------------------------------------------------- resource use
    monitor.records.update(monitor.model_file_sizes())
    monitor.save()

    # ----------------------------------------------------------- artefacts
    state = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "random_seed": config.RANDOM_SEED,
        "dataset": config.DATASET_NAME,
        "dataset_source": config.DATASET_SOURCE,
        "train_file": config.TRAIN_FILE.name,
        "test_file": config.TEST_FILE.name,
        "n_train_records": int(len(train_df)),
        "n_test_records": int(len(test_df)),
        "n_normal_train_records": int(normal_mask.sum()),
        "n_encoded_features": int(X_train.shape[1]),
        "anomaly_threshold": detector.threshold,
        "anomaly_threshold_method": (
            f"{NORMAL_FPR_QUANTILE:.0%} quantile of normal training "
            f"anomaly scores (unsupervised; nominal {NORMAL_FPR_QUANTILE:.0%} "
            "FPR on normal traffic; recall-oriented operating point; "
            "no attack labels or test data used)"),
        "class_weighting": {
            "random_forest": config.RANDOM_FOREST_PARAMS.get("class_weight"),
            "xgboost": config.XGBOOST_CLASS_WEIGHTING,
            "note": ("Cost-sensitive weighting for class imbalance, selected "
                     "on a stratified validation split of KDDTrain+ "
                     "(scripts/experiment_detection.py); KDDTest+ held out."),
        },
        "primary_classifier": primary_name,
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "train_times_seconds": {
            "isolation_forest": round(detector.train_time_seconds or 0, 2),
            **{n: round(c.train_time_seconds or 0, 2)
               for n, c in classifiers.items()},
        },
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PIPELINE_STATE_FILE.write_text(json.dumps(state, indent=2))
    evaluation.save_evaluation(report)
    _write_training_report(report, state)
    log.info("Done. Artefacts: models/, results/metrics/, results/reports/.")
    return 0


def _write_training_report(report: dict, state: dict) -> None:
    lines = [
        "# Training & Evaluation Report (auto-generated)",
        "",
        f"- Trained at: {state['trained_at']}",
        f"- Dataset: {state['dataset']} ({state['train_file']} / "
        f"{state['test_file']})",
        f"- Random seed: {state['random_seed']}",
        f"- Training records: {state['n_train_records']} "
        f"({state['n_normal_train_records']} normal used for Isolation Forest)",
        f"- Testing records: {state['n_test_records']}",
        f"- Encoded features: {state['n_encoded_features']}",
        "",
        "## Stage A: Isolation Forest anomaly detection (KDDTest+)",
        "",
    ]
    ad = report.get("anomaly_detection", {})
    for k in ("roc_auc", "precision", "recall", "false_positive_rate",
              "false_negative_rate", "threshold"):
        lines.append(f"- {k}: {ad.get(k)}")
    cm = ad.get("confusion_matrix", {})
    lines += [f"- confusion matrix: {cm}", ""]
    lines += ["## Stage B: Attack classification (KDDTest+)", ""]
    for name, m in report.get("attack_classification", {}).items():
        lines += [
            f"### {name}",
            f"- accuracy: {m.get('accuracy')}",
            f"- macro F1: {m.get('macro_f1')} | weighted F1: "
            f"{m.get('weighted_f1')}",
            f"- strongest category: {m.get('strongest_category')}",
            f"- weakest category: {m.get('weakest_category')}",
            "",
        ]
    lines += ["## Targets vs actuals (reported separately, not auto-claimed)",
              ""]
    lines.append("```json")
    lines.append(json.dumps(report.get("targets_vs_actuals", {}), indent=2))
    lines.append("```")
    lines += ["", "## Alert triage", "", "```json",
              json.dumps(report.get("triage", {}), indent=2), "```"]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "training_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
