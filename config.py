"""Central configuration for the AI SOC research prototype.

All tunable parameters (paths, seeds, model hyperparameters, severity
mapping, risk-score weights) live here so the pipeline is reproducible
and configurable without editing module code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

TRAIN_FILE = DATA_RAW_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_RAW_DIR / "KDDTest+.txt"

MODEL_FILES = {
    "isolation_forest": MODELS_DIR / "isolation_forest.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost_model.pkl",
    "preprocessing": MODELS_DIR / "preprocessing.pkl",
}
PIPELINE_STATE_FILE = MODELS_DIR / "pipeline_state.json"

# Dashboard-side persisted state (analyst workflow + user configuration).
ALERT_NOTES_FILE = DATA_PROCESSED_DIR / "alert_notes.json"
APP_SETTINGS_FILE = DATA_PROCESSED_DIR / "app_settings.json"
APP_LOG_FILE = RESULTS_DIR / "app_activity.log"

APP_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "NSL-KDD"
DATASET_SOURCE = (
    "Tavallaee et al., 'A Detailed Analysis of the KDD CUP 99 Data Set' (2009). "
    "Files: KDDTrain+.txt / KDDTest+.txt"
)

# NSL-KDD feature names (41 features), plus label and difficulty level.
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
LABEL_COLUMN = "attack_label"        # raw NSL-KDD attack name
DIFFICULTY_COLUMN = "difficulty"    # 42nd column of the "+" files
TARGET_COLUMN = "attack_category"   # mapped 5-class target

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Documented attack -> category mapping (NSL-KDD standard grouping).
ATTACK_CATEGORY_MAP = {
    "normal": "Normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "worm": "DoS",
    "mailbomb": "DoS",
    # Probe / Surveillance
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local (unauthorised access from remote host)
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpguess": "R2L",
    # User to Root (local privilege escalation)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}
ATTACK_CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ---------------------------------------------------------------------------
# Upload validation (dashboard safety limits)
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 60 * 1024 * 1024          # 60 MB
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".txt", ".data"}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": 0.8,
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
# Default anomaly decision threshold on IsolationForest.score_samples()
# (more negative = more anomalous). Records with score <= threshold are
# flagged anomalous. NOTE: train.py overrides this default with a
# threshold selected from NORMAL TRAINING traffic only (the quantile of
# normal anomaly scores given by ANOMALY_NORMAL_FPR_QUANTILE below); the
# effective value is saved with the model and remains adjustable at
# analysis time.
ANOMALY_THRESHOLD = -0.45

# Unsupervised threshold policy: flag the most anomalous 10% of normal
# training traffic (nominal 10% FPR on normal training data). This is a
# recall-oriented operating point chosen on a training validation split
# (scripts/experiment_detection.py): in a SOC a missed attack is costlier
# than a false alarm. No attack labels and no test data are used.
ANOMALY_NORMAL_FPR_QUANTILE = 0.10

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    # Cost-sensitive weighting for the rare R2L/U2R classes; selected on a
    # training validation split (scripts/experiment_detection.py).
    "class_weight": "balanced",
}
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
}
# XGBoost has no class_weight parameter; AttackClassifier.fit applies
# per-record sample weights instead. "balanced" = n_samples / (n_classes *
# class_count), counteracting the extreme R2L/U2R imbalance. Set to None
# to disable. Selected on a training validation split, never on KDDTest+.
XGBOOST_CLASS_WEIGHTING: str | None = "balanced"

# ---------------------------------------------------------------------------
# Severity mapping (initial research rules, configurable; NOT a validated
# real-world SOC risk framework - see README limitations)
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    "Normal": "Informational",
    "Probe": "Medium",
    "DoS": "High",
    "R2L": "High",
    "U2R": "Critical",
}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Informational"]

# ---------------------------------------------------------------------------
# Risk score (transparent weighted formula, 0-100)
#   risk = 100 * (w_sev * severity_weight
#               + w_conf * model_confidence
#               + w_anom * normalised_anomaly_score)
# Weights sum to 1. severity_weight in [0, 1] from SEVERITY_WEIGHTS below.
# Anomaly contribution uses min(1, max(0, (threshold - score) / spread)).
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {"severity": 0.5, "confidence": 0.3, "anomaly": 0.2}
SEVERITY_WEIGHTS = {
    "Critical": 1.0,
    "High": 0.75,
    "Medium": 0.5,
    "Low": 0.25,
    "Informational": 0.0,
}
ANOMALY_SCORE_SPREAD = 0.5  # normalisation range below the threshold

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
SHAP_GLOBAL_SAMPLE_SIZE = 500   # background/sample size for global SHAP
SHAP_LOCAL_TOP_K = 8            # top features shown in per-alert explanation

# ---------------------------------------------------------------------------
# Investigation workflow
# ---------------------------------------------------------------------------
INVESTIGATION_STATUSES = ["New", "Investigating", "Resolved", "False Positive"]

# ---------------------------------------------------------------------------
# Evaluation targets (research targets - reported separately from actuals;
# meeting them is NOT automatically claimed)
# ---------------------------------------------------------------------------
EVALUATION_TARGETS = {
    "anomaly_detection": {"roc_auc": 0.95, "recall": 0.90},
    "attack_classification": {"accuracy": 0.78, "macro_f1": 0.70},
    "resource_efficiency": {
        "mean_prediction_latency_ms_per_record": 1.0,
        "model_size_mb": 50.0,
    },
}
