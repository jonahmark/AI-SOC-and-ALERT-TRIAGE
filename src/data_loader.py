"""Data ingestion and validation for NSL-KDD files and uploaded CSVs.

Responsibilities:
- Load KDDTrain+.txt / KDDTest+.txt with the documented column names.
- Validate structure, dtypes, missing values, duplicates and malformed rows.
- Report data-quality findings explicitly instead of silently dropping data.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = config.FEATURE_NAMES + [
    config.LABEL_COLUMN,
    config.DIFFICULTY_COLUMN,
]
NUMERIC_FEATURES = [
    f for f in config.FEATURE_NAMES if f not in config.CATEGORICAL_FEATURES
]


@dataclass
class ValidationReport:
    """Explicit data-quality report produced by :func:`validate_dataframe`."""

    n_rows: int = 0
    n_columns: int = 0
    missing_values: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    malformed_rows: int = 0
    malformed_row_indices: list = field(default_factory=list)
    unknown_labels: list = field(default_factory=list)
    invalid_numeric_cells: dict = field(default_factory=dict)
    label_distribution: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "malformed_rows": self.malformed_rows,
            "malformed_row_indices": self.malformed_row_indices[:50],
            "unknown_labels": self.unknown_labels,
            "invalid_numeric_cells": self.invalid_numeric_cells,
            "label_distribution": self.label_distribution,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @property
    def is_valid(self) -> bool:
        return not self.errors


def load_nsl_kdd_file(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Load an NSL-KDD '+' file (43 columns: 41 features, label, difficulty).

    Malformed rows are counted and reported in the ValidationReport, then
    excluded with an explicit warning (never silently).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Download KDDTrain+.txt / "
            "KDDTest+.txt from the NSL-KDD dataset page and place them in "
            "data/raw/ (see README)."
        )

    # Fast C engine; explicit fallback counts malformed lines instead of
    # silently skipping them.
    try:
        raw = pd.read_csv(path, header=None, names=EXPECTED_COLUMNS,
                          engine="c")
        n_malformed_lines: list[int] = []
    except pd.errors.ParserError:
        good_lines, n_malformed_lines = [], []
        expected_fields = len(EXPECTED_COLUMNS)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if line.rstrip("\n").count(",") == expected_fields - 1:
                    good_lines.append(line)
                else:
                    n_malformed_lines.append(i)
        raw = pd.read_csv(io.StringIO("".join(good_lines)), header=None,
                          names=EXPECTED_COLUMNS, engine="c")
    logger.info("Loaded %d raw rows from %s", len(raw), path.name)
    df, report = validate_dataframe(raw)
    if n_malformed_lines:
        report.malformed_rows += len(n_malformed_lines)
        report.malformed_row_indices = (
            n_malformed_lines + report.malformed_row_indices
        )[:50]
        report.warnings.append(
            f"{len(n_malformed_lines)} structurally malformed line(s) "
            f"excluded (wrong field count): lines {n_malformed_lines[:10]}."
        )
    return df, report


def load_uploaded_file(
    content: bytes, filename: str
) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Validate and load a user-uploaded CSV/TXT of network records.

    Accepts either the full NSL-KDD 43-column layout, the 42-column layout
    (without difficulty) or a header row using NSL-KDD feature names.
    Returns (None, report) with explicit errors when the file is unusable.
    """
    report = ValidationReport()
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_UPLOAD_EXTENSIONS:
        report.errors.append(
            f"Unsupported file type '{suffix}'. Allowed: "
            f"{sorted(config.ALLOWED_UPLOAD_EXTENSIONS)}"
        )
        return None, report
    if len(content) > config.MAX_UPLOAD_BYTES:
        report.errors.append(
            f"File too large ({len(content) / 1e6:.1f} MB). "
            f"Maximum allowed: {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
        return None, report
    if not content.strip():
        report.errors.append("Uploaded file is empty.")
        return None, report

    try:
        df = pd.read_csv(io.BytesIO(content), header=None, engine="c")
    except pd.errors.ParserError as exc:
        report.errors.append(
            f"Malformed CSV structure (rows with inconsistent field counts): "
            f"{exc}"
        )
        return None, report
    except Exception as exc:  # other parse failures
        report.errors.append(f"Could not parse file as CSV: {exc}")
        return None, report

    # Detect and use a header row when present.
    first = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    if first[0] == "duration" and "protocol_type" in first:
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    elif df.shape[1] == len(EXPECTED_COLUMNS):
        df.columns = EXPECTED_COLUMNS
    elif df.shape[1] == len(EXPECTED_COLUMNS) - 1:
        df.columns = EXPECTED_COLUMNS[:-1]
        df[config.DIFFICULTY_COLUMN] = np.nan
    else:
        report.errors.append(
            f"Expected {len(EXPECTED_COLUMNS)} columns (NSL-KDD layout with "
            f"difficulty) or {len(EXPECTED_COLUMNS) - 1} columns, or a header "
            f"row with NSL-KDD feature names; got {df.shape[1]} columns."
        )
        return None, report

    missing_cols = [c for c in config.FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        report.errors.append(f"Missing required feature columns: {missing_cols}")
        return None, report
    if config.LABEL_COLUMN not in df.columns:
        # Unlabelled upload: allowed, labels filled as unknown for triage.
        df[config.LABEL_COLUMN] = "unlabelled"
        report.warnings.append(
            "No label column found; records treated as unlabelled input."
        )

    df, report2 = validate_dataframe(df, allow_unlabelled=True)
    return df, report2


def validate_dataframe(
    df: pd.DataFrame, allow_unlabelled: bool = False
) -> tuple[pd.DataFrame, ValidationReport]:
    """Structural and content validation; coerces dtypes and reports issues."""
    report = ValidationReport()
    report.n_rows = int(len(df))
    report.n_columns = int(df.shape[1])

    # --- numeric coercion -------------------------------------------------
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_invalid = int(coerced.isna().sum() - pd.Series(df[col]).isna().sum())
        if n_invalid > 0:
            report.invalid_numeric_cells[col] = n_invalid
            report.warnings.append(
                f"Column '{col}': {n_invalid} non-numeric cells coerced to NaN."
            )
        df[col] = coerced

    # --- malformed rows: all features NaN ---------------------------------
    feature_cols = [c for c in config.FEATURE_NAMES if c in df.columns]
    all_nan = df[feature_cols].isna().all(axis=1)
    if all_nan.any():
        report.malformed_rows = int(all_nan.sum())
        report.malformed_row_indices = df.index[all_nan].tolist()
        report.warnings.append(
            f"{report.malformed_rows} fully empty/malformed rows excluded."
        )
        df = df.loc[~all_nan].reset_index(drop=True)
        report.n_rows = int(len(df))

    # --- missing values ----------------------------------------------------
    miss = df.isna().sum()
    report.missing_values = {k: int(v) for k, v in miss.items() if v > 0}

    # --- duplicates ----------------------------------------------------------
    report.duplicate_rows = int(df.duplicated().sum())

    # --- labels ------------------------------------------------------------
    if config.LABEL_COLUMN in df.columns:
        df[config.LABEL_COLUMN] = (
            df[config.LABEL_COLUMN].astype(str).str.strip().str.lower()
        )
        labels = df[config.LABEL_COLUMN]
        known = set(config.ATTACK_CATEGORY_MAP)
        if allow_unlabelled:
            known = known | {"unlabelled"}
        unknown = sorted(set(labels.unique()) - known)
        if unknown:
            report.unknown_labels = unknown
            report.warnings.append(
                f"{len(unknown)} unknown attack label(s) not in the documented "
                f"mapping: {unknown}. These rows are kept with category "
                "'Unknown' and excluded from supervised training."
            )
        df[config.TARGET_COLUMN] = labels.map(
            config.ATTACK_CATEGORY_MAP
        ).fillna("Unknown")
        report.label_distribution = (
            df[config.TARGET_COLUMN].value_counts().to_dict()
        )
        if not allow_unlabelled and unknown:
            report.warnings.append("Unknown labels present in a labelled file.")

    if report.n_rows == 0:
        report.errors.append("No usable rows remain after validation.")
    return df, report
