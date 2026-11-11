"""Shared fixtures: small synthetic NSL-KDD-shaped data for fast unit tests.

Unit tests never depend on the full NSL-KDD download; dataset-level tests
are marked and skipped when the raw files are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def synthetic_nsl_kdd(n: int = 400, seed: int = 0,
                      attack_fraction: float = 0.3) -> pd.DataFrame:
    """Synthetic frame with all 41 NSL-KDD features + label + difficulty."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in config.FEATURE_NAMES:
        if col == "protocol_type":
            data[col] = rng.choice(["tcp", "udp", "icmp"], n)
        elif col == "service":
            data[col] = rng.choice(
                ["http", "smtp", "ftp_data", "private", "other"], n)
        elif col == "flag":
            data[col] = rng.choice(["SF", "S0", "REJ", "RSTR"], n)
        elif col in ("same_srv_rate", "diff_srv_rate", "serror_rate",
                     "rerror_rate", "dst_host_same_srv_rate"):
            data[col] = rng.random(n).round(2)
        else:
            data[col] = rng.integers(0, 100, n)
    n_attack = int(n * attack_fraction)
    labels = np.array(["normal"] * (n - n_attack)
                      + ["neptune"] * (n_attack // 2)
                      + ["satan"] * (n_attack - n_attack // 2), dtype=object)
    rng.shuffle(labels)
    data[config.LABEL_COLUMN] = labels
    data[config.DIFFICULTY_COLUMN] = rng.integers(1, 21, n)
    df = pd.DataFrame(data)
    # make attack rows statistically distinguishable for model tests
    attack_mask = df[config.LABEL_COLUMN] != "normal"
    df.loc[attack_mask, "src_bytes"] += 5000
    df.loc[attack_mask, "count"] = 300
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return synthetic_nsl_kdd()


@pytest.fixture
def labelled_df(synthetic_df) -> pd.DataFrame:
    """Frame with the mapped 5-class target column."""
    df = synthetic_df.copy()
    df[config.TARGET_COLUMN] = df[config.LABEL_COLUMN].map(
        config.ATTACK_CATEGORY_MAP).fillna("Unknown")
    return df


requires_dataset = pytest.mark.skipif(
    not (config.TRAIN_FILE.exists() and config.TEST_FILE.exists()),
    reason="NSL-KDD raw files not present in data/raw/",
)


# --------------------------------------------------------------------- auth
# Credentials used only inside the disposable per-test auth database.
TEST_PASSWORD = "V3ry!Strong-Pass"
ADMIN_PASSWORD = "Adm1n!Str0ng-Pass"


@pytest.fixture(autouse=True)
def analysis_cache_sandbox(tmp_path, monkeypatch):
    """Redirect the persisted analysis cache to a throwaway folder so tests
    never read or write the project's real restart-survival cache."""
    from src import analysis_cache
    monkeypatch.setattr(analysis_cache, "CACHE_DIR",
                        tmp_path / "analysis_cache")


@pytest.fixture(autouse=True)
def auth_sandbox(tmp_path, monkeypatch):
    """Every test gets a throwaway auth database and mail outbox, so no
    test ever reads or writes the project's real user store."""
    from src.auth import db as auth_db
    from src.auth import settings as auth_settings
    from cryptography.fernet import Fernet
    monkeypatch.setattr(auth_settings, "AUTH_DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_settings, "MAIL_OUTBOX_DIR",
                        tmp_path / "mail_outbox")
    monkeypatch.setattr(auth_settings, "EMAIL_MODE", "outbox")
    # Breach screening is opt-in and networked; tests mock it explicitly.
    monkeypatch.setattr(auth_settings, "BREACH_SCREENING", False)
    # Disposable MFA key: tests never use the operator's real AI_SOC_TOTP_KEY.
    monkeypatch.setattr(auth_settings, "TOTP_KEY",
                        Fernet.generate_key().decode("utf-8"))
    auth_db.reset_store_registry()
    yield
    auth_db.reset_store_registry()


@pytest.fixture
def admin_user():
    from src.auth import service
    result = service.provision_admin(
        full_name="Test Administrator", email="admin@example.test",
        username="testadmin", password=ADMIN_PASSWORD)
    assert result.ok, result.errors
    return result.user


@pytest.fixture
def admin_token(admin_user):
    from src.auth import service
    result = service.authenticate(admin_user["username"], ADMIN_PASSWORD)
    assert result.ok, result.errors
    return result.token


@pytest.fixture
def viewer_user():
    from src.auth import service
    result = service.register(
        full_name="Vera Viewer", email="viewer@example.test",
        username="viewer", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD)
    assert result.ok, result.errors
    return result.user


@pytest.fixture
def viewer_token(viewer_user):
    from src.auth import service
    result = service.authenticate(viewer_user["username"], TEST_PASSWORD)
    assert result.ok, result.errors
    return result.token


def latest_mail_link_token() -> str | None:
    """Read the newest development-outbox message and extract its link token.

    The outbox is the documented development delivery mode; parsing the
    real .eml keeps tests honest about what a user would actually receive.
    """
    import re
    from email import policy
    from email.parser import BytesParser

    from src.auth import settings as auth_settings
    files = sorted(auth_settings.MAIL_OUTBOX_DIR.glob("*.eml"))
    if not files:
        return None
    message = BytesParser(policy=policy.default).parsebytes(
        files[-1].read_bytes())
    match = re.search(r"[?&]token=([A-Za-z0-9_\-]+)", message.get_content())
    return match.group(1) if match else None
