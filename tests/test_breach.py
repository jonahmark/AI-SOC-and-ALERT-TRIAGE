"""Breach-password screening tests: the k-anonymity request shape, honest
fail-open behaviour and service integration. Every HTTP call is mocked -
the suite never contacts the real HIBP service.
"""
from __future__ import annotations

import hashlib

import pytest
import requests

from src.auth import breach, db, service
from src.auth import settings as auth_settings
from tests.conftest import TEST_PASSWORD, latest_mail_link_token


class _Response:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _digest(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8")).hexdigest().upper()


@pytest.fixture
def screening_on(monkeypatch):
    monkeypatch.setattr(auth_settings, "BREACH_SCREENING", True)


def _mock_range(monkeypatch, response=None, error=None):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(breach.requests, "get", fake_get)
    return calls


# ------------------------------------------------------------- module level
def test_disabled_by_default_makes_no_request(monkeypatch):
    calls = _mock_range(monkeypatch, _Response(""))
    assert breach.check_password(TEST_PASSWORD) == ("disabled", 0)
    assert calls == []


def test_only_the_five_char_prefix_is_transmitted(monkeypatch, screening_on):
    digest = _digest(TEST_PASSWORD)
    calls = _mock_range(monkeypatch, _Response("AAAA:1\nBBBB:2\n"))
    status, count = breach.check_password(TEST_PASSWORD)
    assert (status, count) == ("clean", 0)
    url, kwargs = calls[0]
    assert url == f"https://api.pwnedpasswords.com/range/{digest[:5]}"
    assert digest[5:] not in url and TEST_PASSWORD not in url
    assert kwargs["timeout"] == auth_settings.BREACH_CHECK_TIMEOUT_SECONDS


def test_matching_suffix_reports_the_occurrence_count(monkeypatch,
                                                      screening_on):
    digest = _digest(TEST_PASSWORD)
    body = f"00000000000000000000000000000000000:1\n{digest[5:]}:4242\n"
    _mock_range(monkeypatch, _Response(body))
    assert breach.check_password(TEST_PASSWORD) == ("breached", 4242)


def test_malformed_count_still_reports_breached(monkeypatch, screening_on):
    digest = _digest(TEST_PASSWORD)
    _mock_range(monkeypatch, _Response(f"{digest[5:]}:not-a-number\n"))
    status, count = breach.check_password(TEST_PASSWORD)
    assert status == "breached" and count >= 1


@pytest.mark.parametrize("error", [
    requests.Timeout("timed out"),
    requests.ConnectionError("no route"),
])
def test_network_failure_is_unreachable_not_clean(monkeypatch, screening_on,
                                                  error):
    _mock_range(monkeypatch, error=error)
    assert breach.check_password(TEST_PASSWORD) == ("unreachable", 0)


def test_http_error_is_unreachable(monkeypatch, screening_on):
    _mock_range(monkeypatch, _Response("", status=503))
    assert breach.check_password(TEST_PASSWORD) == ("unreachable", 0)


# -------------------------------------------------------- service integration
def _breached_response(password: str) -> _Response:
    digest = _digest(password)
    return _Response(f"{digest[5:]}:99\n")


def test_registration_rejects_a_breached_password(monkeypatch, screening_on):
    _mock_range(monkeypatch, _breached_response(TEST_PASSWORD))
    result = service.register(
        full_name="Test Analyst", email="analyst@example.test",
        username="analyst", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD)
    assert not result.ok
    assert any("data breaches" in e for e in result.errors)
    assert db.get_store().query("SELECT * FROM users") == []
    events = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'breach_password_blocked'")
    assert len(events) == 1
    assert TEST_PASSWORD not in events[0]["detail"]


def test_password_change_rejects_a_breached_password(monkeypatch,
                                                     screening_on):
    user = service.register(
        full_name="Test Analyst", email="analyst@example.test",
        username="analyst", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD).user
    breached = "Br3ached!Passphrase-X"
    _mock_range(monkeypatch, _breached_response(breached))
    result = service.change_password(user["user_id"], TEST_PASSWORD,
                                     breached, breached)
    assert not result.ok
    assert any("data breaches" in e for e in result.errors)
    assert service.authenticate("analyst", TEST_PASSWORD).ok


def test_reset_rejects_breached_password_without_consuming_token(
        monkeypatch, screening_on):
    service.register(full_name="Test Analyst", email="analyst@example.test",
                     username="analyst", password=TEST_PASSWORD,
                     confirm_password=TEST_PASSWORD)
    assert service.request_password_reset("analyst@example.test").ok
    token = latest_mail_link_token()
    breached = "Br3ached!Recovery-X"
    _mock_range(monkeypatch, _breached_response(breached))
    result = service.reset_password(token, breached, breached)
    assert not result.ok
    fresh = "Fr3sh!Recovery-Pass"
    _mock_range(monkeypatch, _Response("AAAA:1\n"))
    assert service.reset_password(token, fresh, fresh).ok
    assert service.authenticate("analyst", fresh).ok


def test_unreachable_service_fails_open_and_is_audited(monkeypatch,
                                                       screening_on):
    _mock_range(monkeypatch, error=requests.ConnectionError("offline"))
    result = service.register(
        full_name="Test Analyst", email="analyst@example.test",
        username="analyst", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD)
    assert result.ok, result.errors  # fail-open: outage must not block users
    events = db.get_store().query(
        "SELECT * FROM audit_events WHERE event = 'breach_check_unavailable'")
    assert len(events) == 1
    assert events[0]["detail"] == "fail_open"


def test_screening_off_means_no_request_and_no_audit(monkeypatch):
    calls = _mock_range(monkeypatch, _breached_response(TEST_PASSWORD))
    result = service.register(
        full_name="Test Analyst", email="analyst@example.test",
        username="analyst", password=TEST_PASSWORD,
        confirm_password=TEST_PASSWORD)
    assert result.ok, result.errors
    assert calls == []
    assert db.get_store().query(
        "SELECT * FROM audit_events WHERE event LIKE 'breach%'") == []