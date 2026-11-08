"""Dashboard page registry (single source of truth for navigation).

`build()` creates the `st.Page` objects for the current script run and records
them in `REGISTRY`, so any page can call `st.switch_page(nav.page("..."))`
(Streamlit requires the exact Page instance registered in that run). `st.Page`
only records its target when created inside a script run, so the registry is
rebuilt on every run rather than cached across runs.

When a role is supplied, pages the role cannot access are not registered at
all (clean sidebar); app.py additionally enforces the same permission map
server-side before running any page.
"""
from __future__ import annotations

import streamlit as st

from src.auth import rbac

REGISTRY: dict[str, st.Page] = {}

_DEFS = [
    ("overview", "Overview", ":material/space_dashboard:", "overview"),
    ("traffic_monitor", "Traffic Analysis Monitor", ":material/monitoring:",
     "traffic-analysis-monitor"),
    ("network_analysis", "Network Analysis", ":material/hub:", "network-analysis"),
    ("alert_management", "Alert Management", ":material/notifications_active:",
     "alert-management"),
    ("attack_intelligence", "Attack Intelligence", ":material/radar:",
     "attack-intelligence"),
    ("explainability", "Explainable AI", ":material/psychology:",
     "explainable-ai"),
    ("model_performance", "Model Performance", ":material/insights:",
     "model-performance"),
    ("resource_monitoring", "Resource Monitoring", ":material/memory:",
     "resource-monitoring"),
    ("reports", "Reports", ":material/description:", "reports"),
    ("settings_page", "System Settings", ":material/settings:",
     "system-settings"),
    ("user_management", "User Management", ":material/admin_panel_settings:",
     "user-management"),
]

URL_TO_KEY = {url_path: key for key, _title, _icon, url_path in _DEFS}


def key_for_url(url_path: str) -> str:
    return URL_TO_KEY.get(url_path, "overview")


def build(role: str | None = None) -> dict[str, st.Page]:
    """Create and register the page objects this role may access."""
    from dashboard.pages import (
        alert_management, attack_intelligence, explainability, model_performance,
        network_analysis, overview, reports, resource_monitoring,
        settings_page, traffic_monitor, user_management,
    )
    modules = {
        "overview": overview, "traffic_monitor": traffic_monitor,
        "network_analysis": network_analysis,
        "alert_management": alert_management,
        "attack_intelligence": attack_intelligence,
        "explainability": explainability,
        "model_performance": model_performance,
        "resource_monitoring": resource_monitoring,
        "reports": reports, "settings_page": settings_page,
        "user_management": user_management,
    }
    registry: dict[str, st.Page] = {}
    first = True
    for key, title, icon, url_path in _DEFS:
        if role is not None and not rbac.page_allowed(role, key):
            continue
        registry[key] = st.Page(
            modules[key].render, title=title, icon=icon, url_path=url_path,
            default=first,
        )
        first = False
    REGISTRY.clear()
    REGISTRY.update(registry)
    return registry


def page(name: str) -> st.Page:
    return REGISTRY[name]