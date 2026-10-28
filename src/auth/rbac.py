"""Role-based access control.

Permissions are enforced server-side (service layer and page gate), never
by hiding frontend controls alone. New registrations always receive the
least-privileged role; only an administrator (provisioned via
scripts/provision_admin.py or the User Management page) can grant more.
"""
from __future__ import annotations

ROLES = ("administrator", "analyst", "viewer")
DEFAULT_ROLE = "viewer"

ROLE_LABELS = {
    "administrator": "Administrator",
    "analyst": "SOC Analyst",
    "viewer": "Researcher / Viewer",
}
ROLE_DESCRIPTIONS = {
    "administrator": "Manage users, system settings and authorised "
                     "configurations; view system and audit information.",
    "analyst": "Analyse traffic, investigate and triage alerts, view SHAP "
               "explanations and generate reports.",
    "viewer": "Read-only access to the analytical dashboards, model "
              "evaluation, explainability results and reports.",
}
ROLE_COLORS = {
    "administrator": "#A78BFA",
    "analyst": "#38BDF8",
    "viewer": "#94A3B8",
}

PERMISSIONS: dict[str, set[str]] = {
    "administrator": {
        "view_dashboard", "triage_alerts", "manage_settings", "manage_users",
        "view_audit", "manage_sessions", "generate_reports",
    },
    "analyst": {"view_dashboard", "triage_alerts", "generate_reports"},
    "viewer": {"view_dashboard", "generate_reports"},
}

# Dashboard page key -> permission required to open it.
PAGE_PERMISSIONS = {
    "overview": "view_dashboard",
    "traffic_monitor": "view_dashboard",
    "network_analysis": "view_dashboard",
    "alert_management": "view_dashboard",
    "attack_intelligence": "view_dashboard",
    "explainability": "view_dashboard",
    "model_performance": "view_dashboard",
    "resource_monitoring": "view_dashboard",
    "reports": "generate_reports",
    "settings_page": "manage_settings",
    "user_management": "manage_users",
}


class PermissionDenied(Exception):
    """Raised when a role lacks a permission; messages stay user-safe."""

    def __init__(self, permission: str):
        self.permission = permission
        super().__init__(f"Missing permission: {permission}")


def can(role: str, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, set())


def require(role: str, permission: str) -> None:
    if not can(role, permission):
        raise PermissionDenied(permission)


def page_allowed(role: str, page_key: str) -> bool:
    permission = PAGE_PERMISSIONS.get(page_key)
    return True if permission is None else can(role, permission)