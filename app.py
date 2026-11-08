"""AI-SOC - Intelligent Security Operations dashboard (research prototype).

Streamlit entry point. Unauthenticated visitors only ever see the branded
authentication pages; the shell (brand, navigation, status footer) renders
after a valid server-side session. Every page is permission-checked on the
server, and navigation is filtered to the signed-in role.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

import config
from dashboard import auth_ui, common, components as ui, nav

st.set_page_config(
    page_title="AI-SOC | Intelligent Security Operations",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "AI-SOC research prototype - anomaly detection and alert triage "
            "on the static NSL-KDD benchmark. Not a production IDS/SIEM."
        ),
    },
)

ui.inject_css()

# Warm the model artefacts once per server process so the first sign-in after
# a restart does not pay the cold-load cost. get_models() is @st.cache_resource,
# so this populates the shared cache every page reuses; if artefacts are
# missing it records the errors and the pages show the missing-models notice.
common.get_models()

# Authentication gate: renders login/signup/recovery and st.stop() when
# there is no live session, so no dashboard element runs before sign-in.
user = auth_ui.require_user()

settings = common.get_settings()

ui.sidebar_brand()
auth_ui.sidebar_account(user)
pages = nav.build(role=user["role"])
page = st.navigation(list(pages.values()))
ui.sidebar_footer(*common.artefact_status(settings), version=config.APP_VERSION)

# Defence in depth: even if a page object were reached directly, the
# server refuses it for roles without the required permission.
if not auth_ui.enforce_page_access(user, nav.key_for_url(page.url_path)):
    auth_ui.render_cookie_bridge()
    st.stop()

page.run()
auth_ui.render_cookie_bridge()