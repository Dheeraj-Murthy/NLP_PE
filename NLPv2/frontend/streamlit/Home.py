import streamlit as st

from lib import api_client
from lib.ui_helpers import init_session_state

st.set_page_config(page_title="Legal RAG", page_icon=":material/balance:", layout="wide")
init_session_state()

st.title("Legal RAG", icon=":material/balance:")
st.caption(
    "Retrieval-augmented Q&A over Indian Supreme Court judgments and statutes — "
    "a research tool, not legal advice. Verify every citation against original case law."
)

st.space("large")

nav_cards = [
    ("pages/1_Chat.py", ":material/chat:", "Chat", "Multi-turn conversation with the RAG pipeline"),
    ("pages/2_Single_Query.py", ":material/search:", "Single query", "One-shot Q&A with retrieval/generation controls and metrics"),
    ("pages/3_Document_Upload.py", ":material/upload_file:", "Document upload", "OCR + query an uploaded PDF, DOCX, or TXT"),
    ("pages/4_Citation_Graph.py", ":material/account_tree:", "Citation graph", "Browse landmark cases and precedent networks"),
    ("pages/5_Admin_Debug.py", ":material/tune:", "Admin / debug", "Pipeline status and retrieval-only diagnostics"),
]

# Stretch each tile's page link over its whole card so the entire tile is clickable.
st.html(
    """
    <style>
    [class*="st-key-nav-tile-"] { position: relative; min-height: 7rem; transition: border-color .15s, background-color .15s; }
    [class*="st-key-nav-tile-"]:hover { border-color: #18181B; background-color: #FAFAFA; }
    [class*="st-key-nav-tile-"] [data-testid="stElementContainer"] { position: static; }
    [class*="st-key-nav-tile-"] [data-testid="stPageLink"] a::after { content: ""; position: absolute; inset: 0; z-index: 1; }
    </style>
    """
)

NAV_COLUMNS = 3
for row_start in range(0, len(nav_cards), NAV_COLUMNS):
    cols = st.columns(NAV_COLUMNS)
    for col, (page, icon, title, desc) in zip(cols, nav_cards[row_start : row_start + NAV_COLUMNS]):
        with col, st.container(border=True, height="stretch", key=f"nav-tile-{page[6:-3]}"):
            st.page_link(page, label=f"**{title}**", icon=icon)
            st.caption(desc)

st.space("large")

try:
    health = api_client.health()
    if health.get("status") == "healthy":
        st.badge(
            f"Backend connected — model loaded: {health.get('model_loaded')}",
            icon=":material/check_circle:",
            color="green",
        )
    else:
        st.badge(f"Backend status: {health.get('status')}", icon=":material/warning:", color="orange")
except Exception as e:
    st.error(
        f"Cannot reach backend at `{api_client.BASE_URL}`. "
        f"Check the `RAG_API_URL` environment variable. ({e})",
        icon=":material/error:",
    )

with st.sidebar:
    st.caption(f":material/dns: {api_client.BASE_URL}")
    try:
        status = api_client.status()
        with st.expander("System status", icon=":material/monitoring:"):
            st.json(status)
    except Exception:
        st.caption("Status unavailable")
