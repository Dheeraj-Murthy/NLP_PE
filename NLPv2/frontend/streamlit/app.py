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
    (":material/chat:", "Chat", "Multi-turn conversation with the RAG pipeline"),
    (":material/search:", "Single query", "One-shot Q&A with retrieval/generation controls and metrics"),
    (":material/upload_file:", "Document upload", "OCR + query an uploaded PDF, DOCX, or TXT"),
    (":material/account_tree:", "Citation graph", "Browse landmark cases and precedent networks"),
    (":material/tune:", "Admin / debug", "Pipeline status and retrieval-only diagnostics"),
]

with st.container(horizontal=True):
    for icon, title, desc in nav_cards:
        with st.container(border=True, width=220):
            st.markdown(f"{icon} **{title}**")
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
