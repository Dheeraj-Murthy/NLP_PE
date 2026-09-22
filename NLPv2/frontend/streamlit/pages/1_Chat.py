import streamlit as st

from lib import api_client
from lib.ui_helpers import (
    confidence_badge,
    init_session_state,
    render_api_error,
    render_citations,
    render_model_selector,
    render_sources,
)

st.set_page_config(page_title="Chat — Legal RAG", page_icon=":material/chat:", layout="wide")
init_session_state()

st.title("Chat", icon=":material/chat:")
st.caption(
    "This deployment keeps a single shared conversation session across every user — "
    "not per-browser or per-user. Treat it as a single-user/demo chat.",
)

with st.sidebar:
    model, external_ok, api_key = render_model_selector("chat")
    include_debug = st.toggle("Include debug info", value=False)
    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
        try:
            api_client.chat_clear()
            st.session_state.chat_history = []
            st.rerun()
        except Exception as e:
            render_api_error(e)

if not st.session_state.chat_history:
    selected = st.pills(
        "Try asking:",
        [
            "What are the principles of natural justice?",
            "What is the punishment for murder under Indian law?",
            "Explain the right to life under Article 21",
        ],
        label_visibility="collapsed",
    )
    if selected:
        st.session_state.pending_input = selected
        st.rerun()

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("confidence") is not None:
                confidence_badge(turn["confidence"])
            render_citations(turn.get("citations", []))
            render_sources(turn.get("sources", []))
            if turn.get("debug"):
                with st.expander("Debug info", icon=":material/bug_report:"):
                    st.json(turn["debug"])

user_input = st.chat_input("Ask a legal question...") or st.session_state.pop("pending_input", None)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if model and not external_ok:
            st.warning(
                "Tick the external-model confirmation checkbox in the sidebar first.",
                icon=":material/lock:",
            )
            data = None
        else:
            with st.status(":shimmer[Retrieving and generating]", type="compact") as status:
                try:
                    data = api_client.chat(
                        user_input, include_debug=include_debug,
                        model=model, external_ok=external_ok, api_key=api_key,
                    )
                    status.update(label="Done", state="complete")
                except Exception as e:
                    status.update(label="Failed", state="error")
                    render_api_error(e)
                    data = None

        if data is not None:
            if not data.get("answer_found", True):
                st.warning(data["answer"], icon=":material/search_off:")
            else:
                st.markdown(data["answer"])

            confidence = data.get("confidence")
            if confidence is not None:
                confidence_badge(confidence)
            render_citations(data.get("citations", []))
            render_sources(data.get("sources", []))
            if data.get("debug"):
                with st.expander("Debug info", icon=":material/bug_report:"):
                    st.json(data["debug"])

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data.get("citations", []),
                    "sources": data.get("sources", []),
                    "confidence": confidence,
                    "debug": data.get("debug"),
                }
            )
