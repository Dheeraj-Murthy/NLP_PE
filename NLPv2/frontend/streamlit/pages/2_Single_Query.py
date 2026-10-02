import streamlit as st

from lib import api_client
from lib.ui_helpers import (
    confidence_badge,
    render_api_error,
    render_metrics,
    render_model_selector,
    render_precedent_chains,
    render_sectioned_sources_and_citations,
)

st.set_page_config(page_title="Single query — Legal RAG", page_icon=":material/search:", layout="wide")

st.title("Single query", icon=":material/search:")
st.caption("One-shot grounded Q&A with retrieval/generation controls, full metrics, and precedent chains.")

with st.form("single_query_form"):
    query_text = st.text_area("Question", placeholder="What are the principles of natural justice?")
    with st.container(horizontal=True):
        top_k = st.slider("top_k", min_value=1, max_value=20, value=8)
        threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        include_debug = st.toggle("Include debug info", value=False)
    model, external_ok, api_key, is_external = render_model_selector("single_query")
    submitted = st.form_submit_button("Run query", icon=":material/play_arrow:", type="primary")

if submitted and query_text.strip():
    if is_external and (not external_ok or not model):
        st.warning(
            "Tick the external-model confirmation checkbox and pick a model first."
            if not external_ok
            else "Pick a model first (test your API key if you haven't).",
            icon=":material/lock:",
        )
        st.session_state.last_query_result = None
    else:
        with st.spinner("Retrieving and generating..."):
            try:
                data = api_client.query(
                    query_text, top_k=top_k, threshold=threshold, include_debug=include_debug,
                    model=model, external_ok=external_ok, api_key=api_key,
                )
                st.session_state.last_query_result = data
            except Exception as e:
                render_api_error(e)
                st.session_state.last_query_result = None

result = st.session_state.get("last_query_result")
if result and result.get("error"):
    st.error(result["error"], icon=":material/error:")
elif result:
    with st.container(border=True):
        if not result.get("answer_found", True):
            st.warning(result["answer"], icon=":material/search_off:")
        else:
            st.markdown(result["answer"])

        confidence = result.get("confidence")
        if confidence is not None:
            confidence_badge(confidence)

        render_sectioned_sources_and_citations(result)

    st.space("medium")
    render_metrics(result.get("metrics", {}))

    st.space("medium")
    render_precedent_chains(result.get("precedent_chains", []))

    if result.get("debug"):
        with st.expander("Debug info", icon=":material/bug_report:"):
            st.json(result["debug"])
