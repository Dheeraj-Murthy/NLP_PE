import streamlit as st

from lib import api_client
from lib.ui_helpers import (
    confidence_badge,
    render_api_error,
    render_citations,
    render_metrics,
    render_sources,
)

st.set_page_config(page_title="Document upload — Legal RAG", page_icon=":material/upload_file:", layout="wide")

st.title("Document upload", icon=":material/upload_file:")
st.caption("Upload a PDF, DOCX, or TXT document. OCR runs server-side for scanned PDFs and can take a while.")

uploaded_file = st.file_uploader("Document", type=["pdf", "docx", "txt"])
query_text = st.text_input("Optional question about this document", placeholder="Summarize the key holdings")
with st.container(horizontal=True):
    include_retrieval = st.toggle("Cross-reference case law", value=True)
    include_debug = st.toggle("Include debug info", value=False)

if st.button(
    "Analyze document",
    icon=":material/play_arrow:",
    type="primary",
    disabled=uploaded_file is None,
):
    with st.spinner("Uploading and processing (OCR + generation can take several minutes)..."):
        try:
            data = api_client.upload_document(
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                content_type=uploaded_file.type or "application/octet-stream",
                query_text=query_text or None,
                include_retrieval=include_retrieval,
                include_debug=include_debug,
            )
            st.session_state.last_document_result = data
        except Exception as e:
            render_api_error(e)
            st.session_state.last_document_result = None

result = st.session_state.get("last_document_result")
if result:
    doc = result.get("document", {})
    if doc:
        st.badge(
            f"{doc.get('source', 'document')} — {doc.get('pages', '?')} pages · {doc.get('type', '?')}",
            icon=":material/description:",
            color="blue",
        )
        st.space("small")

    with st.container(border=True):
        if not result.get("answer_found", True):
            st.warning(result.get("answer", "No answer produced."), icon=":material/search_off:")
        else:
            st.markdown(result.get("answer", ""))

        confidence = result.get("confidence")
        if confidence is not None:
            confidence_badge(confidence)

        render_citations(result.get("citations", []))
        render_sources(result.get("sources", []))

    st.space("medium")
    render_metrics(result.get("metrics", {}))

    if result.get("debug"):
        with st.expander("Debug info", icon=":material/bug_report:"):
            st.json(result["debug"])
