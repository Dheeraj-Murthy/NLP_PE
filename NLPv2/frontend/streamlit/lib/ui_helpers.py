from typing import Any, Dict, List

import streamlit as st

from lib.api_client import LegalRAGAPIError


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])


def render_citations(citations: List[str]) -> None:
    if not citations:
        return
    with st.expander(f"Citations ({len(citations)})", icon=":material/gavel:"):
        for c in citations:
            st.markdown(f"- {c}")


def render_sources(sources: List[str]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for s in sources:
            st.markdown(f"- {s}")


def confidence_badge(confidence: float) -> None:
    if confidence >= 0.7:
        st.badge(f"Confidence {confidence:.0%}", icon=":material/check_circle:", color="green")
    elif confidence >= 0.4:
        st.badge(f"Confidence {confidence:.0%}", icon=":material/warning:", color="orange")
    else:
        st.badge(f"Confidence {confidence:.0%}", icon=":material/error:", color="red")


def render_metrics(metrics: Dict[str, Any]) -> None:
    if not metrics:
        return
    labels = [
        ("retrieval_time", "Retrieval", "s"),
        ("generation_time", "Generation", "s"),
        ("total_time", "Total", "s"),
        ("chunks_retrieved", "Chunks", ""),
    ]
    with st.container(horizontal=True):
        for key, label, unit in labels:
            if key in metrics:
                value = metrics[key]
                st.metric(label, f"{value}{unit}" if unit else value, border=True)

    extra = {k: v for k, v in metrics.items() if k not in {t[0] for t in labels}}
    if extra:
        with st.expander("More metrics", icon=":material/tune:"):
            st.json(extra)


def render_precedent_chains(chains: List[Dict[str, Any]]) -> None:
    if not chains:
        return
    st.subheader("Precedent chains", icon=":material/account_tree:")
    for chain in chains:
        label = chain.get("label", f"Case #{chain.get('judgment_id')}")
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(
                f"Cites {chain.get('cites_count', 0)} cases · "
                f"cited by {chain.get('cited_by_count', 0)} later judgments"
            )
            for c in chain.get("cites", []):
                st.markdown(f"→ cites: {c['label']} `{c['relationship']}`")
            for c in chain.get("cited_by", []):
                st.markdown(f"← cited by: {c['label']} `{c['relationship']}`")


def render_api_error(exc: Exception) -> None:
    if isinstance(exc, LegalRAGAPIError):
        st.error(f"Request failed: {exc}", icon=":material/error:")
    else:
        st.error(f"Unexpected error: {exc}", icon=":material/error:")
