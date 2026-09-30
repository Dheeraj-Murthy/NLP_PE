from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from lib.api_client import LegalRAGAPIError

MODEL_OPTIONS = {
    "Qwen (local, default)": (None, False),
    "Claude (Anthropic API)": ("claude-sonnet-5", True),
    "GPT (OpenAI API)": ("gpt-4o", True),
    "Gemini (Google API)": ("gemini-2.5-pro", True),
}


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])


def render_model_selector(key_prefix: str) -> Tuple[Optional[str], bool, Optional[str]]:
    """Model picker + privacy gate. Qwen (local) needs no confirmation;
    picking an external model surfaces a visible warning, requires an
    explicit checkbox before the caller may pass external_ok=True, and lets
    the user optionally supply their own API key for that call instead of
    relying on the server's own ANTHROPIC_API_KEY/OPENAI_API_KEY.

    The key is only ever held in this browser session's st.session_state
    (in-memory, cleared on refresh/close) and sent on the one request it's
    used for — it is never written to disk or persisted server-side (see
    rag_pipeline._resolve_backend, which explicitly avoids caching a
    caller-supplied key)."""
    choice = st.selectbox(
        "Model", list(MODEL_OPTIONS.keys()), key=f"{key_prefix}_model_choice"
    )
    model, is_external = MODEL_OPTIONS[choice]

    external_ok = False
    api_key = None
    if is_external:
        if model.startswith("claude"):
            provider = "Anthropic"
        elif model.startswith("gemini"):
            provider = "Google"
        else:
            provider = "OpenAI"
        st.caption(
            f":material/warning: Sends the retrieved case text and your question to "
            f"{provider}'s API — not local.",
        )
        external_ok = st.checkbox(
            "I understand — send this query externally",
            key=f"{key_prefix}_external_ok",
        )
        api_key = st.text_input(
            f"{provider} API key (optional)",
            type="password",
            key=f"{key_prefix}_api_key",
            help="Leave blank to use the server's own configured key, if any. "
            "Held only in this browser session, never saved.",
        )

    return model, external_ok, (api_key or None)


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

    if metrics.get("model_id"):
        st.caption(f":material/smart_toy: {metrics['model_id']} ({metrics.get('model_version', '?')})")

    extra = {
        k: v for k, v in metrics.items()
        if k not in {t[0] for t in labels} and k not in ("model_id", "model_version")
    }
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
