from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from lib import api_client
from lib.api_client import LegalRAGAPIError

MODEL_OPTIONS = {
    "Qwen (local, default)": (None, False),
    "Claude (Anthropic API)": ("claude-sonnet-5", True),
    "GPT (OpenAI API)": ("gpt-4o", True),
    "Gemini (Google API)": ("gemini-2.5-pro", True),
}


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])


def render_model_selector(key_prefix: str) -> Tuple[Optional[str], bool, Optional[str], bool]:
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

        if model.startswith("gemini"):
            model = _render_gemini_model_picker(key_prefix, api_key, default_model=model)

    return model, external_ok, (api_key or None), is_external


def _render_gemini_model_picker(
    key_prefix: str, api_key: Optional[str], default_model: str
) -> Optional[str]:
    """Test button validates the key against Gemini's ListModels API and, on
    success, shows a dropdown of only the models that key can actually call
    generateContent on. No hardcoded fallback — Google deprecates/renames
    model ids over time (e.g. gemini-2.5-pro 404ing for new keys), so a
    stale pinned default would silently route to a dead model instead of
    one this key can really use. Returns None (blocking send) until a real,
    tested model is chosen."""
    models_state_key = f"{key_prefix}_gemini_models"
    tested_key_state = f"{key_prefix}_gemini_tested_key"

    # Key changed since last successful test — stale model list no longer applies.
    if st.session_state.get(tested_key_state) != api_key:
        st.session_state[models_state_key] = None

    if st.button(
        "Test API key", key=f"{key_prefix}_test_gemini_key", disabled=not api_key
    ):
        try:
            st.session_state[models_state_key] = api_client.gemini_models(api_key)
            st.session_state[tested_key_state] = api_key
        except Exception as e:
            st.session_state[models_state_key] = None
            render_api_error(e)

    available_models = st.session_state.get(models_state_key)
    if available_models:
        st.success(
            f"Key valid — {len(available_models)} model(s) available.",
            icon=":material/check_circle:",
        )
        default_index = (
            available_models.index(default_model)
            if default_model in available_models
            else 0
        )
        return st.selectbox(
            "Gemini model",
            available_models,
            index=default_index,
            key=f"{key_prefix}_gemini_model_choice",
        )

    if available_models is not None:
        # Tested successfully but the key has access to nothing usable.
        st.warning(
            "This key has no models available that support generateContent.",
            icon=":material/block:",
        )
    else:
        st.caption("Test your API key to pick from the models it can actually access.")
    return None


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


def _render_type_expander(title: str, icon: str, citations: List[str], sources: List[str]) -> None:
    with st.expander(f"{title} ({len(citations) or len(sources)})", icon=icon):
        if citations:
            st.markdown("**Citations**")
            for c in citations:
                st.markdown(f"- {c}")
        if sources:
            st.markdown("**Sources**")
            for s in sources:
                st.markdown(f"- {s}")


def render_sectioned_sources_and_citations(data: Dict[str, Any]) -> None:
    """Statutes-first, two-expander layout when citations_by_type/
    sources_by_type are present; falls back to the old flat
    render_citations/render_sources otherwise (e.g. responses from before
    this field existed, or any caller that only ever returns flat lists)."""
    citations_by_type = data.get("citations_by_type")
    sources_by_type = data.get("sources_by_type")

    if not citations_by_type and not sources_by_type:
        render_citations(data.get("citations", []))
        render_sources(data.get("sources", []))
        return

    statute_citations = (citations_by_type or {}).get("statute", [])
    statute_sources = (sources_by_type or {}).get("statute", [])
    judgment_citations = (citations_by_type or {}).get("judgment", [])
    judgment_sources = (sources_by_type or {}).get("judgment", [])

    if statute_citations or statute_sources:
        _render_type_expander(
            "Statutes & Articles", ":material/balance:", statute_citations, statute_sources
        )
    if judgment_citations or judgment_sources:
        _render_type_expander(
            "Case Law", ":material/gavel:", judgment_citations, judgment_sources
        )


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
