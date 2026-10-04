import streamlit as st

from lib import api_client
from lib.ui_helpers import render_api_error


# Graphviz lays the graph out in the browser, and edge labels are by far the
# most expensive part of that layout. Past this many edges, drop them and
# rely on the edges table instead.
EDGE_LABEL_LIMIT = 60


def _build_dot(nodes, edges) -> str:
    show_edge_labels = len(edges) <= EDGE_LABEL_LIMIT
    lines = ["digraph {", "  graph [rankdir=LR];", "  node [shape=box];"]
    for n in nodes:
        label = n.get("label", f"Case #{n['id']}").replace('"', "'")
        fill = "#18181B" if n.get("is_center") else "#F4F4F5"
        font = "#FFFFFF" if n.get("is_center") else "#09090B"
        lines.append(
            f'  "{n["id"]}" [label="{label}", style=filled, fillcolor="{fill}", fontcolor="{font}"];'
        )
    for e in edges:
        if show_edge_labels:
            rel = (e.get("relationship") or "cited").replace('"', "'")
            lines.append(f'  "{e["source"]}" -> "{e["target"]}" [label="{rel}"];')
        else:
            lines.append(f'  "{e["source"]}" -> "{e["target"]}";')
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="Citation graph — Legal RAG", page_icon=":material/account_tree:", layout="wide")

st.title("Citation graph", icon=":material/account_tree:")
st.caption("Browse landmark cases and precedent networks derived from the judgment citation graph.")

NEIGHBOR_PAGE = 25

st.session_state.setdefault("graph_focus", None)
st.session_state.setdefault("graph_trail", [])


def _focus(judgment_id: int, label: str) -> None:
    """Make a case the starting point for the ego graph and citation lists."""
    st.session_state["graph_focus"] = judgment_id
    trail = [t for t in st.session_state["graph_trail"] if t[0] != judgment_id]
    st.session_state["graph_trail"] = (trail + [(judgment_id, label)])[-8:]
    st.session_state["cites_offset"] = 0
    st.session_state["cited_by_offset"] = 0


def _focus_from(widget_key: str, options: dict, reset: bool = False) -> None:
    """Selectbox callback: focus the picked case. Runs only when the user
    changes the selection, so several pickers don't fight each other."""
    picked = st.session_state.get(widget_key)
    if picked in options:
        _focus(*options[picked])
        if reset:
            st.session_state[widget_key] = "—"


@st.cache_data(ttl=60, show_spinner=False)
def _search(q: str):
    return api_client.graph_search(q, limit=15)


@st.cache_data(ttl=60, show_spinner=False)
def _neighbors(judgment_id: int, direction: str, offset: int):
    return api_client.graph_neighbors(judgment_id, direction=direction, limit=NEIGHBOR_PAGE, offset=offset)


query = st.text_input(
    "Find a case",
    placeholder="Case name, citation (e.g. [1950] S.C.R. 940) or judgment ID",
)
if query.strip():
    try:
        results = _search(query.strip())
    except Exception as e:
        render_api_error(e)
        results = []
    if results:
        options = {
            f"{r['label']} (#{r['judgment_id']}, {r['date']})": (r["judgment_id"], r["label"]) for r in results
        }
        st.selectbox(
            f"{len(results)} matches",
            ["—"] + list(options.keys()),
            key="search_pick",
            on_change=_focus_from,
            args=("search_pick", options),
        )
    else:
        st.caption("No matching cases.")

tab_landmarks, tab_ego, tab_neighbors, tab_path = st.tabs(
    ["Landmark cases", "Ego graph", "Cites / Cited by", "Precedent path"]
)

with tab_landmarks:
    st.caption("Top authority cases by PageRank centrality over the citation network.")
    limit = st.number_input("Limit", min_value=1, max_value=100, value=10)

    @st.cache_data(ttl=60)
    def _landmark_cases(limit: int):
        return api_client.graph_landmark_cases(limit=limit)

    try:
        landmarks = _landmark_cases(limit)
    except Exception as e:
        render_api_error(e)
        landmarks = []

    selected_judgment_id = None
    if landmarks:
        st.dataframe(
            landmarks,
            column_order=["judgment_id", "label", "court", "date", "in_degree", "pagerank_score"],
            width="stretch",
        )
        options = {f"{c['label']} (#{c['judgment_id']})": c["judgment_id"] for c in landmarks}
        focus_options = {k: (v, k.rsplit(" (#", 1)[0]) for k, v in options.items()}
        picked = st.selectbox(
            "Focus a case from the table above",
            ["—"] + list(options.keys()),
            key="landmark_pick",
            on_change=_focus_from,
            args=("landmark_pick", focus_options),
        )
        if picked != "—":
            selected_judgment_id = options[picked]
    else:
        st.caption("No landmark cases returned — citation graph may be empty.")

with tab_ego:
    st.caption("N-hop citation network centered on a single judgment.")
    with st.container(horizontal=True):
        judgment_id = st.number_input(
            "Judgment ID",
            min_value=1,
            value=st.session_state["graph_focus"] or selected_judgment_id or 1,
            step=1,
        )
        depth = st.slider("Depth", min_value=1, max_value=3, value=2)
        max_nodes = st.number_input(
            "Max nodes",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Large graphs are slow to draw. Closest and most-cited cases are kept first.",
        )

    if st.button("Load ego graph", icon=":material/play_arrow:", type="primary"):
        try:
            data = api_client.graph_judgment(int(judgment_id), depth=depth, max_nodes=int(max_nodes))
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            if not nodes:
                st.warning(f"Judgment {judgment_id} not found in the citation graph.", icon=":material/search_off:")
            else:
                if data.get("truncated"):
                    st.info(
                        f"Showing {len(nodes)} of {data.get('total_nodes')} cases within {depth} hops. "
                        "Lower the depth or raise Max nodes to see more.",
                        icon=":material/filter_alt:",
                    )
                if len(edges) > EDGE_LABEL_LIMIT:
                    st.caption("Edge labels hidden for large graphs — see the edges table below.")
                st.graphviz_chart(_build_dot(nodes, edges))
                with st.expander(f"Edges ({len(edges)})", icon=":material/list:"):
                    st.dataframe(edges, width="stretch")
        except Exception as e:
            render_api_error(e)

with tab_neighbors:
    st.caption("Cases this judgment cites and cases that cite it, most important first. Open one to go a level deeper.")
    trail = st.session_state["graph_trail"]
    if len(trail) > 1:
        st.caption("Trail: " + " → ".join(f"#{jid}" for jid, _ in trail))
    with st.container(horizontal=True):
        center_id = st.number_input(
            "Judgment ID",
            min_value=1,
            value=st.session_state["graph_focus"] or 1,
            step=1,
            # Keyed on the focus so the box follows it when a case is opened.
            key=f"neighbors_center_{st.session_state['graph_focus']}",
        )
        if st.button("Back", icon=":material/arrow_back:", disabled=len(trail) < 2):
            trail.pop()
            st.session_state["graph_focus"] = trail[-1][0]
            st.session_state["cites_offset"] = st.session_state["cited_by_offset"] = 0
            st.rerun()

    col_cites, col_cited_by = st.columns(2)
    for col, direction, title in (
        (col_cites, "cites", "Cites"),
        (col_cited_by, "cited-by", "Cited by"),
    ):
        offset_key = f"{direction.replace('-', '_')}_offset"
        offset = st.session_state.get(offset_key, 0)
        with col:
            try:
                page = _neighbors(int(center_id), direction, offset)
            except Exception as e:
                render_api_error(e)
                continue
            total = page["total"]
            st.subheader(f"{title} ({total})")
            if not page["items"]:
                st.caption("None found.")
                continue
            st.dataframe(
                page["items"],
                column_order=["judgment_id", "label", "relationship", "date", "pagerank_score"],
                width="stretch",
                hide_index=True,
            )
            with st.container(horizontal=True):
                if st.button("Previous", key=f"{direction}_prev", disabled=offset == 0):
                    st.session_state[offset_key] = max(0, offset - NEIGHBOR_PAGE)
                    st.rerun()
                st.caption(f"{offset + 1}–{offset + len(page['items'])} of {total}")
                if st.button("Next", key=f"{direction}_next", disabled=offset + NEIGHBOR_PAGE >= total):
                    st.session_state[offset_key] = offset + NEIGHBOR_PAGE
                    st.rerun()
            choices = {f"{i['label']} (#{i['judgment_id']})": (i["judgment_id"], i["label"]) for i in page["items"]}
            st.selectbox(
                "Open a case",
                ["—"] + list(choices.keys()),
                key=f"{direction}_open",
                on_change=_focus_from,
                args=(f"{direction}_open", choices, True),
            )

with tab_path:
    st.caption("Shortest citation chain between two judgments.")
    with st.container(horizontal=True):
        source_id = st.number_input("Source judgment ID", min_value=1, value=1, step=1, key="src")
        target_id = st.number_input("Target judgment ID", min_value=1, value=2, step=1, key="tgt")

    if st.button("Find path", icon=":material/route:", type="primary"):
        try:
            path = api_client.graph_path(int(source_id), int(target_id))
            if path is None:
                st.warning("No citation path found between these cases.", icon=":material/search_off:")
            else:
                st.success(" → ".join(str(p) for p in path), icon=":material/check_circle:")
        except Exception as e:
            render_api_error(e)
