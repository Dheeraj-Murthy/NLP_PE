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

tab_landmarks, tab_ego, tab_path = st.tabs(["Landmark cases", "Ego graph", "Precedent path"])

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
        picked = st.selectbox("Focus a case from the table above", ["—"] + list(options.keys()))
        if picked != "—":
            selected_judgment_id = options[picked]
    else:
        st.caption("No landmark cases returned — citation graph may be empty.")

with tab_ego:
    st.caption("N-hop citation network centered on a single judgment.")
    with st.container(horizontal=True):
        judgment_id = st.number_input(
            "Judgment ID", min_value=1, value=selected_judgment_id or 1, step=1
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
