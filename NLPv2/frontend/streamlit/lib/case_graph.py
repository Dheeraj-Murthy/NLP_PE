"""Citation graph, full network or precedent path for a chosen set of cases.

Shared by Chat and Single query (the judgments an answer cites) and by the
tables on the Citation graph page (the rows the user selects).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

from lib import api_client
from lib.mind_map import render_mind_map
from lib.network_view import render_network
from lib.ui_helpers import linked_label, render_api_error

Case = Tuple[int, str]

VIEWS = {
    "mind_map": ":material/account_tree: Citation graph",
    "network": ":material/hub: Full network",
    "path": ":material/route: Precedent path",
}
# More than this many mind maps or merged networks gets slow and unreadable.
MAX_CASES = 6
PER_BRANCH = 12


@st.cache_data(ttl=300, show_spinner=False)
def _mind_map(judgment_id: int, per_branch: int):
    return api_client.graph_mind_map(judgment_id, per_branch=per_branch)


@st.cache_data(ttl=300, show_spinner=False)
def _subgraph(judgment_id: int, depth: int, max_nodes: int):
    return api_client.graph_judgment(judgment_id, depth=depth, max_nodes=max_nodes)


@st.cache_data(ttl=300, show_spinner=False)
def _path(source_id: int, target_id: int) -> Optional[List[int]]:
    return api_client.graph_path(source_id, target_id)


@st.cache_data(ttl=3600, show_spinner=False)
def case_label(judgment_id: int) -> str:
    """A judgment's case name, looked up by ID."""
    try:
        hits = api_client.graph_search(str(judgment_id), limit=1)
    except Exception:
        hits = []
    if hits and hits[0]["judgment_id"] == judgment_id:
        return hits[0]["label"]
    return f"Case #{judgment_id}"


def answer_cases(data: Dict[str, Any]) -> List[Case]:
    """The judgments an answer cites or drew on, first mention first."""
    refs: List[Dict[str, Any]] = []
    for key in ("citation_refs_by_type", "source_refs_by_type"):
        refs += (data.get(key) or {}).get("judgment", [])
    for key in ("citation_refs", "source_refs"):
        refs += [r for r in data.get(key) or [] if r.get("doc_type") != "statute"]
    refs += data.get("precedent_chains") or []
    cases: Dict[int, str] = {}
    for r in refs:
        jid = r.get("judgment_id")
        if jid and int(jid) not in cases:
            cases[int(jid)] = r.get("label") or f"Case #{jid}"
    return list(cases.items())


def selected_cases(rows: Sequence[Dict[str, Any]], event) -> List[Case]:
    """Cases for the rows picked in an st.dataframe(on_select=...)."""
    picked = event.selection.rows if event else []
    return [(rows[i]["judgment_id"], rows[i]["label"]) for i in picked if i < len(rows)]


def merge_networks(graphs: List[Dict[str, Any]], centers: List[int]) -> Dict[str, Any]:
    """One network from several cases' ego graphs. Hops are left out so the
    view measures them from the first case; the others are drawn as centres."""
    nodes: Dict[int, Dict[str, Any]] = {}
    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for g in graphs:
        for n in g.get("nodes", []):
            nodes.setdefault(n["id"], {k: v for k, v in n.items() if k != "hop"})
        for e in g.get("edges", []):
            edges.setdefault((e["source"], e["target"]), e)
    for jid in centers:
        if jid in nodes:
            nodes[jid]["is_center"] = True
    return {
        "center_id": centers[0],
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "truncated": any(g.get("truncated") for g in graphs),
    }


def _chain(a: int, b: int) -> Optional[List[int]]:
    """A citation chain between two cases, in whichever direction exists."""
    return _path(a, b) or _path(b, a)


def _note_unlinked(data: Dict[str, Any], ids: List[int], labels: Dict[int, str]) -> None:
    """Say which chosen cases the drawn network doesn't connect to the first,
    and whether any citation chain links them at all."""
    links: Dict[int, set] = {n["id"]: set() for n in data["nodes"]}
    for e in data["edges"]:
        links[e["source"]].add(e["target"])
        links[e["target"]].add(e["source"])
    reached, stack = {ids[0]}, [ids[0]]
    while stack:
        for m in links.get(stack.pop(), ()):
            if m not in reached:
                reached.add(m)
                stack.append(m)
    first = labels[ids[0]]
    for jid in ids[1:]:
        if jid in reached:
            continue
        try:
            chain = _chain(ids[0], jid)
        except Exception as e:
            render_api_error(e)
            return
        if chain is None:
            st.info(f"There is no possible citation chain between {first} and {labels[jid]}.",
                    icon=":material/link_off:")
        else:
            hops = len(chain) - 1
            st.info(f"{first} and {labels[jid]} are linked by a citation chain of {hops} hop"
                    f"{'' if hops == 1 else 's'}, further out than this network reaches — "
                    "see Precedent path for the chain.", icon=":material/route:")


def render_path(source: Case, target: Case) -> None:
    """Shortest citation chain between two cases, tried in both directions."""
    try:
        with st.spinner("Finding a citation chain…"):
            path = _path(source[0], target[0])
            if path is None:
                path = _path(target[0], source[0])
                if path is not None:
                    source, target = target, source
    except Exception as e:
        render_api_error(e)
        return
    if path is None:
        st.warning(
            f"Neither case leads to the other through citations: no chain from {source[1]} to {target[1]} "
            "or back.",
            icon=":material/search_off:",
        )
        return
    hops = len(path) - 1
    st.success(
        f"{source[1]} reaches {target[1]} in {hops} citation hop{'' if hops == 1 else 's'} — each case cites the next.",
        icon=":material/route:",
    )
    known = dict([source, target])
    for i, jid in enumerate(path):
        label = known.get(jid) or case_label(jid)
        st.markdown(f"{i + 1}. {linked_label(label, {'judgment_id': jid})} `#{jid}`")


def _short(label: str, width: int = 40) -> str:
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"


def render_case_graph_panel(cases: List[Case], key: str, pick: bool = True) -> None:
    """Controls to draw the citation graph (a mind map per case), the full
    network around the cases, or the precedent path between two of them.

    pick: let the user choose among `cases` (answers); off when the cases are
    already a selection (table rows). Nothing is fetched until Build is
    pressed; after that the view follows the controls until it is closed."""
    if not cases:
        return
    labels = dict(cases)
    if pick:
        # Keyed on the cases offered, so a new answer starts a fresh selection.
        ids = st.multiselect(
            "Cases",
            [jid for jid, _ in cases],
            default=[cases[0][0]],
            format_func=lambda jid: f"{labels[jid]} (#{jid})",
            max_selections=MAX_CASES,
            key=f"{key}_cases_{'-'.join(str(jid) for jid, _ in cases)}",
        )
    else:
        ids = [jid for jid, _ in cases][:MAX_CASES]
        if len(cases) > MAX_CASES:
            st.caption(f"Using the first {MAX_CASES} of the {len(cases)} selected cases.")

    with st.container(horizontal=True, vertical_alignment="bottom"):
        view = st.segmented_control(
            "Show", list(VIEWS), format_func=VIEWS.get, default="mind_map", key=f"{key}_view"
        ) or "mind_map"
        if view == "network":
            depth = st.slider("Depth", min_value=1, max_value=3, value=1, key=f"{key}_depth")
            max_nodes = st.number_input(
                "Max nodes per case", min_value=10, max_value=300, value=80, step=10, key=f"{key}_max_nodes"
            )
        on_key = f"{key}_on"
        if st.session_state.get(on_key):
            st.button("Close", icon=":material/close:", key=f"{key}_close",
                      on_click=lambda: st.session_state.update({on_key: False}))
        else:
            st.button("Build", icon=":material/play_arrow:", type="primary", key=f"{key}_build",
                      disabled=not ids, on_click=lambda: st.session_state.update({on_key: True}))

    if not st.session_state.get(on_key):
        return
    if not ids:
        st.caption("Pick at least one case.")
        return

    if view == "mind_map":
        tabs = st.tabs([_short(labels[jid]) for jid in ids]) if len(ids) > 1 else [st.container()]
        for tab, jid in zip(tabs, ids):
            with tab:
                try:
                    with st.spinner("Building citation graph…"):
                        tree = _mind_map(jid, PER_BRANCH)
                    render_mind_map(tree, height=560)
                except Exception as e:
                    render_api_error(e)
    elif view == "network":
        try:
            with st.spinner("Loading network…"):
                graphs = [_subgraph(jid, depth, int(max_nodes)) for jid in ids]
        except Exception as e:
            render_api_error(e)
            return
        data = merge_networks(graphs, ids)
        if not data["nodes"]:
            st.warning("None of these cases is in the citation graph.", icon=":material/search_off:")
            return
        if data["truncated"]:
            st.info("Some neighbourhoods were cut to the closest cases — raise Max nodes or lower the depth to see more.",
                    icon=":material/filter_alt:")
        if len(ids) > 1:
            st.caption(f"Laid out around {labels[ids[0]]}; the other chosen cases are filled in black.")
            _note_unlinked(data, ids, labels)
        render_network(data, height=600)
    else:
        if len(ids) != 2:
            st.caption("Pick exactly two cases to find the precedent path between them.")
            return
        render_path((ids[0], labels[ids[0]]), (ids[1], labels[ids[1]]))


def render_answer_graph(data: Dict[str, Any], key: str) -> None:
    """The panel for an answer's cited judgments, folded away until wanted."""
    cases = answer_cases(data)
    if not cases:
        return
    with st.expander(f"Graph the cited cases ({len(cases)})", icon=":material/account_tree:"):
        render_case_graph_panel(cases, key)
