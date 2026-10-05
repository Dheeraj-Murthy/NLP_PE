"""
Citation Graph Manager for Legal RAG

Serves the directed precedent graph: subgraphs for visual APIs, landmark
cases, precedent chains, PageRank / centrality scores, search and citation
resolution.

Two modes, same public methods and return shapes:
  - "postgres" (default): indexed queries against citation_edges and the
    precomputed judgment_graph_stats table (see graph/pg_store.py). Nothing
    is held in memory, so it scales with the corpus and needs no reload.
  - "memory": the original approach, the whole graph loaded into NetworkX.
    Used when GRAPH_BACKEND=memory, or when a caller fills `self.graph`
    itself and sets `_is_loaded` (as the smoke tests do).
"""

import os
from typing import Dict, List, Any, Optional
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor

from graph.pg_store import PostgresGraphStore

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CitationGraphManager:
    """Citation graph queries, backed by PostgreSQL (default) or an in-memory
    NetworkX copy of it."""

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.db_host = db_host or os.getenv("DB_HOST", "localhost")
        # 5433 matches retriever.py's default and the docker-compose host port
        # mapping — keep these in sync, they drifted apart once already.
        self.db_port = db_port or os.getenv("DB_PORT", "5433")
        self.db_name = db_name or os.getenv("DB_NAME", "legal_rag")
        self.db_user = db_user or os.getenv("DB_USER", "postgres")
        self.db_password = db_password or os.getenv("DB_PASSWORD", "postgres")

        self.graph = nx.DiGraph()
        self._is_loaded = False
        self._pagerank_cache: Optional[tuple] = None

        self.backend = (backend or os.getenv("GRAPH_BACKEND", "postgres")).lower()
        self._pg = PostgresGraphStore(
            {
                "host": self.db_host,
                "port": self.db_port,
                "dbname": self.db_name,
                "user": self.db_user,
                "password": self.db_password,
            }
        )

    def _use_postgres(self) -> bool:
        # A graph someone loaded or filled in memory always wins.
        return self.backend != "memory" and not self._is_loaded

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def _mem_load(self, force_reload: bool = False) -> int:
        """
        Load nodes and edges from PostgreSQL into NetworkX.
        Returns total number of edges loaded.
        """
        if self._is_loaded and not force_reload:
            return self.graph.number_of_edges()

        self.graph.clear()

        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Add judgment nodes with attributes
                cur.execute(
                    "SELECT id, petitioner, respondent, court, date_of_judgment FROM judgments;"
                )
                judgments = cur.fetchall()
                for j in judgments:
                    petitioner = j.get("petitioner") or "Unknown"
                    respondent = j.get("respondent") or "Unknown"
                    label = f"{petitioner} v. {respondent}"
                    self.graph.add_node(
                        j["id"],
                        label=label,
                        court=j.get("court"),
                        date=str(j.get("date_of_judgment")),
                    )

                # Add citation edges
                cur.execute(
                    "SELECT source_judgment_id, target_judgment_id, cited_text, relationship_type FROM citation_edges;"
                )
                edges = cur.fetchall()
                for e in edges:
                    src = e["source_judgment_id"]
                    tgt = e["target_judgment_id"]
                    if tgt is not None and self.graph.has_node(tgt):
                        self.graph.add_edge(
                            src,
                            tgt,
                            cited_text=e["cited_text"],
                            relationship=e["relationship_type"],
                        )

            conn.close()
            self._is_loaded = True
        except Exception as err:
            print(f"Warning: Failed to load citation graph from DB: {err}")

        return self.graph.number_of_edges()

    def _mem_subgraph(
        self, judgment_id: int, depth: int = 2, max_nodes: int = 100
    ) -> Dict[str, Any]:
        """
        Extract an N-hop ego graph around a specific judgment ID, capped at
        max_nodes. Returns JSON-serializable dict of nodes and edges.

        Around well-cited judgments a depth-3 neighbourhood can run into
        thousands of nodes, which the frontend cannot lay out. When the cap is
        hit, nodes closer to the center are kept first and, within a hop
        level, the most-connected ones win. Since every hop level is kept in
        full before the next one is touched, the result stays connected.
        """
        self._mem_load()

        if judgment_id not in self.graph:
            return {
                "nodes": [],
                "edges": [],
                "center_id": judgment_id,
                "total_nodes": 0,
                "truncated": False,
            }

        dist = nx.single_source_shortest_path_length(
            self.graph.to_undirected(as_view=True), judgment_id, cutoff=depth
        )
        total_nodes = len(dist)
        truncated = total_nodes > max_nodes
        if truncated:
            keep = sorted(dist, key=lambda n: (dist[n], -self.graph.degree(n), n))[:max_nodes]
        else:
            keep = list(dist)

        ego_g = self.graph.subgraph(keep)

        nodes = []
        for n, data in ego_g.nodes(data=True):
            nodes.append(
                {
                    "id": n,
                    "label": data.get("label", f"Case #{n}"),
                    "court": data.get("court"),
                    "date": data.get("date"),
                    "is_center": (n == judgment_id),
                }
            )

        edges = []
        for u, v, data in ego_g.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "cited_text": data.get("cited_text", ""),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        return {
            "center_id": judgment_id,
            "nodes": nodes,
            "edges": edges,
            "total_nodes": total_nodes,
            "truncated": truncated,
        }

    def _mem_landmark_cases(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify top authority cases based on PageRank and in-degree centrality.
        """
        self._mem_load()

        if self.graph.number_of_nodes() == 0:
            return []

        pagerank_scores = nx.pagerank(self.graph) if self.graph.number_of_edges() > 0 else {}

        landmarks = []
        for node in self.graph.nodes():
            score = pagerank_scores.get(node, 0.0)
            in_degree = self.graph.in_degree(node)
            data = self.graph.nodes[node]

            landmarks.append(
                {
                    "judgment_id": node,
                    "label": data.get("label", f"Case #{node}"),
                    "court": data.get("court"),
                    "date": data.get("date"),
                    "in_degree": in_degree,
                    "pagerank_score": round(score, 6),
                }
            )

        landmarks.sort(key=lambda x: (x["pagerank_score"], x["in_degree"]), reverse=True)
        return landmarks[:limit]

    def _mem_shortest_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """
        Find shortest citation chain between source and target judgments.
        """
        self._mem_load()

        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _mem_centrality_scores(self) -> Dict[int, float]:
        """
        Return normalized PageRank scores for all judgments (used for retrieval boosting).
        """
        self._mem_load()
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            return {}
        # Cached per graph size: retrieval asks for this on every query.
        shape = (self.graph.number_of_nodes(), self.graph.number_of_edges())
        if self._pagerank_cache is None or self._pagerank_cache[0] != shape:
            self._pagerank_cache = (shape, nx.pagerank(self.graph))
        return self._pagerank_cache[1]

    def _mem_precedent_summary(
        self, judgment_id: int, top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Compact precedent-chain summary for a judgment: how many cases it cites,
        how many cases cite it, and the strongest precedents followed/applied
        later. Returns None if the judgment is not in the graph.
        """
        self._mem_load()
        if judgment_id not in self.graph:
            return None

        out_citations = []
        in_citations = []

        for _src, tgt, data in self.graph.out_edges(judgment_id, data=True):
            out_citations.append(
                {
                    "judgment_id": tgt,
                    "label": self.graph.nodes[tgt].get("label", f"Case #{tgt}"),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        for src, _tgt, data in self.graph.in_edges(judgment_id, data=True):
            in_citations.append(
                {
                    "judgment_id": src,
                    "label": self.graph.nodes[src].get("label", f"Case #{src}"),
                    "relationship": data.get("relationship", "cited"),
                }
            )

        rel_rank = {
            "overruled": 0,
            "followed": 1,
            "distinguished": 2,
            "referred": 3,
            "cited": 4,
        }
        in_citations.sort(
            key=lambda c: (rel_rank.get(c["relationship"], 5), c["judgment_id"])
        )
        out_citations.sort(
            key=lambda c: (rel_rank.get(c["relationship"], 5), c["judgment_id"])
        )

        return {
            "judgment_id": judgment_id,
            "label": self.graph.nodes[judgment_id].get("label", f"Case #{judgment_id}"),
            "cites_count": len(out_citations),
            "cited_by_count": len(in_citations),
            "cites": out_citations[:top_n],
            "cited_by": in_citations[:top_n],
        }

    def _mem_neighbors(
        self, judgment_id: int, direction: str, relationship: Optional[str], limit: int, offset: int
    ) -> Optional[Dict[str, Any]]:
        self._mem_load()
        if judgment_id not in self.graph:
            return None
        edges = (
            self.graph.out_edges(judgment_id, data=True)
            if direction == "cites"
            else self.graph.in_edges(judgment_id, data=True)
        )
        scores = self._mem_centrality_scores()
        items = []
        for src, tgt, data in edges:
            other = tgt if direction == "cites" else src
            rel = data.get("relationship", "cited")
            if other == judgment_id or (relationship and rel != relationship):
                continue
            node = self.graph.nodes[other]
            items.append(
                {
                    "judgment_id": other,
                    "label": node.get("label", f"Case #{other}"),
                    "court": node.get("court"),
                    "date": node.get("date"),
                    "relationship": rel,
                    "cited_text": data.get("cited_text", ""),
                    "pagerank_score": round(scores.get(other, 0.0), 6),
                }
            )
        items.sort(key=lambda i: (-i["pagerank_score"], i["judgment_id"]))
        return {
            "judgment_id": judgment_id,
            "direction": direction,
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "items": items[offset : offset + limit],
        }

    # ------------------------------------------------------------------
    # Public API — same names, arguments and return shapes in both modes.
    # ------------------------------------------------------------------

    def load_graph_from_db(self, force_reload: bool = False) -> int:
        """
        Memory mode: load nodes and edges from PostgreSQL into NetworkX.
        Postgres mode: nothing to load; checks the connection.
        Returns the number of (distinct) citation edges.
        """
        if not self._use_postgres():
            return self._mem_load(force_reload)
        try:
            return self._pg.edge_count()
        except Exception as err:
            print(f"Warning: Failed to reach citation graph in DB: {err}")
            return 0

    def get_subgraph(
        self, judgment_id: int, depth: int = 2, max_nodes: int = 100
    ) -> Dict[str, Any]:
        """
        N-hop ego graph around a judgment, capped at max_nodes (closest
        nodes first, then the most important). Returns nodes, edges,
        total_nodes and truncated.
        """
        if self._use_postgres():
            return self._pg.subgraph(judgment_id, depth=depth, max_nodes=max_nodes)
        return self._mem_subgraph(judgment_id, depth=depth, max_nodes=max_nodes)

    def get_landmark_cases(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Top authority cases by PageRank, then in-degree."""
        if self._use_postgres():
            return self._pg.landmark_cases(limit=limit)
        return self._mem_landmark_cases(limit=limit)

    def get_shortest_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """Shortest citation chain from source to target, or None."""
        if self._use_postgres():
            return self._pg.shortest_path(source_id, target_id)
        return self._mem_shortest_path(source_id, target_id)

    def get_centrality_scores(self) -> Dict[int, float]:
        """PageRank for every judgment with a non-zero score (retrieval boosting)."""
        if self._use_postgres():
            try:
                return self._pg.all_scores()
            except Exception as err:
                print(f"Warning: centrality scores unavailable: {err}")
                return {}
        return self._mem_centrality_scores()

    def centrality_for(self, judgment_ids: List[int]) -> Dict[int, float]:
        """PageRank for just these judgments — what retrieval needs per query,
        without building the full map."""
        ids = [i for i in dict.fromkeys(judgment_ids) if i is not None]
        if self._use_postgres():
            try:
                return self._pg.scores_for(ids)
            except Exception as err:
                print(f"Warning: centrality scores unavailable: {err}")
                return {}
        scores = self._mem_centrality_scores()
        return {i: scores[i] for i in ids if i in scores}

    def get_precedent_summary(
        self, judgment_id: int, top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Compact precedent-chain summary for a judgment: how many cases it
        cites, how many cite it, and the strongest of each. None if the
        judgment is not in the graph.
        """
        if self._use_postgres():
            try:
                return self._pg.precedent_summary(judgment_id, top_n=top_n)
            except Exception as err:
                # Answering a query must not fail because the graph is unavailable.
                print(f"Warning: precedent summary unavailable: {err}")
                return None
        return self._mem_precedent_summary(judgment_id, top_n=top_n)

    def get_neighbors(
        self,
        judgment_id: int,
        direction: str = "cites",
        relationship: Optional[str] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """One page of the cases a judgment cites (direction="cites") or that
        cite it ("cited_by"), most important first. None if it doesn't exist."""
        if direction not in ("cites", "cited_by"):
            raise ValueError("direction must be 'cites' or 'cited_by'")
        if self._use_postgres():
            return self._pg.neighbors(judgment_id, direction, relationship, limit, offset)
        return self._mem_neighbors(judgment_id, direction, relationship, limit, offset)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Cases matching an ID, a reporter citation or a case name. Always
        queries PostgreSQL, in either mode."""
        return self._pg.search(query, limit=limit)

    def resolve_citations(self, citations: List[str]) -> List[Dict[str, Any]]:
        """Resolve citation strings to judgment IDs, exactly as edge building
        does. Always queries PostgreSQL, in either mode."""
        return self._pg.resolve(citations)
