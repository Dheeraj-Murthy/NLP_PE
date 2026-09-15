"""
Citation Graph Manager for Legal RAG

Uses NetworkX and PostgreSQL to manage directed precedent graphs,
calculate PageRank / centrality scores, and extract subgraphs for visual APIs.
"""

import os
from typing import Dict, List, Any, Optional
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class CitationGraphManager:
    """Manages the in-memory NetworkX citation graph synchronized with PostgreSQL."""

    def __init__(
        self,
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_name: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
    ):
        self.db_host = db_host or os.getenv("DB_HOST", "localhost")
        self.db_port = db_port or os.getenv("DB_PORT", "5432")
        self.db_name = db_name or os.getenv("DB_NAME", "legal_rag")
        self.db_user = db_user or os.getenv("DB_USER", "postgres")
        self.db_password = db_password or os.getenv("DB_PASSWORD", "postgres")

        self.graph = nx.DiGraph()
        self._is_loaded = False

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def load_graph_from_db(self, force_reload: bool = False) -> int:
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

    def get_subgraph(self, judgment_id: int, depth: int = 2) -> Dict[str, Any]:
        """
        Extract an N-hop ego graph around a specific judgment ID.
        Returns JSON-serializable dict of nodes and edges.
        """
        self.load_graph_from_db()

        if judgment_id not in self.graph:
            return {"nodes": [], "edges": [], "center_id": judgment_id}

        # Compute ego graph
        ego_g = nx.ego_graph(self.graph, judgment_id, radius=depth, undirected=True)

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

        return {"center_id": judgment_id, "nodes": nodes, "edges": edges}

    def get_landmark_cases(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify top authority cases based on PageRank and in-degree centrality.
        """
        self.load_graph_from_db()

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

    def get_shortest_path(self, source_id: int, target_id: int) -> Optional[List[int]]:
        """
        Find shortest citation chain between source and target judgments.
        """
        self.load_graph_from_db()

        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_centrality_scores(self) -> Dict[int, float]:
        """
        Return normalized PageRank scores for all judgments (used for retrieval boosting).
        """
        self.load_graph_from_db()
        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            return {}
        return nx.pagerank(self.graph)

    def get_precedent_summary(
        self, judgment_id: int, top_n: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Compact precedent-chain summary for a judgment: how many cases it cites,
        how many cases cite it, and the strongest precedents followed/applied
        later. Returns None if the judgment is not in the graph.
        """
        self.load_graph_from_db()
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
