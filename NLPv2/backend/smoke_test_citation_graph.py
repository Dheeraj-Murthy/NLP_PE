"""
Smoke Test for Citation Graph & NetworkX Integration
"""

import sys
import os
import networkx as nx

# Add current directory and ingestion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ingestion"))

from citation_extractor import CitationExtractor
from citation_graph import CitationGraphManager


def test_citation_extractor():
    print("--- Testing CitationExtractor ---")
    extractor = CitationExtractor()
    sample_text = (
        "In the landmark decision of Vijay Kumar v. State of Karnataka (1987), "
        "the court established key principles. This was later overruled in AIR 1992 SC 588."
    )
    extracted = extractor.extract_citations(sample_text)
    print(f"Extracted {len(extracted)} citations from sample text:")
    for item in extracted:
        print(f"  - Cited Text: {item['cited_text']} | Relationship: {item['relationship_type']}")

    assert len(extracted) >= 1, "Should extract at least 1 citation"
    print("✓ CitationExtractor test passed!")


def test_citation_graph_manager():
    print("\n--- Testing CitationGraphManager (Synthetic Graph) ---")
    gm = CitationGraphManager()

    # Manually populate graph nodes & edges to test in-memory algorithms
    gm.graph.add_node(1, label="Kesavananda Bharati v. State of Kerala", court="Supreme Court", date="1973-04-24")
    gm.graph.add_node(2, label="Maneka Gandhi v. Union of India", court="Supreme Court", date="1978-01-25")
    gm.graph.add_node(3, label="Minerva Mills v. Union of India", court="Supreme Court", date="1980-07-31")

    gm.graph.add_edge(2, 1, relationship="followed", cited_text="Kesavananda Bharati case")
    gm.graph.add_edge(3, 1, relationship="followed", cited_text="AIR 1973 SC 1461")
    gm.graph.add_edge(3, 2, relationship="referred", cited_text="Maneka Gandhi judgment")
    gm._is_loaded = True

    # Test Subgraph
    subgraph = gm.get_subgraph(1, depth=2)
    print(f"Subgraph for Case #1: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    assert len(subgraph["nodes"]) == 3, "Subgraph should contain 3 connected nodes"

    # Test Landmark Cases (PageRank)
    landmarks = gm.get_landmark_cases(limit=3)
    print(f"Landmark cases (top authority):")
    for lm in landmarks:
        print(f"  - [{lm['judgment_id']}] {lm['label']} (PageRank: {lm['pagerank_score']}, In-Degree: {lm['in_degree']})")
    assert landmarks[0]["judgment_id"] == 1, "Case #1 should have highest PageRank/in-degree"

    # Test Shortest Path
    path = gm.get_shortest_path(3, 1)
    print(f"Shortest path from Case #3 -> Case #1: {path}")
    assert path == [3, 1], "Path should be direct edge [3, 1]"

    print("✓ CitationGraphManager synthetic test passed!")


if __name__ == "__main__":
    test_citation_extractor()
    test_citation_graph_manager()
    print("\n✓ ALL SMOKE TESTS PASSED SUCCESSFULLY!")
