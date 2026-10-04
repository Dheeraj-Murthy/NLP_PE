-- Citation graph schema. Additive and idempotent: safe to run on a database
-- that already has data, and safe to run more than once.
-- Applied by init_db.sh on fresh setups and by build_citation_edges.py.

CREATE TABLE IF NOT EXISTS citation_edges (
    edge_id SERIAL PRIMARY KEY,
    source_judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    target_judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
    cited_text TEXT NOT NULL,
    relationship_type VARCHAR(50) DEFAULT 'cited',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_citation_edges_source ON citation_edges(source_judgment_id);
CREATE INDEX IF NOT EXISTS idx_citation_edges_target ON citation_edges(target_judgment_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_citation_edges_unique
    ON citation_edges(source_judgment_id, COALESCE(target_judgment_id, -1), cited_text);

-- How each edge's target was found: reporter | exact_name | fuzzy | ambiguous | unresolved.
-- NULL on edges built before the resolver existed.
ALTER TABLE citation_edges ADD COLUMN IF NOT EXISTS resolution_method VARCHAR(20);
ALTER TABLE citation_edges ADD COLUMN IF NOT EXISTS resolution_score REAL;

-- Every normalised name and reporter citation a judgment can be found by.
-- kind = 'case_name' ("state of bombay v f n balsara") or
--        'reporter'  ("air|1953|sc|75", "scr|1955|1|777", "scr|1955||777").
-- One alias may map to several judgments; that is how ambiguity is detected.
CREATE TABLE IF NOT EXISTS judgment_aliases (
    alias_norm TEXT NOT NULL,
    judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
    kind VARCHAR(20) NOT NULL,
    PRIMARY KEY (alias_norm, judgment_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_judgment_aliases_judgment ON judgment_aliases(judgment_id);

-- Precomputed by compute_graph_stats.py so the API never runs PageRank while serving.
CREATE TABLE IF NOT EXISTS judgment_graph_stats (
    judgment_id INTEGER PRIMARY KEY REFERENCES judgments(id) ON DELETE CASCADE,
    in_degree INTEGER NOT NULL DEFAULT 0,
    out_degree INTEGER NOT NULL DEFAULT 0,
    pagerank DOUBLE PRECISION NOT NULL DEFAULT 0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_judgment_graph_stats_pagerank
    ON judgment_graph_stats(pagerank DESC);
