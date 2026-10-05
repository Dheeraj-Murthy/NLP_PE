"""
Citation Extraction Module for Legal RAG

Parses legal citations and precedent mentions from Indian judgment texts,
matching citations against existing DB judgments to construct directed graph edges.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# backend/ is a sibling directory holding the shared resolver (same pattern as ingest.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from graph.citation_resolver import REPORTER_PATTERNS, CitationResolver


class CitationExtractor:
    """Extracts precedent citations and case relationships from legal judgment text."""

    # Case title citations: Petitioner v[s]. Respondent (Year)
    CASE_TITLE_PATTERN = (
        r"([A-Z][A-Za-z0-9\.\s\&]+\s+(?:v\.|vs\.|Versus)\s+[A-Z][A-Za-z0-9\.\s\&]+(?:\s*\(\d{4}\))?)"
    )
    # Reporter citations (AIR 1987 SC 1086, [1950] S.C.R. 940, (1992) 1 SCC 588,
    # A.I.R. 1953 S.C. 75, ...) come from the shared resolver, so extraction and
    # resolution agree on every format; then case titles.
    CITATION_PATTERNS = [p.pattern for p in REPORTER_PATTERNS] + [CASE_TITLE_PATTERN]

    # Keyword patterns for determining relationship type
    RELATIONSHIP_PATTERNS = {
        "overruled": [r"\boverruled\b", r"\boverruling\b", r"\bset aside\b"],
        "followed": [r"\bfollowed\b", r"\bfollowing\b", r"\brelied upon\b", r"\brelied on\b"],
        "distinguished": [r"\bdistinguished\b", r"\bdistinguishing\b"],
        "referred": [r"\breferred to\b", r"\bcited\b", r"\bobserved in\b"],
    }

    def __init__(self):
        # Reporter patterns are case-sensitive ("AIR", not "air"); case titles aren't.
        self.compiled_reporter_res = list(REPORTER_PATTERNS)
        self.compiled_citation_res = self.compiled_reporter_res + [
            re.compile(self.CASE_TITLE_PATTERN, re.IGNORECASE)
        ]
        self.resolver: Optional[CitationResolver] = None

    def build_lookup_index(self, judgments_metadata: List[Dict[str, Any]]) -> None:
        """Build the citation resolver over these judgments (dicts with id,
        petitioner, respondent, date_of_judgment, and optionally header_text
        / citations for their own reporter citations)."""
        self.resolver = CitationResolver(judgments_metadata)

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citation strings and their contextual relationship from text.

        Returns a list of dicts: [{'cited_text': str, 'relationship_type': str, 'context': str}]
        """
        extracted = []
        seen_texts = set()
        reporter_spans: List[Tuple[int, int]] = []

        for pattern_re in self.compiled_citation_res:
            is_reporter = pattern_re in self.compiled_reporter_res
            for match in pattern_re.finditer(text):
                cited_str = match.group(0).strip()
                # Skip short/junk matches
                if len(cited_str) < 5 or cited_str.lower() in seen_texts:
                    continue
                if is_reporter:
                    # Reporter patterns overlap ("1955 1 S.C.R. 777" inside
                    # "[1955] 1 S.C.R. 777"); the first to claim the text wins.
                    start, end = match.span()
                    if any(start < e and s < end for s, e in reporter_spans):
                        continue
                    reporter_spans.append((start, end))

                seen_texts.add(cited_str.lower())

                # Get surrounding context (100 chars before/after)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context_snippet = text[start:end]

                relationship = self._determine_relationship(context_snippet)

                extracted.append(
                    {
                        "cited_text": cited_str,
                        "relationship_type": relationship,
                        "context": context_snippet,
                    }
                )

        return extracted

    def _determine_relationship(self, context: str) -> str:
        """Classify relationship type based on surrounding text context."""
        context_lower = context.lower()

        for rel_type, keywords in self.RELATIONSHIP_PATTERNS.items():
            for kw in keywords:
                if re.search(kw, context_lower):
                    return rel_type

        return "cited"

    def match_target_judgment(
        self, cited_text: str, judgments_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[int]:
        """
        Match a cited text string to a target judgment ID, or None when there
        is no confident match. Uses the index from build_lookup_index(), or
        builds a one-off one from judgments_metadata.
        """
        resolver = self.resolver
        if resolver is None:
            if not judgments_metadata:
                return None
            resolver = CitationResolver(judgments_metadata)
        return resolver.resolve(cited_text).judgment_id
