"""
Citation Extraction Module for Legal RAG

Parses legal citations and precedent mentions from Indian judgment texts,
matching citations against existing DB judgments to construct directed graph edges.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class CitationExtractor:
    """Extracts precedent citations and case relationships from legal judgment text."""

    # Common Indian legal citation formats
    CITATION_PATTERNS = [
        # Reporter citations: AIR 1987 SC 1086, (1992) 1 SCC 588, [1995] 3 SCR 12
        r"(?:AIR\s+\d{4}\s+SC\s+\d+)",
        r"(?:\(\d{4}\)\s+\d+\s+SCC\s+\d+)",
        r"(?:\[\d{4}\]\s+\d+\s+SCR\s+\d+)",
        # Case title citations: Petitioner v[s]. Respondent (Year)
        r"([A-Z][A-Za-z0-9\.\s\&]+\s+(?:v\.|vs\.|Versus)\s+[A-Z][A-Za-z0-9\.\s\&]+(?:\s*\(\d{4}\))?)",
    ]

    # Keyword patterns for determining relationship type
    RELATIONSHIP_PATTERNS = {
        "overruled": [r"\boverruled\b", r"\boverruling\b", r"\bset aside\b"],
        "followed": [r"\bfollowed\b", r"\bfollowing\b", r"\brelied upon\b", r"\brelied on\b"],
        "distinguished": [r"\bdistinguished\b", r"\bdistinguishing\b"],
        "referred": [r"\breferred to\b", r"\bcited\b", r"\bobserved in\b"],
    }

    def __init__(self):
        self.compiled_citation_res = [
            re.compile(p, re.IGNORECASE) for p in self.CITATION_PATTERNS
        ]
        self.lookup_index: Dict[str, int] = {}

    def build_lookup_index(self, judgments_metadata: List[Dict[str, Any]]) -> None:
        """Build O(1) hash table for fast case name and citation matching."""
        self.lookup_index.clear()
        for j in judgments_metadata:
            j_id = j["id"]
            petitioner = (j.get("petitioner") or "").lower().strip()
            respondent = (j.get("respondent") or "").lower().strip()

            if petitioner and len(petitioner) > 3:
                self.lookup_index[petitioner] = j_id
                if respondent and len(respondent) > 3:
                    full_name = f"{petitioner} v. {respondent}"
                    self.lookup_index[full_name] = j_id
                    full_name_vs = f"{petitioner} vs {respondent}"
                    self.lookup_index[full_name_vs] = j_id

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citation strings and their contextual relationship from text.

        Returns a list of dicts: [{'cited_text': str, 'relationship_type': str, 'context': str}]
        """
        extracted = []
        seen_texts = set()

        for pattern_re in self.compiled_citation_res:
            for match in pattern_re.finditer(text):
                cited_str = match.group(0).strip()
                # Skip short/junk matches
                if len(cited_str) < 5 or cited_str.lower() in seen_texts:
                    continue

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
        Try to match a cited text string to a target judgment ID. Uses O(1) index if built.
        """
        cited_clean = re.sub(r"\s+", " ", cited_text).lower().strip()

        # O(1) lookup check
        if cited_clean in self.lookup_index:
            return self.lookup_index[cited_clean]

        # Partial lookup check on indexed keys
        for key, j_id in self.lookup_index.items():
            if len(key) > 5 and key in cited_clean:
                return j_id

        if judgments_metadata:
            for j in judgments_metadata:
                j_id = j["id"]
                petitioner = (j.get("petitioner") or "").lower()
                respondent = (j.get("respondent") or "").lower()

                if petitioner and respondent and len(petitioner) > 2 and len(respondent) > 2:
                    if petitioner in cited_clean and respondent in cited_clean:
                        return j_id

                if petitioner and len(petitioner) > 3 and petitioner in cited_clean:
                    return j_id

        return None
