import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RAGResponse:
    answer: str
    sources: List[str]
    citations: List[str]
    confidence_score: float
    is_answer_found: bool
    retrieval_count: int
    answer_length: int

class PostProcessor:

    NO_ANSWER_PHRASES = [
            "Not found in the provided cases",
            "Not found in provided cases", 
            "Answer not found",
            "Information not found",
            "Cannot be determined from the provided context",
            "Cannot be determined",
            "Not specified in the provided context"
            ]

    def __init__(self, strip_echo: bool = True):
        self.strip_echo = strip_echo

    def process_response(
            self, 
            raw_response: str, 
            retrieved_chunks: List[Dict[str, Any]],
            original_query: str
            ) -> RAGResponse:
        cleaned_answer = self._clean_response(raw_response)

        is_answer_found = self._check_answer_exists(cleaned_answer)
        confidence_score = self._calculate_confidence(cleaned_answer, retrieved_chunks)

        citations = self._extract_citations(cleaned_answer, retrieved_chunks)
        sources = self._get_unique_sources(retrieved_chunks)

        return RAGResponse(
                answer=cleaned_answer,
                sources=sources,
                citations=citations,
                confidence_score=confidence_score,
                is_answer_found=is_answer_found,
                retrieval_count=len(retrieved_chunks),
                answer_length=len(cleaned_answer)
                )

    def _clean_response(self, raw_response: str) -> str:
        if not raw_response:
            return ""

        response = raw_response.strip()

        if self.strip_echo:
            lines = response.split('\n')
            answer_lines = []
            skip_until_answer = True

            for line in lines:
                line = line.strip()

                if skip_until_answer:
                    if line.lower().startswith('answer:'):
                        skip_until_answer = False
                        answer_part = line[7:].strip()
                        if answer_part:
                            answer_lines.append(answer_part)
                    continue

                if line.lower().startswith('context:') or line.lower().startswith('question:'):
                    continue

                answer_lines.append(line)

            response = '\n'.join(answer_lines)

        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)

        return response.strip()

    def _check_answer_exists(self, answer: str) -> bool:
        return True
        if not answer:
            return False

        answer_lower = answer.lower()

        for no_answer_phrase in self.NO_ANSWER_PHRASES:
            if no_answer_phrase.lower() in answer_lower:
                return False

        if len(answer.split()) < 5:
            return False

        return True

    def _calculate_confidence(
            self, 
            answer: str, 
            retrieved_chunks: List[Dict[str, Any]]
            ) -> float:
        if not answer or not retrieved_chunks:
            return 0.0

        is_answer_found = self._check_answer_exists(answer)
        if not is_answer_found:
            return 0.1

        avg_similarity = sum(chunk['similarity'] for chunk in retrieved_chunks) / len(retrieved_chunks)

        answer_words = set(answer.lower().split())
        context_words = set()
        for chunk in retrieved_chunks[:5]:
            context_words.update(chunk['text'].lower().split())

        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0

        confidence = (avg_similarity * 0.6) + (overlap * 0.4)
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    def _extract_citations(
            self, 
            answer: str, 
            retrieved_chunks: List[Dict[str, Any]]
            ) -> List[str]:
        citations = []

        bracket_pattern = r'\[(\d+)\]'
        bracket_matches = re.findall(bracket_pattern, answer)
        for match in bracket_matches:
            idx = int(match) - 1
            if 0 <= idx < len(retrieved_chunks):
                chunk = retrieved_chunks[idx]
                citation = f"{chunk['case']} ({chunk['court']}, {chunk['year']}, {chunk['para']})"
                if citation not in citations:
                    citations.append(citation)

        para_pattern = r'¶(\d+)'
        para_matches = re.findall(para_pattern, answer)
        for match in para_matches:
            for chunk in retrieved_chunks:
                if chunk['para'] == f"¶{match}":
                    citation = f"{chunk['case']} ({chunk['court']}, {chunk['year']}, {chunk['para']})"
                    if citation not in citations:
                        citations.append(citation)
                    break

        return citations

    def _get_unique_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        sources = []
        seen_cases = set()

        for chunk in retrieved_chunks:
            case_key = f"{chunk['case']} ({chunk['year']})"
            if case_key not in seen_cases:
                source = f"{chunk['case']} ({chunk['court']}, {chunk['year']})"
                sources.append(source)
                seen_cases.add(case_key)

        return sources

    def format_response_with_citations(
            self, 
            response: RAGResponse, 
            include_metadata: bool = False
            ) -> str:
        if not response.is_answer_found:
            return response.answer

        formatted = f"Answer:\n{response.answer}"

        if response.citations:
            formatted += "\n\nSources:"
            for citation in response.citations:
                formatted += f"\n- {citation}"

        if include_metadata:
            formatted += f"\n\n---\nConfidence: {response.confidence_score:.2f}"
            formatted += f"\nSources used: {len(response.sources)}"
            formatted += f"\nChunks retrieved: {response.retrieval_count}"

        return formatted

    def get_quality_metrics(self, response: RAGResponse) -> Dict[str, Any]:
        return {
                "answer_length": response.answer_length,
                "confidence_score": response.confidence_score,
                "has_citations": len(response.citations) > 0,
                "citation_count": len(response.citations),
                "source_count": len(response.sources),
                "retrieval_usage": len(response.citations) / response.retrieval_count if response.retrieval_count > 0 else 0,
                "is_comprehensive": response.answer_length > 100,
                "is_precise": 50 < response.answer_length < 500
                }

if __name__ == "__main__":
    processor = PostProcessor()

    sample_chunks = [
            {
                "text": "The Supreme Court held that educational institutions must follow due process when implementing fee structures.",
                "case": "ABC University v. State",
                "court": "Supreme Court of India",
                "year": 2019,
                "para": "¶23",
                "similarity": 0.89
                },
            {
                "text": "The Court emphasized that any increase in fees must be reasonable and proportionate to services provided.",
                "case": "ABC University v. State", 
                "court": "Supreme Court of India",
                "year": 2019,
                "para": "¶24",
                "similarity": 0.85
                }
            ]

    sample_response = "Answer:\nAccording to [1], the Supreme Court held that educational institutions must follow due process when implementing fee structures. The Court emphasized that any increase in fees must be reasonable and proportionate to services provided [2]."

    processed = processor.process_response(sample_response, sample_chunks, "educational fees")

    print("Processed Response:")
    print(processed.answer)

    print(f"\nAnswer Found: {processed.is_answer_found}")
    print(f"Confidence: {processed.confidence_score:.2f}")
    print(f"Citations: {processed.citations}")
    print(f"Sources: {processed.sources}")

    print("\n" + "=" * 50)
    print("Formatted Response:")
    print(processor.format_response_with_citations(processed))

    print("\n" + "=" * 50)
    print("Quality Metrics:")
    metrics = processor.get_quality_metrics(processed)
    for key, value in metrics.items():
        print(f"{key}: {value}")
