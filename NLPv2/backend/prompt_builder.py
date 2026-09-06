from typing import List, Dict, Any

class PromptBuilder:
    
    SYSTEM_PROMPT = (
        "You are a legal assistant. Answer ONLY using the provided context. "
        "Each context item is numbered like [1], [2], etc. — cite the item number "
        "in brackets, e.g. [1], immediately after any claim you draw from it. "
        "If the answer is not present, say: \"Not found in the provided cases.\""
    )
    
    NO_ANSWER_RESPONSE = "Not found in the provided cases."
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def build_rag_prompt(
        self, 
        retrieved_chunks: List[Dict[str, Any]], 
        user_query: str
    ) -> str:
        context_parts = []
        current_length = 0
        min_useful_budget = 200

        for i, chunk in enumerate(retrieved_chunks, 1):
            remaining = self.max_context_length - current_length
            if remaining <= min_useful_budget:
                break

            chunk_text = self._format_chunk_with_metadata(chunk, i)

            if len(chunk_text) > remaining:
                # Truncate the chunk's own text (not the citation/section
                # metadata) to fit what's left, rather than dropping a
                # highly-ranked chunk in favor of a lower-ranked one that
                # merely happens to be shorter.
                overhead = len(chunk_text) - len(chunk["text"])
                truncated = dict(chunk)
                truncated["text"] = chunk["text"][: max(remaining - overhead, 0)] + "..."
                chunk_text = self._format_chunk_with_metadata(truncated, i)

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n\n".join(context_parts)
        
        prompt = f"""<|im_start|>system
{self.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{user_query}<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def _format_chunk_with_metadata(self, chunk: Dict[str, Any], chunk_number: int) -> str:
        citation = f"{chunk['case']} ({chunk['court']}, {chunk['year']}, {chunk['para']})"
        
        return f"""[{chunk_number}] {chunk['text']}

Source: {citation}
Section: {chunk['section']}"""
    
    def build_simple_prompt(self, retrieved_chunks: List[Dict[str, Any]], user_query: str) -> str:
        context_texts = []
        current_length = 0
        min_useful_budget = 200

        for chunk in retrieved_chunks:
            remaining = self.max_context_length - current_length
            if remaining <= min_useful_budget:
                break

            text = chunk['text']
            if len(text) > remaining:
                text = text[:remaining] + "..."

            context_texts.append(text)
            current_length += len(text)

        context = "\n\n".join(context_texts)
        
        prompt = f"""Answer the following legal question using ONLY the provided context. If the answer is not found in the context, say "{self.NO_ANSWER_RESPONSE}".

Context:
{context}

Question: {user_query}

Answer:"""
        
        return prompt
    
    def get_citation_list(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        citations = []
        seen_cases = set()
        
        for chunk in retrieved_chunks:
            case_key = f"{chunk['case']} ({chunk['year']})"
            if case_key not in seen_cases:
                citation = f"{chunk['case']} ({chunk['court']}, {chunk['year']}, {chunk['para']})"
                citations.append(citation)
                seen_cases.add(case_key)
        
        return citations
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
    
    def is_context_too_long(self, prompt: str, max_tokens: int = 3000) -> bool:
        estimated_tokens = self.estimate_tokens(prompt)
        return estimated_tokens > max_tokens

if __name__ == "__main__":
    builder = PromptBuilder()
    
    sample_chunks = [
        {
            "text": "The Supreme Court held that educational institutions must follow due process when implementing fee structures.",
            "case": "ABC University v. State",
            "court": "Supreme Court of India", 
            "year": 2019,
            "para": "¶23",
            "section": "judgment",
            "similarity": 0.89
        },
        {
            "text": "The Karnataka Education Bill was challenged on constitutional grounds for violating fundamental rights.",
            "case": "Karnataka Students Association v. State",
            "court": "Karnataka High Court",
            "year": 2021, 
            "para": "¶11",
            "section": "facts",
            "similarity": 0.85
        }
    ]
    
    query = "What did the Supreme Court say about educational fees in Karnataka?"
    
    prompt = builder.build_rag_prompt(sample_chunks, query)
    print("Generated RAG Prompt:")
    print("=" * 50)
    print(prompt)
    
    print("\n" + "=" * 50)
    print("Citations:")
    for citation in builder.get_citation_list(sample_chunks):
        print(f"- {citation}")
    
    print(f"\nEstimated tokens: {builder.estimate_tokens(prompt):.0f}")
    print(f"Context too long: {builder.is_context_too_long(prompt)}")