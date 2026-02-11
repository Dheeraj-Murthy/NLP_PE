from typing import List, Dict, Any
import time
from dataclasses import dataclass

from retriever import LegalRetriever
from prompt_builder import PromptBuilder
from llm_inference import QwenInference
from post_processor import PostProcessor
from reranker import CrossEncoderReranker


@dataclass
class RAGMetrics:
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    prompt_tokens: int
    confidence_score: float
    answer_found: bool


class LegalRAGPipeline:

    def __init__(
        self,
        db_connection_string: str = "dbname=legal_rag",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        load_llm: bool = True,
        top_k: int = 8,
        similarity_threshold: float = 0.3,
        max_context_length: int = 4000,
        max_new_tokens: int = 512
    ):
        self.retriever = LegalRetriever(db_connection_string)
        self.reranker = CrossEncoderReranker()

        self.stage1_k = 30
        self.stage2_k = top_k
        self.stage1_threshold = 0.2

        self.prompt_builder = PromptBuilder(max_context_length)

        self.llm = None
        if load_llm:
            self.llm = QwenInference(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False
            )

        self.post_processor = PostProcessor()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def query(self, user_query: str, include_debug_info: bool = False) -> Dict[str, Any]:
        start_time = time.time()

        if not self.llm:
            return {
                "error": "LLM not loaded",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0
            }

        try:
            retrieval_start = time.time()

            # ✅ STAGE 1: high-recall retrieval
            candidates = self.retriever.retrieve_relevant_chunks(
                query=user_query,
                top_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold
            )

            # ✅ STAGE 2: reranking
            retrieved_chunks = self.reranker.rerank(
                user_query,
                candidates,
                top_n=self.stage2_k
            )

            retrieval_time = time.time() - retrieval_start

            if not retrieved_chunks:
                return self._create_no_results_response(
                    user_query, include_debug_info, retrieval_time
                )

            prompt = self.prompt_builder.build_rag_prompt(
                retrieved_chunks, user_query
            )

            prompt_tokens = 0
            if self.llm.tokenizer:
                prompt_tokens = self.llm.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.prompt_builder.max_context_length,
                    return_length=True
                )["length"][0]

            generation_start = time.time()
            raw_response = self.llm.generate_response(prompt)
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                raw_response, retrieved_chunks, user_query
            )

            total_time = time.time() - start_time

            result = {
                "answer": self.post_processor.format_response_with_citations(processed),
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "metrics": {
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "chunks_retrieved": len(retrieved_chunks),
                    "prompt_tokens": prompt_tokens
                }
            }

            if include_debug_info:
                result["debug"] = {
                    "retrieved_chunks": retrieved_chunks[:3],
                    "raw_response": raw_response
                }

            return result

        except Exception as e:
            return {
                "error": f"Pipeline failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0
            }

    def _create_no_results_response(
        self, user_query: str, include_debug_info: bool, retrieval_time: float
    ) -> Dict[str, Any]:
        result = {
            "answer": "No relevant legal cases found for your query.",
            "answer_found": False,
            "confidence": 0.0,
            "citations": [],
            "sources": [],
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": 0.0,
                "total_time": round(retrieval_time, 3),
                "chunks_retrieved": 0,
                "prompt_tokens": 0
            }
        }

        if include_debug_info:
            result["debug"] = {"retrieved_chunks": []}

        return result

    def test_retrieval_only(self, query: str) -> Dict[str, Any]:
        try:
            start = time.time()
            chunks = self.retriever.retrieve_relevant_chunks(
                query=query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold
            )
            return {
                "query": query,
                "chunks_found": len(chunks),
                "retrieval_time": round(time.time() - start, 3),
                "retrieved_chunks": chunks[:5]
            }
        except Exception as e:
            return {"error": str(e), "chunks_found": 0}


if __name__ == "__main__":
    print("Initializing Legal RAG Pipeline...")
    pipeline = LegalRAGPipeline()

    queries = [
        "What are the regulations for educational institutions in Karnataka?",
        "How do courts interpret contractual disputes?",
        "What are the principles of natural justice in administrative law?"
    ]

    for q in queries:
        print("\nQUERY:", q)
        out = pipeline.query(q, include_debug_info=True)
        print("FOUND:", out["answer_found"])
        print("CONF:", out["confidence"])
