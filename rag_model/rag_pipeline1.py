from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

from retriever import LegalRetriever
from prompt_builder import PromptBuilder  
from llm_inference import QwenInference
from post_processor import PostProcessor, RAGResponse
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
        self.stage1_k = 30      # candidate pool
        self.stage2_k = top_k   # final chunks after rerank

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
    
    def query(
        self, 
        user_query: str,
        include_debug_info: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            retrieval_start = time.time()
            candidate_chunks = self.retriever.retrieve_candidate_chunks(
                query=user_query,
                candidate_k=self.stage1_k,
                similarity_threshold=0.2
            )

            retrieved_chunks = self.reranker.rerank(
                user_query,
                candidate_chunks,
                top_n=self.stage2_k
            )       

            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_chunks:
                return self._create_no_results_response(user_query, include_debug_info)
            
            prompt = self.prompt_builder.build_rag_prompt(retrieved_chunks, user_query)
            prompt_tokens = len(self.llm.tokenizer.encode(prompt)) if self.llm.tokenizer else 0
            
            generation_start = time.time()
            raw_response = self.llm.generate_response(prompt)
            generation_time = time.time() - generation_start
            
            processed_response = self.post_processor.process_response(
                raw_response, retrieved_chunks, user_query
            )
            
            total_time = time.time() - start_time
            
            metrics = RAGMetrics(
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                chunks_retrieved=len(retrieved_chunks),
                prompt_tokens=prompt_tokens,
                confidence_score=processed_response.confidence_score,
                answer_found=processed_response.is_answer_found
            )
            
            formatted_response = self.post_processor.format_response_with_citations(
                processed_response
            )
            
            result = {
                "answer": formatted_response,
                "answer_found": processed_response.is_answer_found,
                "confidence": processed_response.confidence_score,
                "citations": processed_response.citations,
                "sources": processed_response.sources,
                "metrics": {
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "chunks_retrieved": len(retrieved_chunks),
                    "prompt_tokens": prompt_tokens
                }
            }
            
            if include_debug_info:
                result.update({
                    "debug": {
                        "retrieved_chunks": retrieved_chunks[:3],
                        "raw_response": raw_response,
                        "prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                        "quality_metrics": self.post_processor.get_quality_metrics(processed_response)
                    }
                })
            
            return result
            
        except Exception as e:
            return {
                "error": f"Pipeline failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0
            }
    
    def _create_no_results_response(self, user_query: str, include_debug_info: bool) -> Dict[str, Any]:
        no_answer_response = "No relevant legal cases found for your query. Please try rephrasing your question or using different legal terms."
        
        result = {
            "answer": no_answer_response,
            "answer_found": False,
            "confidence": 0.0,
            "citations": [],
            "sources": [],
            "metrics": {
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0,
                "chunks_retrieved": 0,
                "prompt_tokens": 0
            }
        }
        
        if include_debug_info:
            result["debug"] = {"retrieved_chunks": []}
        
        return result
    
    def batch_query(
        self, 
        queries: List[str],
        include_debug_info: bool = False
    ) -> List[Dict[str, Any]]:
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query}")
            result = self.query(query, include_debug_info)
            results.append(result)
            
            if i < len(queries) - 1:
                time.sleep(0.5)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        memory_info = self.llm.get_memory_info()
        
        return {
            "model_loaded": self.llm.is_model_loaded(),
            "model_name": self.llm.model_name,
            "retriever_config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            },
            "prompt_builder_config": {
                "max_context_length": self.prompt_builder.max_context_length
            },
            "llm_config": {
                "max_new_tokens": self.llm.max_new_tokens,
                "temperature": self.llm.temperature,
                "do_sample": self.llm.do_sample
            },
            "memory_info": memory_info
        }
    
    def test_retrieval_only(self, query: str) -> Dict[str, Any]:
        try:
            start_time = time.time()
            retrieved_chunks = self.retriever.retrieve_relevant_chunks(
                query=query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold
            )
            retrieval_time = time.time() - start_time
            
            stats = self.retriever.get_retrieval_stats(query)
            
            return {
                "query": query,
                "chunks_found": len(retrieved_chunks),
                "retrieval_time": round(retrieval_time, 3),
                "retrieved_chunks": retrieved_chunks[:5],
                "retrieval_stats": stats
            }
            
        except Exception as e:
            return {
                "error": f"Retrieval test failed: {str(e)}",
                "chunks_found": 0
            }

if __name__ == "__main__":
    print("Initializing Legal RAG Pipeline...")
    pipeline = LegalRAGPipeline()
    
    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    status = pipeline.get_system_status()
    print(f"Model loaded: {status['model_loaded']}")
    print(f"Model: {status['model_name']}")
    print(f"Top-K retrieval: {status['retriever_config']['top_k']}")
    print(f"Similarity threshold: {status['retriever_config']['similarity_threshold']}")
    
    print("\n" + "=" * 60)
    print("TESTING SAMPLE QUERIES")
    print("=" * 60)
    
    test_queries = [
        "What are the regulations for educational institutions in Karnataka?",
        "How do courts interpret contractual disputes?",
        "What are the principles of natural justice in administrative law?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        result = pipeline.query(query, include_debug_info=True)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        
        print(f"Answer found: {result['answer_found']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Chunks retrieved: {result['metrics']['chunks_retrieved']}")
        print(f"Total time: {result['metrics']['total_time']}s")
        
        if result['answer_found']:
            print(f"\nAnswer preview: {result['answer'][:200]}...")
            if result['citations']:
                print(f"Citations: {result['citations'][0]}")
        else:
            print(f"Response: {result['answer']}")
    
    print("\n" + "=" * 60)
    print("RETRIEVAL-ONLY TEST")
    print("=" * 60)
    
    retrieval_test = pipeline.test_retrieval_only("Supreme Court education fees")
    print(f"Chunks found: {retrieval_test['chunks_found']}")
    print(f"Retrieval time: {retrieval_test['retrieval_time']}s")
    
    if retrieval_test['retrieved_chunks']:
        print("\nTop chunk preview:")
        chunk = retrieval_test['retrieved_chunks'][0]
        print(f"Case: {chunk['case']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"Content: {chunk['text'][:150]}...")
