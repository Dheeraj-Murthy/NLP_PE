from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, field
from datetime import datetime

from retriever import LegalRetriever
from prompt_builder import PromptBuilder
from llm_inference import QwenInference
from post_processor import PostProcessor, RAGResponse
from reranker import CrossEncoderReranker
from document_processor import DocumentProcessor


@dataclass
class RAGMetrics:
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
    prompt_tokens: int
    confidence_score: float
    answer_found: bool


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ChatSession:
    def __init__(self, max_history: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(
        self, content: str, citations: List[str], confidence: float
    ):
        self.messages.append(
            ChatMessage(
                role="assistant",
                content=content,
                citations=citations,
                confidence=confidence,
            )
        )

    def get_history(self, include_citations: bool = False) -> List[Dict[str, Any]]:
        history = []
        for msg in self.messages:
            item = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            if include_citations and msg.role == "assistant":
                item["citations"] = msg.citations
                item["confidence"] = msg.confidence
            history.append(item)
        return history[-self.max_history :]

    def get_conversation_context(self) -> str:
        context_parts = []
        for msg in self.messages[-self.max_history :]:
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            else:
                context_parts.append(f"Assistant: {msg.content[:200]}...")
        return "\n".join(context_parts)

    def clear(self):
        self.messages.clear()


class LegalRAGPipeline:
    def __init__(
        self,
        db_connection_string: str = "dbname=legal_rag",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        load_llm: bool = True,
        top_k: int = 8,
        similarity_threshold: float = 0.3,
        max_context_length: int = 4000,
        max_new_tokens: int = 512,
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
                do_sample=False,
            )

        self.post_processor = PostProcessor()
        self.document_processor = DocumentProcessor()
        self.chat_session = ChatSession(max_history=10)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def query(
        self, user_query: str, include_debug_info: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()

        if not self.llm:
            return {
                "error": "LLM not loaded",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

        try:
            retrieval_start = time.time()

            # ✅ STAGE 1: high-recall retrieval
            candidate_chunks = self.retriever.retrieve_candidate_chunks(
                query=user_query,
                candidate_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold,
            )

            # ✅ STAGE 2: reranking
            retrieved_chunks = self.reranker.rerank(
                user_query, candidate_chunks, top_n=self.stage2_k
            )

            retrieval_time = time.time() - retrieval_start

            if not retrieved_chunks:
                return self._create_no_results_response(
                    user_query, include_debug_info, retrieval_time
                )

            prompt = self.prompt_builder.build_rag_prompt(retrieved_chunks, user_query)

            prompt_tokens = 0
            if self.llm.tokenizer:
                prompt_tokens = self.llm.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.prompt_builder.max_context_length,
                    return_length=True,
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
                    "prompt_tokens": prompt_tokens,
                },
            }

            if include_debug_info:
                result["debug"] = {
                    "retrieved_chunks": retrieved_chunks[:3],
                    "raw_response": raw_response,
                    "prompt_preview": prompt[:500] + "..."
                    if len(prompt) > 500
                    else prompt,
                    "quality_metrics": self.post_processor.get_quality_metrics(
                        processed
                    ),
                }

            return result

        except Exception as e:
            return {
                "error": f"Pipeline failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

    def _create_no_results_response(
        self, user_query: str, include_debug_info: bool, retrieval_time: float
    ) -> Dict[str, Any]:
        result = {
            "answer": "No relevant legal cases found for your query. Please try rephrasing your question or using different legal terms.",
            "answer_found": False,
            "confidence": 0.0,
            "citations": [],
            "sources": [],
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": 0.0,
                "total_time": round(retrieval_time, 3),
                "chunks_retrieved": 0,
                "prompt_tokens": 0,
            },
        }

        if include_debug_info:
            result["debug"] = {"retrieved_chunks": []}

        return result

    def batch_query(
        self, queries: List[str], include_debug_info: bool = False
    ) -> List[Dict[str, Any]]:
        results = []
        for i, query in enumerate(queries):
            print(f"Processing query {i + 1}/{len(queries)}: {query}")
            result = self.query(query, include_debug_info)
            results.append(result)
            if i < len(queries) - 1:
                time.sleep(0.5)
        return results

    def get_system_status(self) -> Dict[str, Any]:
        memory_info = self.llm.get_memory_info() if self.llm else {}
        return {
            "model_loaded": self.llm.is_model_loaded() if self.llm else False,
            "model_name": self.llm.model_name if self.llm else None,
            "retriever_config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
            },
            "prompt_builder_config": {
                "max_context_length": self.prompt_builder.max_context_length
            },
            "llm_config": {
                "max_new_tokens": self.llm.max_new_tokens if self.llm else None,
                "temperature": self.llm.temperature if self.llm else None,
                "do_sample": self.llm.do_sample if self.llm else None,
            },
            "memory_info": memory_info,
        }

    def query_with_document(
        self,
        document_path: str,
        user_query: Optional[str] = None,
        include_retrieval: bool = True,
        include_debug_info: bool = False,
    ) -> Dict[str, Any]:
        """Query using an attached document (PDF/image) via OCR.

        Alternate path: extracts text from document, optionally combines with
        retrieved context, then generates answer.
        """
        start_time = time.time()

        if not self.llm:
            return {
                "error": "LLM not loaded",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

        try:
            doc_result = self.document_processor.process_document(document_path)

            if not doc_result["success"]:
                return {
                    "error": doc_result["error"],
                    "answer": None,
                    "answer_found": False,
                    "confidence": 0.0,
                }

            ocr_text = doc_result["text"]
            document_info = {
                "source": doc_result["source"],
                "pages": doc_result.get("pages", 1),
                "type": doc_result.get("document_type", "unknown"),
            }

            retrieval_time = 0.0
            retrieved_chunks = []
            if include_retrieval:
                retrieval_start = time.time()
                retrieved_chunks = self.retriever.retrieve_candidate_chunks(
                    query=user_query or ocr_text[:500],
                    candidate_k=self.stage1_k,
                    similarity_threshold=self.stage1_threshold,
                )
                retrieved_chunks = self.reranker.rerank(
                    user_query or ocr_text[:500], retrieved_chunks, top_n=self.stage2_k
                )
                retrieval_time = time.time() - retrieval_start

            prompt = self._build_document_prompt(ocr_text, retrieved_chunks, user_query)

            prompt_tokens = 0
            if self.llm.tokenizer:
                prompt_tokens = self.llm.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.prompt_builder.max_context_length,
                    return_length=True,
                )["length"][0]

            generation_start = time.time()
            raw_response = self.llm.generate_response(prompt)
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                raw_response, retrieved_chunks, user_query or ""
            )

            total_time = time.time() - start_time

            result = {
                "answer": self.post_processor.format_response_with_citations(processed),
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "document": document_info,
                "metrics": {
                    "ocr_time": 0.0,
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "chunks_retrieved": len(retrieved_chunks),
                    "prompt_tokens": prompt_tokens,
                    "document_pages": document_info["pages"],
                },
            }

            if include_debug_info:
                result["debug"] = {
                    "ocr_text_preview": ocr_text[:500] + "..."
                    if len(ocr_text) > 500
                    else ocr_text,
                    "raw_response": raw_response,
                    "retrieved_chunks": retrieved_chunks[:3]
                    if retrieved_chunks
                    else [],
                }

            return result

        except Exception as e:
            return {
                "error": f"Document query failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

    def _build_document_prompt(
        self,
        ocr_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_query: Optional[str],
    ) -> str:
        """Build prompt for document-based query."""
        max_doc_length = 3000
        if len(ocr_text) > max_doc_length:
            ocr_text = ocr_text[:max_doc_length] + "..."

        prompt_parts = []

        if user_query:
            prompt_parts.append(f"User Question: {user_query}\n")

        prompt_parts.append("Attached Document (OCR extracted text):\n")
        prompt_parts.append(ocr_text)

        if retrieved_chunks:
            prompt_parts.append("\n\nRelevant Legal Context from Database:")
            for i, chunk in enumerate(retrieved_chunks[:3], 1):
                prompt_parts.append(
                    f"\n[{i}] {chunk['case']} ({chunk['court']}, {chunk['year']})"
                )
                prompt_parts.append(chunk["text"][:500])

        prompt_parts.append("\n\nInstructions:")
        prompt_parts.append(
            "Based on the attached document and any relevant legal context, "
        )

        if user_query:
            prompt_parts.append(f"answer the user's question: {user_query}")
        else:
            prompt_parts.append(
                "provide a summary and analysis of the document, including any relevant legal principles, precedents, or findings."
            )

        prompt_parts.append(
            "\n\nIf the document contains court judgments or legal cases, identify:"
        )
        prompt_parts.append("- The parties involved")
        prompt_parts.append("- The key issues addressed")
        prompt_parts.append("- The court's reasoning and holding")
        prompt_parts.append("- Any cited precedents")

        prompt_parts.append("\n\nAnswer:")

        return "\n".join(prompt_parts)

    def chat(
        self, user_message: str, include_debug_info: bool = False
    ) -> Dict[str, Any]:
        """Multi-turn chat with conversation history."""
        if not self.llm:
            return {
                "error": "LLM not loaded",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

        self.chat_session.add_user_message(user_message)

        start_time = time.time()

        try:
            retrieval_start = time.time()
            candidate_chunks = self.retriever.retrieve_candidate_chunks(
                query=user_message,
                candidate_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold,
            )
            retrieved_chunks = self.reranker.rerank(
                user_message, candidate_chunks, top_n=self.stage2_k
            )
            retrieval_time = time.time() - retrieval_start

            if not retrieved_chunks:
                no_answer = "I couldn't find relevant legal cases for your query. Could you try rephrasing?"
                self.chat_session.add_assistant_message(no_answer, [], 0.0)
                return {
                    "answer": no_answer,
                    "answer_found": False,
                    "confidence": 0.0,
                    "citations": [],
                    "sources": [],
                    "conversation_history": self.chat_session.get_history(
                        include_citations=True
                    ),
                    "metrics": {
                        "retrieval_time": round(retrieval_time, 3),
                        "generation_time": 0.0,
                        "total_time": round(time.time() - start_time, 3),
                        "chunks_retrieved": 0,
                    },
                }

            conversation_context = self.chat_session.get_conversation_context()
            prompt = self._build_chat_prompt(
                retrieved_chunks, user_message, conversation_context
            )

            prompt_tokens = 0
            if self.llm.tokenizer:
                prompt_tokens = self.llm.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.prompt_builder.max_context_length,
                    return_length=True,
                )["length"][0]

            generation_start = time.time()
            raw_response = self.llm.generate_response(prompt)
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                raw_response, retrieved_chunks, user_message
            )

            total_time = time.time() - start_time

            formatted_answer = self.post_processor.format_response_with_citations(
                processed
            )
            self.chat_session.add_assistant_message(
                formatted_answer, processed.citations, processed.confidence_score
            )

            result = {
                "answer": formatted_answer,
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "conversation_history": self.chat_session.get_history(
                    include_citations=True
                ),
                "metrics": {
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "chunks_retrieved": len(retrieved_chunks),
                    "prompt_tokens": prompt_tokens,
                },
            }

            if include_debug_info:
                result["debug"] = {
                    "retrieved_chunks": retrieved_chunks[:3],
                    "raw_response": raw_response,
                    "conversation_context": conversation_context,
                }

            return result

        except Exception as e:
            return {
                "error": f"Chat failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
                "conversation_history": self.chat_session.get_history(
                    include_citations=True
                ),
            }

    def _build_chat_prompt(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        current_query: str,
        conversation_context: str,
    ) -> str:
        prompt_parts = []

        if conversation_context:
            prompt_parts.append("Previous Conversation:")
            prompt_parts.append(conversation_context)
            prompt_parts.append("\n---\n")

        prompt_parts.append("Relevant Legal Context:")
        for i, chunk in enumerate(retrieved_chunks, 1):
            prompt_parts.append(
                f"\n[{i}] {chunk['case']} ({chunk['court']}, {chunk['year']})"
            )
            prompt_parts.append(chunk["text"][:400])

        prompt_parts.append(f"\n\nCurrent Question: {current_query}")
        prompt_parts.append("\n\nInstructions:")
        prompt_parts.append(
            "Answer the current question based on the legal context above."
        )
        prompt_parts.append(
            "If the question references previous answers, use the conversation history."
        )
        prompt_parts.append("Cite sources using [n] notation.")

        prompt_parts.append("\n\nAnswer:")

        return "\n".join(prompt_parts)

    def clear_chat_history(self):
        self.chat_session.clear()

    def get_chat_history(self) -> List[Dict[str, Any]]:
        return self.chat_session.get_history(include_citations=True)

    def test_retrieval_only(self, query: str) -> Dict[str, Any]:
        try:
            start = time.time()
            chunks = self.retriever.retrieve_candidate_chunks(
                query=query,
                candidate_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
            )
            stats = self.retriever.get_retrieval_stats(query)
            return {
                "query": query,
                "chunks_found": len(chunks),
                "retrieval_time": round(time.time() - start, 3),
                "retrieved_chunks": chunks[:5],
                "retrieval_stats": stats,
            }
        except Exception as e:
            return {"error": str(e), "chunks_found": 0}


if __name__ == "__main__":
    print("Initializing Legal RAG Pipeline...")
    pipeline = LegalRAGPipeline()

    queries = [
        "What are the regulations for educational institutions in Karnataka?",
        "How do courts interpret contractual disputes?",
        "What are the principles of natural justice in administrative law?",
    ]

    for q in queries:
        print("\nQUERY:", q)
        out = pipeline.query(q, include_debug_info=True)
        print("FOUND:", out["answer_found"])
        print("CONF:", out["confidence"])
