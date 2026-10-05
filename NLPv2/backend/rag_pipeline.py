from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, field
from datetime import datetime

from retrieval.retriever import LegalRetriever
from prompt_builder import PromptBuilder
from llm.base import LLMBackend, GenerationResult, PrivacyGateError
from llm.qwen_backend import QwenBackend
from post_processor import PostProcessor, RAGResponse
from retrieval.reranker import CrossEncoderReranker
from document_processor import DocumentProcessor
from retrieval.citation_graph import CitationGraphManager
from tracking import log_query_run
from chat_store import ChatStore, DEFAULT_SESSION
from chat_context import (
    REWRITE_MAX_TOKENS,
    REWRITE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    batches_for_summary,
    carried_documents,
    cited_documents,
    clean_rewrite,
    context_window,
    display_history,
    export_conversation,
    export_markdown,
    fits,
    parse_import,
    history_budget,
    needs_rewrite,
    prompt_history,
    question_with_reading,
    retrieval_query,
    rewrite_request,
    split_for_summary,
    summary_request,
    summary_system_note,
)

# Extra room in the context block for documents cited earlier in a chat.
CARRIED_DOCUMENT_TOKENS = 1200

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"


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

    def clear(self):
        self.messages.clear()


class LegalRAGPipeline:
    DOCUMENT_SYSTEM_PROMPT = (
        PromptBuilder.SYSTEM_PROMPT
        + " When analyzing an attached document, if it contains court judgments or "
        "legal cases, identify the parties involved, the key issues addressed, the "
        "court's reasoning and holding, and any cited precedents."
    )

    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        model_name: str = DEFAULT_QWEN_MODEL,
        load_llm: bool = True,
        top_k: int = 8,
        judgment_top_k: Optional[int] = None,
        statute_top_k: int = 4,
        similarity_threshold: float = 0.3,
        max_context_tokens: int = 1200,
        max_new_tokens: int = 512,
        graph_boost: float = 0.0,
    ):
        self.graph_manager = CitationGraphManager()
        self.retriever = LegalRetriever(
            db_connection_string, graph_manager=self.graph_manager
        )
        self.reranker = CrossEncoderReranker()

        self.stage1_k = 30
        # Statutes and judgments are retrieved, fused, and reranked as two
        # fully independent pipelines (never pooled against each other) so
        # one type can't crowd the other out — see retrieve_judgment_candidates
        # / retrieve_statute_candidates in retrieval/retriever.py.
        self.stage2_k = judgment_top_k if judgment_top_k is not None else top_k
        self.statute_candidate_k = 15
        self.statute_top_k = statute_top_k
        self.stage1_threshold = 0.2
        self.statute_similarity_threshold = 0.2

        self.prompt_builder = PromptBuilder(max_context_tokens)

        self.default_model_id = model_name
        self.max_new_tokens = max_new_tokens
        self.backends: Dict[str, LLMBackend] = {}
        if load_llm:
            self.backends["default"] = QwenBackend(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.9,
                do_sample=True,
            )

        self.post_processor = PostProcessor()
        self.document_processor = DocumentProcessor()
        # Conversations, one per session ID, kept in Postgres (see chat_store.py).
        # How much of each the model sees depends on its context window
        # (chat_context.context_window); older turns get summarised to fit.
        self.chat_store = ChatStore(self.retriever.db_connection_string)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.graph_boost = graph_boost

    def _resolve_backend(
        self, model: Optional[str], external_ok: bool, api_key: Optional[str] = None
    ) -> LLMBackend:
        if not model or model == self.default_model_id or model == "qwen":
            if "default" not in self.backends:
                raise RuntimeError("LLM not loaded")
            return self.backends["default"]

        if model.startswith("claude-"):
            from llm.anthropic_backend import AnthropicBackend

            backend_cls = AnthropicBackend
        elif model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
            from llm.openai_backend import OpenAIBackend

            backend_cls = OpenAIBackend
        elif model.startswith("gemini-"):
            from llm.gemini_backend import GeminiBackend

            backend_cls = GeminiBackend
        else:
            raise ValueError(f"Unknown model: {model}")

        if backend_cls.requires_external_ok and not external_ok:
            raise PrivacyGateError(
                f"Model '{model}' is an external API backend. Pass external_ok=True "
                f"to confirm you accept sending case text to this provider."
            )

        if api_key:
            # A user-supplied key is per-request only — never cached on the
            # shared pipeline instance, since `pipeline` is one global object
            # serving every API request/user (api.py:31). Caching it in
            # self.backends would leak one user's key to every other caller
            # who later picks the same model without supplying their own.
            return backend_cls(model_id=model, api_key=api_key)

        if model in self.backends:
            return self.backends[model]

        backend = backend_cls(model_id=model)  # uses the server's own env-configured key
        self.backends[model] = backend
        return backend

    def _rerank_statutes(
        self, query: str, statute_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reranks statute candidates, but keeps any exact-match row (from an
        explicit 'article N'/'section N' reference in the query) pinned first
        unconditionally — the cross-encoder scores terse statute text against
        prose-heavy queries inconsistently, and would otherwise re-sort the
        guaranteed exact match right out of the final top_n."""
        exact = [c for c in statute_candidates if c.get("exact_match")]
        rest = [c for c in statute_candidates if not c.get("exact_match")]
        remaining_slots = max(self.statute_top_k - len(exact), 0)
        reranked_rest = self.reranker.rerank(query, rest, top_n=remaining_slots)
        return exact[: self.statute_top_k] + reranked_rest

    def _build_metrics(
        self,
        gen: Optional[GenerationResult],
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        chunks_retrieved: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metrics = {
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "chunks_retrieved": chunks_retrieved,
            "prompt_tokens": gen.prompt_tokens if gen else 0,
            "completion_tokens": gen.completion_tokens if gen else 0,
            "model_id": gen.model_id if gen else None,
            "model_version": gen.model_version if gen else None,
        }
        if extra:
            metrics.update(extra)
        return metrics

    def query(
        self,
        user_query: str,
        include_debug_info: bool = False,
        model: Optional[str] = None,
        external_ok: bool = False,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            backend = self._resolve_backend(model, external_ok, api_key)
        except (PrivacyGateError, RuntimeError, ValueError) as e:
            return {
                "error": str(e),
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

        try:
            retrieval_start = time.time()

            # ✅ STAGE 1: high-recall retrieval — judgments and statutes
            # independently, never pooled against each other
            judgment_candidates = self.retriever.retrieve_judgment_candidates(
                query=user_query,
                candidate_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold,
                graph_boost=self.graph_boost,
            )
            statute_candidates = self.retriever.retrieve_statute_candidates(
                query=user_query,
                candidate_k=self.statute_candidate_k,
                similarity_threshold=self.statute_similarity_threshold,
            )

            # ✅ STAGE 2: reranking — statutes first in the final list
            reranked_statutes = self._rerank_statutes(user_query, statute_candidates)
            reranked_judgments = self.reranker.rerank(
                user_query, judgment_candidates, top_n=self.stage2_k
            )
            retrieved_chunks = reranked_statutes + reranked_judgments

            retrieval_time = time.time() - retrieval_start

            if not retrieved_chunks:
                return self._create_no_results_response(
                    user_query, include_debug_info, retrieval_time
                )

            context_block = self.prompt_builder.build_context_block(
                retrieved_chunks, token_counter=backend.count_tokens
            )

            generation_start = time.time()
            gen = backend.generate(
                context_block=context_block,
                user_query=user_query,
                system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                max_new_tokens=self.max_new_tokens,
            )
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                gen.text, retrieved_chunks, user_query
            )

            precedent_chains = self._build_precedent_chains(retrieved_chunks)

            total_time = time.time() - start_time

            extra: Dict[str, Any] = {"graph_boost": self.graph_boost}
            if isinstance(backend, QwenBackend):
                extra["temperature"] = backend.inference.temperature
                extra["do_sample"] = backend.inference.do_sample

            result = {
                "answer": self.post_processor.format_response_with_citations(processed),
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "citations_by_type": processed.citations_by_type,
                "sources_by_type": processed.sources_by_type,
                "precedent_chains": precedent_chains,
                "metrics": self._build_metrics(
                    gen, retrieval_time, generation_time, total_time,
                    len(retrieved_chunks), extra=extra,
                ),
            }

            if include_debug_info:
                result["debug"] = {
                    "retrieved_chunks": retrieved_chunks[:3],
                    "raw_response": gen.text,
                    "prompt_preview": context_block[:500] + "..."
                    if len(context_block) > 500
                    else context_block,
                    "quality_metrics": self.post_processor.get_quality_metrics(
                        processed
                    ),
                }

            log_query_run(
                endpoint="query",
                params={
                    "model_id": gen.model_id,
                    "top_k": self.stage2_k,
                    "similarity_threshold": self.similarity_threshold,
                    "graph_boost": self.graph_boost,
                },
                metrics={**result["metrics"], "confidence": result["confidence"]},
            )

            return result

        except Exception as e:
            return {
                "error": f"Pipeline failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

    def _build_precedent_chains(
        self, retrieved_chunks: List[Dict[str, Any]], max_cases: int = 5
    ) -> List[Dict[str, Any]]:
        """Build a per-case precedent-chain summary from the citation graph,
        keyed by the cases backing the retrieved chunks."""
        chains = []
        seen_ids = set()

        for chunk in retrieved_chunks:
            judgment_id = chunk.get("judgment_id")
            if not judgment_id or judgment_id in seen_ids:
                continue
            seen_ids.add(judgment_id)

            summary = self.graph_manager.get_precedent_summary(judgment_id, top_n=2)
            if summary:
                chains.append(summary)

            if len(chains) >= max_cases:
                break

        return chains

    def _create_no_results_response(
        self, user_query: str, include_debug_info: bool, retrieval_time: float
    ) -> Dict[str, Any]:
        result = {
            "answer": "No relevant legal cases found for your query. Please try rephrasing your question or using different legal terms.",
            "answer_found": False,
            "confidence": 0.0,
            "citations": [],
            "sources": [],
            "citations_by_type": {},
            "sources_by_type": {},
            "metrics": self._build_metrics(
                None, retrieval_time, 0.0, retrieval_time, 0
            ),
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
        default_backend = self.backends.get("default")
        backends_status = {
            key: {"model_id": backend.model_id, "loaded": True}
            for key, backend in self.backends.items()
        }
        memory_info = (
            default_backend.inference.get_memory_info()
            if isinstance(default_backend, QwenBackend)
            else {}
        )
        return {
            "model_loaded": default_backend is not None,
            "model_name": default_backend.model_id if default_backend else None,
            "backends": backends_status,
            "retriever_config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
            },
            "prompt_builder_config": {
                "max_context_tokens": self.prompt_builder.max_context_tokens
            },
            "llm_config": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": default_backend.inference.temperature
                if isinstance(default_backend, QwenBackend)
                else None,
                "do_sample": default_backend.inference.do_sample
                if isinstance(default_backend, QwenBackend)
                else None,
            },
            "memory_info": memory_info,
        }

    def query_with_document(
        self,
        document_path: str,
        user_query: Optional[str] = None,
        include_retrieval: bool = True,
        include_debug_info: bool = False,
        model: Optional[str] = None,
        external_ok: bool = False,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query using an attached document (PDF/image) via OCR.

        Alternate path: extracts text from document, optionally combines with
        retrieved context, then generates answer.
        """
        start_time = time.time()

        try:
            backend = self._resolve_backend(model, external_ok, api_key)
        except (PrivacyGateError, RuntimeError, ValueError) as e:
            return {
                "error": str(e),
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
                doc_query = user_query or ocr_text[:500]
                judgment_candidates = self.retriever.retrieve_judgment_candidates(
                    query=doc_query,
                    candidate_k=self.stage1_k,
                    similarity_threshold=self.stage1_threshold,
                )
                statute_candidates = self.retriever.retrieve_statute_candidates(
                    query=doc_query,
                    candidate_k=self.statute_candidate_k,
                    similarity_threshold=self.statute_similarity_threshold,
                )
                reranked_statutes = self._rerank_statutes(doc_query, statute_candidates)
                reranked_judgments = self.reranker.rerank(
                    doc_query, judgment_candidates, top_n=self.stage2_k
                )
                retrieved_chunks = reranked_statutes + reranked_judgments
                retrieval_time = time.time() - retrieval_start

            context_block = self._build_document_context(
                ocr_text, retrieved_chunks, backend
            )
            effective_query = user_query or (
                "Provide a summary and analysis of the attached document, including "
                "any relevant legal principles, precedents, or findings."
            )

            generation_start = time.time()
            gen = backend.generate(
                context_block=context_block,
                user_query=effective_query,
                system_prompt=self.DOCUMENT_SYSTEM_PROMPT,
                max_new_tokens=self.max_new_tokens,
            )
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                gen.text, retrieved_chunks, user_query or ""
            )

            total_time = time.time() - start_time

            result = {
                "answer": self.post_processor.format_response_with_citations(processed),
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "citations_by_type": processed.citations_by_type,
                "sources_by_type": processed.sources_by_type,
                "document": document_info,
                "metrics": self._build_metrics(
                    gen, retrieval_time, generation_time, total_time,
                    len(retrieved_chunks), extra={"document_pages": document_info["pages"]},
                ),
            }

            if include_debug_info:
                result["debug"] = {
                    "ocr_text_preview": ocr_text[:500] + "..."
                    if len(ocr_text) > 500
                    else ocr_text,
                    "raw_response": gen.text,
                    "retrieved_chunks": retrieved_chunks[:3]
                    if retrieved_chunks
                    else [],
                }

            log_query_run(
                endpoint="document",
                params={
                    "model_id": gen.model_id,
                    "top_k": self.stage2_k,
                    "similarity_threshold": self.similarity_threshold,
                },
                metrics={**result["metrics"], "confidence": result["confidence"]},
            )

            return result

        except Exception as e:
            return {
                "error": f"Document query failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
            }

    def _build_document_context(
        self,
        ocr_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        backend: LLMBackend,
    ) -> str:
        max_doc_length = 3000
        if len(ocr_text) > max_doc_length:
            ocr_text = ocr_text[:max_doc_length] + "..."

        parts = [f"Attached Document (OCR extracted text):\n{ocr_text}"]

        if retrieved_chunks:
            case_context = self.prompt_builder.build_context_block(
                retrieved_chunks, token_counter=backend.count_tokens
            )
            if case_context:
                parts.append(f"Relevant Legal Context from Database:\n{case_context}")

        return "\n\n".join(parts)

    def chat(
        self,
        user_message: str,
        include_debug_info: bool = False,
        model: Optional[str] = None,
        external_ok: bool = False,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-turn chat. Each session_id is its own conversation; None
        uses the shared default session. The model sees the whole
        conversation while it fits its context window, with the oldest turns
        summarised once it doesn't; documents cited in recent answers are
        shown again; follow-up questions are rewritten by the model into
        standalone questions for retrieval, and the model is told that
        reading of the question too. Works the same whichever model answers
        each turn."""
        session_id = session_id or DEFAULT_SESSION
        try:
            backend = self._resolve_backend(model, external_ok, api_key)
        except (PrivacyGateError, RuntimeError, ValueError) as e:
            return {
                "error": str(e),
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
                "session_id": session_id,
            }

        prior = self.chat_store.messages(session_id)
        search_query = self._standalone_query(backend, user_message, prior)
        self.chat_store.add_message(
            session_id, "user", user_message, retrieval_query=search_query
        )

        def history_for_display() -> List[Dict[str, Any]]:
            return display_history(self.chat_store.messages(session_id))

        start_time = time.time()

        try:
            retrieval_start = time.time()
            judgment_candidates = self.retriever.retrieve_judgment_candidates(
                query=search_query,
                candidate_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold,
                graph_boost=self.graph_boost,
            )
            statute_candidates = self.retriever.retrieve_statute_candidates(
                query=search_query,
                candidate_k=self.statute_candidate_k,
                similarity_threshold=self.statute_similarity_threshold,
            )
            reranked_statutes = self._rerank_statutes(search_query, statute_candidates)
            reranked_judgments = self.reranker.rerank(
                search_query, judgment_candidates, top_n=self.stage2_k
            )
            retrieved_chunks = reranked_statutes + reranked_judgments
            # Documents the conversation already relied on, after this turn's
            # results so their [n] numbers don't shift the new ones.
            carried = carried_documents(prior, retrieved_chunks)
            retrieved_chunks = retrieved_chunks + carried
            retrieval_time = time.time() - retrieval_start

            if not retrieved_chunks:
                no_answer = "I couldn't find relevant legal cases for your query. Could you try rephrasing?"
                self.chat_store.add_message(
                    session_id, "assistant", no_answer,
                    details={"citations": [], "confidence": 0.0},
                )
                return {
                    "answer": no_answer,
                    "answer_found": False,
                    "confidence": 0.0,
                    "citations": [],
                    "sources": [],
                    "citations_by_type": {},
                    "sources_by_type": {},
                    "session_id": session_id,
                    "conversation_history": history_for_display(),
                    "metrics": self._build_metrics(
                        None, retrieval_time, 0.0, time.time() - start_time, 0
                    ),
                }

            context_block = self.prompt_builder.build_context_block(
                retrieved_chunks,
                token_counter=backend.count_tokens,
                max_tokens=self.prompt_builder.max_context_tokens
                + (CARRIED_DOCUMENT_TOKENS if carried else 0),
            )

            # Earlier turns only — the current message goes to generate()
            # separately as user_query.
            # The user's words, plus how they read in this conversation, so
            # a follow-up is answered on the conversation's topic rather than
            # on whatever the retrieved documents happen to be about.
            question = question_with_reading(user_message, search_query)
            window = context_window(backend.model_id)
            fixed_tokens = sum(
                backend.count_tokens(t)
                for t in (self.prompt_builder.SYSTEM_PROMPT, context_block, question)
            )
            history, summary, summarized = self._conversation_context(
                session_id,
                prior,
                backend,
                history_budget(window, fixed_tokens, self.max_new_tokens),
                window,
            )
            system_prompt = self.prompt_builder.SYSTEM_PROMPT
            if summary:
                system_prompt += summary_system_note(summary)

            generation_start = time.time()
            gen = backend.generate(
                context_block=context_block,
                user_query=question,
                system_prompt=system_prompt,
                conversation_history=history,
                max_new_tokens=self.max_new_tokens,
            )
            generation_time = time.time() - generation_start

            processed = self.post_processor.process_response(
                gen.text, retrieved_chunks, user_message
            )

            total_time = time.time() - start_time

            formatted_answer = self.post_processor.format_response_with_citations(
                processed
            )
            metrics = self._build_metrics(
                gen, retrieval_time, generation_time, total_time,
                len(retrieved_chunks),
            )
            context_info = {
                "model_context_tokens": window,
                "history_messages": len(history),
                "summarized_messages": summarized,
                "carried_documents": len(carried),
            }
            self.chat_store.add_message(
                session_id,
                "assistant",
                formatted_answer,
                prompt_text=processed.answer,
                details={
                    "citations": processed.citations,
                    "sources": processed.sources,
                    "citations_by_type": processed.citations_by_type,
                    "sources_by_type": processed.sources_by_type,
                    "confidence": processed.confidence_score,
                    "model_id": metrics.get("model_id"),
                    "model_version": metrics.get("model_version"),
                    "cited_documents": cited_documents(gen.text, retrieved_chunks),
                    "context": context_info,
                },
            )

            result = {
                "answer": formatted_answer,
                "answer_found": processed.is_answer_found,
                "confidence": processed.confidence_score,
                "citations": processed.citations,
                "sources": processed.sources,
                "citations_by_type": processed.citations_by_type,
                "sources_by_type": processed.sources_by_type,
                "session_id": session_id,
                "conversation_history": history_for_display(),
                "context": context_info,
                "metrics": metrics,
            }

            if include_debug_info:
                result["debug"] = {
                    "retrieved_chunks": retrieved_chunks[:3],
                    "raw_response": gen.text,
                    "conversation_history": history,
                    "retrieval_query": search_query,
                    "conversation_summary": summary,
                }

            log_query_run(
                endpoint="chat",
                params={
                    "model_id": gen.model_id,
                    "top_k": self.stage2_k,
                    "similarity_threshold": self.similarity_threshold,
                    "graph_boost": self.graph_boost,
                },
                metrics={
                    **result["metrics"],
                    "confidence": result["confidence"],
                    "history_turns": len(history),
                    "summarized_messages": summarized,
                    "carried_documents": len(carried),
                    "follow_up": int(search_query != user_message),
                },
            )

            return result

        except Exception as e:
            return {
                "error": f"Chat failed: {str(e)}",
                "answer": None,
                "answer_found": False,
                "confidence": 0.0,
                "session_id": session_id,
                "conversation_history": history_for_display(),
            }

    def _standalone_query(
        self, backend: LLMBackend, user_message: str, prior: List[Dict[str, Any]]
    ) -> str:
        """The query to retrieve with. In an ongoing chat this turn's model
        rewrites the message as a standalone question; if that fails, the
        follow-up heuristic is used instead."""
        if not needs_rewrite(prior):
            return user_message
        try:
            gen = backend.generate(
                context_block="",
                user_query=rewrite_request(user_message, prior),
                system_prompt=REWRITE_SYSTEM_PROMPT,
                conversation_history=None,
                max_new_tokens=REWRITE_MAX_TOKENS,
            )
            rewritten = clean_rewrite(gen.text)
            if rewritten:
                return rewritten
        except Exception as err:
            print(f"Warning: could not rewrite follow-up query: {err}")
        return retrieval_query(user_message, prior)

    def _conversation_context(
        self,
        session_id: str,
        prior: List[Dict[str, Any]],
        backend: LLMBackend,
        budget: int,
        window: int,
    ):
        """(history for the model, summary or None, number of messages the
        summary covers). The whole conversation if it fits `budget`;
        otherwise older turns are folded into the stored running summary,
        made by this turn's model, and only recent turns go verbatim. If
        summarising fails, the oldest turns are dropped instead so the
        answer still goes through."""
        count = backend.count_tokens
        if fits(prior, count, budget):
            # Everything fits (e.g. after switching to a larger model): the
            # whole conversation word for word, no summary needed.
            return prompt_history(prior, count, budget), None, 0

        stored = self.chat_store.summary(session_id)
        summary, upto = stored["summary"], stored["upto"]
        pending = [m for m in prior if m["message_id"] > upto]

        if not fits(pending, count, budget, summary):
            to_summarise, keep = split_for_summary(pending, count, budget)
            if to_summarise:
                try:
                    summary = self._summarise(
                        backend, summary, to_summarise, window,
                        max_tokens=min(1024, max(256, budget // 4)),
                    )
                    upto = to_summarise[-1]["message_id"]
                    self.chat_store.set_summary(session_id, summary, upto)
                    pending = keep
                except Exception as err:
                    print(f"Warning: could not summarise chat {session_id}: {err}")

        summary_tokens = count(summary) + 20 if summary else 0
        history = prompt_history(pending, count, max(0, budget - summary_tokens))
        summarized = sum(1 for m in prior if m["message_id"] <= upto) if summary else 0
        return history, summary, summarized

    def _summarise(
        self,
        backend: LLMBackend,
        previous: Optional[str],
        messages: List[Dict[str, Any]],
        window: int,
        max_tokens: int,
    ) -> str:
        """Fold messages into the running summary, in as many model calls as
        the window needs."""
        summary = previous
        for batch in batches_for_summary(messages, backend.count_tokens, max(1000, window // 2)):
            gen = backend.generate(
                context_block="",
                user_query=summary_request(summary, batch),
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                conversation_history=None,
                max_new_tokens=max_tokens,
            )
            text = (gen.text or "").strip()
            if not text:
                raise RuntimeError("model returned an empty summary")
            summary = text
        return summary

    def export_chat(self, session_id: Optional[str] = None, fmt: str = "json"):
        """The conversation as a portable JSON dict, or Markdown text."""
        session_id = session_id or DEFAULT_SESSION
        messages = self.chat_store.messages(session_id)
        if fmt == "markdown":
            return export_markdown(messages)
        exported = export_conversation(session_id, messages, self.chat_store.summary(session_id))
        info = self.chat_store.session_info(session_id)
        exported["title"] = info["title"] if info else None
        return exported

    def import_chat(self, data: Dict[str, Any]) -> str:
        """Start a new session from an exported conversation; returns its ID.
        Raises ValueError for a malformed export."""
        messages, summary, covers = parse_import(data)
        title = data.get("title") if isinstance(data.get("title"), str) else None
        return self.chat_store.import_session(messages, summary, covers, title=title)

    def list_chat_sessions(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Chats, most recently active first."""
        return self.chat_store.list_sessions(limit=limit, offset=offset)

    def rename_chat_session(self, session_id: str, title: str) -> bool:
        return self.chat_store.rename(session_id, title)

    def new_chat_session(self) -> str:
        return self.chat_store.new_session()

    def clear_chat_history(self, session_id: Optional[str] = None):
        self.chat_store.clear(session_id or DEFAULT_SESSION)

    def get_chat_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return display_history(self.chat_store.messages(session_id or DEFAULT_SESSION))


    def test_retrieval_only(self, query: str) -> Dict[str, Any]:
        try:
            start = time.time()
            judgment_chunks = self.retriever.retrieve_judgment_candidates(
                query=query,
                candidate_k=self.stage1_k,
                similarity_threshold=self.stage1_threshold,
            )
            statute_chunks = self.retriever.retrieve_statute_candidates(
                query=query,
                candidate_k=self.statute_candidate_k,
                similarity_threshold=self.statute_similarity_threshold,
            )
            chunks = statute_chunks + judgment_chunks
            stats = self.retriever.get_retrieval_stats(query)
            return {
                "query": query,
                "chunks_found": len(chunks),
                "statute_chunks_found": len(statute_chunks),
                "judgment_chunks_found": len(judgment_chunks),
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
