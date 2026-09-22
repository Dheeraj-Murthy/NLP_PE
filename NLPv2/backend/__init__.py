from rag_pipeline import LegalRAGPipeline, ChatSession, ChatMessage
from retrieval.retriever import LegalRetriever
from retrieval.reranker import CrossEncoderReranker
from prompt_builder import PromptBuilder
from llm.qwen_inference import QwenInference
from post_processor import PostProcessor, RAGResponse
from document_processor import DocumentProcessor

__all__ = [
    "LegalRAGPipeline",
    "ChatSession",
    "ChatMessage",
    "LegalRetriever",
    "CrossEncoderReranker",
    "PromptBuilder",
    "QwenInference",
    "PostProcessor",
    "RAGResponse",
    "DocumentProcessor",
]

try:
    from api import app

    __all__.append("app")
except ImportError:
    pass
