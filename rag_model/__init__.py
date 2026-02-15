from rag_pipeline import LegalRAGPipeline, ChatSession, ChatMessage
from retriever import LegalRetriever
from reranker import CrossEncoderReranker
from prompt_builder import PromptBuilder
from llm_inference import QwenInference
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
