from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    model_version: str
    raw: Optional[Dict[str, Any]] = None


class PrivacyGateError(Exception):
    """Raised when a non-local backend is requested without external_ok=True."""


class LLMBackend(ABC):
    requires_external_ok: bool = False

    @abstractmethod
    def generate(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """conversation_history is a list of {"role": "user"|"assistant", "content": str}."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...
