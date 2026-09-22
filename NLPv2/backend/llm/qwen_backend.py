from typing import Any, Dict, List, Optional

from llm.base import GenerationResult, LLMBackend
from llm.qwen_inference import QwenInference


class QwenBackend(LLMBackend):
    requires_external_ok = False

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M", **inference_kwargs):
        self.inference = QwenInference(model_name=model_name, **inference_kwargs)

    def generate(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        prompt = self._build_prompt(context_block, user_query, system_prompt, conversation_history)
        stats = self.inference.generate_response_with_stats(prompt, max_new_tokens=max_new_tokens)
        return GenerationResult(
            text=stats["text"],
            prompt_tokens=stats["prompt_tokens"],
            completion_tokens=stats["completion_tokens"],
            model_id=self.inference.model_name,
            model_version=self._model_version(),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.inference.tokenizer.encode(text))

    @property
    def model_id(self) -> str:
        return self.inference.model_name

    def _model_version(self) -> str:
        commit_hash = getattr(getattr(self.inference.model, "config", None), "_commit_hash", None)
        return commit_hash or "unpinned"

    def _build_prompt(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        turns = ""
        for msg in conversation_history or []:
            turns += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"{turns}"
            f"<|im_start|>user\nContext:\n{context_block}\n\nQuestion:\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
