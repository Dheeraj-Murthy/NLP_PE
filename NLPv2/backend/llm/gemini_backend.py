import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm.base import GenerationResult, LLMBackend

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


class GeminiBackend(LLMBackend):
    requires_external_ok = True

    def __init__(
        self,
        model_id: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self._model_id = model_id
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        # Gemini's `contents` API uses role "model" for prior assistant turns,
        # not "assistant" — everything else in our ChatSession history uses
        # the OpenAI/Anthropic-style "assistant", so translate it here.
        contents = []
        for m in conversation_history or []:
            role = "model" if m["role"] == "assistant" else m["role"]
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=f"Context:\n{context_block}\n\nQuestion:\n{user_query}")],
            )
        )

        resp = self.client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_new_tokens or self.max_new_tokens,
            ),
        )

        usage = resp.usage_metadata

        return GenerationResult(
            text=(resp.text or "").strip(),
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            model_id=self._model_id,
            model_version=getattr(resp, "model_version", None) or self._model_id,
        )

    def count_tokens(self, text: str) -> int:
        # No local tokenizer bundled for Gemini models; word-count heuristic
        # is good enough for context-budget estimation (not billing-accurate).
        return int(len(text.split()) * 1.3)

    @property
    def model_id(self) -> str:
        return self._model_id


def list_available_models(api_key: str) -> List[str]:
    """Models this key can call generateContent on. Raises google.genai.errors.ClientError
    (e.g. API_KEY_INVALID) if the key itself is bad."""
    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        if m.name and "generateContent" in (m.supported_actions or []):
            names.append(m.name.split("/")[-1])
    return sorted(names)
