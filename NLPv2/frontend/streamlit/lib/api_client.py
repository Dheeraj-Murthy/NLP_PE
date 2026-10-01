import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_API_URL = "http://localhost:8000"
DOCUMENT_TIMEOUT = 300  # OCR + 7B generation on an uploaded document can be slow
DEFAULT_TIMEOUT = 60


class LegalRAGAPIError(Exception):
    """Raised for any failure talking to the backend (connection, timeout, non-2xx, or a
    {"success": false} payload)."""


def get_api_base_url() -> str:
    return os.environ.get("RAG_API_URL", DEFAULT_API_URL).rstrip("/")


BASE_URL = get_api_base_url()


def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def _handle_response(resp: requests.Response) -> Dict[str, Any]:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        detail = None
        try:
            detail = resp.json().get("detail")
        except Exception:
            pass
        raise LegalRAGAPIError(detail or str(e)) from e

    payload = resp.json()
    if payload.get("success") is False:
        raise LegalRAGAPIError(payload.get("message", "Request failed"))
    return payload


def health() -> Dict[str, Any]:
    try:
        resp = requests.get(_url("/health"), timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e


def status() -> Dict[str, Any]:
    try:
        resp = requests.get(_url("/status"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def query(
    query_text: str,
    top_k: int = 8,
    threshold: float = 0.3,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "query": query_text,
        "top_k": top_k,
        "threshold": threshold,
        "include_debug": include_debug,
        "external_ok": external_ok,
    }
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    try:
        resp = requests.post(_url("/query"), data=data, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def chat(
    message: str,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    data = {"message": message, "include_debug": include_debug, "external_ok": external_ok}
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    try:
        resp = requests.post(_url("/chat"), data=data, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def gemini_models(api_key: str) -> List[str]:
    try:
        resp = requests.post(
            _url("/models/gemini"), data={"api_key": api_key}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]["models"]


def chat_clear() -> None:
    try:
        resp = requests.post(_url("/chat/clear"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    _handle_response(resp)


def chat_history() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(_url("/chat/history"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def upload_document(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    query_text: Optional[str] = None,
    include_retrieval: bool = True,
    include_debug: bool = False,
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    data = {
        "query": query_text or "",
        "include_retrieval": include_retrieval,
        "include_debug": include_debug,
        "external_ok": external_ok,
    }
    if model:
        data["model"] = model
    if api_key:
        data["api_key"] = api_key
    try:
        resp = requests.post(
            _url("/document"),
            files={"file": (filename, file_bytes, content_type)},
            data=data,
            timeout=DOCUMENT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def retrieval_test(query_text: str) -> Dict[str, Any]:
    try:
        resp = requests.post(
            _url("/retrieval-test"),
            data={"query": query_text},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_judgment(judgment_id: int, depth: int = 2) -> Dict[str, Any]:
    try:
        resp = requests.get(
            _url(f"/graph/judgment/{judgment_id}"),
            params={"depth": depth},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_landmark_cases(limit: int = 10) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            _url("/graph/landmark-cases"), params={"limit": limit}, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e
    return _handle_response(resp)["data"]


def graph_path(source_id: int, target_id: int) -> Optional[List[int]]:
    try:
        resp = requests.get(
            _url("/graph/path"),
            params={"source_id": source_id, "target_id": target_id},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LegalRAGAPIError(str(e)) from e

    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        return None
    return payload["data"]["path"]
