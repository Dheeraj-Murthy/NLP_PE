"""
Per-conversation chat history, stored in PostgreSQL.

Each conversation has its own session ID (a random UUID the API hands out),
so users no longer share one history, and conversations survive API
restarts. Requests that send no session ID use the "default" session, which
keeps the old single shared conversation working for older clients and the
CLI.

If the database can't be reached when the store starts, it falls back to
in-memory sessions (lost on restart) rather than breaking chat.
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2

from db_schema import ensure_schema

DEFAULT_SESSION = "default"
SESSION_ID_RE = re.compile(r"^(default|[0-9a-f]{32})$")
# Sessions untouched for longer than this are deleted when the API starts.
RETENTION_DAYS = int(os.getenv("CHAT_SESSION_RETENTION_DAYS", "30"))

TITLE_MAX_CHARS = 80


def title_from(text: str) -> str:
    """A chat title from its first question: one line, at most TITLE_MAX_CHARS."""
    line = " ".join((text or "").split())
    if len(line) <= TITLE_MAX_CHARS:
        return line or "New chat"
    return line[: TITLE_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"


def is_valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_RE.match(session_id or ""))


class ChatStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._lock = threading.Lock()
        self._memory: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._memory_summaries: Dict[str, Dict[str, Any]] = {}
        self._memory_meta: Dict[str, Dict[str, Any]] = {}
        self._memory_next_id = 1
        self._ready = False

    # -- setup ----------------------------------------------------------------

    def _connect(self):
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = True
        return conn

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                conn = self._connect()
                try:
                    ensure_schema(conn)
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM chat_sessions WHERE session_id <> %s "
                            "AND last_active < now() - make_interval(days => %s)",
                            (DEFAULT_SESSION, RETENTION_DAYS),
                        )
                finally:
                    conn.close()
            except Exception as err:
                print(f"Warning: chat history not persisted (database unavailable): {err}")
                self._memory = {}
            self._ready = True

    @property
    def persistent(self) -> bool:
        self._ensure_ready()
        return self._memory is None

    # -- operations -------------------------------------------------------------

    def new_session(self, title: Optional[str] = None) -> str:
        self._ensure_ready()
        session_id = uuid.uuid4().hex
        if self._memory is not None:
            self._memory[session_id] = []
            now = datetime.now()
            self._memory_meta[session_id] = {"title": title, "created_at": now, "last_active": now}
        else:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO chat_sessions (session_id, title) VALUES (%s, %s)",
                        (session_id, title),
                    )
            finally:
                conn.close()
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        prompt_text: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ensure_ready()
        if self._memory is not None:
            with self._lock:
                message_id = self._memory_next_id
                self._memory_next_id += 1
            self._memory.setdefault(session_id, []).append(
                {
                    "message_id": message_id,
                    "role": role,
                    "content": content,
                    "prompt_text": prompt_text,
                    "retrieval_query": retrieval_query,
                    "details": details or {},
                    "created_at": datetime.now(),
                }
            )
            now = datetime.now()
            meta = self._memory_meta.setdefault(
                session_id, {"title": None, "created_at": now, "last_active": now}
            )
            meta["last_active"] = now
            if role == "user" and not meta["title"]:
                meta["title"] = title_from(content)
            return
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # The first question becomes the title unless one is set.
                first_title = title_from(content) if role == "user" else None
                cur.execute(
                    "INSERT INTO chat_sessions (session_id, title) VALUES (%s, %s) "
                    "ON CONFLICT (session_id) DO UPDATE SET last_active = CURRENT_TIMESTAMP, "
                    "title = COALESCE(chat_sessions.title, EXCLUDED.title)",
                    (session_id, first_title),
                )
                cur.execute(
                    "INSERT INTO chat_messages "
                    "(session_id, role, content, prompt_text, retrieval_query, details) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (session_id, role, content, prompt_text, retrieval_query,
                     json.dumps(details) if details else None),
                )
        finally:
            conn.close()

    def messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Every message in the session, oldest first."""
        self._ensure_ready()
        if self._memory is not None:
            return list(self._memory.get(session_id, []))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT message_id, role, content, prompt_text, retrieval_query, details, created_at "
                    "FROM chat_messages WHERE session_id = %s ORDER BY message_id",
                    (session_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "message_id": r[0],
                "role": r[1],
                "content": r[2],
                "prompt_text": r[3],
                "retrieval_query": r[4],
                "details": r[5] or {},
                "created_at": r[6],
            }
            for r in rows
        ]

    def summary(self, session_id: str) -> Dict[str, Any]:
        """{"summary": str or None, "upto": message_id it covers (0 = none)}."""
        self._ensure_ready()
        if self._memory is not None:
            return dict(self._memory_summaries.get(session_id, {"summary": None, "upto": 0}))
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT summary, summary_upto FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return {"summary": row[0], "upto": row[1]} if row else {"summary": None, "upto": 0}

    def set_summary(self, session_id: str, summary: str, upto: int) -> None:
        self._ensure_ready()
        if self._memory is not None:
            self._memory_summaries[session_id] = {"summary": summary, "upto": upto}
            return
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions (session_id, summary, summary_upto) VALUES (%s, %s, %s) "
                    "ON CONFLICT (session_id) DO UPDATE SET summary = EXCLUDED.summary, "
                    "summary_upto = EXCLUDED.summary_upto",
                    (session_id, summary, upto),
                )
        finally:
            conn.close()

    def import_session(
        self,
        messages: List[Dict[str, Any]],
        summary: Optional[str],
        summary_covers: int,
        title: Optional[str] = None,
    ) -> str:
        """A new session holding these messages (oldest first), with the
        summary covering the first `summary_covers` of them."""
        first_question = next((m["content"] for m in messages if m["role"] == "user"), "")
        session_id = self.new_session(title=title or title_from(first_question))
        if self._memory is not None:
            for m in messages:
                self.add_message(session_id, m["role"], m["content"], m.get("prompt_text"),
                                 m.get("retrieval_query"), m.get("details"))
            ids = [m["message_id"] for m in self._memory[session_id]]
        else:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    ids = []
                    for m in messages:
                        cur.execute(
                            "INSERT INTO chat_messages "
                            "(session_id, role, content, prompt_text, retrieval_query, details) "
                            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING message_id",
                            (session_id, m["role"], m["content"], m.get("prompt_text"),
                             m.get("retrieval_query"),
                             json.dumps(m["details"]) if m.get("details") else None),
                        )
                        ids.append(cur.fetchone()[0])
            finally:
                conn.close()
        if summary and summary_covers:
            self.set_summary(session_id, summary, ids[summary_covers - 1])
        return session_id

    def list_sessions(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Chats with at least one message, most recently active first:
        {"total", "items": [{session_id, title, created_at, last_active,
        message_count}]}. The shared default session is not listed."""
        self._ensure_ready()
        if self._memory is not None:
            rows = [
                (sid, meta["title"], meta["created_at"], meta["last_active"], len(self._memory.get(sid, [])))
                for sid, meta in self._memory_meta.items()
                if sid != DEFAULT_SESSION and self._memory.get(sid)
            ]
            rows.sort(key=lambda r: r[3], reverse=True)
            total, rows = len(rows), rows[offset : offset + limit]
        else:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT s.session_id, s.title, s.created_at, s.last_active, "
                        "COUNT(m.message_id), COUNT(*) OVER () "
                        "FROM chat_sessions s JOIN chat_messages m ON m.session_id = s.session_id "
                        "WHERE s.session_id <> %s "
                        "GROUP BY s.session_id ORDER BY s.last_active DESC, s.session_id "
                        "LIMIT %s OFFSET %s",
                        (DEFAULT_SESSION, limit, offset),
                    )
                    fetched = cur.fetchall()
            finally:
                conn.close()
            total = fetched[0][5] if fetched else 0
            rows = [r[:5] for r in fetched]
        return {
            "total": total,
            "items": [
                {
                    "session_id": sid,
                    "title": title or "New chat",
                    "created_at": created.isoformat() if created else None,
                    "last_active": active.isoformat() if active else None,
                    "message_count": count,
                }
                for sid, title, created, active, count in rows
            ],
        }

    def session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """{"session_id", "title"} or None if the session doesn't exist."""
        self._ensure_ready()
        if self._memory is not None:
            meta = self._memory_meta.get(session_id)
            return {"session_id": session_id, "title": meta["title"]} if meta else None
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT title FROM chat_sessions WHERE session_id = %s", (session_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        return {"session_id": session_id, "title": row[0]} if row else None

    def rename(self, session_id: str, title: str) -> bool:
        """Set a chat's title; False if the chat doesn't exist."""
        self._ensure_ready()
        title = title_from(title)
        if self._memory is not None:
            if session_id not in self._memory_meta:
                return False
            self._memory_meta[session_id]["title"] = title
            return True
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET title = %s WHERE session_id = %s", (title, session_id)
                )
                return cur.rowcount == 1
        finally:
            conn.close()

    def clear(self, session_id: str) -> None:
        self._ensure_ready()
        if self._memory is not None:
            self._memory.pop(session_id, None)
            self._memory_summaries.pop(session_id, None)
            self._memory_meta.pop(session_id, None)
            return
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        finally:
            conn.close()
