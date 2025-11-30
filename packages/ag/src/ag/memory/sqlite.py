from __future__ import annotations

import json
import sqlite3
from enum import StrEnum, auto
from pathlib import Path
from typing import Sequence, override

from ag.common import Usage
from ag.memory.base import Memory
from ag.memory.ephemeral import EphemeralMemory
from ag.models.message import Content, Message, Reasoning, ToolCall, ToolResult


class _MessageType(StrEnum):
    CONTENT = auto()
    REASONING = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()


class SQLiteMemory(Memory):
    def __init__(self, db_path: str | Path, ephemeral_clone: bool = True):
        path = Path(db_path).expanduser()
        if path.as_posix() != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")

        self._init_schema()

        self._ephemeral_copy = EphemeralMemory() if ephemeral_clone else None

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                    ON messages(session_id, id);
                """
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    session_id TEXT PRIMARY KEY,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    input_tokens_cached INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    tool_costs REAL NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    @override
    def get_messages(self, session_id: str) -> Sequence[Message]:
        if self._ephemeral_copy:
            if messages := self._ephemeral_copy.get_messages(session_id):
                return messages

        with self._conn:
            rows = self._conn.execute(
                """
                SELECT
                    payload
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            ).fetchall()

        messages = [self._deserialize_message(row) for row in rows]

        if self._ephemeral_copy:
            # If _ephemeral_copy is enabled but we reached here, it's because this session was
            # not yet loaded into the copy. So let's also fetch the information data and copy
            # this session into the ephemeral version.
            usage = self.get_usage(session_id)
            self._ephemeral_copy.extend(session_id, messages, usage)

        return messages

    @override
    def get_usage(self, session_id: str) -> Usage:
        if self._ephemeral_copy:
            usage = self._ephemeral_copy.get_usage(session_id)
            # the method will return a truthy but zero'd out Usage object if it was not found
            # in memory, so we actually need to check the contents. thankfully, it's impossible
            # for an existing usage to have 0 input tokens
            if usage.input_tokens > 0:
                return usage

        with self._conn:
            row = self._conn.execute(
                """
                SELECT input_tokens, input_tokens_cached, output_tokens, total_cost, tool_costs
                FROM usage
                WHERE session_id = ?;
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return Usage()

        return Usage(
            input_tokens=row["input_tokens"],
            input_tokens_cached=row["input_tokens_cached"],
            output_tokens=row["output_tokens"],
            total_cost=row["total_cost"],
            tool_costs=row["tool_costs"],
        )

    @override
    def extend(
        self,
        session_id: str,
        messages: Sequence[Message],
        usage: Usage,
    ) -> None:
        if self._ephemeral_copy:
            self._ephemeral_copy.extend(session_id, messages, usage)

        serialized_messages = [self._serialize_message(msg) for msg in messages]

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO messages (
                    session_id,
                    payload
                )
                VALUES (?, ?);
                """,
                [(session_id, msg) for msg in serialized_messages],
            )

            self._conn.execute(
                """
                INSERT INTO usage (session_id, input_tokens, input_tokens_cached, output_tokens, total_cost, tool_costs)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    input_tokens = usage.input_tokens + excluded.input_tokens,
                    input_tokens_cached = usage.input_tokens_cached + excluded.input_tokens_cached,
                    output_tokens = usage.output_tokens + excluded.output_tokens,
                    total_cost = usage.total_cost + excluded.total_cost,
                    tool_costs = usage.tool_costs + excluded.tool_costs,
                    created_at = excluded.created_at;
                """,
                (
                    session_id,
                    usage.input_tokens,
                    usage.input_tokens_cached,
                    usage.output_tokens,
                    usage.total_cost,
                    usage.tool_costs,
                ),
            )

    def close(self) -> None:
        self._conn.close()

    def _serialize_message(self, msg: Message) -> str:
        if isinstance(msg, Content):
            return json.dumps(
                {
                    "type": _MessageType.CONTENT.value,
                    "role": msg.role,
                    "text": msg.text,
                }
            )

        if isinstance(msg, Reasoning):
            return json.dumps(
                {
                    "type": _MessageType.REASONING.value,
                    "encrypted_content": msg.encrypted_content,
                    "content": msg.content,
                }
            )

        if isinstance(msg, ToolCall):
            args_json = msg.args or json.dumps(msg.parsed_args)
            return json.dumps(
                {
                    "type": _MessageType.TOOL_CALL.value,
                    "call_id": msg.call_id,
                    "name": msg.name,
                    "args": args_json,
                }
            )

        if isinstance(msg, ToolResult):
            return json.dumps(
                {
                    "type": _MessageType.TOOL_RESULT.value,
                    "call_id": msg.call_id,
                    "result": msg.result,
                }
            )

        # the lsp should have the line below greyed out as unreachable
        raise ValueError(f"received invalid message type: {type(msg)}")

    def _deserialize_message(self, row: sqlite3.Row) -> Message:
        data = json.loads(row["payload"])
        try:
            message_type = _MessageType(data["type"])
        except ValueError:
            raise ValueError(f"Unknown message type stored in SQLite: {data.get('type')}")

        if message_type is _MessageType.CONTENT:
            return Content(role=data["role"], text=data["text"])

        if message_type is _MessageType.REASONING:
            return Reasoning(
                encrypted_content=data["encrypted_content"],
                content=data.get("content", ""),
            )

        if message_type is _MessageType.TOOL_CALL:
            return ToolCall(
                call_id=data["call_id"],
                name=data["name"],
                args=data["args"],
            )

        if message_type is _MessageType.TOOL_RESULT:
            return ToolResult(
                call_id=data["call_id"],
                result=data["result"],
            )

        assert False, "should've already failed in the try/except above"
