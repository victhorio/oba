from collections.abc import Sequence

from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel

from oba.ag.common import Usage


class SessionInfo(BaseModel):
    messages: list[ResponseInputItemParam]
    usage: Usage


class HistoryDb:
    """
    A poc implementation, this should become an abstract base class and
    there should be different implementations, namely in-memory and sqlite.
    For now a very simple, non-configurable in-memory "db".
    """

    def __init__(self):
        # for each session_id we'll have a list of messages to be sent to the model
        self._db: dict[str, SessionInfo] = dict()

    def get_messages(self, session_id: str) -> Sequence[ResponseInputItemParam]:
        if session_id not in self._db:
            return list()
        return self._db[session_id].messages

    def get_usage(self, session_id: str) -> Usage:
        if session_id not in self._db:
            return Usage()
        return self._db[session_id].usage

    def extend(
        self,
        session_id: str,
        messages: Sequence[ResponseInputItemParam],
        usage: Usage,
    ) -> None:
        if session_id not in self._db:
            self._db[session_id] = SessionInfo(messages=list(), usage=Usage())

        self._db[session_id].messages.extend(messages)
        self._db[session_id].usage = self._db[session_id].usage.acc(usage)
