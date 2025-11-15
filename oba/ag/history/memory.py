from typing import Sequence, override

from oba.ag.common import Usage
from oba.ag.history.base import HistoryDb, SessionInfo
from oba.ag.models.types import MessageTypes


class InMemoryDb(HistoryDb):
    def __init__(self):
        # for each session_id we'll have a list of messages to be sent to the model
        self._db: dict[str, SessionInfo] = dict()

    @override
    def get_messages(self, session_id: str) -> Sequence[MessageTypes]:
        if session_id not in self._db:
            return list()
        return self._db[session_id].messages

    @override
    def get_usage(self, session_id: str) -> Usage:
        if session_id not in self._db:
            return Usage()
        return self._db[session_id].usage

    @override
    def extend(
        self,
        session_id: str,
        messages: Sequence[MessageTypes],
        usage: Usage,
    ) -> None:
        if session_id not in self._db:
            self._db[session_id] = SessionInfo(messages=list(), usage=Usage())

        self._db[session_id].messages.extend(messages)
        self._db[session_id].usage = self._db[session_id].usage.acc(usage)
