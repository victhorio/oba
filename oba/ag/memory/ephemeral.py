from typing import Sequence, override

from attrs import define

from oba.ag.common import Usage
from oba.ag.memory.base import Memory
from oba.ag.models.message import Message


class EphemeralMemory(Memory):
    def __init__(self):
        # for each session_id we'll have a list of messages to be sent to the model
        self._db: dict[str, SessionInfo] = dict()

    @override
    def get_messages(self, session_id: str) -> Sequence[Message]:
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
        messages: Sequence[Message],
        usage: Usage,
    ) -> None:
        if session_id not in self._db:
            self._db[session_id] = SessionInfo(messages=list(), usage=Usage())

        self._db[session_id].messages.extend(messages)
        self._db[session_id].usage = self._db[session_id].usage.acc(usage)


@define(slots=True)
class SessionInfo:
    messages: list[Message]
    usage: Usage
