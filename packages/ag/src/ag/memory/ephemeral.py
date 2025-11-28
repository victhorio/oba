from typing import Sequence, override

from ag.common import Usage
from ag.memory.base import Memory
from ag.models.message import Message


class EphemeralMemory(Memory):
    def __init__(self):
        self._messages: dict[str, list[Message]] = dict()
        self._usage: dict[str, Usage] = dict()

    @override
    def get_messages(self, session_id: str) -> Sequence[Message]:
        if session_id not in self._messages:
            return list()
        return self._messages[session_id]

    @override
    def get_usage(self, session_id: str) -> Usage:
        if session_id not in self._usage:
            return Usage()
        return self._usage[session_id]

    @override
    def extend(
        self,
        session_id: str,
        messages: Sequence[Message],
        usage: Usage,
    ) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = list()
            self._usage[session_id] = Usage()

        self._messages[session_id].extend(messages)
        self._usage[session_id] = self._usage[session_id].acc(usage)
