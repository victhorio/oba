from abc import ABC, abstractmethod
from collections.abc import Sequence

from attrs import define

from oba.ag.common import Usage
from oba.ag.models.message import Message


@define
class SessionInfo:
    messages: list[Message]
    usage: Usage


class Memory(ABC):
    @abstractmethod
    def get_messages(self, session_id: str) -> Sequence[Message]: ...

    @abstractmethod
    def get_usage(self, session_id: str) -> Usage: ...

    @abstractmethod
    def extend(
        self,
        session_id: str,
        messages: Sequence[Message],
        usage: Usage,
    ) -> None: ...
