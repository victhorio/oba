from abc import ABC, abstractmethod
from collections.abc import Sequence

from ag.common import Usage
from ag.models.message import Message


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

    @abstractmethod
    def add_tool_cost(self, session_id: str, cost: float) -> None: ...
