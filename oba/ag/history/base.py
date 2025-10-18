from abc import ABC, abstractmethod
from collections.abc import Sequence

from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel

from oba.ag.common import Usage


class SessionInfo(BaseModel):
    messages: list[ResponseInputItemParam]
    usage: Usage


class HistoryDb(ABC):
    @abstractmethod
    def get_messages(self, session_id: str) -> Sequence[ResponseInputItemParam]: ...

    @abstractmethod
    def get_usage(self, session_id: str) -> Usage: ...

    @abstractmethod
    def extend(
        self,
        session_id: str,
        messages: Sequence[ResponseInputItemParam],
        usage: Usage,
    ) -> None: ...
