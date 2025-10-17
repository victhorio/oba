from collections.abc import Sequence

from openai.types.responses import ResponseInputItemParam


class HistoryDb:
    """
    A poc implementation, this should become an abstract base class and
    there should be different implementations, namely in-memory and sqlite.
    For now a very simple, non-configurable in-memory "db".
    """

    def __init__(self):
        # for each session_id we'll have a list of messages to be sent to the model
        self._db: dict[str, list[ResponseInputItemParam]] = dict()

    def get_history(self, session_id: str) -> Sequence[ResponseInputItemParam]:
        return tuple(self._db.get(session_id, []))

    def extend_history(self, session_id: str, messages: list[ResponseInputItemParam]) -> None:
        if session_id not in self._db:
            self._db[session_id] = []
        self._db[session_id].extend(messages)
