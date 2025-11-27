import inspect
from typing import Any, Awaitable, Callable

from attrs import define, field
from pydantic import BaseModel

ToolCallable = Callable[..., Awaitable[tuple[str, float]]] | Callable[..., Awaitable[str]]


def _wrap_in_async_if_needed(
    callable: ToolCallable | Callable[..., str],
) -> ToolCallable:
    if inspect.iscoroutinefunction(callable):
        return callable

    # we've already excluded the ToolCallable possibility for `callable` with the inspection above
    # so the only thing left to be is Callable[..., str]. Let's help pyright accept this fact.
    callable_sync: Callable[..., str] = callable  # pyright: ignore[reportAssignmentType]

    async def new_callable(*args: Any, **kwargs: Any) -> str:
        return callable_sync(*args, **kwargs)

    return new_callable


@define
class Tool:
    spec: type[BaseModel]
    callable: ToolCallable = field(converter=_wrap_in_async_if_needed)
