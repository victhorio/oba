import inspect
from typing import Any, Awaitable, Callable

from attrs import define, field
from pydantic import BaseModel

ToolCallable = Callable[..., Awaitable[str]]


def _wrap_in_async_if_needed(
    callable: Callable[..., str] | Callable[..., Awaitable[str]],
) -> ToolCallable:
    if inspect.iscoroutinefunction(callable):
        return callable

    # we've already checked above if it was Callable[..., Awaitable[str]] but
    # pyright isn't smart enough to know, so we use the line below to "force"
    # it to understand
    callable_sync: Callable[..., str] = callable  # pyright: ignore[reportAssignmentType]

    async def new_callable(*args: Any, **kwargs: Any) -> str:
        return callable_sync(*args, **kwargs)

    return new_callable


@define
class Tool:
    spec: type[BaseModel]
    callable: ToolCallable = field(converter=_wrap_in_async_if_needed)
