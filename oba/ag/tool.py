from typing import Callable, Coroutine

from attrs import define
from pydantic import BaseModel

ToolCallable = Callable[..., str] | Callable[..., Coroutine[None, None, str]]


@define
class Tool:
    spec: type[BaseModel]
    callable: ToolCallable
