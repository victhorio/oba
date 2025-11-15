from typing import Callable

from attrs import define
from pydantic import BaseModel

ToolCallable = Callable[..., str]


@define
class Tool:
    spec: type[BaseModel]
    callable: ToolCallable
