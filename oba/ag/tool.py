from typing import Callable

from pydantic import BaseModel

ToolCallable = Callable[..., str]


class Tool(BaseModel):
    spec: type[BaseModel]
    callable: ToolCallable


def tool(
    spec: type[BaseModel],
    callable: ToolCallable,
) -> Tool:
    return Tool(spec=spec, callable=callable)
