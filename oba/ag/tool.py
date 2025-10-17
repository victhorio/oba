from textwrap import dedent
from typing import Callable
from pydantic import BaseModel
from openai.types.responses import FunctionToolParam


ToolCallable = Callable[..., str]


class Tool(BaseModel):
    spec: type[BaseModel]
    callable: ToolCallable
    _oai_spec: FunctionToolParam | None = None

    def to_oai(self) -> FunctionToolParam:
        if self._oai_spec:
            return self._oai_spec

        name = self.spec.__name__
        description = self.spec.__doc__ or ""
        description = dedent(description).strip()

        properties: dict[str, object] = self.spec.model_json_schema()["properties"]  # pyright: ignore[reportAny]
        parameters: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": [k for k in properties],
            "additionalProperties": False,
        }

        return FunctionToolParam(
            name=name,
            parameters=parameters,
            strict=True,
            type="function",
            description=description,
        )


def tool(
    spec: type[BaseModel],
    callable: ToolCallable,
) -> Tool:
    return Tool(spec=spec, callable=callable)
