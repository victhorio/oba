import warnings

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.metrics import Metrics as AgnoMetrics
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from pydantic import BaseModel

from oba import prompts
from oba.configs import Config


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    @classmethod
    def from_agno(cls, agno_metrics: AgnoMetrics, model: str) -> "Usage":
        return cls(
            input_tokens=agno_metrics.input_tokens,
            output_tokens=agno_metrics.output_tokens,
            reasoning_tokens=agno_metrics.reasoning_tokens,
            total_tokens=agno_metrics.total_tokens,
            total_cost=cls._calculate_cost(agno_metrics, model),
        )

    @staticmethod
    def _calculate_cost(agno_metrics: AgnoMetrics, model: str) -> float:
        if model not in _MODEL_COSTS:
            warnings.warn(
                f"Unknown model '{model}' for cost calculation, defaulting to 0.0"
            )
            return 0.0

        costs = _MODEL_COSTS[model]
        input_cost = (agno_metrics.input_tokens / 1e6) * costs["input"]
        output_cost = (agno_metrics.output_tokens / 1e6) * costs["output"]
        return input_cost + output_cost


class Metrics(BaseModel):
    ttft: float | None
    duration: float | None


class Response(BaseModel):
    """
    Represents a processed response from the agent.
    """

    content: str
    usage: Usage
    metrics: Metrics


def new(settings: Config) -> Agent:
    model = OpenAIResponses(
        id=settings.model_id,
        system_prompt=prompts.load("system_prompt"),
        reasoning_effort="minimal",
    )

    return Agent(
        model=model,
        retries=0,
        markdown=False,
        db=InMemoryDb(),
        add_history_to_context=True,
        num_history_runs=settings.max_history_turns,
    )


async def send_message(agent: Agent, message: str) -> Response:
    assert agent.model is not None, "Agent model is not set"

    run_output: RunOutput = await agent.arun(input=message)  # type: ignore
    if not isinstance(run_output.content, str):
        raise ValueError("RunOutput.content is not a string")
    if not run_output.metrics:
        raise ValueError("RunOutput.metrics is None")

    return Response(
        content=run_output.content,
        usage=Usage.from_agno(run_output.metrics, agent.model.id),
        metrics=Metrics(
            ttft=run_output.metrics.time_to_first_token,
            duration=run_output.metrics.duration,
        ),
    )


_MODEL_COSTS = {
    "gpt-5-mini": {"input": 0.250, "output": 2.000},
}
