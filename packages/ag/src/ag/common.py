from attrs import define


@define
class Usage:
    input_tokens: int = 0
    input_tokens_cached: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    tool_costs: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            input_tokens_cached=self.input_tokens_cached + other.input_tokens_cached,
            output_tokens=self.output_tokens + other.output_tokens,
            total_cost=self.total_cost + other.total_cost,
            tool_costs=self.tool_costs + other.tool_costs,
        )
