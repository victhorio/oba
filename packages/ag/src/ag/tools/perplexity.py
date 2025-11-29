import json
import os
import warnings

import httpx
from pydantic import BaseModel, Field

from ag.tool import Tool


def create_web_search_tool(client: httpx.AsyncClient) -> Tool:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set")

    WEB_SEARCH_FLAT_COST = 5 / 1000  # 5 USD per 1k requests

    async def _web_search(query: str) -> tuple[str, float]:
        response = await client.post(
            _PERPLEXITY_SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "max_results": 5,
                "max_tokens_per_page": 1024,
            },
            timeout=10,
        )

        if not response.is_success:
            return (
                f"[WebSearch API returned status code {response.status_code}]",
                0.0,
            )

        if results := response.json().get("results"):
            return json.dumps(results), WEB_SEARCH_FLAT_COST
        else:
            raise AssertionError("WebSearch API returned no `results` key")

    return Tool(spec=WebSearch, callable=_web_search)


def create_agentic_web_search_tool(client: httpx.AsyncClient) -> Tool:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set")

    async def _agentic_web_search(prompt: str, reasoning: bool) -> tuple[str, float]:
        response = await client.post(
            _PERPLEXITY_CHAT_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "sonar-pro" if not reasoning else "sonar-reasoning-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "web_search_options": {
                    "search_context_size": "low" if not reasoning else "medium",
                },
            },
            timeout=30 if not reasoning else 60,
        )

        if not response.is_success:
            return (
                f"[AgenticWebSearch API returned status code {response.status_code}]",
                0.0,
            )

        data = response.json()

        dollar_cost = data["usage"]["cost"]["total_cost"]

        # Citations are a list of links that are cited in the response by 1-based index. So in the
        # actual response we may have instances like "[2][3]", "[10]", etc. We can enhance this
        # slightly by also getting the title/date from the Search Results, which is always in the
        # same order anyway, making "citations" redundant.
        search_results: list[dict[str, str]] = data["search_results"]
        content: str = data["choices"][0]["message"]["content"]

        if reasoning:
            # we need to strip the `<think>...</think>` block of the content
            marker = "</think>"
            marker_idx = content.find(marker)
            if marker_idx == -1:
                warnings.warn("unexpectedly found response with no </think> block")
            else:
                content = content[marker_idx + len(marker) :]

        enriched_citations: list[str] = [
            f"- [{i + 1}] {sr['title']} ({sr.get('date', 'N/A')}) [{sr['url']}]"
            for i, sr in enumerate(search_results)
        ]

        return (
            f"<result>\n{content.strip()}\n</result>\n\n<references>\n{'\n'.join(enriched_citations)}\n</references>",
            dollar_cost,
        )

    return Tool(spec=AgenticWebSearch, callable=_agentic_web_search)


class WebSearch(BaseModel):
    """
    Use this tool to search information on the web based on a search query.

    Prefer to use specific queries. For example, "artificial intelligence medical diagnosis
    accuracy" is much better than "AI medical".


    You will get up to 1024 tokens worth of content for the top-5 most relevant results.
    """

    query: str = Field(description="The search query used for the web search")


class AgenticWebSearch(BaseModel):
    """
    This tool allows you to have an agent search the web for information based on a prompt.
    This leverages Perplexity's grounded Sonar LLM to search the web and provide a useful answer
    grounded in the search results that are also included in the response.

    Use `prompt` as a message to the Sonar LLM indicating its task/the answer it needs to provide.

    Use `reasoning` to enable a more capable reasoning agent that can search through more sources
    and provide more built-in inference on top of the search results. This is suitable for
    reasoning, inference and speculation on top of the search results - not for simple factual
    queries/information retrieval.
    """

    prompt: str = Field(
        description="The prompt for the agent to use to search the web and provide a useful answer to you"
    )

    reasoning: bool = Field(
        description="Whether to enable a more capable reasoning agent that can search through more sources and provide more built-in inference on top of the search results",
    )


_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
_PERPLEXITY_CHAT_URL = "https://api.perplexity.ai/chat/completions"
