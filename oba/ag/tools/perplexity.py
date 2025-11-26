import json
import os
import warnings

import httpx
from pydantic import BaseModel, Field

from ..tool import Tool


def create_web_search_tool(client: httpx.AsyncClient) -> Tool:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set")

    async def _web_search(query: str) -> str:
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
            return f"[WebSearch API returned status code {response.status_code}]"

        if results := response.json().get("results"):
            return json.dumps(results)
        else:
            raise AssertionError("WebSearch API returned no `results` key")

    return Tool(spec=WebSearch, callable=_web_search)


def create_agentic_web_search_tool(client: httpx.AsyncClient) -> Tool:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is not set")

    async def _agentic_web_search(query: str, enhanced: bool) -> str:
        response = await client.post(
            _PERPLEXITY_CHAT_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "sonar" if not enhanced else "sonar-reasoning-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "web_search_options": {
                    "search_context_size": "low" if not enhanced else "medium",
                },
            },
            timeout=20 if not enhanced else 45,
        )

        if not response.is_success:
            return f"[AgenticWebSearch API returned status code {response.status_code}]"

        data = response.json()

        # TODO: actually use this information
        _ = data["usage"]["cost"]["total_cost"]  # dollar cost

        # Citations are a list of links that are cited in the response by 1-based index. So in the
        # actual response we may have instances like "[2][3]", "[10]", etc. We can enhance this
        # slightly by also getting the title/date from the Search Results, which is always in the
        # same order anyway, making "citations" redundant.
        search_results: list[dict[str, str]] = data["search_results"]
        content: str = data["choices"][0]["message"]["content"]

        if enhanced:
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

        return f"<result>\n{content.strip()}\n</result>\n\n<references>\n{'\n'.join(enriched_citations)}\n</references>"

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
    Use this tool to have an agent search the web for information based on a query.
    The agent will read the actual contents itself and return compiled/relevant information in a
    digestible format, as well as provide references.

    The `enhanced` flag can help enable more capable reasoning agent that can read/search more
    sources, and provide more insighful answers.  Use the appropriate flag for the task at hand.

    Examples of simple queries: "latest news about apple", "what's the top 5 largest market cap companies", etc.
    Examples of enhanced queries: "what's the likelihood of Bolsonaro going to jail?", "what's the expected impact of the US government shutdown?", etc.
    """

    query: str = Field(description="The query/question/request for the agent to search the web for")

    enhanced: bool = Field(
        description="Whether to enable a more capable reasoning agent that can read/search more sources and provide more insightful answers",
    )


_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
_PERPLEXITY_CHAT_URL = "https://api.perplexity.ai/chat/completions"
