import json
import os

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
        )

        if not response.is_success:
            return f"[WebSearch API returned status code {response.status_code}]"

        if results := response.json().get("results"):
            return json.dumps(results)
        else:
            raise AssertionError("WebSearch API returned no `results` key")

    return Tool(spec=WebSearch, callable=_web_search)


class WebSearch(BaseModel):
    """
    Use this tool to search information on the web based on a search query.

    Prefer to use specific queries. For example, "artificial intelligence medical diagnosis
    accuracy" is much better than "AI medical".

    You will get up to 1024 tokens worth of content for the top-5 most relevant results.
    """

    query: str = Field(..., description="The search query used for the web search")


_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
