import os
from typing import Literal

from attrs import define
from httpx import AsyncClient

EmbeddingModelID = Literal["text-embedding-3-small", "text-embedding-3-large"]


@define
class EmbeddingsResult:
    embeddings: list[list[float]]
    dollar_cost: float


class OpenAIEmbeddings:
    def __init__(
        self,
        model_id: EmbeddingModelID,
        *,
        api_key: str | None = None,
        client: AsyncClient | None = None,
    ):
        self.model_id: EmbeddingModelID = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("either pass api_key or set OPENAI_API_KEY in env")

        self.client = client or AsyncClient()

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingsResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: dict[str, object] = {
            "input": inputs,
            "model": self.model_id,
            "encoding_format": "float",  # TODO: base64 for efficiency?
        }

        if dimensions:
            payload["dimensions"] = dimensions

        response = await self.client.post(
            _EMBEDDINGS_ENDPOINT,
            headers=headers,
            json=payload,
        )

        response.raise_for_status()

        response_dict = response.json()
        return EmbeddingsResult(
            embeddings=[data["embedding"] for data in response_dict["data"]],
            dollar_cost=_COST_PER_MODEL[self.model_id] * response_dict["usage"]["prompt_tokens"],
        )


_EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"
_COST_PER_MODEL: dict[EmbeddingModelID, float] = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,
}
